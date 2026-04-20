# conda activate biapy_env_test, which is biapy_env and the following packages:
# pip install gdown==5.1.0 --quiet
import sys
import os
import gdown
import urllib
import yaml
from zipfile import ZipFile
from subprocess import Popen
import numpy as np
import argparse
import time 
import requests
import collections.abc

# ---------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Check BiaPy code consistency",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--out_folder", type=str, help="Output folder")
parser.add_argument("--gpus", type=str, help="GPUs to use")
parser.add_argument("--biapy_folder", default="", help="BiaPy code directory to test")
args = parser.parse_args()

gpu = args.gpus.split(",")[0] 
gpus = args.gpus 
gpu_list = gpus.split(",")
ngpus = len(gpu_list)

print(f"Using GPU: '{gpu}' (single-gpu checks) ; GPUs: '{gpus}' (multi-gpu checks)")

data_folder = os.path.join(os.getcwd(), args.out_folder) 
biapy_folder = os.getcwd() if args.biapy_folder == "" else args.biapy_folder

print(f"Out folder: {data_folder}")
print(f"Running in folder: {biapy_folder}")

bmz_folder = os.path.join(data_folder, "bmz_files")
bmz_script = os.path.join(biapy_folder, "biapy", "utils", "scripts", "export_bmz_test.py")
results_folder = os.path.join(data_folder, "output")

if not os.path.exists(biapy_folder):
    raise ValueError(f"BiaPy not found in: {biapy_folder}")

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def download_drive_file(drive_link, out_filename, attempts=5):
    """ Try a few times to download a file from Drive using gdown """
    for i in range(attempts):
        print(f"Trying to download {drive_link} (attempt {i+1})")
        try:
            gdown.download(drive_link, out_filename, quiet=True)
        except Exception as e:
            print(e)
            time.sleep(5)
        if os.path.exists(out_filename):
            break

def download_onedrive_file(drive_link, out_filename, attempts=5):
    for i in range(attempts):
        print(f"Trying to download {drive_link} (attempt {i+1})")
        try:
            response = requests.get(drive_link, stream=True)
            if response.status_code == 200:
                with open(out_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                print(f"Failed to download. Status code: {response.status_code}")
        except Exception as e:
            print(e)
            time.sleep(5)
        if os.path.exists(out_filename):
            break

def check_bmz_file_created(last_lines, pattern_to_find, bmz_pck_to_find):
    """
    Checks BMZ model creation. E.g. "Package path: *.zip"
    """
    text_found = False
    for line in last_lines:
        if pattern_to_find in line and "zip" in line:
            text_found = True

    package_found = True if os.path.exists(bmz_pck_to_find) else False
    return (text_found and package_found)

def check_bmz_weight_agreement(last_lines, pattern_to_find):
    """
    Checks BMZ model weight agreement. E.g. "weights.pytorch_state_dict" in the logs.
    Returns False if there is an error (i.e. the pattern is found but it is related to a disagreement), False otherwise.
    """
    for line in last_lines:
        if pattern_to_find in line:
            if "❌" in line or "disagrees with" in line:
                return False

    return True

def check_value(last_lines, pattern_to_find, ref_value, gt=True):
    """
    Checks just one value. E.g. 'Test Foreground IoU (merge patches): 0.45622628648145197' "
    """
    finished_good = False
    for line in last_lines:
        if pattern_to_find in line:
            val = float(line.split(' ')[-1].replace('\n',''))
            if gt and val >= ref_value:
                finished_good = True
            elif not gt and val < ref_value:
                finished_good = True
            break
    return finished_good

def check_DatasetMatching(last_lines, pattern_to_find, ref_value, gt=True, value_to_check=1, metric="f1"):
    """
    Check just one value that can appear more than once (control the one you want to check with 'value_to_check').
    E.g.:
        DatasetMatching(criterion='iou', thresh=0.3, fp=1262, tp=357, fn=78, precision=0.2205064854848672,
        recall=0.8206896551724138, accuracy=0.2103712433706541, f1=0.3476144109055501, n_true=435,
        n_pred=1619, mean_true_score=0.5633060849946121, mean_matched_score=0.6863813640690651,
        panoptic_quality=0.23859605352741603, by_image=False)
    """
    finished_good = False
    c = 1
    for line in last_lines:
        if pattern_to_find in line:
            if c == value_to_check:
                for part in line.split(' '):
                    if metric in part:
                        val = float(part.split('=')[-1][:-1])
                        if gt and val >= ref_value:
                            finished_good = True
                        elif not gt and val < ref_value:
                            finished_good = True
                        break
            else:
                c += 1
    return finished_good

def check_finished(test_info):
    # get the last lines of the output file
    jobdir = os.path.join(results_folder, test_info["jobname"])
    jobout_file = os.path.join(jobdir, test_info["jobname"]+"_1")
    logfile = open(jobout_file, 'r')
    last_lines = logfile.readlines()
    last_lines = last_lines[-min(300,len(last_lines)):]

    # Check if the final message appears there
    finished_good = False
    for line in last_lines:
        if "FINISHED JOB" in line:
            finished_good = True
            break
    logfile.close()

    return finished_good, last_lines

def print_result(finished_good, jobname, int_checks):
    # Print the final message to the user accordingly
    if all(finished_good):
        print(f"** {jobname} job: [OK] ({int_checks} internal checks passed)")
    else:
        print(f"** {jobname} job: [ERROR] ({sum(finished_good)} of {int_checks} internal checks passed)")
    print("######")


def runjob(test_info, results_folder, yaml_file, biapy_folder, multigpu=False, bmz_by_command=False, bmz_package=None, reuse_original_bmz_config=False):
    # Declare the log file
    jobdir = os.path.join(results_folder, test_info["jobname"])
    jobout_file = os.path.join(jobdir, test_info["jobname"]+"_1")
    os.makedirs(jobdir, exist_ok=True)
    logfile = open(jobout_file, 'w')
    
    # Run the process and wait until finishes
    os.chdir(biapy_folder)
    print(f"Log: {jobout_file}")
    if bmz_by_command and bmz_package is not None:
        os.makedirs(bmz_folder, exist_ok=True)
        cmd = ["python", "-u", bmz_script, 
               "--code_dir", biapy_folder,
               "--jobname", test_info["jobname"],
               "--config", yaml_file, 
               "--result_dir", results_folder, 
               "--model_name", str(bmz_package.split(".")[:-1][0]),
               "--bmz_folder", bmz_folder,
               "--gpu", gpu]
        if reuse_original_bmz_config:
            cmd += ["--reuse_original_bmz_config"]
    else:
        if multigpu:
            cmd = ["python", "-u", "-m", "torch.distributed.run", "--nproc_per_node="+str(ngpus),
                f"--master-port={np.random.randint(low=1500, high=7000, size=1)[0]}", "main.py",
                "--config", yaml_file, "--result_dir", results_folder, "--name", test_info["jobname"], "--run_id", "1",
                "--gpu", gpus]
        else:
            cmd = ["python", "-u", "main.py", "--config", yaml_file, "--result_dir", results_folder, "--name",
                test_info["jobname"], "--run_id", "1", "--gpu", gpu]
            
    print(f"Command: {' '.join(cmd)}")
    print("Running job . . .")
    process = Popen(cmd, stdout=logfile, stderr=logfile)
    process.wait()

    logfile.close()

def update_nested_dict(d, u):
    """
    Recursively updates a nested dictionary. 
    Allows overwriting specific YAML keys without clearing out the rest of the parent block.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

# ---------------------------------------------------------
# TEST DEFINITIONS
# ---------------------------------------------------------       
all_test_info = {}
all_test_info["Test1"] = {
    "enable": True,
    "jobname": "test1",
    "description": "2D Semantic seg. Lucchi++. Basic DA. unet. 2D stack as 3D. Post-proc: z-filtering. BMZ export through YAML.",
    "yaml": "test_1.yaml",
    "yaml_modifications": {
        "DATA": {
            "TRAIN": {
                "PATH": os.path.join(data_folder, "semantic_seg", "fibsem_epfl_2D", "train", "raw"),
                "GT_PATH": os.path.join(data_folder, "semantic_seg", "fibsem_epfl_2D", "train", "label"),
                "IN_MEMORY": True,
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "semantic_seg", "fibsem_epfl_2D", "test", "raw"),
                "GT_PATH": os.path.join(data_folder, "semantic_seg", "fibsem_epfl_2D", "test", "label"),
                "IN_MEMORY": False,
                "LOAD_GT": True,
            },
        },
        "AUGMENTOR": {
            "CONTRAST": True,
            "BRIGHTNESS": True
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 4,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "hrnet32",
            "BMZ": {
                "EXPORT": {
                    "ENABLE": True,
                    "MODEL_NAME": "lucchi2Dsegmentation",
                    "DESCRIPTION": "2D mitochondria segmentation",
                    "AUTHORS": [{"name": "Daniel Franco-Barranco", "github_user": "danifranco"}],
                    'TAGS': ["mitochondria",  "electron microscopy"],
                    "CITE": [{"text": "model", "doi": "10.1109/TPAMI.2020.2983686"}, {"text": "training library", "doi": "10.1038/s41592-025-02699-y"}],
                    'DATASET_INFO': [{"name": "lucchi++", "doi": "10.5281/zenodo.17829532", "image_modality": "electron microscopy"}],
                    'MODEL_VERSION': "0.1.0"
                },  
            },
        },
        "LOSS": {   
            "TYPE": "W_CE_DICE",
        },
        "TEST": {
            "ENABLE": True,
            "AUGMENTATION": True,
            "FULL_IMG": True,
            "ANALIZE_2D_IMGS_AS_3D_STACK": True,
            "POST_PROCESSING": {
                "MEDIAN_FILTER": True,
                "MEDIAN_FILTER_AXIS": ["z"],
                "MEDIAN_FILTER_SIZE": [5]
            },
        },
    },
    "internal_checks": [
        {"type": "regular", "pattern": "Test Foreground IoU (per image)", "gt": True, "value": 0.7},
        {"type": "regular", "pattern": "Test Foreground IoU (as 3D stack - post-processing)", "gt": True, "value": 0.7},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "lucchi2Dsegmentation.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},     
    ],
}

all_test_info["Test2"] = {
    "enable": True,
    "jobname": "test2",
    "description": "3D Semantic seg. Lucchi++. attention_unet. Basic DA.",
    "yaml": "test_2.yaml",
    "yaml_modifications": {
        "AUGMENTOR": {
            "CONTRAST": True,
            "BRIGHTNESS": True
        },
        "DATA": {
            "TRAIN": {
                "PATH": os.path.join(data_folder, "semantic_seg", "fibsem_epfl_3D", "train", "raw"),
                "GT_PATH": os.path.join(data_folder, "semantic_seg", "fibsem_epfl_3D", "train", "label"),
                "IN_MEMORY": True,
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "semantic_seg", "fibsem_epfl_3D", "test", "raw"),
                "GT_PATH": os.path.join(data_folder, "semantic_seg", "fibsem_epfl_3D", "test", "label"),
                "IN_MEMORY": False,
                "LOAD_GT": True,
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 5,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "stunet",
            "STUNET": {
                "VARIANT": "base",
                "PRETRAINED": True
            },
        },
        "TEST": {
            "ENABLE": True,
            "REDUCE_MEMORY": True,
        }
    },
    "bmz_by_command": True,
    "bmz_package": "lucchi3Dsegmentation.zip",
    "internal_checks": [
        {"pattern": "Test Foreground IoU (merge patches)", "gt": True, "value": 0.50},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "lucchi3Dsegmentation.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},    
    ]
}

all_test_info["Test3"] = {
    "enable": True,
    "jobname": "test3",
    "description": "2D Instance seg. Stardist 2D data. Basic DA. BC (auto). resunet++. "
        "Post-proc: Clear border + remove instances by properties.",
    "yaml": "test_3.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "TYPE": "INSTANCE_SEG",
            "INSTANCE_SEG": {
                "DATA_CHANNELS": "BC",
                "DATA_MW_TH_TYPE": "auto",
                "DATA_CHANNEL_WEIGHTS": "(0.5, 1)",
                "WATERSHED": {
                    "DATA_REMOVE_BEFORE_MW": True,
                    "DATA_REMOVE_SMALL_OBJ_BEFORE": 10,
                },
            },  
        },      
        "DATA": {
            "TRAIN": {
                "PATH": os.path.join(data_folder, "instance_seg", "Stardist_v2_2D", "train", "raw"),
                "GT_PATH": os.path.join(data_folder, "instance_seg", "Stardist_v2_2D", "train", "label"),
                "IN_MEMORY": True,
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "instance_seg", "Stardist_v2_2D", "test", "raw"),
                "GT_PATH": os.path.join(data_folder, "instance_seg", "Stardist_v2_2D", "test", "label"),
                "IN_MEMORY": False,
                "LOAD_GT": True,
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 20,
            "PATIENCE": -1,
        },
        "AUGMENTOR": {
            "CONTRAST": True,
            "BRIGHTNESS": True
        },
        "MODEL": {
            "ARCHITECTURE": "resunet++",
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
            "POST_PROCESSING": {
                "CLEAR_BORDER": True, 
                "MEASURE_PROPERTIES": {
                    "ENABLE": True,
                    "REMOVE_BY_PROPERTIES": {
                        "ENABLE": True,
                        "PROPS": [['circularity', 'area']],
                        "VALUES": [[0.2, 100]],
                        "SIGNS": [['lt', 'lt']]
                    }
                }
            },
        },        
    },
    "bmz_by_command": True,
    "bmz_package": "stardist2D_instance_segmentation.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Test IoU (F channel) (merge patches):", "gt": True, "value": 0.4},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1", "gt": True, "value": 0.8},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 2, "metric": "f1", "gt": True, "value": 0.7},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "stardist2D_instance_segmentation.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},  
    ]
}

all_test_info["Test4"] = {
    "enable": True,
    "jobname": "test4",
    "description": "3D Instance seg. Demo 3D data. Basic DA. BCD (manual). resunet. Watershed multiple options. Post-proc: Clear border",
    "yaml": "test_4.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "TYPE": "INSTANCE_SEG",
            "INSTANCE_SEG": {
                "DATA_CHANNELS": "BCD",
                "DATA_MW_TH_TYPE": "manual",
                "DATA_CHANNEL_WEIGHTS": "(0.5, 1, 1)",
                "DATA_MW_TH_BINARY_MASK": 0.4,
                "DATA_MW_TH_CONTOUR": 0.25, 
                "DATA_MW_TH_DISTANCE": 0.5,
                "DATA_REMOVE_SMALL_OBJ_BEFORE": 20,
                "DATA_REMOVE_BEFORE_MW": True,
                "SEED_MORPH_SEQUENCE": ['erode','dilate'],
                "SEED_MORPH_RADIUS": [2,2],
                "FORE_EROSION_RADIUS": 4,
                "FORE_DILATION_RADIUS": 4,
                "DATA_CHECK_MW": True,
                "WATERSHED_BY_2D_SLICES": True
            }
        },
        "AUGMENTOR": {
            "ELASTIC": True
        },
        "DATA": {
            "TRAIN": {
                "PATH": os.path.join(data_folder, "instance_seg", "demo3D_3D", "train", "raw"),
                "GT_PATH": os.path.join(data_folder, "instance_seg", "demo3D_3D", "train", "label"),
                "IN_MEMORY": True,
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "instance_seg", "demo3D_3D", "test", "raw"),
                "GT_PATH": os.path.join(data_folder, "instance_seg", "demo3D_3D", "test", "label"),
                "IN_MEMORY": False,
                "LOAD_GT": True
            }
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 20,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "resunet++",
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
            "POST_PROCESSING": {
                "CLEAR_BORDER": True
            }
        },
    },
    "bmz_by_command": True,
    "bmz_package": "demo3D_instance_segmentation.zip",
    "internal_checks": [
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1",
            "gt": True, "value": 0.7},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 2, "metric": "f1",
            "gt": True, "value": 0.7}, # Post-processing
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "demo3D_instance_segmentation.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},  
    ]
}

all_test_info["Test5"] = {
    "enable": True,
    "jobname": "test5",
    "description": "3D Instance seg. Cyst data. Basic DA. BCM (auto). resunet. Post-proc: Clear border + Voronoi + remove by props",
    "yaml": "test_5.yaml",
    "yaml_modifications": {
        'PROBLEM': {
            'INSTANCE_SEG': {
                'DATA_CHANNELS': 'BCM',
                'DATA_MW_TH_TYPE': "auto",
                'DATA_REMOVE_SMALL_OBJ_BEFORE': 10,
                'DATA_REMOVE_BEFORE_MW': True,
                'SEED_MORPH_SEQUENCE': ['erode','dilate'],
                'SEED_MORPH_RADIUS': [2,2],
                'FORE_EROSION_RADIUS': 4,
                'FORE_DILATION_RADIUS': 4,
            },
        },
        "DATA": {
            'REFLECT_TO_COMPLETE_SHAPE': True,
            'PATCH_SIZE': "(80, 80, 80, 1)",
            "TRAIN": {
                "PATH": os.path.join(data_folder, "instance_seg", "CartoCell", "train", "raw"),
                "GT_PATH": os.path.join(data_folder, "instance_seg", "CartoCell", "train", "label"),
                "IN_MEMORY": True,
            },
            "VAL": {
                "PATH": os.path.join(data_folder, "instance_seg", "CartoCell", "validation", "raw"),
                "GT_PATH": os.path.join(data_folder, "instance_seg", "CartoCell", "validation", "label"),
                "IN_MEMORY": True,
                "FROM_TRAIN": False
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "instance_seg", "CartoCell", "test", "raw"),
                "GT_PATH": os.path.join(data_folder, "instance_seg", "CartoCell", "test", "label"),
                "IN_MEMORY": False,
                "LOAD_GT": True
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 3,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "resunet",
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
            "POST_PROCESSING": {
                "CLEAR_BORDER": True,
                "VORONOI_ON_MASK": True,
                "MEASURE_PROPERTIES": {
                    "ENABLE": True,
                    "REMOVE_BY_PROPERTIES": {
                        "ENABLE": True,
                        "PROPS": [['sphericity'], ['area']],
                        "VALUES": [[0.3], [100]],
                        "SIGNS": [['lt'], ['lt']]
                    }
                }
            }
        },    
    },
    "internal_checks": [
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1",
            "gt": True, "value": 0.4},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 2, "metric": "f1",
            "gt": True, "value": 0.1}, # Post-processing
    ]
}

all_test_info["Test6"] = {
    "enable": True,
    "jobname": "test6",
    "description": "2D Detection. Stardist v2 2D data. zero_mean_unit_variance norm, percentile clip. Basic DA. Export model to BMZ. "
        "multiresunet. Post-proc: remove close points + det watershed",
    "yaml": "test_6.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "TYPE": "DETECTION",
            "DETECTION": {
                "CENTRAL_POINT_DILATION": [3],
                "CHECK_POINTS_CREATED": False,
                "DATA_CHECK_MW": True
            },
        },
        "DATA": {
            "NORMALIZATION": {
                "PERC_CLIP": {
                    "ENABLE": True,
                    "LOWER_PERC": 0.1,
                    "UPPER_PERC": 99.8
                },
                "TYPE": "zero_mean_unit_variance"
            },
            "TRAIN": {
                "PATH": os.path.join(data_folder, "detection", "Stardist_v2_detection", "train", "raw"),
                "GT_PATH": os.path.join(data_folder, "detection", "Stardist_v2_detection", "train", "label"),
                "IN_MEMORY": True,
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "detection", "Stardist_v2_detection", "test", "raw"),
                "GT_PATH": os.path.join(data_folder, "detection", "Stardist_v2_detection", "test", "label"),
                "IN_MEMORY": False,
                "LOAD_GT": True,
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 20,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "multiresunet",
            "LOAD_CHECKPOINT": False,
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
            "DET_POINT_CREATION_FUNCTION": "peak_local_max",
            "DET_PEAK_LOCAL_MAX_MIN_DISTANCE": 1,
            "DET_MIN_TH_TO_BE_PEAK": 0.7,
            "POST_PROCESSING": {
                "REMOVE_CLOSE_POINTS": True,
                "REMOVE_CLOSE_POINTS_RADIUS": 5,
                "DET_WATERSHED": True,
                "DET_WATERSHED_FIRST_DILATION": [2,2],
                "MEASURE_PROPERTIES": {
                    "ENABLE": True,
                    "REMOVE_BY_PROPERTIES": {
                        "ENABLE": True,
                        "PROPS": [['circularity']],
                        "VALUES": [[0.4]],
                        "SIGNS": [['lt']]
                    }
                }
            }
        },
    },
    "bmz_by_command": True,
    "bmz_package": "biapy_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Test Foreground IoU (merge patches):", "gt": True, "value": 0.5},
        {"type": "regular", "pattern": "Test F1 (merge patches)", "gt": True, "value": 0.85},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "biapy_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},  
    ]
}

all_test_info["Test8"] = {
    "enable": True,
    "jobname": "test8",
    "description": "3D Detection. NucMM-Z 3D data. zero_mean_unit_variance norm, percentile clip. Basic DA. "
        "unetr. Post-proc: remove close points + det watershed",
    "yaml": "test_8.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "DETECTION": {
                "CENTRAL_POINT_DILATION": [3],
                "CHECK_POINTS_CREATED": False,
                "DATA_CHECK_MW": True
            },
        },
       "DATA": {
            "NORMALIZATION": {
                "PERC_CLIP": {
                    "ENABLE": True,
                    "LOWER_PERC": 0.1,
                    "UPPER_PERC": 99.8
                },
                "TYPE": "zero_mean_unit_variance"
            },
            "TRAIN": {
                "PATH": os.path.join(data_folder, "detection", "NucMM-Z", "train", "raw"),
                "GT_PATH": os.path.join(data_folder, "detection", "NucMM-Z", "train", "label"),
                "IN_MEMORY": True,
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "detection", "NucMM-Z", "test", "raw"),
                "GT_PATH": os.path.join(data_folder, "detection", "NucMM-Z", "test", "label"),
                "IN_MEMORY": False,
                "LOAD_GT": True,
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 20,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "unetr",
            "LOAD_CHECKPOINT": False,
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
            "DET_POINT_CREATION_FUNCTION": "blob_log",
            "DET_PEAK_LOCAL_MAX_MIN_DISTANCE": 1,
            "DET_MIN_TH_TO_BE_PEAK": 0.7,
            "DET_TOLERANCE": 8,
            "POST_PROCESSING": {
                "REMOVE_CLOSE_POINTS": True,
                "REMOVE_CLOSE_POINTS_RADIUS": 3,
                "DET_WATERSHED": True,
                "DET_WATERSHED_FIRST_DILATION": [2,2,1],
                "MEASURE_PROPERTIES": {
                    "ENABLE": True,
                    "REMOVE_BY_PROPERTIES": {
                        "ENABLE": True,
                        "PROPS": [['sphericity']],
                        "VALUES": [[0.5]],
                        "SIGNS": [['lt']]
                    }
                }
            }
        },
    },
    "bmz_by_command": True,
    "bmz_package": "NucMM-Z_3D_detection.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Test F1 (merge patches)", "gt": True, "value": 0.2},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "NucMM-Z_3D_detection.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},  
    ]
}

all_test_info["Test9"] = {
    "enable": True,
    "jobname": "test9",
    "description": "3D Detection. Zarr 3D data (Brainglobe). zero_mean_unit_variance norm, percentile norm, per image. "
        "filter samples: foreground + mean. warmupcosine. Basic DA. resunet. test by chunks: Zarr. Post-proc: remove close points",
    "yaml": "test_9.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "DETECTION": {
                "CENTRAL_POINT_DILATION": [2],
                "CHECK_POINTS_CREATED": False,
            },
        },
       "DATA": {
            "PATCH_SIZE": "(20, 128, 128, 2)",
            "NORMALIZATION": {
                "TYPE": "zero_mean_unit_variance",
                "PERC_CLIP": {
                    "ENABLE": True,
                    "LOWER_PERC": 0.1,
                    "UPPER_PERC": 99.8
                }
            },
            "TRAIN": {
                "INPUT_IMG_AXES_ORDER": 'ZYXC',
                "FILTER_SAMPLES": {
                    "ENABLE": True,
                    "PROPS": [['foreground']],
                    "VALUES": [[1.0e-22]],
                    "SIGNS": [["lt"]]
                },
                "PATH": os.path.join(data_folder, "detection", "Brainglobe", "train", "raw"),
                "GT_PATH": os.path.join(data_folder, "detection", "Brainglobe", "train", "label"),   
                "IN_MEMORY": True,
            },
            "VAL": {
                "SPLIT_TRAIN": 0.1
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "detection", "Brainglobe", "test", "raw"),
                "GT_PATH": os.path.join(data_folder, "detection", "Brainglobe", "test", "label"),    
                "IN_MEMORY": False,
                "LOAD_GT": True,
                "PADDING": "(0,18,18)"  
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 100,
            "BATCH_SIZE": 2,
            "PATIENCE": 50,
            "LR": 0.0008,
            "LR_SCHEDULER": {
                "NAME": 'warmupcosine',
                "MIN_LR": 0.0001,
                "WARMUP_COSINE_DECAY_EPOCHS": 15
            },
        },
        "MODEL": { 
            "ARCHITECTURE": "hrnet18",
            "HRNET": {
                "Z_DOWN": False
            },
            "LOAD_CHECKPOINT": False,
        },
        "LOSS": {
            "CLASS_REBALANCE": True
        },
        "AUGMENTOR": {
            "RANDOM_ROT": True,
            "AFFINE_MODE": 'reflect',
            "ZFLIP": True
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
            "DET_MIN_TH_TO_BE_PEAK": 0.5,
            "DET_TOLERANCE": 8,
            "VERBOSE": True,
            "BY_CHUNKS": {
                "ENABLE": True,
                "FORMAT": "Zarr",
                "SAVE_OUT_TIF": True,
                "INPUT_IMG_AXES_ORDER": 'ZYXC',
                "WORKFLOW_PROCESS": {
                    "ENABLE": True,
                    "TYPE": "chunk_by_chunk"
                }
            },
            "POST_PROCESSING": {
                "REMOVE_CLOSE_POINTS": True,
                "REMOVE_CLOSE_POINTS_RADIUS": 3,
            },
        },
    },
    "internal_checks": [
        {"type": "regular", "pattern": "Test F1 (merge patches)", "gt": True, "value": 0.15},
    ]
}

all_test_info["Test10"] = {
    "enable": True,
    "jobname": "test10",
    "description": "2D Denoising. LongBeach data (N2V RGB data). zero_mean_unit_variance norm. Basic DA.",
    "yaml": "test_10.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "DENOISING": {
                "N2V_STRUCTMASK": True,
            },
        },
        "DATA": {
            "PATCH_SIZE": "(64, 64, 3)",
            "NORMALIZATION": {
                "TYPE": "zero_mean_unit_variance"
            },
            "TRAIN": {
                "PATH": os.path.join(data_folder, "denoising", "Noise2Void_RGB", "train"),
                "IN_MEMORY": False,
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "denoising", "Noise2Void_RGB", "test"),
                "IN_MEMORY": False,
                "LOAD_GT": True,
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 10,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "unet",
            "LOAD_CHECKPOINT": False,
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
        },
    }, 
    "bmz_by_command": True,
    "bmz_package": "long_beach_denoising.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation MSE:", "gt": False, "value": 0.6},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "long_beach_denoising.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"}, 
    ]
}

all_test_info["Test11"] = {
    "enable": True,
    "jobname": "test11",
    "description": "3D Denoising. Flywing 3D data. zero_mean_unit_variance norm. Basic DA. "
        "resunet. Post-proc: remove close points + det watershed",
    "yaml": "test_11.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "DENOISING": {
                "N2V_STRUCTMASK": True,
            },
        },
        "DATA": {
            "NORMALIZATION": {
                "TYPE": "zero_mean_unit_variance"
            },
            "TRAIN": {
                "PATH": os.path.join(data_folder, "denoising", "flywing3D", "train"),
                "IN_MEMORY": True,
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "denoising", "flywing3D", "test"),
                "IN_MEMORY": True,
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 10,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "unet",
            "FEATURE_MAPS": [16, 32, 64, 128, 256],
            "LOAD_CHECKPOINT": False,
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
        },
    },
    "bmz_by_command": True,
    "bmz_package": "flywing3D.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation MSE:", "gt": False, "value": 0.6},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "flywing3D.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},    
    ]
}

all_test_info["Test12"] = {
    "enable": True,
    "jobname": "test12",
    "description": "2D super-resolution. SR 2D data. Cross-val. Basic DA. DFCAN",
    "yaml": "test_12.yaml",
    "yaml_modifications": {
        "DATA": {
            "TRAIN": {
                "PATH": os.path.join(data_folder, "sr", "sr_data_2D", "train", "LR"),
                "GT_PATH": os.path.join(data_folder, "sr", "sr_data_2D", "train", "HR"),
                "IN_MEMORY": True,
                "FILTER_SAMPLES": {
                    "ENABLE": True,
                    "PROPS": [['mean']],
                    "VALUES": [[10000]],
                    "SIGNS": [["lt"]]
                }
            },
            "VAL": {
                "CROSS_VAL": True,
                "CROSS_VAL_NFOLD": 5,
                "CROSS_VAL_FOLD": 2
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "sr", "sr_data_2D", "test", "LR"),
                "GT_PATH": os.path.join(data_folder, "sr", "sr_data_2D", "test", "HR"),
                "IN_MEMORY": False,
                "USE_VAL_AS_TEST": True
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 20,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "dfcan",
            "LOAD_CHECKPOINT": False,
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
        },
    },
    "bmz_by_command": True,
    "bmz_package": "biapy_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 22.0},
        {"type": "regular", "pattern": "Test PSNR (merge patches)", "gt": True, "value": 23.0},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "biapy_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test13"] = {
    "enable": True,
    "jobname": "test13",
    "description": "3D super-resolution. SR 3D data. Cross-val. Basic DA. resunet. one-cycle",
    "yaml": "test_13.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "SUPER_RESOLUTION": {
                "UPSCALING": "(1,1,1)"
            }
        },
        "DATA": {
            "PATCH_SIZE": "(6,128,128,1)",
            "TRAIN": {
                "PATH": os.path.join(data_folder, "sr", "sr_data_3D", "train", "LR"),
                "GT_PATH": os.path.join(data_folder, "sr", "sr_data_3D", "train", "HR"),
                "IN_MEMORY": True,
            },
            "VAL": {
                "CROSS_VAL": True,
                "CROSS_VAL_NFOLD": 5,
                "CROSS_VAL_FOLD": 4
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "sr", "sr_data_3D", "test", "LR"),
                "GT_PATH": os.path.join(data_folder, "sr", "sr_data_3D", "test", "HR"),
                "IN_MEMORY": False,
                "USE_VAL_AS_TEST": True
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 20,
            "PATIENCE": -1,
            "LR_SCHEDULER": {
                "NAME": 'onecycle'
            },
            "LR": 0.001,
            "BATCH_SIZE": 16
        },
        "MODEL": {
            "ARCHITECTURE": "resunet",
            "Z_DOWN": [1,1],
            "LOAD_CHECKPOINT": False,
            "UNET_SR_UPSAMPLE_POSITION": "post",
            "FEATURE_MAPS": [32, 64, 128]    
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
        },
    },
    "bmz_by_command": True,
    "bmz_package": "biapy_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 20.0},
        {"type": "regular", "pattern": "Test PSNR (merge patches)", "gt": True, "value": 20.0},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "biapy_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test14"] = {
    "enable": True,
    "jobname": "test14",
    "description": "2D self-supervision. Lucchi data. Cross-val. Basic DA. rcan. Export BMZ model.",
    "yaml": "test_14.yaml",
    "yaml_modifications": {
        "DATA": {
            "NORMALIZATION": {
                "TYPE": "div",
            },
            "PATCH_SIZE": "(256,256,1)",
            "TRAIN": {
                "PATH": os.path.join(data_folder, "ssl", "fibsem_epfl_2D", "data", "train", "raw"),
                "IN_MEMORY": True,
            },
            "VAL": {
                "CROSS_VAL": True,
                "CROSS_VAL_NFOLD": 5,
                "CROSS_VAL_FOLD": 1
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "ssl", "fibsem_epfl_2D", "data", "test", "raw"),
                "IN_MEMORY": False,
                "USE_VAL_AS_TEST": True,
                "LOAD_GT": True
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 1,
            "PATIENCE": 1,
            "BATCH_SIZE": 6
        },
        "MODEL": {
            "ARCHITECTURE": "rcan",
            "LOAD_CHECKPOINT": True
        },
        "TEST": {
            "ENABLE": True,
        },
        "PATHS": {
            "CHECKPOINT_FILE": os.path.join(data_folder, "checkpoints", "test14_checkpoint.pth")
        }
    },
    "bmz_by_command": True,
    "bmz_package": "biapy_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 19.0},
        {"type": "regular", "pattern": "Test PSNR (merge patches):", "gt": True, "value": 22.0},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "biapy_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test15"] = {
    "enable": True,
    "jobname": "test15",
    "description": "2D self-supervision. Lucchi data. Cross-val. Basic DA. mae, masking: random",
    "yaml": "test_15.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "SELF_SUPERVISED": {
                "PRETEXT_TASK": "masking",
            }
        },
        "DATA": {
            "PATCH_SIZE": "(128,128,1)",
            "TRAIN": {
                "PATH": os.path.join(data_folder, "ssl", "fibsem_epfl_2D", "data", "train", "raw"),
                "IN_MEMORY": True,
            },
            "VAL": {
                "CROSS_VAL": True,
                "CROSS_VAL_NFOLD": 5,
                "CROSS_VAL_FOLD": 1
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "ssl", "fibsem_epfl_2D", "data", "test", "raw"),
                "IN_MEMORY": False,
                "USE_VAL_AS_TEST": True,
                "LOAD_GT": True
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 20,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "mae",
            "MAE_MASK_TYPE": "random",
            "LOAD_CHECKPOINT": False,
        },
        "TEST": {
            "ENABLE": True,
        },
    },
    "bmz_by_command": True,
    "bmz_package": "biapy_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 12},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "biapy_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test16"] = {
    "enable": True,
    "jobname": "test16",
    "description": "2D self-supervision. Lucchi data. Cross-val. Basic DA. mae, masking: grid",
    "yaml": "test16.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "SELF_SUPERVISED": {
                "PRETEXT_TASK": "masking",
            }
        },
        "DATA": {
            "PATCH_SIZE": "(128,128,1)",
            "TRAIN": {
                "PATH": os.path.join(data_folder, "ssl", "fibsem_epfl_2D", "data", "train", "raw"),
                "IN_MEMORY": True,
            },
            "VAL": {
                "CROSS_VAL": True,
                "CROSS_VAL_NFOLD": 5,
                "CROSS_VAL_FOLD": 1
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "ssl", "fibsem_epfl_2D", "data", "test", "raw"),
                "IN_MEMORY": False,
                "USE_VAL_AS_TEST": True,
                "LOAD_GT": True
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 20,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "mae",
            "MAE_MASK_TYPE": "grid",
            "LOAD_CHECKPOINT": False,
        },
        "TEST": {
            "ENABLE": True,
        },
    },
    "bmz_by_command": True,
    "bmz_package": "biapy_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 12},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "biapy_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test17"] = {
    "enable": True,
    "jobname": "test17",
    "description": "3D self-supervision. Lucchi data. Basic DA. resunet++",
    "yaml": "test17.yaml",
    "yaml_modifications": {
        'PROBLEM': {
            'SELF_SUPERVISED': {
                'PRETEXT_TASK': 'crappify',
            }
        },
        'DATA': { 
            'PATCH_SIZE': "(20,128,128,1)",
            'TRAIN': {
                'PATH': os.path.join(data_folder, "ssl", "fibsem_epfl_3D", "data", "train", "raw"),
                'IN_MEMORY': True,
            },
            'TEST': {
                'PATH': os.path.join(data_folder, "ssl", "fibsem_epfl_3D", "data", "test", "raw"),
                'IN_MEMORY': False,
                'PADDING': "(4,16,16)"
            },
        },
        'TRAIN': {
            'ENABLE': True,
            'EPOCHS': 20, 
            'BATCH_SIZE': 1,
            'PATIENCE': -1
        },
        'MODEL': {
            'ARCHITECTURE': 'resunet++',
            'Z_DOWN': [1,1,1,1],
            'LOAD_CHECKPOINT': False
        },
        'TEST': {
            'ENABLE': True,
        }
    },
    "bmz_by_command": True,
    "bmz_package": "biapy_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 10.0},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "biapy_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test18"] = {
    "enable": True,
    "jobname": "test18",
    "description": "3D self-supervision. Lucchi data. Cross-val. Basic DA. mae, masking: random",
    "yaml": "test18.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "SELF_SUPERVISED": {
                "PRETEXT_TASK": "masking",
            }
        },
        "DATA": {
            "PATCH_SIZE": "(20,128,128,1)",
            "TRAIN": {
                "PATH": os.path.join(data_folder, "ssl", "fibsem_epfl_3D", "data", "train", "raw"),
                "IN_MEMORY": True,
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "ssl", "fibsem_epfl_3D", "data", "test", "raw"),
                "IN_MEMORY": False,
                "PADDING": "(0,0,0)"
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 20,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "mae",
            "MAE_MASK_TYPE": "random",
            "LOAD_CHECKPOINT": False,
        },
        "AUGMENTOR": {
            "ENABLE": True,
            "RANDOM_ROT": True,
        },
        "TEST": {
            "ENABLE": True,
        },
    },
    "bmz_by_command": True,
    "bmz_package": "biapy_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 13.0},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "biapy_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test19"] = {
    "enable": True,
    "jobname": "test19",
    "description": "2D classification. DermaMNIST 2D data. preprocess: resize, Cross-val. Basic DA. ViT",
    "yaml": "test19.yaml",
    "yaml_modifications": {
        "DATA": {
            "PREPROCESS": {
                "TRAIN": True,
                "VAL": True,
                "TEST": True,
                "RESIZE": {
                    "ENABLE": True,
                    "OUTPUT_SHAPE": "(56,56)"
                }
            },
            "PATCH_SIZE": "(56,56,3)",
            "TRAIN": {
                "PATH": os.path.join(data_folder, "classification", "DermaMNIST_2D", "data", "train"),
                "IN_MEMORY": True,
            },
            "VAL": {
                "FROM_TRAIN": True,
                "CROSS_VAL": True,
                "CROSS_VAL_NFOLD": 5,
                "CROSS_VAL_FOLD": 1
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "classification", "DermaMNIST_2D", "data", "test"),
                "IN_MEMORY": True,
                "USE_VAL_AS_TEST": True,
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 5,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "vit",
            "LOAD_CHECKPOINT": False,
            "N_CLASSES": 7
        },
        "TEST": {
            "ENABLE": True,
        },
    },
    "bmz_by_command": True,
    "bmz_package": "biapy_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation Top 5 accuracy:", "gt": True, "value": 0.9},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "biapy_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test20"] = {
    "enable": True,
    "jobname": "test20",
    "description": "2D classification. butterfly data. preprocess: resize. Basic DA. efficientnet_b1",
    "yaml": "test20.yaml",
    "yaml_modifications": {
        "DATA": {
             "PREPROCESS": {
                "TRAIN": True,
                "VAL": True,
                "TEST": True,   
                "RESIZE": {
                    "ENABLE": True,
                    "OUTPUT_SHAPE": "(56,56)"
                }
            },
            "PATCH_SIZE": "(56,56,3)",
            "TRAIN": {
                "PATH": os.path.join(data_folder, "classification", "butterfly_data", "data", "train"),
                "IN_MEMORY": True,
            },
            "VAL": {
                "FROM_TRAIN": True,
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "classification", "butterfly_data", "data", "test"),
                "IN_MEMORY": True,
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 5,
            "PATIENCE": -1, 
        },
        "MODEL": {
            "ARCHITECTURE": "efficientnet_b1",
            "LOAD_CHECKPOINT": False,
            "N_CLASSES": 75
        },
        "TEST": {   
            "ENABLE": True,
        },
    },
    "internal_checks": [
        {"type": "regular", "pattern": "Validation Top 5 accuracy:", "gt": True, "value": 0.7},
    ]
}

all_test_info["Test21"] = {
    "enable": True,
    "jobname": "test21",
    "description": "3D classification. DermaMNIST 3D data. preprocess: resize, Cross-val. Basic DA. simple_cnn",
    "yaml": "test21.yaml",
    "yaml_modifications": {
        "DATA": {
             "PREPROCESS": {
                "TRAIN": True,
                "VAL": True,
                "TEST": True,
                "RESIZE": {
                    "ENABLE": True,
                    "OUTPUT_SHAPE": "(56,56,56)"
                }
            },
            "PATCH_SIZE": "(56,56,56,1)",
            "TRAIN": {
                "PATH": os.path.join(data_folder, "classification", "DermaMNIST_3D", "data", "train"),
                "IN_MEMORY": True,
            },
            "VAL": {
                "FROM_TRAIN": True,
                "CROSS_VAL": True,
                "CROSS_VAL_NFOLD": 5,
                "CROSS_VAL_FOLD": 3
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "classification", "DermaMNIST_3D", "data", "test"),
                "IN_MEMORY": True,
                "USE_VAL_AS_TEST": True,
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 5,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "simple_cnn",
            "LOAD_CHECKPOINT": False,
            "N_CLASSES": 11
        },
        "TEST": {
            "ENABLE": True,
        },
    },
    "bmz_by_command": True,
    "bmz_package": "biapy_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation Top 5 accuracy:", "gt": True, "value": 0.7},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "biapy_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test22"] = {
    "enable": True,
    "jobname": "test22",
    "description": "2D image to image. Dapi 2D data. preprocess: resize, Cross-val. Basic DA. multiresunet",
    "yaml": "test22.yaml",
    "yaml_modifications": {
        "DATA": {
             "NORMALIZATION": {
                "TYPE": "div",
            },
            "PATCH_SIZE": "(256,256,1)",
            "TRAIN": {
                "PATH": os.path.join(data_folder, "image_to_image", "Dapi_dataset", "data", "train", "raw"),
                "GT_PATH": os.path.join(data_folder, "image_to_image", "Dapi_dataset", "data", "train", "target"),
                "IN_MEMORY": True,
            },
            "VAL": {
                "FROM_TRAIN": True,
                "CROSS_VAL": False,
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "image_to_image", "Dapi_dataset", "data", "test", "raw"),
                "GT_PATH": os.path.join(data_folder, "image_to_image", "Dapi_dataset", "data", "test", "target"),
                "IN_MEMORY": True,
                "PADDING": "(40,40)"
            }
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 10,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": "multiresunet",
            "LOAD_CHECKPOINT": False,
        },
        "TEST": {
            "ENABLE": True,
        },
    },
    "bmz_by_command": True,
    "bmz_package": "biapy_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Test PSNR (merge patches):", "gt": True, "value": 19.0},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "biapy_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test23"] = {
    "enable": True,
    "jobname": "test23",
    "description": "2D image to image. lightmycells 2D data. extract random. val and train not in memory. Basic DA. UNETR",
    "yaml": "test23.yaml",
    "yaml_modifications": {
        "DATA": {
             "NORMALIZATION": {
                "TYPE": "zero_mean_unit_variance"
            },
            "REFLECT_TO_COMPLETE_SHAPE": True,
            "PATCH_SIZE": "(1024, 1024, 1)",
            "TRAIN": {
                "PATH": os.path.join(data_folder, "image_to_image", "reduced_actin_lightmycells", "actin", "raw"),
                "GT_PATH": os.path.join(data_folder, "image_to_image", "reduced_actin_lightmycells", "actin", "label"),
                "IN_MEMORY": False,
            },
            "VAL": {
                "FROM_TRAIN": False,
                "PATH": os.path.join(data_folder, "image_to_image", "reduced_actin_lightmycells", "actin", "raw"),
                "GT_PATH": os.path.join(data_folder,"image_to_image", "reduced_actin_lightmycells", "actin", "label"),
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "image_to_image", "reduced_actin_lightmycells", "actin", "raw"),
                "GT_PATH": os.path.join(data_folder, "image_to_image", "reduced_actin_lightmycells", "actin", "label"),
                "IN_MEMORY": False,
                "PADDING": "(200,200)"
            }
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 10,
            "PATIENCE": -1,
            "LR_SCHEDULER": {
                "WARMUP_COSINE_DECAY_EPOCHS": 5
            }
        },
        "AUGMENTOR": {
            "GRIDMASK": False,
            "ROT90": True
        },
        "MODEL": {
            "ARCHITECTURE": "unetr",
            "LOAD_CHECKPOINT": False
        },
        "TEST": {
            "ENABLE": True,
        },
        "PATHS": {},
    },
    "bmz_by_command": True,
    "bmz_package": "lightmycells_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 5.5},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "lightmycells_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test24"] = {
    "enable": True,
    "jobname": "test24",
    "description": "3D Instance seg. Zarr 3D data SNEMI. in memory false. input zarr multiple data raw: 'volumes.raw'"
        "warmupcosine. inference, by chunks, zarr multiple data, workflow process: entire pred.",
    "yaml": "test_24.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "INSTANCE_SEG": {
                "DATA_CHANNELS": 'BC',
                "DATA_MW_TH_TYPE": "manual",
                "DATA_MW_TH_BINARY_MASK": 0.9,
                "DATA_MW_TH_CONTOUR": 0.1,
                "DATA_REMOVE_SMALL_OBJ_BEFORE": 20,
                "DATA_REMOVE_BEFORE_MW": True,
                "DATA_CHANNEL_WEIGHTS": "(0.3, 1)"
            },
        },
        "DATA": {
            "PATCH_SIZE": "(20, 256, 256, 1)",
            "TRAIN": {
                "PATH": os.path.join(data_folder, "instance_seg", "snemi_zarr", "data", "train", "zarr"),
                "IN_MEMORY": False,
                "INPUT_IMG_AXES_ORDER": 'ZYX',
                "INPUT_MASK_AXES_ORDER": 'ZYX',
                "INPUT_ZARR_MULTIPLE_DATA": True,
                "INPUT_ZARR_MULTIPLE_DATA_RAW_PATH": 'volumes.raw',
                "INPUT_ZARR_MULTIPLE_DATA_GT_PATH": 'volumes.labels.neuron_ids'
            },
            "VAL": {
                "FROM_TRAIN": True, 
                "IN_MEMORY": False,
                "SPLIT_TRAIN": 0.1
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "instance_seg", "snemi_zarr", "data", "train", "zarr"),
                "IN_MEMORY": False,
                "LOAD_GT": True
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 80,
            "PATIENCE": -1,
            "OPTIMIZER": "ADAMW",
            "LR": 1.E-4,
            "LR_SCHEDULER": {
                "NAME": 'warmupcosine',
                "MIN_LR": 5.E-6,
                "WARMUP_COSINE_DECAY_EPOCHS": 15
            }
        },
        "MODEL": {
            "ARCHITECTURE": 'hrnet32',
            "HRNET": {
                "Z_DOWN": False
            }
        },
        "AUGMENTOR": {
            "BRIGHTNESS": True,
            "CONTRAST": True,
            "MISALIGNMENT": True,
            "MISSING_SECTIONS": True,
            "ELASTIC": True
        },
        "TEST": {
            "ENABLE": True,
            "BY_CHUNKS": {
                "ENABLE": True,
                "FORMAT": "Zarr",
                "SAVE_OUT_TIF": False,
                "INPUT_IMG_AXES_ORDER": 'ZYX',
                "INPUT_ZARR_MULTIPLE_DATA": True,   
                "INPUT_ZARR_MULTIPLE_DATA_RAW_PATH": 'volumes.raw',
                "INPUT_ZARR_MULTIPLE_DATA_GT_PATH": 'volumes.labels.neuron_ids',
                "WORKFLOW_PROCESS": {
                    "ENABLE": True,
                    "TYPE": "entire_pred"
                }
            }
        },
    },
    "bmz_by_command": True,
    "bmz_package": "biapy_model.zip",
    "internal_checks": [
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1", "gt": True, "value": 0.25},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "biapy_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test25"] = {
    "enable": True,
    "jobname": "test25",
    "description": "3D Image to image. Nuclear_Pore_complex_3D data. in memory true. val 0.1 of train.",
    "yaml": "test_25.yaml",
    "yaml_modifications": {
        "DATA": {
             "PATCH_SIZE": "(6,128,128,1)",
             "TRAIN": {
                "PATH": os.path.join(data_folder, "image_to_image", "Nuclear_Pore_complex_3D", "data", "train", "raw"),
                "GT_PATH": os.path.join(data_folder, "image_to_image", "Nuclear_Pore_complex_3D", "data", "train", "label"),  
                "IN_MEMORY": True,
            },
            "VAL": {
                "FROM_TRAIN": True,
                "SPLIT_TRAIN": 0.1,
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "image_to_image", "Nuclear_Pore_complex_3D", "data", "test", "raw"),
                "GT_PATH": os.path.join(data_folder, "image_to_image", "Nuclear_Pore_complex_3D", "data", "test", "label"),
                "IN_MEMORY": False,
                "PADDING": "(0,24,24)"
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 15,
            "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": 'resunet',
            "Z_DOWN": [1,1,1,1],
            "LOAD_CHECKPOINT": False
        },
        "TEST": {
            "ENABLE": True,
        },
    },
    "bmz_by_command": True,
    "bmz_package": "biapy_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 17.0},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "biapy_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test26"] = {
    "enable": True,
    "jobname": "test26",
    "description": "2D instance segmentation. BMZ 'stupendous-blowfish' model import, inference and export. "
        "zero_mean_unit_variance + format_version: 0.5.3 ",
    "yaml": "test_26.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "INSTANCE_SEG": {
                "DATA_CHANNELS": 'A',
                "DATA_MW_TH_TYPE": "auto",
                "DATA_CHANNEL_WEIGHTS": (1),
            },
        },
        "DATA": {
            "TEST": {
                "PATH": os.path.join(data_folder, "instance_seg", "dsb2018", "test", "images"),
                "GT_PATH": os.path.join(data_folder, "instance_seg", "dsb2018", "test", "masks"),
                "IN_MEMORY": False,
                "LOAD_GT": True
            },
        },
        "TRAIN": {
            "ENABLE": False,
        },
        "MODEL": {
            "SOURCE": 'bmz',
            "BMZ": {
                "SOURCE_MODEL_ID": 'stupendous-blowfish'
            }
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
        },
    }, 
    "bmz_by_command": True,
    "bmz_package": "test_model_stupendous-blowfish.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Test IoU (F channel) (merge patches):", "gt": True, "value": 0.6},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1", "gt": True, "value": 0.85},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name": "test_model_stupendous-blowfish.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test27"] = {
    "enable": True,
    "jobname": "test27",
    "description": "2D instance segmentation. BMZ 'hiding-blowfish' model import, inference and export."
        "scale_range + format_version: 0.4.10",
    "yaml": "test_27.yaml",
    "yaml_modifications": {
        "PROBEM": {
            "INSTANCE_SEG": {
                "DATA_CHANNELS": 'Db',
                "DATA_MW_TH_TYPE": "auto",
            },
        },
        "DATA": {
            "TEST": {
                "PATH": os.path.join(data_folder, "instance_seg", "MitoEM_human_2d_toy_data", "toy", "train", "raw"),
                "GT_PATH": os.path.join(data_folder, "instance_seg", "MitoEM_human_2d_toy_data", "toy", "train", "label"),
                "IN_MEMORY": False,
                "LOAD_GT": True
            },
        },
        "TRAIN": {
            "ENABLE": False,
        },
        "MODEL": {
            "SOURCE": 'bmz',
            "BMZ": {
                "SOURCE_MODEL_ID": 'hiding-blowfish'
            }
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
        },
    },
    "bmz_by_command": True,
    "bmz_package": "test_model_hiding-blowfish.zip",
    "internal_checks": [
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name": "test_model_hiding-blowfish.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test28"] = {
    "enable": True,
    "jobname": "test28",
    "description": "2D instance segmentation. BMZ 'frank-boar' model import, finetunning and export (reusing model original info)."
        "zero_mean_unit_variance + format_version: 0.5.3 ",
    "yaml": "test_28.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "INSTANCE_SEG": {
                "DATA_CHANNELS": 'BC',
                "DATA_MW_TH_TYPE": "auto",
                "DATA_CHANNEL_WEIGHTS": (0.5, 1),
            },
        },
        "DATA": {
            "TRAIN": {
                "PATH": os.path.join(data_folder, "instance_seg", "Stardist_v2_2D", "data", "train", "raw"),
                "GT_PATH": os.path.join(data_folder, "instance_seg", "Stardist_v2_2D", "data", "train", "label"),
                "IN_MEMORY": True,
            },
            "TEST": {   
                "PATH": os.path.join(data_folder, "instance_seg", "Stardist_v2_2D", "data", "test", "raw"),
                "GT_PATH": os.path.join(data_folder, "instance_seg", "Stardist_v2_2D", "data", "test", "label"),
                "IN_MEMORY": False,
                "LOAD_GT": True
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 5,
            "PATIENCE": -1,
        },
        "MODEL": {
            "SOURCE": 'bmz',
            "BMZ": {
                "SOURCE_MODEL_ID": 'frank-boar'
            }
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
        },
    },
    "bmz_by_command": True,
    "bmz_package": "2D U-NeXt V1 for nucleus segmentation.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Test IoU (F channel) (merge patches):", "gt": True, "value": 0.7},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1", "gt": True, "value": 0.85},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name": "2D U-NeXt V1 for nucleus segmentation.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}


all_test_info["Test29"] = {
    "enable": True,
    "jobname": "test29",
    "description": "2D Instance seg. Conic 2D data (multihead). Basic DA. BC (auto). resunet++. ",
    "yaml": "test_29.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "INSTANCE_SEG": {
                 "DATA_CHANNELS": 'BC',
                 "DATA_MW_TH_TYPE": "auto"
            },
        },
        "DATA": {
             "PATCH_SIZE": "(256,256,3)",
             "NORMALIZATION": {
                "TYPE": "zero_mean_unit_variance"
            },
            "TRAIN": {
                "PATH": os.path.join(data_folder, "instance_seg", "Conic", "conic_instance_subset", "train", "raw"),
                "GT_PATH": os.path.join(data_folder, "instance_seg", "Conic", "conic_instance_subset", "train", "label"),
                "IN_MEMORY": True,
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "instance_seg", "Conic", "conic_instance_subset", "tiny_test", "raw"),
                "GT_PATH": os.path.join(data_folder, "instance_seg", "Conic", "conic_instance_subset", "tiny_test", "label"),
                "IN_MEMORY": False,
                "LOAD_GT": True
            },
        },
        "TRAIN": {
            "ENABLE": True,
            "EPOCHS": 20,
            "PATIENCE": 5,
            "BATCH_SIZE": 6,
            "LR": 0.001,
            "LR_SCHEDULER": {
                "NAME": 'onecycle'
            }
        },
        "MODEL": {
            "ARCHITECTURE": 'unext_v1',
            "N_CLASSES": 7,
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
        },
    },
    "bmz_by_command": True,
    "bmz_package": "Conic_segmentation_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Test IoU (F channel) (merge patches):", "gt": True, "value": 0.35},
        {"type": "regular", "pattern": "Merge patches classification IoU:", "gt": True, "value": 0.1},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1", "gt": True, "value": 0.3},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name": "Conic_segmentation_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}

all_test_info["Test30"] = {
    "enable": True,
    "jobname": "test30",
    "description": "3D Instance seg. Cyst data. BCM. BMZ pretrained model: 'venomous-swan'. Post-proc: Clear border + Voronoi",
    "yaml": "test_30.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "INSTANCE_SEG": {
                "DATA_MW_TH_TYPE": "auto",
            },
        },
        "DATA": {
            "REFLECT_TO_COMPLETE_SHAPE": True,
            "PATCH_SIZE": "(80, 80, 80, 1)",
            "TEST": {
                "PATH": os.path.join(data_folder, "instance_seg", "CartoCell", "data", "validation", "x"),
                "GT_PATH": os.path.join(data_folder, "instance_seg", "CartoCell", "data", "validation", "y"),
                "IN_MEMORY": True,
                "LOAD_GT": True
            },
        },
        "TRAIN": {
            "ENABLE": False,
        },
        "MODEL": {
            "SOURCE": 'bmz',
            "BMZ": {
                "SOURCE_MODEL_ID": 'venomous-swan'
            }
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
            "POST_PROCESSING": {
                "CLEAR_BORDER": True,
                "VORONOI_ON_MASK": True
            }
        },
    },
    "internal_checks": [
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1", "gt": True, "value": 0.8},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 2, "metric": "f1", "gt": True, "value": 0.85}, # Post-processing
    ]
}

all_test_info["Test31"] = {
    "enable": True,
    "jobname": "test31",
    "description": "3D Detection. Achucarro data. points+classes. Export model to BMZ",
    "yaml": "test_31.yaml",
    "yaml_modifications": {
        "PROBLEM": {
            "DETECTION": {
                "CENTRAL_POINT_DILATION": [3],
                "CHECK_POINTS_CREATED": False,
                "DATA_CHECK_MW": False
            }
        },
        "DATA": {
            "NORMALIZATION": {
                "PERC_CLIP": {
                    "ENABLE": True,
                    "LOWER_PERC": 0.1,
                    "UPPER_PERC": 99.8
                },
                "TYPE": 'zero_mean_unit_variance'
            },
            "PATCH_SIZE": "(30, 128, 128, 3)",
            "N_CLASSES": 3,
            "TRAIN": {
                "PATH": os.path.join(data_folder, "detection", "achucarro_data", "data", "train", "raw"),
                "GT_PATH": os.path.join(data_folder, "detection", "achucarro_data", "data", "train", "label"),
                "IN_MEMORY": True,
            },
            "TEST": {
                "PATH": os.path.join(data_folder, "detection", "achucarro_data", "data", "test", "raw"),
                "GT_PATH": os.path.join(data_folder, "detection", "achucarro_data", "data", "test", "label"),
                "IN_MEMORY": False,
                "LOAD_GT": True,
                "PADDING": "(4,18,18)"
            },
        },
        "TRAIN": {
             "ENABLE": True,
             "EPOCHS": 50,
             "PATIENCE": -1,
        },
        "MODEL": {
            "ARCHITECTURE": 'unet',
            "Z_DOWN": [1,1,1],
            "LOAD_CHECKPOINT": False
        },
        "TEST": {
            "ENABLE": True,
            "FULL_IMG": False,
            "DET_PEAK_LOCAL_MAX_MIN_DISTANCE": 1,
            "DET_MIN_TH_TO_BE_PEAK": 0.5,
            "DET_TOLERANCE": 8
        },
    },
    "bmz_by_command": True,
    "bmz_package": "achucarro_det_classes_model.zip",
    "internal_checks": [
        {"type": "regular", "pattern": "Test F1 (merge patches)", "gt": True, "value": 0.55},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "achucarro_det_classes_model.zip"},
        {"type": "BMZ_weight_agreement", "pattern": "weights.pytorch_state_dict", "value": "✔️"},
    ]
}


# ---------------------------------------------------------
# 4. DATASET DEFINITIONS
# ---------------------------------------------------------
DATASETS = [
    {
        "folder_name": "semantic_seg",
        "data": [
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/semantic_segmentation/2d_semantic_segmentation.yaml",
                "template_local": "2d_semantic_segmentation.yaml",
                "drive_link": "https://drive.google.com/uc?id=1DfUoVHf__xk-s4BWSKbkfKYMnES-9RJt",
                "filename": "fibsem_epfl_2D.zip",
                "outpath": "fibsem_epfl_2D"
            },
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/semantic_segmentation/3d_semantic_segmentation.yaml",
                "template_local": "3d_semantic_segmentation.yaml",
                "drive_link": "https://drive.google.com/uc?id=10Cf11PtERq4pDHCJroekxu_hf10EZzwG",
                "filename": "fibsem_epfl_3D.zip",
                "outpath": "fibsem_epfl_3D"
            }
        ]
    },
    {
        "folder_name": "instance_seg",
        "data": [
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/instance_segmentation/2d_instance_segmentation.yaml",
                "template_local": "2d_instance_segmentation.yaml",
                "drive_link": "https://drive.google.com/uc?id=1b7_WDDGEEaEoIpO_1EefVr0w0VQaetmg",
                "filename": "Stardist_v2_2D.zip",
                "outpath": "Stardist_v2_2D"
            },
            {
                "drive_link": "https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip",
                "filename": "dsb2018.zip",
                "outpath": "dsb2018"
            },
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/instance_segmentation/3d_instance_segmentation.yaml",
                "template_local": "3d_instance_segmentation.yaml",
                "drive_link": "https://drive.google.com/uc?id=1fdL35ZTNw5hhiKau1gadaGu-rc5ZU_C7",
                "filename": "demo3D_3D.zip",
                "outpath": "demo3D_3D"
            },
            {
                "drive_link": "https://zenodo.org/records/10973241/files/CartoCell.zip?download=1",
                "filename": "CartoCell.zip",
                "outpath": "CartoCell"
            },
            {
                "drive_link": "https://drive.google.com/uc?id=1Ralex5SvYUZbXoDkWoaCjb6d_iWuuOHp",
                "filename": "snemi_zarr.zip",
                "outpath": "snemi_zarr"
            },
            {
                "drive_link": "https://drive.google.com/uc?id=1xrSsK23-2KfxCanaNJD7dldewWboKIw5",
                "filename": "MitoEM_human_2d_toy_data.zip",
                "outpath": "MitoEM_human_2d_toy_data"
            },
            {
                "drive_link": "https://drive.google.com/uc?id=1QGV0gP8N8B8-EmcAPNQAudr2dqXYzhss",
                "filename": "Conic.zip",
                "outpath": "Conic"
            }
        ]
    },
    {
        "folder_name": "detection",
        "data": [
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/detection/2d_detection.yaml",
                "template_local": "2d_detection.yaml",
                "drive_link": "https://drive.google.com/uc?id=1pWqQhcWY15b5fVLZDkPS-vnE-RU6NlYf",
                "filename": "Stardist_v2_detection.zip",
                "outpath": "Stardist_v2_detection"
            },
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/detection/3d_detection.yaml",
                "template_local": "3d_detection.yaml",
                "drive_link": "https://drive.google.com/uc?id=19P4AcvBPJXeW7QRj92Jh1keunGa5fi8d",
                "filename": "NucMM-Z_training.zip",
                "outpath": "NucMM-Z_training"
            },
            {
                "drive_link": "https://drive.google.com/uc?id=1veBueUuYi_mWbSky_4mtzfKBpO00SvWR",
                "filename": "brainglobe_small_data.zip",
                "outpath": "brainglobe_small_data"
            },
            {
                "drive_link": "https://upvehueus-my.sharepoint.com/:u:/g/personal/ignacio_arganda_ehu_eus/IQDcqg87-HIaTKsCdzT3_cyDAbNsr6WDiuvFsSfz_gisO-s?e=hcrpIG&download=1",
                "filename": "achucarro_data.zip",
                "outpath": "achucarro_data",
            }
        ]
    },
    {
        "folder_name": "denoising",
        "data": [
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/denoising/2d_denoising.yaml",
                "template_local": "2d_denoising.yaml",
                "drive_link": "https://drive.google.com/uc?id=1ZCNBWkOJc4XOtfKHP7M0g1yIVzqtwS76",
                "filename": "Noise2Void_RGB.zip",
                "outpath": "Noise2Void_RGB"
            },
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/denoising/3d_denoising.yaml",
                "template_local": "3d_denoising.yaml",
                "drive_link": "https://drive.google.com/uc?id=1OIjnUoJKdnbClBlpzk7V5R8wtoLont-r",
                "filename": "flywing3D.zip",
                "outpath": "flywing3D"
            }
        ]
    },
    {
        "folder_name": "sr",
        "data": [
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/super-resolution/2d_super-resolution.yaml",
                "template_local": "2d_super_resolution.yaml",
                "drive_link": "https://drive.google.com/uc?id=1rtrR_jt8hcBEqvwx_amFBNR7CMP5NXLo",
                "filename": "sr_data_2D.zip",
                "outpath": "sr_data_2D"
            },
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/super-resolution/3d_super-resolution.yaml",
                "template_local": "3d_super_resolution.yaml",
                "drive_link": "https://drive.google.com/uc?id=1TfQVK7arJiRAVmKHRebsfi8NEas8ni4s",
                "filename": "sr_data_3D.zip",
                "outpath": "sr_data_3D"
            }
        ]
    },
    {
        "folder_name": "ssl",
        "data": [
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/self-supervised/2d_self-supervised.yaml",
                "template_local": "2d_self_supervision.yaml",
                "drive_link": "https://drive.google.com/uc?id=1DfUoVHf__xk-s4BWSKbkfKYMnES-9RJt",
                "filename": "fibsem_epfl_2D.zip",
                "outpath": "fibsem_epfl_2D"
            },
            {
                "drive_link": "https://drive.google.com/uc?id=1bLB-oYx0JFAvSGv1Fa0F-vK26U_HlPtQ",
                "filename": "test14_checkpoint.pth",
                "outpath": "checkpoints" 
            },
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/self-supervised/3d_self-supervised.yaml",
                "template_local": "3d_self_supervision.yaml",
                "drive_link": "https://drive.google.com/uc?id=10Cf11PtERq4pDHCJroekxu_hf10EZzwG",
                "filename": "fibsem_epfl_3D.zip",
                "outpath": "fibsem_epfl_3D"
            }
        ]
    },
    {
        "folder_name": "classification",
        "data": [
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/classification/2d_classification.yaml",
                "template_local": "2d_classification.yaml",
                "drive_link": "https://drive.google.com/uc?id=15_pnH4_tJcwhOhNqFsm26NQuJbNbFSIN",
                "filename": "DermaMNIST_2D.zip",
                "outpath": "DermaMNIST_2D"
            },
            {
                "drive_link": "https://drive.google.com/uc?id=1m4_3UAgUsZ8FDjB4HyfA50Sht7_XkfdB",
                "filename": "butterfly_data.zip",
                "outpath": "butterfly_data"
            },
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/classification/3d_classification.yaml",
                "template_local": "3d_classification.yaml",
                "drive_link": "https://drive.google.com/uc?id=1pypWJ4Z9sRLPlVHbG6zpwmS6COkm3wUg",
                "filename": "DermaMNIST_3D.zip",
                "outpath": "DermaMNIST_3D"
            }
        ]
    },
    {
        "folder_name": "image_to_image",
        "data": [
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/image-to-image/2d_image-to-image.yaml",
                "template_local": "2d_image_to_image.yaml",
                "drive_link": "https://drive.google.com/uc?id=1L8AXNjh0_updVI3-v1duf6CbcZb8uZK7",
                "filename": "Dapi_dataset.zip",
                "outpath": "Dapi_dataset"
            },
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/image-to-image/lightmycells/lightmycells_actin.yaml",
                "template_local": "2d_image_to_image_light.yaml",
                "drive_link": "https://drive.google.com/uc?id=1SU4u-bcM1ZaDzEYg-d8W3zP6Yq2o8eKV",
                "filename": "reduced_actin_lightmycells.zip",
                "outpath": "reduced_actin_lightmycells"
            },
            {
                "template": "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/image-to-image/3d_image-to-image.yaml",
                "template_local": "3d_image_to_image.yaml",
                "drive_link": "https://drive.google.com/uc?id=1jL0bn2X3OFaV5T-6KR1g6fPDllH-LWzm",
                "filename": "Nuclear_Pore_complex_3D.zip",
                "outpath": "Nuclear_Pore_complex_3D"
            }
        ]
    }
]

# Universal Data Preparation Loop
for category in DATASETS:
    target_folder = os.path.join(data_folder, category["folder_name"])
    os.makedirs(target_folder, exist_ok=True)
    
    for item in category["data"]:
        # Download Template
        template_local_path = os.path.join(target_folder, item["template_local"])
        if not os.path.exists(template_local_path) and "template" in item:
            urllib.request.urlretrieve(item["template"], template_local_path)
            
        # Download and Extract Data
        out_filename = os.path.join(target_folder, item["filename"])
        out_path = os.path.join(target_folder, item["outpath"])
        
        if not os.path.exists(out_path):
            if "onedrive" in item.get("drive_link", "").lower() or "sharepoint" in item.get("drive_link", "").lower():
                download_onedrive_file(item["drive_link"], out_filename)
            else:
                download_drive_file(item["drive_link"], out_filename)
                
            # Unzip if downloaded file is a zip
            if os.path.exists(out_filename) and out_filename.endswith(".zip"):
                with ZipFile(out_filename, 'r') as zip_ref:
                    zip_ref.extractall(target_folder)

# ---------------------------------------------------------
# 5. TEST EXECUTION LOOP
# ---------------------------------------------------------
test_results = []
count_correct = 0

# Dynamic Loop: Runs through Test 1 to Test 36
for test_key, test_info in all_test_info.items():
    if not test_info.get("enable", True):
        continue
        
    print(f"\n==============================================")
    print(f"Running {test_key}: {test_info['description']}")
    print(f"==============================================")
    
    try:
        correct = True
        results = []
        
        # 0. Adapt paths and templates based on test info
        yaml_out_file = os.path.join(data_folder, test_info["yaml"])
        template_file = test_info["template_path"]
        
        # Load the base template
        with open(template_file, 'r') as f:
            biapy_config = yaml.safe_load(f)
            
        # Update with specific test modifications
        if "yaml_modifications" in test_info:
            biapy_config = update_nested_dict(biapy_config, test_info["yaml_modifications"])
            
        # Save the specific YAML for this test
        with open(yaml_out_file, 'w') as f:
            yaml.dump(biapy_config, f, default_flow_style=False)
        print(f"Generated specific config: {yaml_out_file}")

        # 1. Run the BiaPy Execution
        bmz_package_name = None
        for checks in all_test_info["Test7"]["internal_checks"]:
            if checks["type"] != "regular":
                bmz_package_name = checks['bmz_package_name']
                break
        assert bmz_package_name is not None, "bmz_package_name not found"
        args = {
            "test_info": test_info,
            "results_folder": results_folder,
            "yaml_file": yaml_out_file,
            "biapy_folder": biapy_folder,
            "multigpu": test_info.get("multigpu", True),
            "bmz_by_command": test_info.get("bmz_by_command", False),
            "bmz_package": test_info.get("bmz_package", None),
            "reuse_original_bmz_config": test_info.get("reuse_original_bmz_config", False)
        }
        runjob(**args)
        
        # Verify BiaPy process finished successfully
        res, last_lines = check_finished(all_test_info["Test1"])
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)

        # 2. Dynamic Checking Loop
        int_checks = 1
        for checks in test_info["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            elif checks["type"] == "DatasetMatching":
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], 
                                                     gt=checks["gt"], value_to_check=checks["nApparition"], 
                                                     metric=checks["metric"]))
            elif checks["type"] == "BMZ":
                bmz_file_path = os.path.join(bmz_folder, checks['bmz_package_name'])
                results.append(check_bmz_file_created(last_lines, checks["pattern"], bmz_file_path))
            elif checks["type"] == "BMZ_weight_agreement":
                results.append(check_bmz_weight_agreement(last_lines, checks["pattern"]))

            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(
                    checks["pattern"], checks.get("gt", ""), checks.get("value", "")
                ))

        # 3. Print test results and append to main list
        print_result(results, test_info["jobname"], int_checks)
        test_results.append(correct)

    except Exception as e:
        print(f"An error occurred during {test_key} execution.")
        print(e)
        test_results.append(False)

# ---------------------------------------------------------
# 6. FINAL SUMMARY
# ---------------------------------------------------------
print("\nFinish tests!!")
for res in test_results:
    if res:
        count_correct += 1

print(f"Test passed: ({count_correct}/{len(test_results)})")

if count_correct == len(test_results):
    sys.exit(0)
else:
    sys.exit(1)
