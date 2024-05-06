# conda activsate BiaPy_env_test, which is BiaPy_env and the following packages:
# pip install gdown==5.1.0 --quiet

import os
import gdown
import requests
import urllib
import yaml
import sys
from zipfile import ZipFile 
from subprocess import Popen, PIPE
from time import sleep
import numpy as np

gpu = "0"
data_folder = "/data/dfranco/biapy_checks"
results_folder = os.path.join(data_folder,  "output")
biapy_folder = "/data/dfranco/BiaPy"

###############
# Semantic seg.
###############
semantic_folder = os.path.join(data_folder, "semantic_seg")

# 2D
semantic_2d_data_drive_link = "https://drive.google.com/uc?id=1DfUoVHf__xk-s4BWSKbkfKYMnES-9RJt"
semantic_2d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/semantic_segmentation/2d_semantic_segmentation.yaml"
semantic_2d_template_local = os.path.join(semantic_folder, "2d_semantic_segmentation.yaml")
semantic_2d_data_filename = "fibsem_epfl_2D.zip"
semantic_2d_data_outpath = os.path.join(semantic_folder, "fibsem_epfl_2D")
# 3D
semantic_3d_data_drive_link = "https://drive.google.com/uc?id=10Cf11PtERq4pDHCJroekxu_hf10EZzwG"
semantic_3d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/semantic_segmentation/3d_semantic_segmentation.yaml"
semantic_3d_template_local = os.path.join(semantic_folder, "3d_semantic_segmentation.yaml")
semantic_3d_data_filename = "fibsem_epfl_3D.zip"
semantic_3d_data_outpath = os.path.join(semantic_folder, "fibsem_epfl_3D")

###############
# Instance seg.
###############
inst_seg_folder = os.path.join(data_folder, "instance_seg")

# 2D
instance_seg_2d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/instance_segmentation/2d_instance_segmentation.yaml"
instance_seg_2d_template_local = os.path.join(inst_seg_folder, "2d_instance_segmentation.yaml")
instance_seg_2d_data_drive_link = "https://drive.google.com/uc?id=1b7_WDDGEEaEoIpO_1EefVr0w0VQaetmg"
instance_seg_2d_data_filename = "Stardist_v2_2D.zip"
instance_seg_2d_data_outpath = os.path.join(inst_seg_folder, "Stardist_v2_2D")
# 3D
instance_seg_3d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/instance_segmentation/3d_instance_segmentation.yaml"
instance_seg_3d_template_local = os.path.join(inst_seg_folder, "3d_instance_segmentation.yaml")
instance_seg_3d_data_drive_link = "https://drive.google.com/uc?id=1fdL35ZTNw5hhiKau1gadaGu-rc5ZU_C7"
instance_seg_3d_data_filename = "demo3D_3D.zip"
instance_seg_3d_data_outpath = os.path.join(inst_seg_folder, "demo3D_3D")

instance_seg_cyst_data_zenodo_link = "https://zenodo.org/records/10973241/files/CartoCell.zip?download=1"
instance_seg_cyst_data_filename = "CartoCell.zip"
instance_seg_cyst_data_outpath = os.path.join(inst_seg_folder, "CartoCell")

###########
# Detection
###########
detection_folder = os.path.join(data_folder, "detection")

# 2D
detection_2d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/detection/2d_detection.yaml"
detection_2d_template_local = os.path.join(detection_folder, "2d_detection.yaml")
detection_2d_data_drive_link = "https://drive.google.com/uc?id=1pWqQhcWY15b5fVLZDkPS-vnE-RU6NlYf"
detection_2d_data_filename = "Stardist_v2_detection.zip"
detection_2d_data_outpath = os.path.join(detection_folder, "Stardist_v2_detection")
# 3D
detection_3d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/detection/3d_detection.yaml"
detection_3d_template_local = os.path.join(detection_folder, "3d_detection.yaml")
detection_3d_data_drive_link = "https://drive.google.com/uc?id=19P4AcvBPJXeW7QRj92Jh1keunGa5fi8d"
detection_3d_data_filename = "NucMM-Z_training.zip"
detection_3d_data_outpath = os.path.join(detection_folder, "NucMM-Z_training")

detection_3d_brainglobe_data_drive_link = "https://drive.google.com/uc?id=1veBueUuYi_mWbSky_4mtzfKBpO00SvWR"
detection_3d_brainglobe_data_filename = "brainglobe_small_data.zip"
detection_3d_brainglobe_data_outpath = os.path.join(detection_folder, "brainglobe_small_data")

###########
# Denoising
###########
denoising_folder = os.path.join(data_folder, "denoising")

# 2D
denoising_2d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/denoising/2d_denoising.yaml"
denoising_2d_template_local = os.path.join(denoising_folder, "2d_denoising.yaml")
denoising_2d_data_drive_link = "https://drive.google.com/uc?id=1TFvOySOiIgVIv9p4pbHdEbai-d2YGDvV"
denoising_2d_data_filename = "convallaria2D.zip"
denoising_2d_data_outpath = os.path.join(denoising_folder, "convallaria2D")
# 3D
denoising_3d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/denoising/3d_denoising.yaml"
denoising_3d_template_local = os.path.join(denoising_folder, "3d_denoising.yaml")
denoising_3d_data_drive_link = "https://drive.google.com/uc?id=1OIjnUoJKdnbClBlpzk7V5R8wtoLont-r"
denoising_3d_data_filename = "flywing3D.zip"
denoising_3d_data_outpath = os.path.join(denoising_folder, "flywing3D")


if not os.path.exists(biapy_folder): 
    raise ValueError(f"BiaPy not found in: {biapy_folder}")

all_test_info = {}
all_test_info["Test1"] = {
    "enable": False,
    "run_experiment": False,
    "jobname": "test1",
    "description": "2D Semantic seg. Lucchi++. Basic DA. Extract random crops (probability map). unet. 2D stack as 3D. Post-proc: z-filtering.",
    "yaml": "test_1.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test Foreground IoU (per image)", "gt": True, "value": 0.7},
        {"type": "regular", "pattern": "Test Foreground IoU (as 3D stack - post-processing)", "gt": True, "value": 0.7},
    ]
}

all_test_info["Test2"] = {
    "enable": False,
    "run_experiment": False,
    "jobname": "test2",
    "description": "3D Semantic seg. Lucchi++. attention_unet. Basic DA.",
    "yaml": "test_2.yaml",
    "internal_checks": [
        {"pattern": "Test Foreground IoU (merge patches)", "gt": True, "value": 0.7},
    ]
}

all_test_info["Test3"] = {
    "enable": False,
    "run_experiment": True,
    "jobname": "test3",
    "description": "2D Semantic seg. Lucchi++. Basic DA. 5 epochs. seunet. FULL_IMG False",
    "yaml": "test_3.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test Foreground IoU (merge patches)", "gt": True, "value": 0.5},
    ]
}

all_test_info["Test4"] = {
    "enable": False,
    "run_experiment": False,
    "jobname": "test4",
    "description": "2D Instance seg. Stardist 2D data. Basic DA. BC (auto). Replicate 2. resunet++. "
        "Post-proc: Clear border + remove instances by properties",
    "yaml": "test_4.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test Foreground IoU (merge patches):", "gt": True, "value": 0.3},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1", 
            "gt": True, "value": 0.7},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 2, "metric": "f1",
            "gt": False, "value": 0.3}, # Post-processing
    ]
}

all_test_info["Test5"] = {
    "enable": False,
    "run_experiment": False,
    "jobname": "test5",
    "description": "3D Instance seg. Demo 3D data. Basic DA. BCD (manual). resunet. Watershed multiple options. Post-proc: Clear border",
    "yaml": "test_5.yaml",
    "internal_checks": [
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1", 
            "gt": True, "value": 0.6},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 2, "metric": "f1",
            "gt": True, "value": 0.6}, # Post-processing
    ]
}

all_test_info["Test6"] = {
    "enable": False,
    "run_experiment": False,
    "jobname": "test6",
    "description": "3D Instance seg. Cyst data. Basic DA. BCM (auto). resunet. Post-proc: Clear border + Voronoi + remove by props",
    "yaml": "test_6.yaml",
    "internal_checks": [
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1", 
            "gt": True, "value": 0.4},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 2, "metric": "f1",
            "gt": True, "value": 0.1}, # Post-processing
    ]
}

all_test_info["Test7"] = {
    "enable": False,
    "run_experiment": False,
    "jobname": "test7",
    "description": "2D Detection. Stardist v2 2D data. custom norm, dataset, percentile clip. Basic DA. "
        "multiresunet. Post-proc: remove close points + det weatershed",
    "yaml": "test_7.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test Foreground IoU (merge patches):", "gt": True, "value": 0.2},
        {"type": "regular", "pattern": "Detection - Test F1 (merge patches)", "gt": True, "value": 0.4},
    ]
}

all_test_info["Test8"] = {
    "enable": False,
    "run_experiment": False,
    "jobname": "test8",
    "description": "3D Detection. NucMM-Z 3D data. custom norm, dataset, percentile clip. Basic DA. "
        "unetr. Post-proc: remove close points + det weatershed",
    "yaml": "test_8.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Detection - Test F1 (merge patches)", "gt": True, "value": 0.2},
    ]
}

all_test_info["Test11"] = {
    "enable": True,
    "run_experiment": True,
    "jobname": "test11",
    "description": "3D Detection. Zarr 3D data (Brainglobe). custom norm, percentile norm, per image. "
        "MINIMUM_FOREGROUND_PER. warmupcosine. Basic DA. resunet. test by chunks: Zarr. Post-proc: remove close points",
    "yaml": "test_11.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Detection - Test F1 (merge patches)", "gt": True, "value": 0.1},
    ]
}

all_test_info["Test9"] = {
    "enable": False,
    "run_experiment": False,
    "jobname": "test9",
    "description": "2D Denoising. Convallaria data. custom norm, dataset. Basic DA."
        "unetr. Post-proc: remove close points + det weatershed",
    "yaml": "test_9.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation MSE:", "gt": False, "value": 1.},
    ]
}

all_test_info["Test10"] = {
    "enable": False,
    "run_experiment": False,
    "jobname": "test10",
    "description": "3D Denoising. Flywing 3D data. custom norm, dataset. Basic DA. "
        "resunet. Post-proc: remove close points + det weatershed",
    "yaml": "test_10.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation MSE:", "gt": False, "value": 1.},
    ]
}

###################
# Semantic seg.
###################

# General things: 2D Data + YAML donwload
if not os.path.exists(semantic_2d_data_outpath) and (all_test_info["Test1"]["enable"] or\
    all_test_info["Test3"]["enable"]):
    print("Downloading 2D semantic seg. data . . .")
    
    os.makedirs(semantic_folder, exist_ok=True)
    os.chdir(semantic_folder)
    gdown.download(semantic_2d_data_drive_link, semantic_2d_data_filename, quiet=True)

    with ZipFile(os.path.join(semantic_folder, semantic_2d_data_filename), 'r') as zObject: 
        zObject.extractall(path=semantic_2d_data_outpath) 

    if not os.path.exists(semantic_2d_template_local):
        print("Downloading semantic seg. YAML . . .") 
        _, _ = urllib.request.urlretrieve(semantic_2d_template, filename=semantic_2d_template_local)

# General things: 3D Data + YAML donwload
if not os.path.exists(semantic_3d_data_outpath) and all_test_info["Test2"]["enable"]:
    print("Downloading 3D semantic seg. data . . .")
    
    os.makedirs(semantic_folder, exist_ok=True)
    os.chdir(semantic_folder)
    gdown.download(semantic_3d_data_drive_link, semantic_3d_data_filename, quiet=True)

    with ZipFile(os.path.join(semantic_folder, semantic_3d_data_filename), 'r') as zObject: 
        zObject.extractall(path=semantic_3d_data_outpath) 

    if not os.path.exists(semantic_3d_template_local):
        print("Downloading semantic seg. YAML . . .") 
        _, _ = urllib.request.urlretrieve(semantic_3d_template, filename=semantic_3d_template_local)

###############
# Instance seg.
###############

# General things: 2D Data + YAML donwload
if not os.path.exists(instance_seg_2d_data_outpath) and all_test_info["Test4"]["enable"]:
    print("Downloading 2D instance seg. data . . .")
    
    os.makedirs(inst_seg_folder, exist_ok=True)
    os.chdir(inst_seg_folder)
    gdown.download(instance_seg_2d_data_drive_link, instance_seg_2d_data_filename, quiet=True)

    with ZipFile(os.path.join(inst_seg_folder, instance_seg_2d_data_filename), 'r') as zObject: 
        zObject.extractall(path=instance_seg_2d_data_outpath) 

    if not os.path.exists(instance_seg_2d_template_local):
        print("Downloading instance seg. YAML . . .") 
        _, _ = urllib.request.urlretrieve(instance_seg_2d_template, filename=instance_seg_2d_template_local)

# General things: 3D Data + YAML donwload
if not os.path.exists(instance_seg_3d_data_outpath) and all_test_info["Test5"]["enable"]:
    print("Downloading 3D instance seg. data . . .")
    
    os.makedirs(inst_seg_folder, exist_ok=True)
    os.chdir(inst_seg_folder)
    gdown.download(instance_seg_3d_data_drive_link, instance_seg_3d_data_filename, quiet=True)

    with ZipFile(os.path.join(inst_seg_folder, instance_seg_3d_data_filename), 'r') as zObject: 
        zObject.extractall(path=instance_seg_3d_data_outpath) 

    if not os.path.exists(instance_seg_3d_template_local):
        print("Downloading instance seg. YAML . . .") 
        _, _ = urllib.request.urlretrieve(instance_seg_3d_template, filename=instance_seg_3d_template_local)

# General things: Cyst Data + YAML donwload
if not os.path.exists(instance_seg_cyst_data_outpath) and all_test_info["Test6"]["enable"]:
    print("Downloading cyst instance seg. data . . .")
    
    os.makedirs(inst_seg_folder, exist_ok=True)
    os.chdir(inst_seg_folder)
    _, _ = urllib.request.urlretrieve(instance_seg_cyst_data_zenodo_link, filename=instance_seg_cyst_data_filename)

    with ZipFile(os.path.join(inst_seg_folder, instance_seg_cyst_data_filename), 'r') as zObject: 
        zObject.extractall(path=inst_seg_folder) 

    if not os.path.exists(instance_seg_3d_template_local):
        print("Downloading instance seg. YAML . . .") 
        _, _ = urllib.request.urlretrieve(instance_seg_3d_template, filename=instance_seg_3d_template_local)

###########
# Detection
###########

# General things: 2D Data + YAML donwload
if not os.path.exists(detection_2d_data_outpath) and all_test_info["Test7"]["enable"]:
    print("Downloading 2D detection data . . .")
    
    os.makedirs(detection_folder, exist_ok=True)
    os.chdir(detection_folder)
    gdown.download(detection_2d_data_drive_link, detection_2d_data_filename, quiet=True)

    with ZipFile(os.path.join(detection_folder, detection_2d_data_filename), 'r') as zObject: 
        zObject.extractall(path=detection_2d_data_outpath) 

    if not os.path.exists(detection_2d_template_local):
        print("Downloading detection YAML . . .") 
        _, _ = urllib.request.urlretrieve(detection_2d_template, filename=detection_2d_template_local)

# General things: 3D Data + YAML donwload
if not os.path.exists(detection_3d_data_outpath) and all_test_info["Test8"]["enable"]:
    print("Downloading 3D detection data . . .")
    
    os.makedirs(detection_folder, exist_ok=True)
    os.chdir(detection_folder)
    gdown.download(detection_3d_data_drive_link, detection_3d_data_filename, quiet=True)

    with ZipFile(os.path.join(detection_folder, detection_3d_data_filename), 'r') as zObject: 
        zObject.extractall(path=detection_3d_data_outpath) 

    if not os.path.exists(detection_3d_template_local):
        print("Downloading detection YAML . . .") 
        _, _ = urllib.request.urlretrieve(detection_3d_template, filename=detection_3d_template_local)

# General things: 3D Brainglobe Data + YAML donwload
if not os.path.exists(detection_3d_brainglobe_data_outpath) and all_test_info["Test11"]["enable"]:
    print("Downloading 3D Brainglobe detection data . . .")
    
    os.makedirs(detection_folder, exist_ok=True)
    os.chdir(detection_folder)
    gdown.download(detection_3d_brainglobe_data_drive_link, detection_3d_brainglobe_data_filename, quiet=True)

    with ZipFile(os.path.join(detection_folder, detection_3d_brainglobe_data_filename), 'r') as zObject: 
        zObject.extractall(path=detection_3d_brainglobe_data_outpath) 

    if not os.path.exists(detection_3d_template_local):
        print("Downloading detection YAML . . .") 
        _, _ = urllib.request.urlretrieve(detection_3d_template, filename=detection_3d_template_local)

###########
# Denoising
###########

# General things: 2D Data + YAML donwload
if not os.path.exists(denoising_2d_data_outpath) and all_test_info["Test9"]["enable"]:
    print("Downloading 2D denoising data . . .")
    
    os.makedirs(denoising_folder, exist_ok=True)
    os.chdir(denoising_folder)
    gdown.download(denoising_2d_data_drive_link, denoising_2d_data_filename, quiet=True)

    with ZipFile(os.path.join(denoising_folder, denoising_2d_data_filename), 'r') as zObject: 
        zObject.extractall(path=denoising_2d_data_outpath) 

    if not os.path.exists(denoising_2d_template_local):
        print("Downloading denoising YAML . . .") 
        _, _ = urllib.request.urlretrieve(denoising_2d_template, filename=denoising_2d_template_local)

# General things: 3D Data + YAML donwload
if not os.path.exists(denoising_3d_data_outpath) and all_test_info["Test10"]["enable"]:
    print("Downloading 3D denoising data . . .")
    
    os.makedirs(denoising_folder, exist_ok=True)
    os.chdir(denoising_folder)
    gdown.download(denoising_3d_data_drive_link, denoising_3d_data_filename, quiet=True)

    with ZipFile(os.path.join(denoising_folder, denoising_3d_data_filename), 'r') as zObject: 
        zObject.extractall(path=denoising_3d_data_outpath) 

    if not os.path.exists(denoising_3d_template_local):
        print("Downloading denoising YAML . . .") 
        _, _ = urllib.request.urlretrieve(denoising_3d_template, filename=denoising_3d_template_local)


def print_inventory(dct): 
    for item, amount in dct.items():  # dct.iteritems() in Python 2
        if not isinstance(amount, list):
            print("{}: {}".format(item, amount))
        else:
            print("{}:".format(item))
            for i in amount:
                print("\t{}:".format(i))

def check_finished(test_info, jobname):
    # get the last lines of the output file
    jobdir = os.path.join(results_folder, test_info["jobname"])
    jobout_file = os.path.join(jobdir, test_info["jobname"]+"_1")
    logfile = open(jobout_file, 'r')
    last_lines = logfile.readlines()
    last_lines = last_lines[-min(50,len(last_lines)):]

    # Check if the final message appears there 
    finished_good = False 
    for line in last_lines:
        if "FINISHED JOB" in line:
            finished_good = True 
            break
    logfile.close()

    return finished_good, last_lines

def check_value(last_lines, pattern_to_find, ref_value, gt=True):
    """ 
    Check just one value. E.g. 'Test Foreground IoU (merge patches): 0.45622628648145197' "
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

def print_result(finished_good, jobname, int_checks):
    # Print the final message to the user accordingly
    if all(finished_good):
        print(f"** {jobname} job: [OK] ({int_checks} internal checks passed)")
    else: 
        print(f"** {jobname} job: [ERROR] ({sum(finished_good)} of {int_checks} internal checks passed)")
    print("######")
    
def runjob(test_info, results_folder, yaml_file, biapy_folder, multigpu=False):
    # Declare the log file 
    jobdir = os.path.join(results_folder, test_info["jobname"])
    jobout_file = os.path.join(jobdir, test_info["jobname"]+"_1")
    os.makedirs(jobdir, exist_ok=True)
    logfile = open(jobout_file, 'w')
    
    # Run the process and wait until finishes
    os.chdir(biapy_folder)
    print(f"Log: {jobout_file}")
    if multigpu:
        cmd = ["python", "-u", "-m", "torch.distributed.run", "--nproc_per_node=2", 
            f"--master-port={np.random.randint(low=1500, high=7000, size=1)[0]}", "main.py", 
            "--config", yaml_file, "--result_dir", results_folder, "--name", test_info["jobname"], "--run_id", "1", 
            "--gpu", "0,1"]
    else:
        cmd = ["python", "-u", "main.py", "--config", yaml_file, "--result_dir", results_folder, "--name", 
            test_info["jobname"], "--run_id", "1", "--gpu", gpu]
    print(f"Command: {' '.join(cmd)}")
    print("Running job . . .")
    process = Popen(cmd, stdout=logfile, stderr=logfile)
    process.wait()
    logfile.close()

#~~~~~~~~~~~~
# Test 1 
#~~~~~~~~~~~~
if all_test_info["Test1"]["enable"]:
    print("######")
    print("Running Test 1")
    print_inventory(all_test_info["Test1"])

    #*******************
    # File preparation 
    #*******************
    # Open config file
    with open(semantic_2d_template_local, 'r') as stream:
        try:
            biapy_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(exc)
            
    biapy_config['DATA']['EXTRACT_RANDOM_PATCH'] = True
    biapy_config['DATA']['PROBABILITY_MAP'] = True

    biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(semantic_2d_data_outpath, "data", "train", "x")
    biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(semantic_2d_data_outpath, "data", "train", "y")
    biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
    biapy_config['DATA']['TEST']['PATH'] = os.path.join(semantic_2d_data_outpath, "data", "test", "x")
    biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(semantic_2d_data_outpath, "data", "test", "y")
    biapy_config['DATA']['TEST']['IN_MEMORY'] = False
    biapy_config['DATA']['TEST']['LOAD_GT'] = True

    biapy_config['AUGMENTOR']['CONTRAST'] = True 
    biapy_config['AUGMENTOR']['BRIGHTNESS'] = True 

    biapy_config['TRAIN']['ENABLE'] = True
    biapy_config['TRAIN']['EPOCHS'] = 50
    biapy_config['TRAIN']['PATIENCE'] = -1 

    biapy_config['MODEL']['ARCHITECTURE'] = 'unet'

    biapy_config['TEST']['ENABLE'] = True
    biapy_config['TEST']['AUGMENTATION'] = True
    biapy_config['TEST']['FULL_IMG'] = True
    biapy_config['TEST']['ANALIZE_2D_IMGS_AS_3D_STACK'] = True
    biapy_config['TEST']['POST_PROCESSING'] = {}
    biapy_config['TEST']['POST_PROCESSING']['YZ_FILTERING'] = True

    # Save file
    test_file = os.path.join(semantic_folder, all_test_info["Test1"]["yaml"])
    with open(test_file, 'w') as outfile:
        yaml.dump(biapy_config, outfile, default_flow_style=False)

    # Run  
    if all_test_info["Test1"]["run_experiment"]:
        runjob(all_test_info["Test1"], results_folder, test_file, biapy_folder)

    # Check  
    results = []
    res, last_lines = check_finished(all_test_info["Test1"], "Test 1")
    if not res:
        print("Internal check not passed: seems that it didn't finish")
    results.append(res)
    int_checks = 1
    for checks in all_test_info["Test1"]["internal_checks"]: 
        results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
        int_checks += 1
        if not results[-1]:
            print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

    # Test result  
    print_result(results, all_test_info["Test1"]["jobname"], int_checks)


#~~~~~~~~~~~~
# Test 2 
#~~~~~~~~~~~~
if all_test_info["Test2"]["enable"]:
    print("######")
    print("Running Test 2")
    print_inventory(all_test_info["Test2"])

    #*******************
    # File preparation 
    #*******************
    # Open config file
    with open(semantic_3d_template_local, 'r') as stream:
        try:
            biapy_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(exc)

    biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(semantic_3d_data_outpath, "data", "train", "x")
    biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(semantic_3d_data_outpath, "data", "train", "y")
    biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
    biapy_config['DATA']['TEST']['PATH'] = os.path.join(semantic_3d_data_outpath, "data", "test", "x")
    biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(semantic_3d_data_outpath, "data", "test", "y")
    biapy_config['DATA']['TEST']['IN_MEMORY'] = False
    biapy_config['DATA']['TEST']['LOAD_GT'] = True

    biapy_config['AUGMENTOR']['CONTRAST'] = True 
    biapy_config['AUGMENTOR']['BRIGHTNESS'] = True 

    biapy_config['TRAIN']['ENABLE'] = True
    biapy_config['TRAIN']['EPOCHS'] = 30
    biapy_config['TRAIN']['PATIENCE'] = -1 

    biapy_config['MODEL']['ARCHITECTURE'] = 'attention_unet'

    biapy_config['TEST']['ENABLE'] = True

    # Save file
    test_file = os.path.join(semantic_folder, all_test_info["Test2"]["yaml"])
    with open(test_file, 'w') as outfile:
        yaml.dump(biapy_config, outfile, default_flow_style=False)

    # Run  
    if all_test_info["Test2"]["run_experiment"]:
        runjob(all_test_info["Test2"], results_folder, test_file, biapy_folder)

    # Check  
    results = []
    res, last_lines = check_finished(all_test_info["Test2"], "Test 2")
    if not res:
        print("Internal check not passed: seems that it didn't finish")
    results.append(res)
    int_checks = 1
    for checks in all_test_info["Test2"]["internal_checks"]: 
        results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
        int_checks += 1
        if not results[-1]:
            print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

    # Test result  
    print_result(results, all_test_info["Test2"]["jobname"], int_checks)

#~~~~~~~~~~~~
# Test 3 
#~~~~~~~~~~~~
if all_test_info["Test3"]["enable"]:
    print("######")
    print("Running Test 3")
    print_inventory(all_test_info["Test3"])

    #*******************
    # File preparation 
    #*******************
    # Open config file
    with open(semantic_2d_template_local, 'r') as stream:
        try:
            biapy_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(exc)

    biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(semantic_2d_data_outpath, "data", "train", "x")
    biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(semantic_2d_data_outpath, "data", "train", "y")
    biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
    biapy_config['DATA']['TEST']['PATH'] = os.path.join(semantic_2d_data_outpath, "data", "test", "x")
    biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(semantic_2d_data_outpath, "data", "test", "y")
    biapy_config['DATA']['TEST']['IN_MEMORY'] = True
    biapy_config['DATA']['TEST']['LOAD_GT'] = True

    biapy_config['AUGMENTOR']['CONTRAST'] = True 
    biapy_config['AUGMENTOR']['BRIGHTNESS'] = True 

    biapy_config['TRAIN']['ENABLE'] = True
    biapy_config['TRAIN']['EPOCHS'] = 2
    biapy_config['TRAIN']['PATIENCE'] = -1 

    biapy_config['MODEL']['ARCHITECTURE'] = 'seunet'

    biapy_config['TEST']['ENABLE'] = True
    biapy_config['TEST']['FULL_IMG'] = False

    # Save file
    test_file = os.path.join(semantic_folder, all_test_info["Test3"]["yaml"])
    with open(test_file, 'w') as outfile:
        yaml.dump(biapy_config, outfile, default_flow_style=False)

    # Run  
    if all_test_info["Test3"]["run_experiment"]:
        runjob(all_test_info["Test3"], results_folder, test_file, biapy_folder)

    # Check  
    results = []
    res, last_lines = check_finished(all_test_info["Test3"], "Test 2")
    if not res:
        print("Internal check not passed: seems that it didn't finish")
    results.append(res)
    int_checks = 1
    for checks in all_test_info["Test3"]["internal_checks"]: 
        results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
        int_checks += 1
        if not results[-1]:
            print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

    # Test result  
    print_result(results, all_test_info["Test3"]["jobname"], int_checks)

# Multiclass semantic seg
# DATA.TEST.ARGMAX_TO_OUTPUT = True

#~~~~~~~~~~~~
# Test 4 
#~~~~~~~~~~~~
if all_test_info["Test4"]["enable"]:
    print("######")
    print("Running Test 4")
    print_inventory(all_test_info["Test4"])

    #*******************
    # File preparation 
    #*******************
    # Open config file
    with open(instance_seg_2d_template_local, 'r') as stream:
        try:
            biapy_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(exc)

    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_CHANNELS'] = 'BC'
    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_MW_TH_TYPE'] = "auto"
    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_CHANNEL_WEIGHTS'] = "(0.5, 1)"

    biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(instance_seg_2d_data_outpath, "data", "train", "x")
    biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(instance_seg_2d_data_outpath, "data", "train", "y")
    biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
    biapy_config['DATA']['TRAIN']['REPLICATE'] = 2
    biapy_config['DATA']['TEST']['PATH'] = os.path.join(instance_seg_2d_data_outpath, "data", "test", "x")
    biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(instance_seg_2d_data_outpath, "data", "test", "y")
    biapy_config['DATA']['TEST']['IN_MEMORY'] = False
    biapy_config['DATA']['TEST']['LOAD_GT'] = True

    biapy_config['TRAIN']['ENABLE'] = True
    biapy_config['TRAIN']['EPOCHS'] = 20
    biapy_config['TRAIN']['PATIENCE'] = -1 

    biapy_config['MODEL']['ARCHITECTURE'] = 'resunet++'

    biapy_config['TEST']['ENABLE'] = True
    biapy_config['TEST']['FULL_IMG'] = False
    biapy_config['TEST']['POST_PROCESSING'] = {}
    biapy_config['TEST']['POST_PROCESSING']['CLEAR_BORDER'] = True
    
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES'] = {}
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['ENABLE'] = True
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES'] = {} 
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['ENABLE'] = True
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['PROPS'] = [['circularity', 'area']]
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['VALUES'] = [[0.5, 100]]
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['SIGN'] = [['gt', 'gt']]

    # Save file
    test_file = os.path.join(inst_seg_folder, all_test_info["Test4"]["yaml"])
    with open(test_file, 'w') as outfile:
        yaml.dump(biapy_config, outfile, default_flow_style=False)

    # Run  
    if all_test_info["Test4"]["run_experiment"]:
        runjob(all_test_info["Test4"], results_folder, test_file, biapy_folder)

    # Check  
    results = []
    res, last_lines = check_finished(all_test_info["Test4"], "Test 4")
    if not res:
        print("Internal check not passed: seems that it didn't finish")
    results.append(res)
    int_checks = 1
    for checks in all_test_info["Test4"]["internal_checks"]: 
        if checks["type"] == "regular":
            results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
        else:
            results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"], 
                value_to_check=checks["nApparition"], metric=checks["metric"]))
        int_checks += 1
        if not results[-1]:
            print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

    # Test result  
    print_result(results, all_test_info["Test4"]["jobname"], int_checks)


#~~~~~~~~~~~~
# Test 5 
#~~~~~~~~~~~~
if all_test_info["Test5"]["enable"]:
    print("######")
    print("Running Test 5")
    print_inventory(all_test_info["Test5"])

    #*******************
    # File preparation 
    #*******************
    # Open config file
    with open(instance_seg_3d_template_local, 'r') as stream:
        try:
            biapy_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(exc)

    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_CHANNELS'] = 'BCD'
    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_MW_TH_TYPE'] = "manual"
    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_CHANNEL_WEIGHTS'] = "(0.5, 1, 1)"

    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_MW_TH_BINARY_MASK'] = 0.4
    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_MW_TH_CONTOUR'] = 0.1
    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_MW_TH_DISTANCE'] = 1.8
    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_REMOVE_SMALL_OBJ_BEFORE'] = 20
    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_REMOVE_BEFORE_MW'] = True
    biapy_config['PROBLEM']['INSTANCE_SEG']['SEED_MORPH_SEQUENCE'] = ['erode','dilate']
    biapy_config['PROBLEM']['INSTANCE_SEG']['SEED_MORPH_RADIUS'] = [2,2]
    biapy_config['PROBLEM']['INSTANCE_SEG']['FORE_EROSION_RADIUS'] = 4
    biapy_config['PROBLEM']['INSTANCE_SEG']['FORE_DILATION_RADIUS'] = 4
    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_CHECK_MW'] = True
    biapy_config['PROBLEM']['INSTANCE_SEG']['WATERSHED_BY_2D_SLICES'] = True

    biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(instance_seg_3d_data_outpath, "data", "train", "x")
    biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(instance_seg_3d_data_outpath, "data", "train", "y")
    biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
    biapy_config['DATA']['TEST']['PATH'] = os.path.join(instance_seg_3d_data_outpath, "data", "test", "x")
    biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(instance_seg_3d_data_outpath, "data", "test", "y")
    biapy_config['DATA']['TEST']['IN_MEMORY'] = False
    biapy_config['DATA']['TEST']['LOAD_GT'] = True

    biapy_config['TRAIN']['ENABLE'] = True
    biapy_config['TRAIN']['EPOCHS'] = 20
    biapy_config['TRAIN']['PATIENCE'] = -1 

    biapy_config['MODEL']['ARCHITECTURE'] = 'resunet'

    biapy_config['TEST']['ENABLE'] = True
    biapy_config['TEST']['FULL_IMG'] = False
    biapy_config['TEST']['POST_PROCESSING'] = {}
    biapy_config['TEST']['POST_PROCESSING']['CLEAR_BORDER'] = True    
    
    # Save file
    test_file = os.path.join(inst_seg_folder, all_test_info["Test5"]["yaml"])
    with open(test_file, 'w') as outfile:
        yaml.dump(biapy_config, outfile, default_flow_style=False)

    # Run  
    if all_test_info["Test5"]["run_experiment"]:
        runjob(all_test_info["Test5"], results_folder, test_file, biapy_folder)

    # Check  
    results = []
    res, last_lines = check_finished(all_test_info["Test5"], "Test 5")
    if not res:
        print("Internal check not passed: seems that it didn't finish")
    results.append(res)
    int_checks = 1
    for checks in all_test_info["Test5"]["internal_checks"]: 
        if checks["type"] == "regular":
            results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
        else:
            results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"], 
                value_to_check=checks["nApparition"], metric=checks["metric"]))
        int_checks += 1
        if not results[-1]:
            print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

    # Test result  
    print_result(results, all_test_info["Test5"]["jobname"], int_checks)

#~~~~~~~~~~~~
# Test 6 
#~~~~~~~~~~~~
if all_test_info["Test6"]["enable"]:
    print("######")
    print("Running Test 6")
    print_inventory(all_test_info["Test6"])

    #*******************
    # File preparation 
    #*******************
    # Open config file
    with open(instance_seg_3d_template_local, 'r') as stream:
        try:
            biapy_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(exc)

    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_CHANNELS'] = 'BCM'
    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_MW_TH_TYPE'] = "auto"

    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_REMOVE_SMALL_OBJ_BEFORE'] = 10
    biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_REMOVE_BEFORE_MW'] = True
    biapy_config['PROBLEM']['INSTANCE_SEG']['SEED_MORPH_SEQUENCE'] = ['erode','dilate']
    biapy_config['PROBLEM']['INSTANCE_SEG']['SEED_MORPH_RADIUS'] = [2,2]
    biapy_config['PROBLEM']['INSTANCE_SEG']['FORE_EROSION_RADIUS'] = 4
    biapy_config['PROBLEM']['INSTANCE_SEG']['FORE_DILATION_RADIUS'] = 4

    biapy_config['DATA']['REFLECT_TO_COMPLETE_SHAPE'] = True
    biapy_config['DATA']['PATCH_SIZE'] = "(80, 80, 80, 1)"
    biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(instance_seg_cyst_data_outpath, "train_M2", "x")
    biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(instance_seg_cyst_data_outpath, "train_M2", "y")
    biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
    biapy_config['DATA']['VAL']['PATH'] = os.path.join(instance_seg_cyst_data_outpath, "validation", "x")
    biapy_config['DATA']['VAL']['GT_PATH'] = os.path.join(instance_seg_cyst_data_outpath, "validation", "y")
    biapy_config['DATA']['VAL']['IN_MEMORY'] = True
    biapy_config['DATA']['VAL']['FROM_TRAIN'] = False
    biapy_config['DATA']['TEST']['PATH'] = os.path.join(instance_seg_cyst_data_outpath, "validation", "x")
    biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(instance_seg_cyst_data_outpath, "validation", "y")
    biapy_config['DATA']['TEST']['IN_MEMORY'] = True
    biapy_config['DATA']['TEST']['LOAD_GT'] = True

    biapy_config['TRAIN']['ENABLE'] = True
    biapy_config['TRAIN']['EPOCHS'] = 5
    biapy_config['TRAIN']['PATIENCE'] = -1 

    biapy_config['MODEL']['ARCHITECTURE'] = 'resunet'

    biapy_config['TEST']['ENABLE'] = True
    biapy_config['TEST']['FULL_IMG'] = False
    biapy_config['TEST']['POST_PROCESSING'] = {}

    biapy_config['TEST']['POST_PROCESSING']['CLEAR_BORDER'] = True    
    biapy_config['TEST']['POST_PROCESSING']['VORONOI_ON_MASK'] = True

    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES'] = {}
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['ENABLE'] = True
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES'] = {} 
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['ENABLE'] = True
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['PROPS'] = [['sphericity'], ['area']]
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['VALUES'] = [[0.3], [100]]
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['SIGN'] = [['lt'], ['lt']]

    # Save file
    test_file = os.path.join(inst_seg_folder, all_test_info["Test6"]["yaml"])
    with open(test_file, 'w') as outfile:
        yaml.dump(biapy_config, outfile, default_flow_style=False)

    # Run  
    if all_test_info["Test6"]["run_experiment"]:
        runjob(all_test_info["Test6"], results_folder, test_file, biapy_folder)

    # Check  
    results = []
    res, last_lines = check_finished(all_test_info["Test6"], "Test 6")
    if not res:
        print("Internal check not passed: seems that it didn't finish")
    results.append(res)
    int_checks = 1
    for checks in all_test_info["Test6"]["internal_checks"]: 
        if checks["type"] == "regular":
            results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
        else:
            results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"], 
                value_to_check=checks["nApparition"], metric=checks["metric"]))
        int_checks += 1
        if not results[-1]:
            print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

    # Test result  
    print_result(results, all_test_info["Test6"]["jobname"], int_checks)

#~~~~~~~~~~~~
# Test 7 
#~~~~~~~~~~~~
if all_test_info["Test7"]["enable"]:
    print("######")
    print("Running Test 7")
    print_inventory(all_test_info["Test7"])

    #*******************
    # File preparation 
    #*******************
    # Open config file
    with open(detection_2d_template_local, 'r') as stream:
        try:
            biapy_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(exc)

    biapy_config['PROBLEM']['DETECTION'] = {}
    biapy_config['PROBLEM']['DETECTION']['CENTRAL_POINT_DILATION'] = 3
    biapy_config['PROBLEM']['DETECTION']['CHECK_POINTS_CREATED'] = False
    biapy_config['PROBLEM']['DETECTION']['DATA_CHECK_MW'] = True

    biapy_config['DATA']['NORMALIZATION'] = {}
    biapy_config['DATA']['NORMALIZATION']['PERC_CLIP'] = True
    biapy_config['DATA']['NORMALIZATION']['PERC_LOWER'] = 0.1
    biapy_config['DATA']['NORMALIZATION']['PERC_UPPER'] = 99.8
    biapy_config['DATA']['NORMALIZATION']['TYPE'] = 'custom'
    biapy_config['DATA']['NORMALIZATION']['APPLICATION_MODE'] = 'dataset'

    biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(detection_2d_data_outpath, "data", "train", "x")
    biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(detection_2d_data_outpath, "data", "train", "y")
    biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
    biapy_config['DATA']['TEST']['PATH'] = os.path.join(detection_2d_data_outpath, "data", "test", "x")
    biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(detection_2d_data_outpath, "data", "test", "y")
    biapy_config['DATA']['TEST']['IN_MEMORY'] = False
    biapy_config['DATA']['TEST']['LOAD_GT'] = True

    biapy_config['TRAIN']['ENABLE'] = True
    biapy_config['TRAIN']['EPOCHS'] = 20
    biapy_config['TRAIN']['PATIENCE'] = -1 

    biapy_config['MODEL']['ARCHITECTURE'] = 'multiresunet'
    biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

    biapy_config['TEST']['ENABLE'] = True
    biapy_config['TEST']['FULL_IMG'] = False
    biapy_config['TEST']['DET_POINT_CREATION_FUNCTION'] = 'peak_local_max'
    biapy_config['TEST']['DET_PEAK_LOCAL_MAX_MIN_DISTANCE'] = 1
    biapy_config['TEST']['DET_MIN_TH_TO_BE_PEAK'] = [0.7]
    
    biapy_config['TEST']['POST_PROCESSING'] = {}
    biapy_config['TEST']['POST_PROCESSING']['REMOVE_CLOSE_POINTS'] = True
    biapy_config['TEST']['POST_PROCESSING']['REMOVE_CLOSE_POINTS_RADIUS'] = [5]
    biapy_config['TEST']['POST_PROCESSING']['DET_WATERSHED'] = True
    biapy_config['TEST']['POST_PROCESSING']['DET_WATERSHED_FIRST_DILATION'] = [[2,2]]

    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES'] = {}
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['ENABLE'] = True
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES'] = {}
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['ENABLE'] = True
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['PROPS'] = [['circularity']]
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['VALUES'] = [[0.4]]
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['SIGN'] = [['lt']]

    # Save file
    test_file = os.path.join(detection_folder, all_test_info["Test7"]["yaml"])
    with open(test_file, 'w') as outfile:
        yaml.dump(biapy_config, outfile, default_flow_style=False)

    # Run  
    if all_test_info["Test7"]["run_experiment"]:
        runjob(all_test_info["Test7"], results_folder, test_file, biapy_folder)

    # Check  
    results = []
    res, last_lines = check_finished(all_test_info["Test7"], "Test 7")
    if not res:
        print("Internal check not passed: seems that it didn't finish")
    results.append(res)
    int_checks = 1
    for checks in all_test_info["Test7"]["internal_checks"]: 
        if checks["type"] == "regular":
            results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
        else:
            results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"], 
                value_to_check=checks["nApparition"], metric=checks["metric"]))
        int_checks += 1
        if not results[-1]:
            print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

    # Test result  
    print_result(results, all_test_info["Test7"]["jobname"], int_checks)

#~~~~~~~~~~~~
# Test 8 
#~~~~~~~~~~~~
if all_test_info["Test8"]["enable"]:
    print("######")
    print("Running Test 8")
    print_inventory(all_test_info["Test8"])

    #*******************
    # File preparation 
    #*******************
    # Open config file
    with open(detection_3d_template_local, 'r') as stream:
        try:
            biapy_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(exc)

    biapy_config['PROBLEM']['DETECTION'] = {}
    biapy_config['PROBLEM']['DETECTION']['CENTRAL_POINT_DILATION'] = 3
    biapy_config['PROBLEM']['DETECTION']['CHECK_POINTS_CREATED'] = False
    biapy_config['PROBLEM']['DETECTION']['DATA_CHECK_MW'] = True

    biapy_config['DATA']['NORMALIZATION'] = {}
    biapy_config['DATA']['NORMALIZATION']['PERC_CLIP'] = True
    biapy_config['DATA']['NORMALIZATION']['PERC_LOWER'] = 0.1
    biapy_config['DATA']['NORMALIZATION']['PERC_UPPER'] = 99.8
    biapy_config['DATA']['NORMALIZATION']['TYPE'] = 'custom'
    biapy_config['DATA']['NORMALIZATION']['APPLICATION_MODE'] = 'image'

    biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(detection_3d_data_outpath, "data", "train", "x")
    biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(detection_3d_data_outpath, "data", "train", "y")
    biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
    biapy_config['DATA']['TEST']['PATH'] = os.path.join(detection_3d_data_outpath, "data", "test", "x")
    biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(detection_3d_data_outpath, "data", "test", "y")
    biapy_config['DATA']['TEST']['IN_MEMORY'] = False
    biapy_config['DATA']['TEST']['LOAD_GT'] = True

    biapy_config['TRAIN']['ENABLE'] = True
    biapy_config['TRAIN']['EPOCHS'] = 50
    biapy_config['TRAIN']['PATIENCE'] = -1 

    biapy_config['MODEL']['ARCHITECTURE'] = 'unetr'
    biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

    biapy_config['TEST']['ENABLE'] = True
    biapy_config['TEST']['FULL_IMG'] = False
    biapy_config['TEST']['DET_POINT_CREATION_FUNCTION'] = 'blob_log'
    biapy_config['TEST']['DET_PEAK_LOCAL_MAX_MIN_DISTANCE'] = 1
    biapy_config['TEST']['DET_MIN_TH_TO_BE_PEAK'] = [0.7]
    biapy_config['TEST']['DET_TOLERANCE'] = [8]

    biapy_config['TEST']['POST_PROCESSING'] = {}
    biapy_config['TEST']['POST_PROCESSING']['REMOVE_CLOSE_POINTS'] = True
    biapy_config['TEST']['POST_PROCESSING']['REMOVE_CLOSE_POINTS_RADIUS'] = [3]
    biapy_config['TEST']['POST_PROCESSING']['DET_WATERSHED'] = True
    biapy_config['TEST']['POST_PROCESSING']['DET_WATERSHED_FIRST_DILATION'] = [[2,2,1]]

    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES'] = {}
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['ENABLE'] = True
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES'] = {}
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['ENABLE'] = True
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['PROPS'] = [['sphericity']]
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['VALUES'] = [[0.5]]
    biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['SIGN'] = [['lt']]

    # Save file
    test_file = os.path.join(detection_folder, all_test_info["Test8"]["yaml"])
    with open(test_file, 'w') as outfile:
        yaml.dump(biapy_config, outfile, default_flow_style=False)

    # Run  
    if all_test_info["Test8"]["run_experiment"]:
        runjob(all_test_info["Test8"], results_folder, test_file, biapy_folder)

    # Check  
    results = []
    res, last_lines = check_finished(all_test_info["Test8"], "Test 8")
    if not res:
        print("Internal check not passed: seems that it didn't finish")
    results.append(res)
    int_checks = 1
    for checks in all_test_info["Test8"]["internal_checks"]: 
        if checks["type"] == "regular":
            results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
        else:
            results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"], 
                value_to_check=checks["nApparition"], metric=checks["metric"]))
        int_checks += 1
        if not results[-1]:
            print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

    # Test result  
    print_result(results, all_test_info["Test8"]["jobname"], int_checks)


#~~~~~~~~~~~~
# Test 11 
#~~~~~~~~~~~~
if all_test_info["Test11"]["enable"]:
    print("######")
    print("Running Test 11")
    print_inventory(all_test_info["Test11"])

    #*******************
    # File preparation 
    #*******************
    # Open config file
    with open(detection_3d_template_local, 'r') as stream:
        try:
            biapy_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(exc)

    biapy_config['PROBLEM']['DETECTION'] = {}
    biapy_config['PROBLEM']['DETECTION']['CENTRAL_POINT_DILATION'] = 2
    biapy_config['PROBLEM']['DETECTION']['CHECK_POINTS_CREATED'] = False

    biapy_config['DATA']['PATCH_SIZE'] = "(20, 128, 128, 2)"
    biapy_config['DATA']['NORMALIZATION'] = {}
    biapy_config['DATA']['NORMALIZATION']['PERC_CLIP'] = True
    biapy_config['DATA']['NORMALIZATION']['PERC_LOWER'] = 0.1
    biapy_config['DATA']['NORMALIZATION']['PERC_UPPER'] = 99.8
    biapy_config['DATA']['NORMALIZATION']['APPLICATION_MODE'] = 'image'

    biapy_config['DATA']['TRAIN']['INPUT_IMG_AXES_ORDER'] = 'ZYXC'
    biapy_config['DATA']['TRAIN']['MINIMUM_FOREGROUND_PER'] = 0.0000000000000000000001
    biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(detection_3d_brainglobe_data_outpath, "data", "3D_ch2ch4_Zarr")
    biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(detection_3d_brainglobe_data_outpath, "data", "y")
    biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
    biapy_config['DATA']['VAL']['SPLIT_TRAIN'] = 0.1
    biapy_config['DATA']['TEST']['PATH'] = os.path.join(detection_3d_brainglobe_data_outpath, "data", "3D_ch2ch4_Zarr")
    biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(detection_3d_brainglobe_data_outpath, "data", "y")
    biapy_config['DATA']['TEST']['IN_MEMORY'] = False
    biapy_config['DATA']['TEST']['LOAD_GT'] = True
    biapy_config['DATA']['TEST']['PADDING'] = "(0,18,18)"

    biapy_config['TRAIN']['ENABLE'] = True
    biapy_config['TRAIN']['EPOCHS'] = 100
    biapy_config['TRAIN']['BATCH_SIZE'] = 1
    biapy_config['TRAIN']['PATIENCE'] = 20 
    biapy_config['TRAIN']['LR_SCHEDULER'] = {} 
    biapy_config['TRAIN']['LR_SCHEDULER']['NAME'] = 'warmupcosine'  
    biapy_config['TRAIN']['LR_SCHEDULER']['MIN_LR'] = 5.E-6  
    biapy_config['TRAIN']['LR_SCHEDULER']['WARMUP_COSINE_DECAY_EPOCHS'] = 15

    biapy_config['MODEL']['ARCHITECTURE'] = 'resunet'
    biapy_config['MODEL']['Z_DOWN'] = [1,1,1,1]
    del biapy_config['MODEL']['FEATURE_MAPS']
    biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

    biapy_config['AUGMENTOR']['RANDOM_ROT'] = True
    biapy_config['AUGMENTOR']['AFFINE_MODE'] = 'reflect'
    biapy_config['AUGMENTOR']['ZFLIP'] = True

    biapy_config['TEST']['ENABLE'] = True
    biapy_config['TEST']['FULL_IMG'] = False
    biapy_config['TEST']['DET_MIN_TH_TO_BE_PEAK'] = [0.2]
    biapy_config['TEST']['DET_TOLERANCE'] = [8]
    biapy_config['TEST']['VERBOSE'] = True

    biapy_config['TEST']['BY_CHUNKS'] = {}
    biapy_config['TEST']['BY_CHUNKS']['ENABLE'] = True
    biapy_config['TEST']['BY_CHUNKS']['FORMAT'] = "Zarr"
    biapy_config['TEST']['BY_CHUNKS']['SAVE_OUT_TIF'] = True
    biapy_config['TEST']['BY_CHUNKS']['INPUT_IMG_AXES_ORDER'] = 'ZYXC'
    biapy_config['TEST']['BY_CHUNKS']['WORKFLOW_PROCESS'] = {}
    biapy_config['TEST']['BY_CHUNKS']['WORKFLOW_PROCESS']['ENABLE'] = True
    biapy_config['TEST']['BY_CHUNKS']['WORKFLOW_PROCESS']['TYPE'] = "chunk_by_chunk" 

    biapy_config['TEST']['POST_PROCESSING'] = {}
    biapy_config['TEST']['POST_PROCESSING']['REMOVE_CLOSE_POINTS'] = True
    biapy_config['TEST']['POST_PROCESSING']['REMOVE_CLOSE_POINTS_RADIUS'] = [3]

    # Save file
    test_file = os.path.join(detection_folder, all_test_info["Test11"]["yaml"])
    with open(test_file, 'w') as outfile:
        yaml.dump(biapy_config, outfile, default_flow_style=False)

    # Run  
    if all_test_info["Test11"]["run_experiment"]:
        runjob(all_test_info["Test11"], results_folder, test_file, biapy_folder, multigpu=True)

    # Check  
    results = []
    res, last_lines = check_finished(all_test_info["Test11"], "Test 11")
    if not res:
        print("Internal check not passed: seems that it didn't finish")
    results.append(res)
    int_checks = 1
    for checks in all_test_info["Test11"]["internal_checks"]: 
        if checks["type"] == "regular":
            results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
        else:
            results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"], 
                value_to_check=checks["nApparition"], metric=checks["metric"]))
        int_checks += 1
        if not results[-1]:
            print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

    # Test result  
    print_result(results, all_test_info["Test11"]["jobname"], int_checks)


#~~~~~~~~~~~~
# Test 9 
#~~~~~~~~~~~~
if all_test_info["Test9"]["enable"]:
    print("######")
    print("Running Test 9")
    print_inventory(all_test_info["Test9"])

    #*******************
    # File preparation 
    #*******************
    # Open config file
    with open(denoising_2d_template_local, 'r') as stream:
        try:
            biapy_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(exc)

    biapy_config['PROBLEM']['DENOISING']['N2V_STRUCTMASK'] = True

    biapy_config['DATA']['NORMALIZATION'] = {}
    biapy_config['DATA']['NORMALIZATION']['TYPE'] = 'custom'
    biapy_config['DATA']['NORMALIZATION']['APPLICATION_MODE'] = 'image'

    biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(denoising_2d_data_outpath, "data", "train")
    biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
    biapy_config['DATA']['TEST']['PATH'] = os.path.join(denoising_2d_data_outpath, "data", "test")
    biapy_config['DATA']['TEST']['IN_MEMORY'] = False

    biapy_config['TRAIN']['ENABLE'] = True
    biapy_config['TRAIN']['EPOCHS'] = 20
    biapy_config['TRAIN']['PATIENCE'] = -1 

    biapy_config['MODEL']['ARCHITECTURE'] = 'unetr'
    biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

    biapy_config['TEST']['ENABLE'] = True
    biapy_config['TEST']['FULL_IMG'] = False

    # Save file
    test_file = os.path.join(denoising_folder, all_test_info["Test9"]["yaml"])
    with open(test_file, 'w') as outfile:
        yaml.dump(biapy_config, outfile, default_flow_style=False)

    # Run  
    if all_test_info["Test9"]["run_experiment"]:
        runjob(all_test_info["Test9"], results_folder, test_file, biapy_folder)

    # Check  
    results = []
    res, last_lines = check_finished(all_test_info["Test9"], "Test 9")
    if not res:
        print("Internal check not passed: seems that it didn't finish")
    results.append(res)
    int_checks = 1
    for checks in all_test_info["Test9"]["internal_checks"]: 
        if checks["type"] == "regular":
            results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
        else:
            results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"], 
                value_to_check=checks["nApparition"], metric=checks["metric"]))
        int_checks += 1
        if not results[-1]:
            print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

    # Test result  
    print_result(results, all_test_info["Test9"]["jobname"], int_checks)

#~~~~~~~~~~~~
# Test 10 
#~~~~~~~~~~~~
if all_test_info["Test10"]["enable"]:
    print("######")
    print("Running Test 10")
    print_inventory(all_test_info["Test10"])

    #*******************
    # File preparation 
    #*******************
    # Open config file
    with open(denoising_3d_template_local, 'r') as stream:
        try:
            biapy_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(exc)

    biapy_config['PROBLEM']['DENOISING']['N2V_STRUCTMASK'] = False

    biapy_config['DATA']['NORMALIZATION'] = {}
    biapy_config['DATA']['NORMALIZATION']['TYPE'] = 'custom'
    biapy_config['DATA']['NORMALIZATION']['APPLICATION_MODE'] = 'dataset'

    biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(denoising_3d_data_outpath, "data", "train")
    biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
    biapy_config['DATA']['TEST']['PATH'] = os.path.join(denoising_3d_data_outpath, "data", "test")
    biapy_config['DATA']['TEST']['IN_MEMORY'] = False

    biapy_config['TRAIN']['ENABLE'] = True
    biapy_config['TRAIN']['EPOCHS'] = 20
    biapy_config['TRAIN']['PATIENCE'] = -1 

    biapy_config['MODEL']['ARCHITECTURE'] = 'unet'
    biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

    biapy_config['TEST']['ENABLE'] = True
    biapy_config['TEST']['FULL_IMG'] = False

    # Save file
    test_file = os.path.join(denoising_folder, all_test_info["Test10"]["yaml"])
    with open(test_file, 'w') as outfile:
        yaml.dump(biapy_config, outfile, default_flow_style=False)

    # Run  
    if all_test_info["Test10"]["run_experiment"]:
        runjob(all_test_info["Test10"], results_folder, test_file, biapy_folder)

    # Check  
    results = []
    res, last_lines = check_finished(all_test_info["Test10"], "Test 10")
    if not res:
        print("Internal check not passed: seems that it didn't finish")
    results.append(res)
    int_checks = 1
    for checks in all_test_info["Test10"]["internal_checks"]: 
        if checks["type"] == "regular":
            results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
        else:
            results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"], 
                value_to_check=checks["nApparition"], metric=checks["metric"]))
        int_checks += 1
        if not results[-1]:
            print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

    # Test result  
    print_result(results, all_test_info["Test10"]["jobname"], int_checks)

import pdb; pdb.set_trace()
