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

parser = argparse.ArgumentParser(description="Check BiaPy code consistency",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--out_folder", type=str, help="Output folder")
parser.add_argument("--gpus", type=str, help="GPUs to use")
parser.add_argument("--biapy_folder", default="", help="BiaPy code directory to test")
args = parser.parse_args()

gpu = args.gpus.split(",")[0] # "0"
gpus = args.gpus # "0,1" # For those tests that use more than one
print(f"Using GPU: '{gpu}' (single-gpu checks) ; GPUs: '{gpus}' (multi-gpu checks)")
data_folder = os.path.join(os.getcwd(), args.out_folder) # "/data/dfranco/biapy_checks"
print(f"Out folder: {data_folder}")
if args.biapy_folder == "":
    biapy_folder = os.getcwd()# "/data/dfranco/BiaPy"
else:
    biapy_folder = args.biapy_folder
print(f"Running in folder: {biapy_folder}")
bmz_folder = os.path.join(data_folder, "bmz_files")
bmz_script = os.path.join(biapy_folder, "biapy", "utils", "scripts", "export_bmz_test.py")

all_test_info = {}
all_test_info["Test1"] = {
    "enable": True,
    "jobname": "test1",
    "description": "2D Semantic seg. Lucchi++. Basic DA. unet. 2D stack as 3D. Post-proc: z-filtering.",
    "yaml": "test_1.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test Foreground IoU (per image)", "gt": True, "value": 0.7},
        {"type": "regular", "pattern": "Test Foreground IoU (as 3D stack - post-processing)", "gt": True, "value": 0.7},
    ]
}

all_test_info["Test2"] = {
    "enable": True,
    "jobname": "test2",
    "description": "3D Semantic seg. Lucchi++. attention_unet. Basic DA.",
    "yaml": "test_2.yaml",
    "internal_checks": [
        {"pattern": "Test Foreground IoU (merge patches)", "gt": True, "value": 0.55},
    ]
}

all_test_info["Test3"] = {
    "enable": True,
    "jobname": "test3",
    "description": "2D Semantic seg. Lucchi++. Basic DA. 5 epochs. seunet. FULL_IMG False",
    "yaml": "test_3.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test Foreground IoU (merge patches)", "gt": True, "value": 0.5},
    ]
}

all_test_info["Test4"] = {
    "enable": True,
    "jobname": "test4",
    "description": "2D Instance seg. Stardist 2D data. Basic DA. BC (auto). resunet++. "
        "Post-proc: Clear border + remove instances by properties (leave only the bad ones).",
    "yaml": "test_4.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test IoU (F channel) (merge patches):", "gt": True, "value": 0.4},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1",
            "gt": True, "value": 0.50},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 2, "metric": "f1",
            "gt": False, "value": 0.3}, # Post-processing (leave bad instances only)
    ]
}

all_test_info["Test5"] = {
    "enable": True,
    "jobname": "test5",
    "description": "3D Instance seg. Demo 3D data. Basic DA. BCD (manual). resunet. Watershed multiple options. Post-proc: Clear border",
    "yaml": "test_5.yaml",
    "internal_checks": [
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1",
            "gt": True, "value": 0.45},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 2, "metric": "f1",
            "gt": True, "value": 0.45}, # Post-processing
    ]
}

all_test_info["Test6"] = {
    "enable": True,
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
    "enable": True,
    "jobname": "test7",
    "description": "2D Detection. Stardist v2 2D data. zero_mean_unit_variance norm, percentile clip. Basic DA. "
        "multiresunet. Post-proc: remove close points + det weatershed",
    "yaml": "test_7.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test Foreground IoU (merge patches):", "gt": True, "value": 0.3},
        {"type": "regular", "pattern": "Test F1 (merge patches)", "gt": True, "value": 0.7},
    ]
}

all_test_info["Test8"] = {
    "enable": True,
    "jobname": "test8",
    "description": "3D Detection. NucMM-Z 3D data. zero_mean_unit_variance norm, percentile clip. Basic DA. "
        "unetr. Post-proc: remove close points + det weatershed",
    "yaml": "test_8.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test F1 (merge patches)", "gt": True, "value": 0.2},
    ]
}

all_test_info["Test11"] = {
    "enable": True,
    "jobname": "test11",
    "description": "3D Detection. Zarr 3D data (Brainglobe). zero_mean_unit_variance norm, percentile norm, per image. "
        "filter samples: foreground + mean. warmupcosine. Basic DA. resunet. test by chunks: Zarr. Post-proc: remove close points",
    "yaml": "test_11.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test F1 (merge patches)", "gt": True, "value": 0.15},
    ]
}

all_test_info["Test9"] = {
    "enable": True,
    "jobname": "test9",
    "description": "2D Denoising. Convallaria data. zero_mean_unit_variance norm. Basic DA."
        "unetr. Post-proc: remove close points + det weatershed",
    "yaml": "test_9.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation MSE:", "gt": False, "value": 1.},
    ]
}

all_test_info["Test10"] = {
    "enable": True,
    "jobname": "test10",
    "description": "3D Denoising. Flywing 3D data. zero_mean_unit_variance norm. Basic DA. "
        "resunet. Post-proc: remove close points + det weatershed",
    "yaml": "test_10.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation MSE:", "gt": False, "value": 1.},
    ]
}

all_test_info["Test12"] = {
    "enable": True,
    "jobname": "test12",
    "description": "2D super-resolution. SR 2D data. Cross-val. Basic DA. DFCAN",
    "yaml": "test_12.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 22.0},
        {"type": "regular", "pattern": "Test PSNR (merge patches)", "gt": True, "value": 23.0},
    ]
}

all_test_info["Test13"] = {
    "enable": True,
    "jobname": "test13",
    "description": "3D super-resolution. SR 3D data. Cross-val. Basic DA. resunet. one-cycle",
    "yaml": "test_13.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 20.0},
        {"type": "regular", "pattern": "Test PSNR (merge patches)", "gt": True, "value": 18.0},
    ]
}

all_test_info["Test14"] = {
    "enable": True,
    "jobname": "test14",
    "description": "2D self-supervision. Lucchi data. Cross-val. Basic DA. rcan",
    "yaml": "test_14.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 19.0},
        {"type": "regular", "pattern": "Test PSNR (merge patches):", "gt": True, "value": 19.0},
    ]
}

all_test_info["Test15"] = {
    "enable": True,
    "jobname": "test15",
    "description": "2D self-supervision. Lucchi data. Cross-val. Basic DA. mae, masking: random",
    "yaml": "test_15.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 12},
    ]
}

all_test_info["Test16"] = {
    "enable": True,
    "jobname": "test16",
    "description": "2D self-supervision. Lucchi data. Cross-val. Basic DA. mae, masking: grid",
    "yaml": "test16.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 12},
    ]
}

all_test_info["Test17"] = {
    "enable": True,
    "jobname": "test17",
    "description": "3D self-supervision. Lucchi data. Basic DA. resunet++",
    "yaml": "test17.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 10.0},
    ]
}

all_test_info["Test18"] = {
    "enable": True,
    "jobname": "test18",
    "description": "3D self-supervision. Lucchi data. Cross-val. Basic DA. mae, masking: random",
    "yaml": "test18.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": -5.0},
    ]
}

all_test_info["Test19"] = {
    "enable": True,
    "jobname": "test19",
    "description": "3D self-supervision. Lucchi data. Cross-val. Basic DA. mae, masking: grid",
    "yaml": "test19.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": -5.0},
    ]
}

all_test_info["Test20"] = {
    "enable": True,
    "jobname": "test20",
    "description": "2D classification. DermaMNIST 2D data. preprocess: resize, Cross-val. Basic DA. ViT",
    "yaml": "test20.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation Top 5 accuracy:", "gt": True, "value": 0.9},
    ]
}

all_test_info["Test21"] = {
    "enable": True,
    "jobname": "test21",
    "description": "2D classification. butterfly data. preprocess: resize. Basic DA. efficientnet_b1",
    "yaml": "test21.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation Top 5 accuracy:", "gt": True, "value": 0.7},
    ]
}

all_test_info["Test22"] = {
    "enable": True,
    "jobname": "test22",
    "description": "3D classification. DermaMNIST 3D data. preprocess: resize, Cross-val. Basic DA. simple_cnn",
    "yaml": "test22.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation Top 5 accuracy:", "gt": True, "value": 0.7},
    ]
}

all_test_info["Test23"] = {
    "enable": True,
    "jobname": "test23",
    "description": "3D classification. DermaMNIST 3D data. preprocess: resize. Basic DA. simple_cnn",
    "yaml": "test23.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation Top 5 accuracy:", "gt": True, "value": 0.7},
    ]
}

all_test_info["Test24"] = {
    "enable": True,
    "jobname": "test24",
    "description": "2D image to image. Dapi 2D data. preprocess: resize, Cross-val. Basic DA. multiresunet",
    "yaml": "test24.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test PSNR (merge patches):", "gt": True, "value": 19.0},
    ]
}

all_test_info["Test25"] = {
    "enable": True,
    "jobname": "test25",
    "description": "2D image to image. lightmycells 2D data. extract random. val and train not in memory. Basic DA. UNETR",
    "yaml": "test25.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 5.5},
    ]
}

all_test_info["Test26"] = {
    "enable": True,
    "jobname": "test26",
    "description": "3D Detection. Zarr 3D data (Brainglobe). in memory false. zero_mean_unit_variance norm, percentile norm, per image. "
        "filter_samples: foreground. warmupcosine. Basic DA. resunet. test by chunks: Zarr. Post-proc: remove close points",
    "yaml": "test_26.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test F1 (merge patches)", "gt": True, "value": 0.35},
    ]
}

all_test_info["Test27"] = {
    "enable": True,
    "jobname": "test27",
    "description": "3D Instance seg. Zarr 3D data SNEMI. in memory false. input zarr multiple data raw: 'volumes.raw'"
        "warmupcosine. inference, by chunks, zarr multiple data, workflow process: entire pred.",
    "yaml": "test_27.yaml",
    "internal_checks": [
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1",
            "gt": True, "value": 0.25},
    ]
}

all_test_info["Test28"] = {
    "enable": True,
    "jobname": "test28",
    "description": "3D Image to image. Nuclear_Pore_complex_3D data. in memory true. val 0.1 of train.",
    "yaml": "test_28.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 17.0},
    ]
}

all_test_info["Test29"] = {
    "enable": True,
    "jobname": "test29",
    "description": "2D instance segmentation. BMZ 'stupendous-blowfish' model import, inference and export. "
        "zero_mean_unit_variance + format_version: 0.5.3 ",
    "yaml": "test_29.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test IoU (F channel) (merge patches):", "gt": True, "value": 0.6},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name": "test_model_stupendous-blowfish.zip"},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1",
            "gt": True, "value": 0.85},
    ]
}

all_test_info["Test30"] = {
    "enable": True,
    "jobname": "test30",
    "description": "2D instance segmentation. BMZ 'hiding-blowfish' model import, inference and export."
        "scale_range + format_version: 0.4.10",
    "yaml": "test_30.yaml",
    "internal_checks": [
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name": "test_model_hiding-blowfish.zip"},
    ]
}

all_test_info["Test31"] = {
    "enable": True,
    "jobname": "test31",
    "description": "2D instance segmentation. BMZ 'frank-boar' model import, finetunning and export (reusing model original info)."
        "zero_mean_unit_variance + format_version: 0.5.3 ",
    "yaml": "test_31.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test IoU (F channel) (merge patches):", "gt": True, "value": 0.7},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name": "2D U-NeXt V1 for nucleus segmentation.zip"},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1",
            "gt": True, "value": 0.9},
    ]
}

all_test_info["Test32"] = {
    "enable": True,
    "jobname": "test32",
    "description": "2D instance segmentation. Export BiaPy model to BMZ format",
    "yaml": "test_32.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test IoU (F channel) (merge patches):", "gt": True, "value": 0.35},
        {"type": "BMZ", "pattern": "Package path:", "bmz_package_name":  "biapy_model.zip"},
    ]
}


all_test_info["Test33"] = {
    "enable": True,
    "jobname": "test33",
    "description": "2D image to image. lightmycells 2D data. preprocess: resize. Val in memory, train not in memory. Basic DA. attention_unet",
    "yaml": "test_33.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Validation PSNR:", "gt": True, "value": 8.5},
    ]
}

all_test_info["Test34"] = {
    "enable": True,
    "jobname": "test34",
    "description": "2D Instance seg. Conic 2D data (multihead). Basic DA. BC (auto). resunet++. ",
    "yaml": "test_34.yaml",
    "internal_checks": [
        {"type": "regular", "pattern": "Test IoU (F channel) (merge patches):", "gt": True, "value": 0.35},
        {"type": "regular", "pattern": "Merge patches classification IoU:", "gt": True, "value": 0.1},
        {"type": "DatasetMatching", "pattern": "DatasetMatching(criterion='iou', thresh=0.3,", "nApparition": 1, "metric": "f1",
            "gt": True, "value": 0.45},
    ]
}

################################################
# NO-DEVS: DO NOT TOUCH BELOW THIS LINE
################################################

results_folder = os.path.join(data_folder,  "output")

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

instance_seg_2d_affable_shark_data_link = "https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip"
instance_seg_2d_affable_shark_data_filename = "dsb2018.zip"
instance_seg_2d_affable_shark_data_outpath = os.path.join(inst_seg_folder, "dsb2018")

# 3D
instance_seg_3d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/instance_segmentation/3d_instance_segmentation.yaml"
instance_seg_3d_template_local = os.path.join(inst_seg_folder, "3d_instance_segmentation.yaml")
instance_seg_3d_data_drive_link = "https://drive.google.com/uc?id=1fdL35ZTNw5hhiKau1gadaGu-rc5ZU_C7"
instance_seg_3d_data_filename = "demo3D_3D.zip"
instance_seg_3d_data_outpath = os.path.join(inst_seg_folder, "demo3D_3D")

instance_seg_cyst_data_zenodo_link = "https://zenodo.org/records/10973241/files/CartoCell.zip?download=1"
instance_seg_cyst_data_filename = "CartoCell.zip"
instance_seg_cyst_data_outpath = os.path.join(inst_seg_folder, "CartoCell")

instance_seg_snemi_zarr_data_drive_link = "https://drive.google.com/uc?id=1Ralex5SvYUZbXoDkWoaCjb6d_iWuuOHp"
instance_seg_snemi_zarr_data_filename = "snemi_zarr.zip"
instance_seg_snemi_zarr_data_outpath = os.path.join(inst_seg_folder, "snemi_zarr")

# MitoEM (for BMZ tests)
instance_seg_mitoem_data_drive_link = "https://drive.google.com/uc?id=1xrSsK23-2KfxCanaNJD7dldewWboKIw5"
instance_seg_mitoem_data_filename = "MitoEM_human_2d_toy_data.zip"
instance_seg_mitoem_data_outpath = os.path.join(inst_seg_folder, "MitoEM_human_2d_toy_data")

# Conic for multihead
instance_seg_conic_data_drive_link = "https://drive.google.com/uc?id=1QGV0gP8N8B8-EmcAPNQAudr2dqXYzhss"
instance_seg_conic_data_filename = "Conic.zip"
instance_seg_conic_data_outpath = os.path.join(inst_seg_folder, "Conic")

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
denoising_2d_data_drive_link = "https://drive.google.com/uc?id=1ZCNBWkOJc4XOtfKHP7M0g1yIVzqtwS76"
denoising_2d_data_filename = "Noise2Void_RGB.zip"
denoising_2d_data_outpath = os.path.join(denoising_folder, "Noise2Void_RGB")
# 3D
denoising_3d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/denoising/3d_denoising.yaml"
denoising_3d_template_local = os.path.join(denoising_folder, "3d_denoising.yaml")
denoising_3d_data_drive_link = "https://drive.google.com/uc?id=1OIjnUoJKdnbClBlpzk7V5R8wtoLont-r"
denoising_3d_data_filename = "flywing3D.zip"
denoising_3d_data_outpath = os.path.join(denoising_folder, "flywing3D")

###########
# SR
###########
super_resolution_folder = os.path.join(data_folder, "sr")

# 2D
super_resolution_2d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/super-resolution/2d_super-resolution.yaml"
super_resolution_2d_template_local = os.path.join(super_resolution_folder, "2d_super_resolution.yaml")
super_resolution_2d_data_drive_link = "https://drive.google.com/uc?id=1rtrR_jt8hcBEqvwx_amFBNR7CMP5NXLo"
super_resolution_2d_data_filename = "sr_data_2D.zip"
super_resolution_2d_data_outpath = os.path.join(super_resolution_folder, "sr_data_2D")
# 3D
super_resolution_3d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/super-resolution/3d_super-resolution.yaml"
super_resolution_3d_template_local = os.path.join(super_resolution_folder, "3d_super_resolution.yaml")
super_resolution_3d_data_drive_link = "https://drive.google.com/uc?id=1TfQVK7arJiRAVmKHRebsfi8NEas8ni4s"
super_resolution_3d_data_filename = "sr_data_3D.zip"
super_resolution_3d_data_outpath = os.path.join(super_resolution_folder, "sr_data_3D")

###########
# SSL
###########
self_supervision_folder = os.path.join(data_folder, "ssl")

# 2D
self_supervision_2d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/self-supervised/2d_self-supervised.yaml"
self_supervision_2d_template_local = os.path.join(self_supervision_folder, "2d_self_supervision.yaml")
self_supervision_2d_data_drive_link = semantic_2d_data_drive_link
self_supervision_2d_data_filename = "fibsem_epfl_2D.zip"
self_supervision_2d_data_outpath = os.path.join(self_supervision_folder, "fibsem_epfl_2D")
self_supervision_2d_checkpoint_link = "https://drive.google.com/uc?id=1bLB-oYx0JFAvSGv1Fa0F-vK26U_HlPtQ"
self_supervision_2d_checkpoint_test14 = os.path.join(self_supervision_folder, "test14_checkpoint.pth")
# 3D
self_supervision_3d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/self-supervised/3d_self-supervised.yaml"
self_supervision_3d_template_local = os.path.join(self_supervision_folder, "3d_self_supervision.yaml")
self_supervision_3d_data_drive_link = semantic_3d_data_drive_link
self_supervision_3d_data_filename = "fibsem_epfl_3D.zip"
self_supervision_3d_data_outpath = os.path.join(self_supervision_folder, "fibsem_epfl_3D")


###########
# Classification
###########
classification_folder = os.path.join(data_folder, "classification")

# 2D
classification_2d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/classification/2d_classification.yaml"
classification_2d_template_local = os.path.join(classification_folder, "2d_classification.yaml")
classification_2d_data_drive_link = "https://drive.google.com/uc?id=15_pnH4_tJcwhOhNqFsm26NQuJbNbFSIN"
classification_2d_data_filename = "DermaMNIST_2D.zip"
classification_2d_data_outpath = os.path.join(classification_folder, "DermaMNIST_2D")

# 2D (natural images)
classification_butterfly_2d_data_drive_link = "https://drive.google.com/uc?id=1m4_3UAgUsZ8FDjB4HyfA50Sht7_XkfdB"
classification_butterfly_2d_data_filename = "butterfly_data.zip"
classification_butterfly_2d_data_outpath = os.path.join(classification_folder, "butterfly_data")

# 3D
classification_3d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/classification/3d_classification.yaml"
classification_3d_template_local = os.path.join(classification_folder, "3d_classification.yaml")
classification_3d_data_drive_link = "https://drive.google.com/uc?id=1pypWJ4Z9sRLPlVHbG6zpwmS6COkm3wUg"
classification_3d_data_filename = "DermaMNIST_3D.zip"
classification_3d_data_outpath = os.path.join(classification_folder, "DermaMNIST_3D")


###########
# I2I
###########
image_to_image_folder = os.path.join(data_folder, "image_to_image")

# 2D
image_to_image_2d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/image-to-image/2d_image-to-image.yaml"
image_to_image_2d_template_local = os.path.join(image_to_image_folder, "2d_image_to_image.yaml")
image_to_image_2d_data_drive_link = "https://drive.google.com/uc?id=1L8AXNjh0_updVI3-v1duf6CbcZb8uZK7"
image_to_image_2d_data_filename = "Dapi_dataset.zip"
image_to_image_2d_data_outpath = os.path.join(image_to_image_folder, "Dapi_dataset")

# 2D ligthmycells
image_to_image_light_2d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/image-to-image/lightmycells/lightmycells_actin.yaml"
image_to_image_light_2d_template_local = os.path.join(image_to_image_folder, "2d_image_to_image_light.yaml")
image_to_image_light_2d_data_drive_link = "https://drive.google.com/uc?id=1SU4u-bcM1ZaDzEYg-d8W3zP6Yq2o8eKV"
image_to_image_light_2d_data_filename = "reduced_actin_lightmycells.zip"
image_to_image_light_2d_data_outpath = os.path.join(image_to_image_folder, "reduced_actin_lightmycells")

# 3D
image_to_image_3d_template = "https://raw.githubusercontent.com/BiaPyX/BiaPy/master/templates/image-to-image/3d_image-to-image.yaml"
image_to_image_3d_template_local = os.path.join(image_to_image_folder, "3d_image_to_image.yaml")
image_to_image_3d_data_drive_link = "https://drive.google.com/uc?id=1jL0bn2X3OFaV5T-6KR1g6fPDllH-LWzm"
image_to_image_3d_data_filename = "Nuclear_Pore_complex_3D.zip"
image_to_image_3d_data_outpath = os.path.join(image_to_image_folder, "Nuclear_Pore_complex_3D")

def download_drive_file(drive_link, out_filename, attempts=5):
    """
    Try a few times to donwload a file from Drive using gdown (as sometimes it crashes randomly)
    """
    for i in range(attempts):
        print(f"Trying to download {drive_link} (attempt {i+1})")
        try:
            gdown.download(drive_link, out_filename, quiet=True)
        except Exception as e: 
            print(e)
            time.sleep(5)

        if os.path.exists(out_filename):
            break

if not os.path.exists(biapy_folder):
    raise ValueError(f"BiaPy not found in: {biapy_folder}")

###################
# Semantic seg.
###################

# General things: 2D Data + YAML donwload
if not os.path.exists(semantic_2d_data_outpath) and (all_test_info["Test1"]["enable"] or\
    all_test_info["Test3"]["enable"]):
    print("Downloading 2D semantic seg. data . . .")

    os.makedirs(semantic_folder, exist_ok=True)
    os.chdir(semantic_folder)

    download_drive_file(semantic_2d_data_drive_link, semantic_2d_data_filename)

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
    download_drive_file(semantic_3d_data_drive_link, semantic_3d_data_filename)

    with ZipFile(os.path.join(semantic_folder, semantic_3d_data_filename), 'r') as zObject:
        zObject.extractall(path=semantic_3d_data_outpath)

    if not os.path.exists(semantic_3d_template_local):
        print("Downloading semantic seg. YAML . . .")
        _, _ = urllib.request.urlretrieve(semantic_3d_template, filename=semantic_3d_template_local)

###############
# Instance seg.
###############

# General things: 2D Data + YAML donwload
if ( not os.path.exists(instance_seg_2d_data_outpath) and
    (
        all_test_info["Test4"]["enable"] 
        or all_test_info["Test31"]["enable"]
    )
):
    print("Downloading 2D instance seg. data [Test3/Test31] . . .")
    os.makedirs(inst_seg_folder, exist_ok=True)
    os.chdir(inst_seg_folder)

    download_drive_file(instance_seg_2d_data_drive_link, instance_seg_2d_data_filename)

    with ZipFile(os.path.join(inst_seg_folder, instance_seg_2d_data_filename), 'r') as zObject:
        zObject.extractall(path=instance_seg_2d_data_outpath)

if (
    not os.path.exists(instance_seg_2d_affable_shark_data_outpath)
    and all_test_info["Test29"]["enable"] 
):
    print("Downloading 2D instance seg. data [Test29] (affable-shark data) . . .")
    os.makedirs(inst_seg_folder, exist_ok=True)
    os.chdir(inst_seg_folder)

    _, _ = urllib.request.urlretrieve(instance_seg_2d_affable_shark_data_link, filename=instance_seg_2d_affable_shark_data_filename)

    with ZipFile(os.path.join(inst_seg_folder, instance_seg_2d_affable_shark_data_filename), 'r') as zObject:
        zObject.extractall(path=instance_seg_2d_affable_shark_data_outpath)

if (
    not os.path.exists(instance_seg_conic_data_outpath)
    and all_test_info["Test34"]["enable"]
):
    print("Downloading 2D instance seg. data [Test34] (CoNIC data) . . .")
    os.makedirs(inst_seg_folder, exist_ok=True)
    os.chdir(inst_seg_folder)

    download_drive_file(instance_seg_conic_data_drive_link, instance_seg_conic_data_filename)

    with ZipFile(os.path.join(inst_seg_folder, instance_seg_conic_data_filename), 'r') as zObject:
        zObject.extractall(path=instance_seg_conic_data_outpath)

if (
    (all_test_info["Test4"]["enable"] 
    or all_test_info["Test29"]["enable"] 
    or all_test_info["Test31"]["enable"]
    or all_test_info["Test34"]["enable"])
    and not os.path.exists(instance_seg_2d_template_local)
):
    print("Downloading 2D instance seg. data . . .")
    os.makedirs(inst_seg_folder, exist_ok=True)
    os.chdir(inst_seg_folder)

    if not os.path.exists(instance_seg_2d_template_local):
        print("Downloading instance seg. YAML . . .")
        _, _ = urllib.request.urlretrieve(instance_seg_2d_template, filename=instance_seg_2d_template_local)

# General things: 3D Data + YAML donwload
if not os.path.exists(instance_seg_3d_data_outpath) and all_test_info["Test5"]["enable"]:
    print("Downloading 3D instance seg. data . . .")

    os.makedirs(inst_seg_folder, exist_ok=True)
    os.chdir(inst_seg_folder)
    download_drive_file(instance_seg_3d_data_drive_link, instance_seg_3d_data_filename)

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

# General things: SNEMI 3D Data + YAML donwload
if not os.path.exists(instance_seg_snemi_zarr_data_outpath) and all_test_info["Test27"]["enable"]:
    print("Downloading 3D snemi zarr data . . .")

    os.makedirs(inst_seg_folder, exist_ok=True)
    os.chdir(inst_seg_folder)
    download_drive_file(instance_seg_snemi_zarr_data_drive_link, instance_seg_snemi_zarr_data_filename)

    with ZipFile(os.path.join(inst_seg_folder, instance_seg_snemi_zarr_data_filename), 'r') as zObject:
        zObject.extractall(path=instance_seg_snemi_zarr_data_outpath)

    if not os.path.exists(instance_seg_3d_template_local):
        print("Downloading instance seg. YAML . . .")
        _, _ = urllib.request.urlretrieve(instance_seg_3d_template, filename=instance_seg_3d_template_local)

# General things: MitoEM 2D Data + YAML donwload
if not os.path.exists(instance_seg_mitoem_data_outpath) and (all_test_info["Test30"]["enable"] or\
    all_test_info["Test32"]["enable"]):
    print("Downloading 2D MitoEM data . . .")

    os.makedirs(inst_seg_folder, exist_ok=True)
    os.chdir(inst_seg_folder)
    download_drive_file(instance_seg_mitoem_data_drive_link, instance_seg_mitoem_data_filename)

    with ZipFile(os.path.join(inst_seg_folder, instance_seg_mitoem_data_filename), 'r') as zObject:
        zObject.extractall(path=instance_seg_mitoem_data_outpath)

    if not os.path.exists(instance_seg_2d_template_local):
        print("Downloading instance seg. YAML . . .")
        _, _ = urllib.request.urlretrieve(instance_seg_2d_template, filename=instance_seg_2d_template_local)

###########
# Detection
###########

# General things: 2D Data + YAML donwload
if not os.path.exists(detection_2d_data_outpath) and all_test_info["Test7"]["enable"]:
    print("Downloading 2D detection data . . .")

    os.makedirs(detection_folder, exist_ok=True)
    os.chdir(detection_folder)
    download_drive_file(detection_2d_data_drive_link, detection_2d_data_filename)
    
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
    download_drive_file(detection_3d_data_drive_link, detection_3d_data_filename)

    with ZipFile(os.path.join(detection_folder, detection_3d_data_filename), 'r') as zObject:
        zObject.extractall(path=detection_3d_data_outpath)

    if not os.path.exists(detection_3d_template_local):
        print("Downloading detection YAML . . .")
        _, _ = urllib.request.urlretrieve(detection_3d_template, filename=detection_3d_template_local)

# General things: 3D Brainglobe Data + YAML donwload
if not os.path.exists(detection_3d_brainglobe_data_outpath) and (all_test_info["Test11"]["enable"] or\
    all_test_info["Test26"]["enable"]):
    print("Downloading 3D Brainglobe detection data . . .")

    os.makedirs(detection_folder, exist_ok=True)
    os.chdir(detection_folder)
    download_drive_file(detection_3d_brainglobe_data_drive_link, detection_3d_brainglobe_data_filename)

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
    download_drive_file(denoising_2d_data_drive_link, denoising_2d_data_filename)

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
    download_drive_file(denoising_3d_data_drive_link, denoising_3d_data_filename)

    with ZipFile(os.path.join(denoising_folder, denoising_3d_data_filename), 'r') as zObject:
        zObject.extractall(path=denoising_3d_data_outpath)

    if not os.path.exists(denoising_3d_template_local):
        print("Downloading denoising YAML . . .")
        _, _ = urllib.request.urlretrieve(denoising_3d_template, filename=denoising_3d_template_local)

###########
# SR
###########

# General things: 2D Data + YAML donwload
if not os.path.exists(super_resolution_2d_data_outpath) and all_test_info["Test12"]["enable"]:
    print("Downloading 2D super_resolution data . . .")

    os.makedirs(super_resolution_folder, exist_ok=True)
    os.chdir(super_resolution_folder)
    download_drive_file(super_resolution_2d_data_drive_link, super_resolution_2d_data_filename)

    with ZipFile(os.path.join(super_resolution_folder, super_resolution_2d_data_filename), 'r') as zObject:
        zObject.extractall(path=super_resolution_2d_data_outpath)

    if not os.path.exists(super_resolution_2d_template_local):
        print("Downloading super_resolution YAML . . .")
        _, _ = urllib.request.urlretrieve(super_resolution_2d_template, filename=super_resolution_2d_template_local)

# General things: 3D Data + YAML donwload
if not os.path.exists(super_resolution_3d_data_outpath) and all_test_info["Test13"]["enable"]:
    print("Downloading 3D super_resolution data . . .")

    os.makedirs(super_resolution_folder, exist_ok=True)
    os.chdir(super_resolution_folder)
    download_drive_file(super_resolution_3d_data_drive_link, super_resolution_3d_data_filename)

    with ZipFile(os.path.join(super_resolution_folder, super_resolution_3d_data_filename), 'r') as zObject:
        zObject.extractall(path=super_resolution_3d_data_outpath)

    if not os.path.exists(super_resolution_3d_template_local):
        print("Downloading super_resolution YAML . . .")
        _, _ = urllib.request.urlretrieve(super_resolution_3d_template, filename=super_resolution_3d_template_local)

###########
# SSl
###########

# General things: 2D Data + YAML donwload
if not os.path.exists(self_supervision_2d_data_outpath) and (all_test_info["Test14"]["enable"] \
    or all_test_info["Test15"]["enable"] or all_test_info["Test16"]["enable"]):
    print("Downloading 2D self_supervision data . . .")

    os.makedirs(self_supervision_folder, exist_ok=True)
    os.chdir(self_supervision_folder)
    download_drive_file(self_supervision_2d_data_drive_link, self_supervision_2d_data_filename)

    with ZipFile(os.path.join(self_supervision_folder, self_supervision_2d_data_filename), 'r') as zObject:
        zObject.extractall(path=self_supervision_2d_data_outpath)

    if not os.path.exists(self_supervision_2d_template_local):
        print("Downloading self_supervision YAML . . .")
        _, _ = urllib.request.urlretrieve(self_supervision_2d_template, filename=self_supervision_2d_template_local)
    
    if not os.path.exists(self_supervision_2d_checkpoint_test14):
        print("Downloading test14 checkpoint . . .")
        download_drive_file(self_supervision_2d_checkpoint_link, self_supervision_2d_checkpoint_test14)

# General things: 3D Data + YAML donwload
if not os.path.exists(self_supervision_3d_data_outpath) and (all_test_info["Test17"]["enable"] \
    or all_test_info["Test18"]["enable"] or all_test_info["Test19"]["enable"]):
    print("Downloading 3D self_supervision data . . .")

    os.makedirs(self_supervision_folder, exist_ok=True)
    os.chdir(self_supervision_folder)
    download_drive_file(self_supervision_3d_data_drive_link, self_supervision_3d_data_filename)

    with ZipFile(os.path.join(self_supervision_folder, self_supervision_3d_data_filename), 'r') as zObject:
        zObject.extractall(path=self_supervision_3d_data_outpath)

    if not os.path.exists(self_supervision_3d_template_local):
        print("Downloading self_supervision YAML . . .")
        _, _ = urllib.request.urlretrieve(self_supervision_3d_template, filename=self_supervision_3d_template_local)

###########
# Classification
###########

# General things: 2D Data + YAML donwload
if not os.path.exists(classification_2d_data_outpath) and (all_test_info["Test20"]["enable"] \
    or all_test_info["Test21"]["enable"]):
    print("Downloading 2D classification data . . .")

    os.makedirs(classification_folder, exist_ok=True)
    os.chdir(classification_folder)
    download_drive_file(classification_2d_data_drive_link, classification_2d_data_filename)

    with ZipFile(os.path.join(classification_folder, classification_2d_data_filename), 'r') as zObject:
        zObject.extractall(path=classification_2d_data_outpath)

    # Butterfly data
    os.chdir(classification_folder)
    download_drive_file(classification_butterfly_2d_data_drive_link, classification_butterfly_2d_data_filename)

    with ZipFile(os.path.join(classification_folder, classification_butterfly_2d_data_filename), 'r') as zObject:
        zObject.extractall(path=classification_butterfly_2d_data_outpath)

    if not os.path.exists(classification_2d_template_local):
        print("Downloading classification YAML . . .")
        _, _ = urllib.request.urlretrieve(classification_2d_template, filename=classification_2d_template_local)

# General things: 3D Data + YAML donwload
if not os.path.exists(classification_3d_data_outpath) and all_test_info["Test22"]["enable"]:
    print("Downloading 3D classification data . . .")

    os.makedirs(classification_folder, exist_ok=True)
    os.chdir(classification_folder)
    download_drive_file(classification_3d_data_drive_link, classification_3d_data_filename)

    with ZipFile(os.path.join(classification_folder, classification_3d_data_filename), 'r') as zObject:
        zObject.extractall(path=classification_3d_data_outpath)

    if not os.path.exists(classification_3d_template_local):
        print("Downloading classification YAML . . .")
        _, _ = urllib.request.urlretrieve(classification_3d_template, filename=classification_3d_template_local)

###########
# I2I
###########

# General things: 2D Data + YAML donwload
if not os.path.exists(image_to_image_2d_data_outpath) and (all_test_info["Test24"]["enable"] \
    or all_test_info["Test25"]["enable"] or all_test_info["Test33"]["enable"]):
    print("Downloading 2D image_to_image data . . .")

    os.makedirs(image_to_image_folder, exist_ok=True)
    os.chdir(image_to_image_folder)
    download_drive_file(image_to_image_2d_data_drive_link, image_to_image_2d_data_filename)

    with ZipFile(os.path.join(image_to_image_folder, image_to_image_2d_data_filename), 'r') as zObject:
        zObject.extractall(path=image_to_image_2d_data_outpath)

    # Lightmycells data
    os.chdir(image_to_image_folder)
    download_drive_file(image_to_image_light_2d_data_drive_link, image_to_image_light_2d_data_filename)

    with ZipFile(os.path.join(image_to_image_folder, image_to_image_light_2d_data_filename), 'r') as zObject:
        zObject.extractall(path=image_to_image_light_2d_data_outpath)

    if not os.path.exists(image_to_image_2d_template_local):
        print("Downloading image_to_image YAML . . .")
        _, _ = urllib.request.urlretrieve(image_to_image_2d_template, filename=image_to_image_2d_template_local)

    if not os.path.exists(image_to_image_light_2d_template_local):
        print("Downloading image_to_image YAML . . .")
        _, _ = urllib.request.urlretrieve(image_to_image_light_2d_template, filename=image_to_image_light_2d_template_local)

# General things: 3D Data + YAML donwload
if not os.path.exists(image_to_image_3d_data_outpath) and all_test_info["Test28"]["enable"]:
    print("Downloading 3D image to image data . . .")

    os.makedirs(image_to_image_folder, exist_ok=True)
    os.chdir(image_to_image_folder)
    download_drive_file(image_to_image_3d_data_drive_link, image_to_image_3d_data_filename)

    with ZipFile(os.path.join(image_to_image_folder, image_to_image_3d_data_filename), 'r') as zObject:
        zObject.extractall(path=image_to_image_3d_data_outpath)

    if not os.path.exists(image_to_image_3d_template_local):
        print("Downloading image to image YAML . . .")
        _, _ = urllib.request.urlretrieve(image_to_image_3d_template, filename=image_to_image_3d_template_local)


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
    last_lines = last_lines[-min(300,len(last_lines)):]

    # Check if the final message appears there
    finished_good = False
    for line in last_lines:
        if "FINISHED JOB" in line:
            finished_good = True
            break
    logfile.close()

    return finished_good, last_lines

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

def print_result(finished_good, jobname, int_checks):
    # Print the final message to the user accordingly
    if all(finished_good):
        print(f"** {jobname} job: [OK] ({int_checks} internal checks passed)")
    else:
        print(f"** {jobname} job: [ERROR] ({sum(finished_good)} of {int_checks} internal checks passed)")
    print("######")

def runjob(test_info, results_folder, yaml_file, biapy_folder, multigpu=False, bmz=False, bmz_package=None, reuse_original_bmz_config=False):
    # Declare the log file
    jobdir = os.path.join(results_folder, test_info["jobname"])
    jobout_file = os.path.join(jobdir, test_info["jobname"]+"_1")
    os.makedirs(jobdir, exist_ok=True)
    logfile = open(jobout_file, 'w')
    
    # Run the process and wait until finishes
    os.chdir(biapy_folder)
    print(f"Log: {jobout_file}")
    if bmz and bmz_package is not None:
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
            cmd = ["python", "-u", "-m", "torch.distributed.run", "--nproc_per_node=2",
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


test_results = []

#~~~~~~~~~~~~
# Test 1
#~~~~~~~~~~~~
try:
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

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(semantic_2d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(semantic_2d_data_outpath, "data", "train", "label")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(semantic_2d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(semantic_2d_data_outpath, "data", "test", "label")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['LOAD_GT'] = True

        biapy_config['AUGMENTOR']['CONTRAST'] = True
        biapy_config['AUGMENTOR']['BRIGHTNESS'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 50
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'hrnet48'

        biapy_config['LOSS'] = {}
        biapy_config['LOSS']['TYPE'] = "W_CE_DICE"
        
        biapy_config['TEST']['ENABLE'] = True
        biapy_config['TEST']['AUGMENTATION'] = True
        biapy_config['TEST']['FULL_IMG'] = True
        biapy_config['TEST']['ANALIZE_2D_IMGS_AS_3D_STACK'] = True
        biapy_config['TEST']['POST_PROCESSING'] = {}
        biapy_config['TEST']['POST_PROCESSING']['MEDIAN_FILTER'] = True
        biapy_config['TEST']['POST_PROCESSING']['MEDIAN_FILTER_AXIS'] = ["z"]
        biapy_config['TEST']['POST_PROCESSING']['MEDIAN_FILTER_SIZE'] = [5]

        # Save file
        test_file = os.path.join(semantic_folder, all_test_info["Test1"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test1"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test1"], "Test 1")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test1"]["internal_checks"]:
            results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test1"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 1 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 2
#~~~~~~~~~~~~
try:
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

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(semantic_3d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(semantic_3d_data_outpath, "data", "train", "label")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(semantic_3d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(semantic_3d_data_outpath, "data", "test", "label")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['LOAD_GT'] = True

        biapy_config['AUGMENTOR']['CONTRAST'] = True
        biapy_config['AUGMENTOR']['BRIGHTNESS'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 30
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'attention_unet'

        biapy_config['TEST']['ENABLE'] = True
        biapy_config['TEST']['REDUCE_MEMORY'] = True
        
        # Save file
        test_file = os.path.join(semantic_folder, all_test_info["Test2"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test2"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test2"], "Test 2")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test2"]["internal_checks"]:
            results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test2"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 2 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 3
#~~~~~~~~~~~~
try:
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

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(semantic_2d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(semantic_2d_data_outpath, "data", "train", "label")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(semantic_2d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(semantic_2d_data_outpath, "data", "test", "label")
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
        runjob(all_test_info["Test3"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test3"], "Test 2")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test3"]["internal_checks"]:
            results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test3"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 3 execution.")
    print(e)
    test_results.append(False)


#~~~~~~~~~~~~
# Test 4
#~~~~~~~~~~~~
try:
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
        biapy_config['PROBLEM']['INSTANCE_SEG']["WATERSHED"] = {}
        biapy_config['PROBLEM']['INSTANCE_SEG']["WATERSHED"]["DATA_REMOVE_BEFORE_MW"] = True
        biapy_config['PROBLEM']['INSTANCE_SEG']["WATERSHED"]["DATA_REMOVE_SMALL_OBJ_BEFORE"] = 10
        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(instance_seg_2d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(instance_seg_2d_data_outpath, "data", "train", "label")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(instance_seg_2d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(instance_seg_2d_data_outpath, "data", "test", "label")
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
        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['SIGNS'] = [['gt', 'gt']]

        # Save file
        test_file = os.path.join(inst_seg_folder, all_test_info["Test4"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test4"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test4"], "Test 4")
        if not res:
            correct = False
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
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test4"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 4 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 5
#~~~~~~~~~~~~
try:
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
        biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_MW_TH_CONTOUR'] = 0.25
        biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_MW_TH_DISTANCE'] = 0.5
        biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_REMOVE_SMALL_OBJ_BEFORE'] = 20
        biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_REMOVE_BEFORE_MW'] = True
        biapy_config['PROBLEM']['INSTANCE_SEG']['SEED_MORPH_SEQUENCE'] = ['erode','dilate']
        biapy_config['PROBLEM']['INSTANCE_SEG']['SEED_MORPH_RADIUS'] = [2,2]
        biapy_config['PROBLEM']['INSTANCE_SEG']['FORE_EROSION_RADIUS'] = 4
        biapy_config['PROBLEM']['INSTANCE_SEG']['FORE_DILATION_RADIUS'] = 4
        biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_CHECK_MW'] = True
        biapy_config['PROBLEM']['INSTANCE_SEG']['WATERSHED_BY_2D_SLICES'] = True

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(instance_seg_3d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(instance_seg_3d_data_outpath, "data", "train", "label")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(instance_seg_3d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(instance_seg_3d_data_outpath, "data", "test", "label")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['LOAD_GT'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 20
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'unext_v2'

        biapy_config['TEST']['ENABLE'] = True
        biapy_config['TEST']['FULL_IMG'] = False
        biapy_config['TEST']['POST_PROCESSING'] = {}
        biapy_config['TEST']['POST_PROCESSING']['CLEAR_BORDER'] = True

        # Save file
        test_file = os.path.join(inst_seg_folder, all_test_info["Test5"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test5"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test5"], "Test 5")
        if not res:
            correct = False
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
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test5"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 5 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 6
#~~~~~~~~~~~~
try:
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
        biapy_config['TRAIN']['EPOCHS'] = 3
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
        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['SIGNS'] = [['lt'], ['lt']]

        # Save file
        test_file = os.path.join(inst_seg_folder, all_test_info["Test6"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test6"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test6"], "Test 6")
        if not res:
            correct = False
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
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test6"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 6 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 7
#~~~~~~~~~~~~
try:
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
        biapy_config['PROBLEM']['DETECTION']['CENTRAL_POINT_DILATION'] = [3]
        biapy_config['PROBLEM']['DETECTION']['CHECK_POINTS_CREATED'] = False
        biapy_config['PROBLEM']['DETECTION']['DATA_CHECK_MW'] = True

        biapy_config['DATA']['NORMALIZATION'] = {}
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP'] = {}
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP']['ENABLE'] = True
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP']['LOWER_PERC'] = 0.1
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP']['UPPER_PERC'] = 99.8
        biapy_config['DATA']['NORMALIZATION']['TYPE'] = 'zero_mean_unit_variance'

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(detection_2d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(detection_2d_data_outpath, "data", "train", "label")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(detection_2d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(detection_2d_data_outpath, "data", "test", "label")
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
        biapy_config['TEST']['DET_MIN_TH_TO_BE_PEAK'] = 0.7

        biapy_config['TEST']['POST_PROCESSING'] = {}
        biapy_config['TEST']['POST_PROCESSING']['REMOVE_CLOSE_POINTS'] = True
        biapy_config['TEST']['POST_PROCESSING']['REMOVE_CLOSE_POINTS_RADIUS'] = 5
        biapy_config['TEST']['POST_PROCESSING']['DET_WATERSHED'] = True
        biapy_config['TEST']['POST_PROCESSING']['DET_WATERSHED_FIRST_DILATION'] = [2,2]

        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES'] = {}
        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['ENABLE'] = True
        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES'] = {}
        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['ENABLE'] = True
        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['PROPS'] = [['circularity']]
        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['VALUES'] = [[0.4]]
        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['SIGNS'] = [['lt']]

        # Save file
        test_file = os.path.join(detection_folder, all_test_info["Test7"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test7"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test7"], "Test 7")
        if not res:
            correct = False
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
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test7"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 7 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 8
#~~~~~~~~~~~~
try:
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
        biapy_config['PROBLEM']['DETECTION']['CENTRAL_POINT_DILATION'] = [3]
        biapy_config['PROBLEM']['DETECTION']['CHECK_POINTS_CREATED'] = False
        biapy_config['PROBLEM']['DETECTION']['DATA_CHECK_MW'] = True

        biapy_config['DATA']['NORMALIZATION'] = {}
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP'] = {}
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP']['ENABLE'] = True
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP']['LOWER_PERC'] = 0.1
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP']['UPPER_PERC'] = 99.8
        biapy_config['DATA']['NORMALIZATION']['TYPE'] = 'zero_mean_unit_variance'

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(detection_3d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(detection_3d_data_outpath, "data", "train", "label")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(detection_3d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(detection_3d_data_outpath, "data", "test", "label")
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
        biapy_config['TEST']['DET_MIN_TH_TO_BE_PEAK'] = 0.7
        biapy_config['TEST']['DET_TOLERANCE'] = 8

        biapy_config['TEST']['POST_PROCESSING'] = {}
        biapy_config['TEST']['POST_PROCESSING']['REMOVE_CLOSE_POINTS'] = True
        biapy_config['TEST']['POST_PROCESSING']['REMOVE_CLOSE_POINTS_RADIUS'] = 3
        biapy_config['TEST']['POST_PROCESSING']['DET_WATERSHED'] = True
        biapy_config['TEST']['POST_PROCESSING']['DET_WATERSHED_FIRST_DILATION'] = [2,2,1]

        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES'] = {}
        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['ENABLE'] = True
        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES'] = {}
        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['ENABLE'] = True
        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['PROPS'] = [['sphericity']]
        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['VALUES'] = [[0.5]]
        biapy_config['TEST']['POST_PROCESSING']['MEASURE_PROPERTIES']['REMOVE_BY_PROPERTIES']['SIGNS'] = [['lt']]

        # Save file
        test_file = os.path.join(detection_folder, all_test_info["Test8"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test8"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test8"], "Test 8")
        if not res:
            correct = False
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
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test8"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 8 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 11
#~~~~~~~~~~~~
try:
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
        biapy_config['PROBLEM']['DETECTION']['CENTRAL_POINT_DILATION'] = [2]
        biapy_config['PROBLEM']['DETECTION']['CHECK_POINTS_CREATED'] = False

        biapy_config['DATA']['PATCH_SIZE'] = "(20, 128, 128, 2)"
        biapy_config['DATA']['NORMALIZATION'] = {}
        biapy_config['DATA']['NORMALIZATION']['TYPE'] = "scale_range"
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP'] = {}
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP']['ENABLE'] = True
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP']['LOWER_PERC'] = 0.1
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP']['UPPER_PERC'] = 99.8

        biapy_config['DATA']['TRAIN']['INPUT_IMG_AXES_ORDER'] = 'ZYXC'
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES'] = {}
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES']['ENABLE'] = True
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES']['PROPS'] = [['foreground']]
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES']['VALUES'] = [[1.0e-22]]
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES']['SIGNS'] = [["lt"]]
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

        biapy_config['MODEL']['ARCHITECTURE'] = 'hrnet48'
        biapy_config['MODEL']['HRNET_48'] = {}
        biapy_config['MODEL']['HRNET_48']['Z_DOWN'] = False
        del biapy_config['MODEL']['FEATURE_MAPS']
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

        biapy_config['AUGMENTOR']['RANDOM_ROT'] = True
        biapy_config['AUGMENTOR']['AFFINE_MODE'] = 'reflect'
        biapy_config['AUGMENTOR']['ZFLIP'] = True

        biapy_config['TEST']['ENABLE'] = True
        biapy_config['TEST']['FULL_IMG'] = False
        biapy_config['TEST']['DET_MIN_TH_TO_BE_PEAK'] = 0.5
        biapy_config['TEST']['DET_TOLERANCE'] = 8
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
        biapy_config['TEST']['POST_PROCESSING']['REMOVE_CLOSE_POINTS_RADIUS'] = 3

        # Save file
        test_file = os.path.join(detection_folder, all_test_info["Test11"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test11"], results_folder, test_file, biapy_folder, multigpu=True)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test11"], "Test 11")
        if not res:
            correct = False
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
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test11"]["jobname"], int_checks)
        test_results.append(correct)

except Exception as e:
    print("An error occurred during Test 11 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 9
#~~~~~~~~~~~~
try:
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

        biapy_config['DATA']['PATCH_SIZE'] = "(64, 64, 3)"
        biapy_config['DATA']['NORMALIZATION'] = {}
        biapy_config['DATA']['NORMALIZATION']['TYPE'] = 'zero_mean_unit_variance'

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(denoising_2d_data_outpath, "Noise2Void_RGB")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(denoising_2d_data_outpath, "Noise2Void_RGB")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 10
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'unet'
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

        biapy_config['TEST']['ENABLE'] = True
        biapy_config['TEST']['FULL_IMG'] = False

        # Save file
        test_file = os.path.join(denoising_folder, all_test_info["Test9"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test9"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test9"], "Test 9")
        if not res:
            correct = False
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
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test9"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 9 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 10
#~~~~~~~~~~~~
try:
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
        biapy_config['DATA']['NORMALIZATION']['TYPE'] = 'zero_mean_unit_variance'

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
        runjob(all_test_info["Test10"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test10"], "Test 10")
        if not res:
            correct = False
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
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test10"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 10 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 12
#~~~~~~~~~~~~
try:
    if all_test_info["Test12"]["enable"]:
        print("######")
        print("Running Test 12")
        print_inventory(all_test_info["Test12"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(super_resolution_2d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(super_resolution_2d_data_outpath, "data", "train", "LR")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(super_resolution_2d_data_outpath, "data", "train", "HR")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL_NFOLD'] = 5
        biapy_config['DATA']['VAL']['CROSS_VAL_FOLD'] = 2
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(super_resolution_2d_data_outpath, "data", "test", "LR")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(super_resolution_2d_data_outpath, "data", "test", "HR")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['USE_VAL_AS_TEST'] = True
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES'] = {}
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES']['ENABLE'] = True
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES']['PROPS'] = [['mean']]
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES']['VALUES'] = [[10000]]
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES']['SIGNS'] = [["lt"]]

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 20
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'dfcan'
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

        biapy_config['TEST']['ENABLE'] = True
        biapy_config['TEST']['FULL_IMG'] = False

        # Save file
        test_file = os.path.join(super_resolution_folder, all_test_info["Test12"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test12"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test12"], "Test 12")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test12"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test12"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 12 execution.")
    print(e)
    test_results.append(False)
#~~~~~~~~~~~~
# Test 13
#~~~~~~~~~~~~
try:
    if all_test_info["Test13"]["enable"]:
        print("######")
        print("Running Test 13")
        print_inventory(all_test_info["Test13"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(super_resolution_3d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['PROBLEM']['SUPER_RESOLUTION']['UPSCALING'] = "(1,1,1)"

        biapy_config['DATA']['PATCH_SIZE'] = "(6,128,128,1)"
        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(super_resolution_3d_data_outpath, "data", "train", "LR")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(super_resolution_3d_data_outpath, "data", "train", "HR")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL_NFOLD'] = 5
        biapy_config['DATA']['VAL']['CROSS_VAL_FOLD'] = 4
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(super_resolution_3d_data_outpath, "data", "test", "LR")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(super_resolution_3d_data_outpath, "data", "test", "HR")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['USE_VAL_AS_TEST'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 20
        biapy_config['TRAIN']['PATIENCE'] = -1
        biapy_config['TRAIN']['LR_SCHEDULER'] = {}
        biapy_config['TRAIN']['LR_SCHEDULER']['NAME'] = 'onecycle'
        biapy_config['TRAIN']['LR'] = 0.001
        biapy_config['TRAIN']['BATCH_SIZE'] = 16

        biapy_config['MODEL']['ARCHITECTURE'] = 'resunet'
        biapy_config['MODEL']['Z_DOWN'] = [1,1]
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False
        biapy_config['MODEL']['UNET_SR_UPSAMPLE_POSITION'] = "post"
        biapy_config['MODEL']['FEATURE_MAPS'] = [32, 64, 128]    

        biapy_config['TEST']['ENABLE'] = True

        # Save file
        test_file = os.path.join(super_resolution_folder, all_test_info["Test13"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test13"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test13"], "Test 13")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test13"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test13"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 13 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 14
#~~~~~~~~~~~~
try:
    if all_test_info["Test14"]["enable"]:
        print("######")
        print("Running Test 14")
        print_inventory(all_test_info["Test14"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(self_supervision_2d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['DATA']['NORMALIZATION'] = {}
        biapy_config['DATA']['NORMALIZATION']['TYPE'] = "div"
        biapy_config['DATA']['PATCH_SIZE'] = "(256,256,1)"
        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(self_supervision_2d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL_NFOLD'] = 5
        biapy_config['DATA']['VAL']['CROSS_VAL_FOLD'] = 1
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(self_supervision_2d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['USE_VAL_AS_TEST'] = True
        biapy_config['DATA']['TEST']['LOAD_GT'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 1
        biapy_config['TRAIN']['PATIENCE'] = 1
        biapy_config['TRAIN']['BATCH_SIZE'] = 6

        biapy_config['MODEL']['ARCHITECTURE'] = 'rcan'
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = True

        biapy_config['TEST']['ENABLE'] = True

        biapy_config['PATHS'] = {}
        biapy_config['PATHS']['CHECKPOINT_FILE'] = self_supervision_2d_checkpoint_test14
         
        # Save file
        test_file = os.path.join(self_supervision_folder, all_test_info["Test14"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test14"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test14"], "Test 14")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test14"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test14"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 14 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 15
#~~~~~~~~~~~~
try:
    if all_test_info["Test15"]["enable"]:
        print("######")
        print("Running Test 15")
        print_inventory(all_test_info["Test15"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(self_supervision_2d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['PROBLEM']['SELF_SUPERVISED']['PRETEXT_TASK'] = 'masking'

        biapy_config['DATA']['PATCH_SIZE'] = "(128,128,1)"
        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(self_supervision_2d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL_NFOLD'] = 5
        biapy_config['DATA']['VAL']['CROSS_VAL_FOLD'] = 1
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(self_supervision_2d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['USE_VAL_AS_TEST'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 20
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'mae'
        biapy_config['MODEL']['MAE_MASK_TYPE'] = "random"
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

        biapy_config['TEST']['ENABLE'] = True

        # Save file
        test_file = os.path.join(self_supervision_folder, all_test_info["Test15"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test15"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test15"], "Test 15")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test15"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test15"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 15 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 16
#~~~~~~~~~~~~
try:
    if all_test_info["Test16"]["enable"]:
        print("######")
        print("Running Test 16")
        print_inventory(all_test_info["Test16"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(self_supervision_2d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['PROBLEM']['SELF_SUPERVISED']['PRETEXT_TASK'] = 'masking'

        biapy_config['DATA']['PATCH_SIZE'] = "(128,128,1)"
        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(self_supervision_2d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL_NFOLD'] = 5
        biapy_config['DATA']['VAL']['CROSS_VAL_FOLD'] = 1
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(self_supervision_2d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['USE_VAL_AS_TEST'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 20
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'mae'
        biapy_config['MODEL']['MAE_MASK_TYPE'] = "grid"
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

        biapy_config['TEST']['ENABLE'] = True

        # Save file
        test_file = os.path.join(self_supervision_folder, all_test_info["Test16"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test16"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test16"], "Test 16")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test16"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test16"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 16 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 17
#~~~~~~~~~~~~
try:
    if all_test_info["Test17"]["enable"]:
        print("######")
        print("Running Test 17")
        print_inventory(all_test_info["Test17"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(self_supervision_3d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['PROBLEM']['SELF_SUPERVISED']['PRETEXT_TASK'] = 'crappify'

        biapy_config['DATA']['PATCH_SIZE'] = "(20,128,128,1)"
        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(self_supervision_3d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(self_supervision_3d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['PADDING'] = "(4,16,16)"
        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 20
        biapy_config['TRAIN']['BATCH_SIZE'] = 1
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'resunet++'
        biapy_config['MODEL']['Z_DOWN'] = [1,1,1,1]

        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

        biapy_config['TEST']['ENABLE'] = True

        # Save file
        test_file = os.path.join(self_supervision_folder, all_test_info["Test17"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test17"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test17"], "Test 17")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test17"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test17"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:  
    print("An error occurred during Test 17 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 18
#~~~~~~~~~~~~
try:
    if all_test_info["Test18"]["enable"]:
        print("######")
        print("Running Test 18")
        print_inventory(all_test_info["Test18"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(self_supervision_3d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['PROBLEM']['SELF_SUPERVISED']['PRETEXT_TASK'] = 'masking'

        biapy_config['DATA']['PATCH_SIZE'] = "(80,80,80,1)"
        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(self_supervision_3d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(self_supervision_3d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['PADDING'] = "(0,0,0)"

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 20
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'mae'
        biapy_config['MODEL']['MAE_MASK_TYPE'] = "random"
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

        biapy_config['AUGMENTOR']['ENABLE'] = True
        biapy_config['AUGMENTOR']['RANDOM_ROT'] = True

        biapy_config['TEST']['ENABLE'] = True


        # Save file
        test_file = os.path.join(self_supervision_folder, all_test_info["Test18"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test18"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test18"], "Test 18")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test18"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test18"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 18 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 19
#~~~~~~~~~~~~
try:
    if all_test_info["Test19"]["enable"]:
        print("######")
        print("Running Test 19")
        print_inventory(all_test_info["Test19"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(self_supervision_3d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['PROBLEM']['SELF_SUPERVISED']['PRETEXT_TASK'] = 'masking'

        biapy_config['DATA']['PATCH_SIZE'] = "(80,80,80,1)"
        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(self_supervision_3d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(self_supervision_3d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['PADDING'] = "(0,0,0)"

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 20
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'mae'
        biapy_config['MODEL']['MAE_MASK_TYPE'] = "grid"
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

        biapy_config['TEST']['ENABLE'] = True

        # Save file
        test_file = os.path.join(self_supervision_folder, all_test_info["Test19"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test19"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test19"], "Test 19")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test19"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test19"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 19 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 20
#~~~~~~~~~~~~
try:
    if all_test_info["Test20"]["enable"]:
        print("######")
        print("Running Test 20")
        print_inventory(all_test_info["Test20"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(classification_2d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['DATA']['PREPROCESS'] = {}
        biapy_config['DATA']['PREPROCESS']['TRAIN'] = True
        biapy_config['DATA']['PREPROCESS']['VAL'] = True
        biapy_config['DATA']['PREPROCESS']['TEST'] = True
        biapy_config['DATA']['PREPROCESS']['RESIZE'] = {}
        biapy_config['DATA']['PREPROCESS']['RESIZE']['ENABLE'] = True
        biapy_config['DATA']['PREPROCESS']['RESIZE']['OUTPUT_SHAPE'] = "(56,56)"

        biapy_config['DATA']['PATCH_SIZE'] = "(56,56,3)"

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(classification_2d_data_outpath, "data", "train")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['VAL']['FROM_TRAIN'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL_NFOLD'] = 5
        biapy_config['DATA']['VAL']['CROSS_VAL_FOLD'] = 1
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(classification_2d_data_outpath, "data", "test")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['USE_VAL_AS_TEST'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 5
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'vit'
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False
        biapy_config['MODEL']['N_CLASSES'] = 7

        biapy_config['TEST']['ENABLE'] = True

        # Save file
        test_file = os.path.join(classification_folder, all_test_info["Test20"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test20"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test20"], "Test 20")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test20"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test20"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 20 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 21
#~~~~~~~~~~~~
try:
    if all_test_info["Test21"]["enable"]:
        print("######")
        print("Running Test 21")
        print_inventory(all_test_info["Test21"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(classification_2d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['DATA']['PREPROCESS'] = {}
        biapy_config['DATA']['PREPROCESS']['TRAIN'] = True
        biapy_config['DATA']['PREPROCESS']['VAL'] = True
        biapy_config['DATA']['PREPROCESS']['TEST'] = True
        biapy_config['DATA']['PREPROCESS']['RESIZE'] = {}
        biapy_config['DATA']['PREPROCESS']['RESIZE']['ENABLE'] = True
        biapy_config['DATA']['PREPROCESS']['RESIZE']['OUTPUT_SHAPE'] = "(56,56)"

        biapy_config['DATA']['PATCH_SIZE'] = "(56,56,3)"

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(classification_butterfly_2d_data_outpath, "data", "train")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['VAL']['FROM_TRAIN'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(classification_butterfly_2d_data_outpath, "data", "test")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 5
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'efficientnet_b1'
        biapy_config['MODEL']['N_CLASSES'] = 75
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

        biapy_config['TEST']['ENABLE'] = True

        # Save file
        test_file = os.path.join(classification_folder, all_test_info["Test21"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test21"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test21"], "Test 21")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test21"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test21"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 21 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 22
#~~~~~~~~~~~~
try:
    if all_test_info["Test22"]["enable"]:
        print("######")
        print("Running Test 22")
        print_inventory(all_test_info["Test22"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(classification_3d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['DATA']['PREPROCESS'] = {}
        biapy_config['DATA']['PREPROCESS']['TRAIN'] = True
        biapy_config['DATA']['PREPROCESS']['VAL'] = True
        biapy_config['DATA']['PREPROCESS']['TEST'] = True
        biapy_config['DATA']['PREPROCESS']['RESIZE'] = {}
        biapy_config['DATA']['PREPROCESS']['RESIZE']['ENABLE'] = True
        biapy_config['DATA']['PREPROCESS']['RESIZE']['OUTPUT_SHAPE'] = "(56,56,56)"

        biapy_config['DATA']['PATCH_SIZE'] = "(56,56,56,1)"

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(classification_3d_data_outpath, "data", "train")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['VAL']['FROM_TRAIN'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL_NFOLD'] = 5
        biapy_config['DATA']['VAL']['CROSS_VAL_FOLD'] = 3
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(classification_3d_data_outpath, "data", "test")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['USE_VAL_AS_TEST'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 5
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'simple_cnn'
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False
        biapy_config['MODEL']['N_CLASSES'] = 11

        biapy_config['TEST']['ENABLE'] = True

        # Save file
        test_file = os.path.join(classification_folder, all_test_info["Test22"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test22"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test22"], "Test 22")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test22"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test22"]["jobname"], int_checks)
        test_results.append(correct)

except Exception as e:
    print("An error occurred during Test 22 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 23
#~~~~~~~~~~~~
try:
    if all_test_info["Test23"]["enable"]:
        print("######")
        print("Running Test 23")
        print_inventory(all_test_info["Test23"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(classification_3d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['DATA']['PREPROCESS'] = {}
        biapy_config['DATA']['PREPROCESS']['TRAIN'] = True
        biapy_config['DATA']['PREPROCESS']['VAL'] = True
        biapy_config['DATA']['PREPROCESS']['TEST'] = True
        biapy_config['DATA']['PREPROCESS']['RESIZE'] = {}
        biapy_config['DATA']['PREPROCESS']['RESIZE']['ENABLE'] = True
        biapy_config['DATA']['PREPROCESS']['RESIZE']['OUTPUT_SHAPE'] = "(56,56,56)"

        biapy_config['DATA']['PATCH_SIZE'] = "(56,56,56,1)"

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(classification_3d_data_outpath, "data", "train")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['VAL']['FROM_TRAIN'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(classification_3d_data_outpath, "data", "test")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 5
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'simple_cnn'
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False
        biapy_config['MODEL']['N_CLASSES'] = 11

        biapy_config['TEST']['ENABLE'] = True

        # Save file
        test_file = os.path.join(classification_folder, all_test_info["Test23"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test23"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test23"], "Test 23")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test23"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test23"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 23 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 24
#~~~~~~~~~~~~
try:
    if all_test_info["Test24"]["enable"]:
        print("######")
        print("Running Test 24")
        print_inventory(all_test_info["Test24"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(image_to_image_2d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['DATA']['NORMALIZATION'] = {}
        biapy_config['DATA']['NORMALIZATION']['TYPE'] = "div"
        biapy_config['DATA']['REFLECT_TO_COMPLETE_SHAPE'] = False
        biapy_config['DATA']['PATCH_SIZE'] = "(256, 256, 1)"
        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(image_to_image_2d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(image_to_image_2d_data_outpath, "data", "train", "target")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['VAL']['FROM_TRAIN'] = True
        biapy_config['DATA']['VAL']['CROSS_VAL'] = False
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(image_to_image_2d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(image_to_image_2d_data_outpath, "data", "test", "target")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PADDING'] = "(40,40)"

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 10
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'multiresunet'
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

        biapy_config['TEST']['ENABLE'] = True

        # Save file
        test_file = os.path.join(image_to_image_folder, all_test_info["Test24"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test24"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test24"], "Test 24")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test24"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test24"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 24 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 25
#~~~~~~~~~~~~
try:
    if all_test_info["Test25"]["enable"]:
        print("######")
        print("Running Test 25")
        print_inventory(all_test_info["Test25"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(image_to_image_light_2d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['DATA']['NORMALIZATION']['TYPE'] = "zero_mean_unit_variance"
        biapy_config['DATA']['REFLECT_TO_COMPLETE_SHAPE'] = True
        biapy_config['DATA']['PATCH_SIZE'] = "(1024, 1024, 1)"
        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(image_to_image_light_2d_data_outpath, "reduced_actin_lightmycells", "actin", "raw")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(image_to_image_light_2d_data_outpath, "reduced_actin_lightmycells", "actin", "label")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = False
        biapy_config['DATA']['VAL']['FROM_TRAIN'] = False
        biapy_config['DATA']['VAL']['PATH'] = os.path.join(image_to_image_light_2d_data_outpath, "reduced_actin_lightmycells", "actin", "raw")
        biapy_config['DATA']['VAL']['GT_PATH'] = os.path.join(image_to_image_light_2d_data_outpath, "reduced_actin_lightmycells", "actin", "label")
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(image_to_image_light_2d_data_outpath, "reduced_actin_lightmycells", "actin", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(image_to_image_light_2d_data_outpath, "reduced_actin_lightmycells", "actin", "label")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['PADDING'] = "(200,200)"

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 10
        biapy_config['TRAIN']['PATIENCE'] = -1
        biapy_config['TRAIN']['LR_SCHEDULER']['WARMUP_COSINE_DECAY_EPOCHS'] = 5

        biapy_config['AUGMENTOR']['GRIDMASK'] = False
        biapy_config['AUGMENTOR']['ROT90'] = True

        biapy_config['MODEL']['ARCHITECTURE'] = 'unetr'
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

        biapy_config['TEST']['ENABLE'] = True

        del biapy_config['PATHS']

        # Save file
        test_file = os.path.join(image_to_image_folder, all_test_info["Test25"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test25"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test25"], "Test 25")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test25"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test25"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 25 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 26
#~~~~~~~~~~~~
try:
    if all_test_info["Test26"]["enable"]:
        print("######")
        print("Running Test 26")
        print_inventory(all_test_info["Test26"])

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
        biapy_config['PROBLEM']['DETECTION']['CENTRAL_POINT_DILATION'] = [2]
        biapy_config['PROBLEM']['DETECTION']['CHECK_POINTS_CREATED'] = False

        biapy_config['DATA']['PATCH_SIZE'] = "(20, 128, 128, 2)"
        biapy_config['DATA']['NORMALIZATION'] = {}
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP'] = {}
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP']['ENABLE'] = True
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP']['LOWER_PERC'] = 0.1
        biapy_config['DATA']['NORMALIZATION']['PERC_CLIP']['UPPER_PERC'] = 99.8

        biapy_config['DATA']['TRAIN']['INPUT_IMG_AXES_ORDER'] = 'ZYXC'
        biapy_config['DATA']['TRAIN']['INPUT_MASK_AXES_ORDER'] = 'TZCYX'
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES'] = {}
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES']['ENABLE'] = True
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES']['PROPS'] = [['foreground'], ["mean"]]
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES']['VALUES'] = [[1.0e-22], [0.1]]
        biapy_config['DATA']['TRAIN']['FILTER_SAMPLES']['SIGNS'] = [["lt"], ["lt"]]
        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(detection_3d_brainglobe_data_outpath, "data", "3D_ch2ch4_Zarr")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(detection_3d_brainglobe_data_outpath, "data", "y")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = False
        biapy_config['DATA']['VAL']['FROM_TRAIN'] = False
        biapy_config['DATA']['VAL']['PATH'] = os.path.join(detection_3d_brainglobe_data_outpath, "data", "3D_ch2ch4_Zarr")
        biapy_config['DATA']['VAL']['GT_PATH'] = os.path.join(detection_3d_brainglobe_data_outpath, "data", "y")
        biapy_config['DATA']['VAL']['IN_MEMORY'] = False
        biapy_config['DATA']['VAL']['INPUT_IMG_AXES_ORDER'] = 'ZYXC'
        biapy_config['DATA']['VAL']['INPUT_MASK_AXES_ORDER'] = 'TZCYX'
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(detection_3d_brainglobe_data_outpath, "data", "3D_ch2ch4_Zarr")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(detection_3d_brainglobe_data_outpath, "data", "y")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['LOAD_GT'] = True
        biapy_config['DATA']['TEST']['PADDING'] = "(4,18,18)"

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 100
        biapy_config['TRAIN']['BATCH_SIZE'] = 1
        biapy_config['TRAIN']['PATIENCE'] = 20
        biapy_config['TRAIN']['LR'] = 0.0001
        biapy_config['TRAIN']['LR_SCHEDULER'] = {}
        biapy_config['TRAIN']['LR_SCHEDULER']['NAME'] = 'warmupcosine'
        biapy_config['TRAIN']['LR_SCHEDULER']['MIN_LR'] = 5.E-6
        biapy_config['TRAIN']['LR_SCHEDULER']['WARMUP_COSINE_DECAY_EPOCHS'] = 5

        biapy_config['MODEL']['ARCHITECTURE'] = 'resunet'
        biapy_config['MODEL']['Z_DOWN'] = [1,1,1,1]
        del biapy_config['MODEL']['FEATURE_MAPS']
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

        biapy_config['AUGMENTOR']['RANDOM_ROT'] = True
        biapy_config['AUGMENTOR']['AFFINE_MODE'] = 'reflect'
        biapy_config['AUGMENTOR']['ZFLIP'] = True

        biapy_config['TEST']['ENABLE'] = True
        biapy_config['TEST']['FULL_IMG'] = False
        biapy_config['TEST']['DET_MIN_TH_TO_BE_PEAK'] = 0.2
        biapy_config['TEST']['DET_TOLERANCE'] = 8
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
        biapy_config['TEST']['POST_PROCESSING']['REMOVE_CLOSE_POINTS_RADIUS'] = 3

        # Save file
        test_file = os.path.join(detection_folder, all_test_info["Test26"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test26"], results_folder, test_file, biapy_folder, multigpu=True)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test26"], "Test 26")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test26"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test26"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 26 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 27
#~~~~~~~~~~~~
try:
    if all_test_info["Test27"]["enable"]:
        print("######")
        print("Running Test 27")
        print_inventory(all_test_info["Test27"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(instance_seg_3d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_CHANNELS'] = 'BC'
        biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_MW_TH_TYPE'] = "manual"
        biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_MW_TH_BINARY_MASK'] = 0.9
        biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_MW_TH_CONTOUR'] = 0.1
        biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_REMOVE_SMALL_OBJ_BEFORE'] = 20
        biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_REMOVE_BEFORE_MW'] = True
        biapy_config['PROBLEM']['INSTANCE_SEG']['DATA_CHANNEL_WEIGHTS'] = "(0.3, 1)"

        biapy_config['DATA']['PATCH_SIZE'] = "(20, 256, 256, 1)"
        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(instance_seg_snemi_zarr_data_outpath, "data", "train", "zarr")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = False
        biapy_config['DATA']['TRAIN']['INPUT_IMG_AXES_ORDER'] = 'ZYX'
        biapy_config['DATA']['TRAIN']['INPUT_MASK_AXES_ORDER'] = 'ZYX'
        biapy_config['DATA']['TRAIN']['INPUT_ZARR_MULTIPLE_DATA'] = True
        biapy_config['DATA']['TRAIN']['INPUT_ZARR_MULTIPLE_DATA_RAW_PATH'] = 'volumes.raw'
        biapy_config['DATA']['TRAIN']['INPUT_ZARR_MULTIPLE_DATA_GT_PATH'] = 'volumes.labels.neuron_ids'
        biapy_config['DATA']['VAL']['FROM_TRAIN'] = True
        biapy_config['DATA']['VAL']['IN_MEMORY'] = False
        biapy_config['DATA']['VAL']['SPLIT_TRAIN'] = 0.1
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(instance_seg_snemi_zarr_data_outpath, "data", "train", "zarr")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['LOAD_GT'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 80
        biapy_config['TRAIN']['PATIENCE'] = -1
        biapy_config['TRAIN']['OPTIMIZER'] = "ADAMW"
        biapy_config['TRAIN']['LR'] = 1.E-4
        biapy_config['TRAIN']['LR_SCHEDULER'] = {}
        biapy_config['TRAIN']['LR_SCHEDULER']['NAME'] = 'warmupcosine'
        biapy_config['TRAIN']['LR_SCHEDULER']['MIN_LR'] = 5.E-6
        biapy_config['TRAIN']['LR_SCHEDULER']['WARMUP_COSINE_DECAY_EPOCHS'] = 15

        biapy_config['MODEL']['ARCHITECTURE'] = 'resunet'
        biapy_config['AUGMENTOR']['BRIGHTNESS'] = True
        biapy_config['AUGMENTOR']['CONTRAST'] = True
        biapy_config['AUGMENTOR']['MISALIGNMENT'] = True
        biapy_config['AUGMENTOR']['MISSING_SECTIONS'] = True
        biapy_config['AUGMENTOR']['ELASTIC'] = True

        biapy_config['TEST']['ENABLE'] = True
        biapy_config['TEST']['BY_CHUNKS'] = {}
        biapy_config['TEST']['BY_CHUNKS']['ENABLE'] = True
        biapy_config['TEST']['BY_CHUNKS']['FORMAT'] = "Zarr"
        biapy_config['TEST']['BY_CHUNKS']['SAVE_OUT_TIF'] = False
        biapy_config['TEST']['BY_CHUNKS']['INPUT_IMG_AXES_ORDER'] = 'ZYX'
        biapy_config['TEST']['BY_CHUNKS']['INPUT_ZARR_MULTIPLE_DATA'] = True
        biapy_config['TEST']['BY_CHUNKS']['INPUT_ZARR_MULTIPLE_DATA_RAW_PATH'] = 'volumes.raw'
        biapy_config['TEST']['BY_CHUNKS']['INPUT_ZARR_MULTIPLE_DATA_GT_PATH'] = 'volumes.labels.neuron_ids'
        biapy_config['TEST']['BY_CHUNKS']['WORKFLOW_PROCESS'] = {}
        biapy_config['TEST']['BY_CHUNKS']['WORKFLOW_PROCESS']['ENABLE'] = True
        biapy_config['TEST']['BY_CHUNKS']['WORKFLOW_PROCESS']['TYPE'] = "entire_pred"

        # Save file
        test_file = os.path.join(inst_seg_folder, all_test_info["Test27"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test27"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test27"], "Test 27")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test27"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test27"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 27 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 28
#~~~~~~~~~~~~
try:
    if all_test_info["Test28"]["enable"]:
        print("######")
        print("Running Test 28")
        print_inventory(all_test_info["Test28"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(image_to_image_3d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['DATA']['PATCH_SIZE'] = "(6,128,128,1)"
        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(image_to_image_3d_data_outpath, "Nuclear_Pore_complex_3D", "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(image_to_image_3d_data_outpath, "Nuclear_Pore_complex_3D", "data", "train", "label")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['VAL']['FROM_TRAIN'] = True
        biapy_config['DATA']['VAL']['SPLIT_TRAIN'] = 0.1
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(image_to_image_3d_data_outpath, "Nuclear_Pore_complex_3D", "data", "test", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(image_to_image_3d_data_outpath, "Nuclear_Pore_complex_3D", "data", "test", "label")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['PADDING'] = "(0,24,24)"

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 15
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['ARCHITECTURE'] = 'resunet'
        biapy_config['MODEL']['Z_DOWN'] = [1,1,1,1]
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

        biapy_config['TEST']['ENABLE'] = True

        # Save file
        test_file = os.path.join(image_to_image_folder, all_test_info["Test28"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test28"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test28"], "Test 28")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test28"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test28"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 28 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 29
#~~~~~~~~~~~~
try:
    if all_test_info["Test29"]["enable"]:
        print("######")
        print("Running Test 29")
        print_inventory(all_test_info["Test29"])

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

        biapy_config['DATA']['TEST']['PATH'] = os.path.join(instance_seg_2d_affable_shark_data_outpath, "dsb2018", "test", "images")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(instance_seg_2d_affable_shark_data_outpath, "dsb2018", "test", "masks")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['LOAD_GT'] = True

        biapy_config['TRAIN']['ENABLE'] = False

        biapy_config['MODEL']['SOURCE'] = 'bmz'
        biapy_config['MODEL']['BMZ'] = {}
        biapy_config['MODEL']['BMZ']['SOURCE_MODEL_ID'] = 'stupendous-blowfish' 

        biapy_config['TEST']['ENABLE'] = True
        biapy_config['TEST']['FULL_IMG'] = False

        # Save file
        test_file = os.path.join(inst_seg_folder, all_test_info["Test29"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        bmz_package_name = None
        for checks in all_test_info["Test29"]["internal_checks"]:
            if checks["type"] != "regular":
                bmz_package_name = checks['bmz_package_name']
                break
        assert bmz_package_name is not None, "bmz_package_name not found"

        runjob(all_test_info["Test29"], results_folder, test_file, biapy_folder, bmz=True, bmz_package=bmz_package_name)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test29"], "Test 29")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test29"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            elif checks["type"] == "DatasetMatching":
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            else: # BMZ
                results.append(check_bmz_file_created(last_lines, checks["pattern"], os.path.join(bmz_folder, checks['bmz_package_name'])))
            int_checks += 1
            if not results[-1]:
                correct = False
                if checks["type"] in ["regular","DatasetMatching"]:
                    print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))
                else:
                    print("Internal check not passed: BMZ model not found: {}".format(os.path.join(bmz_folder, checks['bmz_package_name'])))

        # Test result
        print_result(results, all_test_info["Test29"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 29 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 30
#~~~~~~~~~~~~
try:
    if all_test_info["Test30"]["enable"]:
        print("######")
        print("Running Test 30")
        print_inventory(all_test_info["Test30"])

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

        biapy_config['DATA']['TEST']['PATH'] = os.path.join(instance_seg_mitoem_data_outpath, "MitoEM_human_2d_toy_data", "toy", "train", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(instance_seg_mitoem_data_outpath, "MitoEM_human_2d_toy_data", "toy", "train", "label")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['LOAD_GT'] = True

        biapy_config['TRAIN']['ENABLE'] = False

        biapy_config['MODEL']['SOURCE'] = 'bmz'
        biapy_config['MODEL']['BMZ'] = {}
        biapy_config['MODEL']['BMZ']['SOURCE_MODEL_ID'] = 'hiding-blowfish'

        biapy_config['TEST']['ENABLE'] = True
        biapy_config['TEST']['FULL_IMG'] = False

        # Save file
        test_file = os.path.join(inst_seg_folder, all_test_info["Test30"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        bmz_package_name = None
        for checks in all_test_info["Test30"]["internal_checks"]:
            if checks["type"] != "regular":
                bmz_package_name = checks['bmz_package_name']
                break
        assert bmz_package_name is not None, "bmz_package_name not found"
        runjob(all_test_info["Test30"], results_folder, test_file, biapy_folder, bmz=True, bmz_package=bmz_package_name)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test30"], "Test 30")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test30"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            elif checks["type"] == "DatasetMatching":
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            else: # BMZ
                results.append(check_bmz_file_created(last_lines, checks["pattern"], os.path.join(bmz_folder, checks['bmz_package_name'])))

            int_checks += 1
            if not results[-1]:
                correct = False
                if checks["type"] in ["regular","DatasetMatching"]:
                    print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))
                else:
                    print("Internal check not passed: BMZ model not found: {}".format(os.path.join(bmz_folder, checks['bmz_package_name'])))

        # Test result
        print_result(results, all_test_info["Test30"]["jobname"], int_checks)
        test_results.append(correct)

except Exception as e:
    print("An error occurred during Test 30 execution.")
    print(e)
    test_results.append(False)
#~~~~~~~~~~~~
# Test 31
#~~~~~~~~~~~~
try:
    if all_test_info["Test31"]["enable"]:
        print("######")
        print("Running Test 31")
        print_inventory(all_test_info["Test31"])

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

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(instance_seg_2d_data_outpath, "data", "train", "raw")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(instance_seg_2d_data_outpath, "data", "train", "label")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(instance_seg_2d_data_outpath, "data", "test", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(instance_seg_2d_data_outpath, "data", "test", "label")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['LOAD_GT'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 5
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['MODEL']['SOURCE'] = 'bmz'
        biapy_config['MODEL']['BMZ'] = {}
        biapy_config['MODEL']['BMZ']['SOURCE_MODEL_ID'] = 'frank-boar' 

        biapy_config['TEST']['ENABLE'] = True
        biapy_config['TEST']['FULL_IMG'] = False

        # Save file
        test_file = os.path.join(inst_seg_folder, all_test_info["Test31"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        bmz_package_name = None
        for checks in all_test_info["Test31"]["internal_checks"]:
            if checks["type"] != "regular":
                bmz_package_name = checks['bmz_package_name']
                break
        assert bmz_package_name is not None, "bmz_package_name not found"
        runjob(all_test_info["Test31"], results_folder, test_file, biapy_folder, bmz=True, bmz_package=bmz_package_name,
            reuse_original_bmz_config=True)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test31"], "Test 31")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test31"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            elif checks["type"] == "DatasetMatching":
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            else: # BMZ
                results.append(check_bmz_file_created(last_lines, checks["pattern"], os.path.join(bmz_folder, checks['bmz_package_name'])))

            int_checks += 1
            if not results[-1]:
                correct = False
                if checks["type"] in ["regular","DatasetMatching"]:
                    print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))
                else:
                    print("Internal check not passed: BMZ model not found: {}".format(os.path.join(bmz_folder, checks['bmz_package_name'])))

        # Test result
        print_result(results, all_test_info["Test31"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 31 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 32
#~~~~~~~~~~~~
try:
    if all_test_info["Test32"]["enable"]:
        print("######")
        print("Running Test 32")
        print_inventory(all_test_info["Test32"])

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

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(instance_seg_mitoem_data_outpath, "MitoEM_human_2d_toy_data", "toy", "train", "raw")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(instance_seg_mitoem_data_outpath, "MitoEM_human_2d_toy_data", "toy", "train", "label")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(instance_seg_mitoem_data_outpath, "MitoEM_human_2d_toy_data", "toy", "train", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(instance_seg_mitoem_data_outpath, "MitoEM_human_2d_toy_data", "toy", "train", "label")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['LOAD_GT'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 50
        biapy_config['TRAIN']['PATIENCE'] = -1

        biapy_config['TEST']['ENABLE'] = True
        biapy_config['TEST']['FULL_IMG'] = False

        # Save file
        test_file = os.path.join(inst_seg_folder, all_test_info["Test32"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        bmz_package_name = None
        for checks in all_test_info["Test32"]["internal_checks"]:
            if checks["type"] != "regular":
                bmz_package_name = checks['bmz_package_name']
                break
        assert bmz_package_name is not None, "bmz_package_name not found"
        runjob(all_test_info["Test32"], results_folder, test_file, biapy_folder, bmz=True, bmz_package=bmz_package_name)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test32"], "Test 32")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test32"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            elif checks["type"] == "DatasetMatching":
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            else: # BMZ
                results.append(check_bmz_file_created(last_lines, checks["pattern"], os.path.join(bmz_folder, checks['bmz_package_name'])))

            int_checks += 1
            if not results[-1]:
                correct = False
                if checks["type"] in ["regular","DatasetMatching"]:
                    print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))
                else:
                    print("Internal check not passed: BMZ model not found: {}".format(os.path.join(bmz_folder, checks['bmz_package_name'])))

        # Test result
        print_result(results, all_test_info["Test32"]["jobname"], int_checks)
        test_results.append(correct)

except Exception as e:
    print("An error occurred during Test 32 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 33
#~~~~~~~~~~~~
try:
    if all_test_info["Test33"]["enable"]:
        print("######")
        print("Running Test 33")
        print_inventory(all_test_info["Test33"])

        #*******************
        # File preparation
        #*******************
        # Open config file
        with open(image_to_image_light_2d_template_local, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(exc)

        biapy_config['DATA']['NORMALIZATION']['TYPE'] = "zero_mean_unit_variance"
        biapy_config['DATA']['REFLECT_TO_COMPLETE_SHAPE'] = True
        biapy_config['DATA']['PATCH_SIZE'] = "(1024, 1024, 1)"
        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(image_to_image_light_2d_data_outpath, "reduced_actin_lightmycells", "actin", "raw")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(image_to_image_light_2d_data_outpath, "reduced_actin_lightmycells", "actin", "label")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['VAL']['FROM_TRAIN'] = False
        biapy_config['DATA']['VAL']['PATH'] = os.path.join(image_to_image_light_2d_data_outpath, "reduced_actin_lightmycells", "actin", "raw")
        biapy_config['DATA']['VAL']['GT_PATH'] = os.path.join(image_to_image_light_2d_data_outpath, "reduced_actin_lightmycells", "actin", "label")
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(image_to_image_light_2d_data_outpath, "reduced_actin_lightmycells", "actin", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(image_to_image_light_2d_data_outpath, "reduced_actin_lightmycells", "actin", "label")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PADDING'] = "(200,200)"

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 50
        biapy_config['TRAIN']['PATIENCE'] = -1
        biapy_config['TRAIN']['LR_SCHEDULER']['WARMUP_COSINE_DECAY_EPOCHS'] = 5

        biapy_config['AUGMENTOR']['GRIDMASK'] = False
        biapy_config['AUGMENTOR']['ROT90'] = True

        biapy_config['MODEL']['ARCHITECTURE'] = 'attention_unet'
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = False

        biapy_config['TEST']['ENABLE'] = True

        del biapy_config['PATHS']

        # Save file
        test_file = os.path.join(image_to_image_folder, all_test_info["Test33"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test33"], results_folder, test_file, biapy_folder)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test33"], "Test 33")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test33"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test33"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 33 execution.")
    print(e)
    test_results.append(False)

#~~~~~~~~~~~~
# Test 34
#~~~~~~~~~~~~
try:
    if all_test_info["Test34"]["enable"]:
        print("######")
        print("Running Test 34")
        print_inventory(all_test_info["Test34"])

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

        biapy_config['DATA']['PATCH_SIZE'] = "(256,256,3)"
        biapy_config['DATA']['NORMALIZATION'] = {}
        biapy_config['DATA']['NORMALIZATION']['TYPE'] = 'zero_mean_unit_variance'

        biapy_config['DATA']['TRAIN']['PATH'] = os.path.join(instance_seg_conic_data_outpath, "conic_instance_subset", "train", "raw")
        biapy_config['DATA']['TRAIN']['GT_PATH'] = os.path.join(instance_seg_conic_data_outpath, "conic_instance_subset", "train", "label")
        biapy_config['DATA']['TRAIN']['IN_MEMORY'] = True
        biapy_config['DATA']['TEST']['PATH'] = os.path.join(instance_seg_conic_data_outpath, "conic_instance_subset", "tiny_test", "raw")
        biapy_config['DATA']['TEST']['GT_PATH'] = os.path.join(instance_seg_conic_data_outpath, "conic_instance_subset", "tiny_test", "label")
        biapy_config['DATA']['TEST']['IN_MEMORY'] = False
        biapy_config['DATA']['TEST']['LOAD_GT'] = True

        biapy_config['TRAIN']['ENABLE'] = True
        biapy_config['TRAIN']['EPOCHS'] = 5
        biapy_config['TRAIN']['PATIENCE'] = 5
        biapy_config['TRAIN']['BATCH_SIZE'] = 4

        biapy_config['MODEL']['ARCHITECTURE'] = 'resunet'
        biapy_config['MODEL']['N_CLASSES'] = 7

        biapy_config['TEST']['ENABLE'] = True
        biapy_config['TEST']['FULL_IMG'] = False

        # Save file
        test_file = os.path.join(inst_seg_folder, all_test_info["Test34"]["yaml"])
        with open(test_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

        # Run
        runjob(all_test_info["Test34"], results_folder, test_file, biapy_folder, multigpu=True)

        # Check
        results = []
        correct = True
        res, last_lines = check_finished(all_test_info["Test34"], "Test 34")
        if not res:
            correct = False
            print("Internal check not passed: seems that it didn't finish")
        results.append(res)
        int_checks = 1
        for checks in all_test_info["Test34"]["internal_checks"]:
            if checks["type"] == "regular":
                results.append(check_value(last_lines, checks["pattern"], checks["value"], checks["gt"]))
            else:
                results.append(check_DatasetMatching(last_lines, checks["pattern"], checks["value"], gt=checks["gt"],
                    value_to_check=checks["nApparition"], metric=checks["metric"]))
            int_checks += 1
            if not results[-1]:
                correct = False
                print("Internal check not passed: {} {} {}".format(checks["pattern"], checks["gt"], checks["value"]))

        # Test result
        print_result(results, all_test_info["Test34"]["jobname"], int_checks)
        test_results.append(correct)
except Exception as e:
    print("An error occurred during Test 34 execution.")
    print(e)
    test_results.append(False)

print("Finish tests!!")

count_correct = 0
for res in test_results:
    if res:
        count_correct += 1

print(f"Test passed: ({count_correct}/{len(test_results)})")

if count_correct == len(test_results):
    sys.exit(0)
else:
    sys.exit(1)
