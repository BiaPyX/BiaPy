import os
import sys
import h5py
import numpy as np
from skimage.io import imread
from tqdm import tqdm

job_name = 'cyst1'
code_dir = "/scratch/dfranco/thesis/data2/dfranco/EM_Image_Segmentation"
map_code_dir = "/scratch/dfranco/thesis/data2/dfranco/mAP_3Dvolume"
input_dir = "/scratch/dfranco/thesis/data2/dfranco/exp_results/cyst1/results/cyst1_1/per_image_instances"
gt_dir = "/scratch/dfranco/thesis/data2/dfranco/datasets/sevilla/cyst/cysts_dataset_150/train/y"
partial_base_dir = "/scratch/dfranco/thesis/data2/dfranco/exp_results/partial_map_files"
partial_files_dir = os.path.join(partial_base_dir, job_name)
gt_partial_files_dir = partial_base_dir+'_gt'

sys.path.insert(0, code_dir)
from data.data_3D_manipulation import crop_3D_data_with_overlap
from utils.util import save_tif_pair_discard

# Prepare mAP call
sys.path.insert(0, map_code_dir)
from demo_modified import main as mAP_calculation
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

mAP_50_total = 0
mAP_75_total = 0
os.makedirs(partial_files_dir, exist_ok=True)
os.makedirs(gt_partial_files_dir, exist_ok=True)
ids = sorted(next(os.walk(input_dir))[2])
for n, id_ in enumerate(ids):
    img = imread(os.path.join(input_dir, id_)).astype(np.int64)
    print("Analizing file {}".format(os.path.join(input_dir, id_)))
    file_dir = os.path.join(partial_files_dir, id_)
    os.makedirs(file_dir, exist_ok=True)

    # Convert the prediction into an .h5 file
    h5file_name = os.path.join(file_dir, os.path.splitext(id_)[0]+'.h5')
    print("Creating prediction h5 file to calculate mAP: {}".format(h5file_name))
    h5f = h5py.File(h5file_name, 'w')
    h5f.create_dataset('dataset', data=img, compression="lzf")
    h5f.close()

    # Create GT H5 file if it does not exist
    gt_f = os.path.join(gt_partial_files_dir, os.path.splitext(id_)[0]+'.h5')
    if not os.path.isfile(gt_f):
        test_file = os.path.join(gt_dir, id_)
        print("GT .h5 file needed for mAP calculation is not found in {} so it will be created "
              "from its mask: {}".format(gt_f, test_file))

        if not os.path.isfile(test_file):
            raise ValueError("The mask is supossed to have the same name as the image")

        _Y = imread(test_file).squeeze()
        print("_Y: {}".format(_Y.shape))

        print("Saving .h5 GT data from array shape: {}".format(_Y.shape))
        os.makedirs(gt_partial_files_dir, exist_ok=True)
        h5f = h5py.File(gt_f, 'w')
        h5f.create_dataset('dataset', data=_Y, compression="lzf")
        h5f.close()

    # Calculate mAP
    args = Namespace(gt_seg=gt_f, predict_seg=h5file_name, predict_score='', threshold="5e3, 3e4", threshold_crumb=64,
                     chunk_size=250, output_name=file_dir, do_txt=1, do_eval=1, slices="-1")
    mAP_calculation(args)

    with open(os.path.join(partial_files_dir, id_, 'nucmm_map.txt'), "r") as read_obj:
        for line in read_obj:
            if 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] =' in line:
                mAP_50_total += float(line.split()[-1])
            if 'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] =' in line:
                mAP_75_total += float(line.split()[-1])

mAP_50_total = mAP_50_total / len(ids)
mAP_75_total = mAP_75_total / len(ids)
print("Average Precision (AP) - IoU=0.50 : {}".format(mAP_50_total))
print("Average Precision (AP) - IoU=0.75 : {}".format(mAP_75_total))

