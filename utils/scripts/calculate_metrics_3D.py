import os
import sys
import h5py
import numpy as np
from skimage.io import imread

code_dir = "/home/user/BiaPy"
map_code_dir = "/home/user/mAP_3Dvolume"
input_dir = "/home/user/input"
gt_dir = "/home/user/input_gt"
temporal_dir= "/home/user/TEMP"

iou = True
mAP = True
matching_stats = True
matching_stats_ths = [0.3, 0.5, 0.75]
matching_segcompare = True
verbose = True


gt_partial_files_dir = os.path.join(temporal_dir, "GT")
os.makedirs(gt_partial_files_dir, exist_ok=True)

sys.path.insert(0, code_dir)
from utils.matching import matching, match_using_segCompare
from utils.util import save_tif_pair_discard, wrapper_matching_dataset_lazy, wrapper_matching_segCompare
from engine.metrics import jaccard_index_numpy, voc_calculation

if iou:
    all_iou = 0
    all_ov_iou = 0  # (overall IoU)
if mAP:
    # Prepare mAP call
    sys.path.insert(0, map_code_dir)
    from demo_modified import main as mAP_calculation
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    mAP_50_total = 0
    mAP_75_total = 0
if matching_stats:
    all_matching_stats = []
if matching_segcompare:
    all_matching_stats_segCompare = []

ids = sorted(next(os.walk(input_dir))[2])
for n, id_ in enumerate(ids):
    img = imread(os.path.join(input_dir, id_)).astype(np.int64)
    mask = imread(os.path.join(gt_dir, id_)).astype(np.int64)

    print(" ")
    print("#######################################")
    print("Analizing file {}".format(os.path.join(input_dir, id_)))
    file_dir = os.path.join(temporal_dir, id_)
    os.makedirs(file_dir, exist_ok=True)

    ###########################
    # IoU & VOC (overall IoU) #
    ###########################
    if iou:
        _iou = jaccard_index_numpy((mask>0.5).astype(np.uint8), (img>0.5).astype(np.uint8))
        all_iou += _iou
        _ov_iou = voc_calculation((mask>0.5).astype(np.uint8), (img>0.5).astype(np.uint8), _iou)
        all_ov_iou += _ov_iou
        if verbose: print("Foreground IoU: {} - Overall IoU {}".format(_iou,_ov_iou))

    #######
    # mAP #
    #######
    if mAP:
        # Convert the prediction into an .h5 file
        h5file_name = os.path.join(file_dir, os.path.splitext(id_)[0]+'.h5')
        if verbose: print("Creating prediction h5 file to calculate mAP: {}".format(h5file_name))
        h5f = h5py.File(h5file_name, 'w')
        h5f.create_dataset('dataset', data=img, compression="lzf")
        h5f.close()

        # Create GT H5 file if it does not exist
        gt_f = os.path.join(gt_partial_files_dir, os.path.splitext(id_)[0]+'.h5')
        if not os.path.isfile(gt_f):
            test_file = os.path.join(gt_dir, id_)
            if verbose: print("GT .h5 file needed for mAP calculation is not found in {} so it will be created "
                              "from its mask: {}".format(gt_f, test_file))

            if not os.path.isfile(test_file):
                raise ValueError("The mask is supossed to have the same name as the image")

            if verbose: print("Saving .h5 GT data from array shape: {}".format(mask.squeeze().shape))
            os.makedirs(gt_partial_files_dir, exist_ok=True)
            h5f = h5py.File(gt_f, 'w')
            h5f.create_dataset('dataset', data=mask.squeeze(), compression="lzf")
            h5f.close()

        # Calculate mAP
        args = Namespace(gt_seg=gt_f, predict_seg=h5file_name, predict_score='', threshold="5e3, 3e4", threshold_crumb=64,
                         chunk_size=250, output_name=file_dir, do_txt=1, do_eval=1, slices="-1")
        mAP_calculation(args)

        with open(os.path.join(temporal_dir, id_, 'nucmm_map.txt'), "r") as read_obj:
            for line in read_obj:
                if 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] =' in line:
                    mAP_50_total += float(line.split()[-1])
                if 'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] =' in line:
                    mAP_75_total += float(line.split()[-1])

    ##################
    # matching stats #
    ##################
    if matching_stats:
        r_stats = matching(mask, img, thresh=matching_stats_ths, report_matches=False)
        if verbose: print(r_stats)
        all_matching_stats.append(r_stats)

    #######################
    # matching segcompare #
    #######################
    if matching_segcompare:
        r_stats_segCompare = match_using_segCompare(mask, img)
        if verbose: print(r_stats_segCompare)
        all_matching_stats_segCompare.append(r_stats_segCompare)

print("#################")
print("# FINAL RESULTS #")
print("#################")
print("")

if iou:
    all_iou = all_iou / len(ids)
    all_ov_iou = all_ov_iou / len(ids)
    print("~~~~~~ Foreground IoU & Overall IoU ~~~~~~")
    print("Foreground IoU: {}".format(all_iou))
    print("Overall IoU: {}".format(all_ov_iou))
    print("")

if mAP:
    mAP_50_total = mAP_50_total / len(ids)
    mAP_75_total = mAP_75_total / len(ids)
    print("~~~~~~ mAP ~~~~~~")
    print("Average Precision (AP) - IoU=0.50 : {}".format(mAP_50_total))
    print("Average Precision (AP) - IoU=0.75 : {}".format(mAP_75_total))
    print("")

if matching_stats:
    stats = wrapper_matching_dataset_lazy(all_matching_stats, matching_stats_ths)
    print("~~~~~~ Matching stats ~~~~~~")
    for i in range(len(matching_stats_ths)):
        print("IoU TH={}".format(matching_stats_ths[i]))
        print(stats[i])
    print("")

if matching_segcompare:
    print("~~~~~~ Matching segcompare ~~~~~~")
    stats_segCompare = wrapper_matching_segCompare(all_matching_stats_segCompare)
    print("segCompare segmentation rates:")
    print(stats_segCompare)
    print("")

print("Remember to remove the temporal folder: {}".format(temporal_dir))
