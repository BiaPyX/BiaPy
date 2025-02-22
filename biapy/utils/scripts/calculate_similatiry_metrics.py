import os
import sys
import torch
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from biapy.data.pre_processing import norm_range01, percentile_clip
from biapy.data.data_manipulation import read_img_as_ndarray

parser = argparse.ArgumentParser(description="Calculate SR/I2I workflow metrics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-input_dir", "--input_dir", required=True, help="Predicted image directory")
parser.add_argument("-gt_dir", "--gt_dir", required=True, help="Ground truth image directory")
args = vars(parser.parse_args())

# python calculate_I2I_metrics.py --input_dir /scratch/dfranco/thesis/data2/dfranco/datasets/FuseMyCells/prepared_data/val/raw --gt_dir /scratch/dfranco/thesis/data2/dfranco/datasets/FuseMyCells/prepared_data/val/label

metrics_to_calculate = ["psnr", "ssim", "mae"]
metrics_funcs_to_calculate = [
       PeakSignalNoiseRatio(), 
       StructuralSimilarityIndexMeasure(),
       MeanAbsoluteError()
    ]
out_metrics = {}

ids = sorted(next(os.walk(args['input_dir']))[2])
gt_ids = sorted(next(os.walk(args['gt_dir']))[2])
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    print(f"Analizing image: {id_}")

    # img read
    img_path = os.path.join(args['input_dir'], id_)
    img = read_img_as_ndarray(img_path, is_3d=True)

    # img norm
    img, _, _ = percentile_clip(img, lower=2., upper=99.8)
    img, _ = norm_range01(img, div_using_max_and_scale=True)
    img = torch.from_numpy(img.copy().astype(np.float32)).permute((3,0,1,2)).unsqueeze(0)

    # mask read
    mask_path = os.path.join(args['gt_dir'], gt_ids[n])
    mask = read_img_as_ndarray(mask_path, is_3d=True)

    # mask norm
    mask, _, _ = percentile_clip(mask, lower=2., upper=99.8)
    mask, _ = norm_range01(mask, div_using_max_and_scale=True)
    mask = torch.from_numpy(mask.copy().astype(np.float32)).permute((3,0,1,2)).unsqueeze(0)

    for m_name, metric in zip(metrics_to_calculate,metrics_funcs_to_calculate):
        if m_name in ["mse", "mae"]:
            val = metric(img, mask)
        elif m_name == "ssim":
            val = metric(img.to(torch.float32), mask.to(torch.float32))
        elif m_name == "psnr":
            val = metric(img*255, mask*255)

        val = val.item() if not torch.isnan(val) else 0
        if m_name not in out_metrics:
            out_metrics[m_name] = []
        out_metrics[m_name].append(val)

print("Out metrics: {}".format(out_metrics))

for n, id_ in enumerate(ids):
    print("{} - PSNR: {} - SSIM: {} - MAE: {}".format(id_,out_metrics["psnr"][n],out_metrics["ssim"][n],out_metrics["mae"][n]))

print("Mean PSNR: {}".format(np.array(out_metrics["psnr"]).mean()))
print("Mean SSIM: {}".format(np.array(out_metrics["ssim"]).mean()))
print("Mean MAE: {}".format(np.array(out_metrics["mae"]).mean()))

print("FINISHED!")