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
from biapy.data.data_manipulation import read_img_as_ndarray, save_tif, norm_range01, percentile_clip
from skimage.metrics import structural_similarity

parser = argparse.ArgumentParser(description="Calculate SR/I2I workflow metrics",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-input_dir", "--input_dir", required=True, help="Predicted image directory")
parser.add_argument("-output_dir", "--output_dir", required=True, help="Predicted image directory")
parser.add_argument("-gt_dir", "--gt_dir", required=True, help="Ground truth image directory")
args = vars(parser.parse_args())

# python -u ~/thesis/data2/dfranco/BiaPy/biapy/utils/scripts/calculate_similatiry_metrics.py --input_dir /scratch/dfranco/thesis/data2/dfranco/exp_results/fusemycells_2/results/fusemycells_2_1/per_image --output_dir /scratch/dfranco/thesis/data2/dfranco/datasets/FuseMyCells/out_metrics/fusemycells_2 --gt_dir /scratch/dfranco/thesis/data2/dfranco/datasets/FuseMyCells/prepared_data/val/label


# SPECIFIC TO FUSEMYCELLS CHALLENGE
patch_size = (43,100)
data_info = [ 
   {
    "id": "012",
    "z": 114,
    "y": 300,
    "x": 388,
    "scale": 2,
   },
   {
    "id": "134",
    "z": 27,
    "y": 1260,
    "x": 357,
    "scale": 4,
   },
    {
    "id": "322",
    "z": 50,
    "y": 612,
    "x": 468,
    "scale": 4,
   },
       {
    "id": "395",
    "z": 314,
    "y": 408,
    "x": 1344,
    "scale": 3,
   },
]

def ssim(img1, img2):
  return structural_similarity(img1,img2,data_range=1.,full=True, gaussian_weights=True, use_sample_covariance=False, sigma=1.5)

# SPECIFIC TO FUSEMYCELLS CHALLENGE


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
# for n, id_ in enumerate(ids):
    print(f"Analizing image: {id_}")

    # img read
    img_path = os.path.join(args['input_dir'], id_)
    img = read_img_as_ndarray(img_path, is_3d=True)

    # img norm
    img, _, _ = percentile_clip(img, lower=2., upper=99.8)
    img, _ = norm_range01(img, div_using_max_and_scale=True)
    img = img.astype(np.float32)

    # mask read
    mask_path = os.path.join(args['gt_dir'], gt_ids[n])
    mask = read_img_as_ndarray(mask_path, is_3d=True)

    # mask norm
    mask, _, _ = percentile_clip(mask, lower=2., upper=99.8)
    mask, _ = norm_range01(mask, div_using_max_and_scale=True)
    mask = mask.astype(np.float32)

    # images for the paper's figure 
    sample = None
    for data in data_info:
        if data["id"] in id_:
            sample = data
            break 
    if sample is not None:
        index, ssim_map = ssim(mask.squeeze(), img.squeeze())
        
        fname = os.path.splitext(id_)[0]
        image_folder = os.path.join(args['output_dir'], fname)
        aux = img[sample['z']]
        save_tif(np.expand_dims(aux,0), image_folder, [fname+"_slice{}_x.tif".format(sample['z'])])
        aux = mask[sample['z']]
        save_tif(np.expand_dims(aux,0), image_folder, [fname+"_slice{}_y.tif".format(sample['z'])])
        aux = ssim_map[sample['z']]
        save_tif(np.expand_dims(np.expand_dims(aux,0),-1), image_folder, [fname+"_slice{}_ssim.tif".format(sample['z'])])

        spatch_size = [x* sample['scale'] for x in patch_size]
        aux = img[sample['z'],sample['y']:sample['y']+spatch_size[0],sample['x']:sample['x']+spatch_size[1]]
        save_tif(np.expand_dims(aux,0), image_folder, [fname+"_slice{}_x_patch.tif".format(sample['z'])])
        aux = mask[sample['z'],sample['y']:sample['y']+spatch_size[0],sample['x']:sample['x']+spatch_size[1]]
        save_tif(np.expand_dims(aux,0), image_folder, [fname+"_slice{}_y_patch.tif".format(sample['z'])])
        aux = ssim_map[sample['z'],sample['y']:sample['y']+spatch_size[0],sample['x']:sample['x']+spatch_size[1]]
        save_tif(np.expand_dims(np.expand_dims(aux,0),-1), image_folder, [fname+"_slice{}_ssim_patch.tif".format(sample['z'])])
        del aux, ssim_map

    mask = torch.from_numpy(mask).permute((3,0,1,2)).unsqueeze(0)
    img = torch.from_numpy(img).permute((3,0,1,2)).unsqueeze(0)

    for m_name, metric in zip(metrics_to_calculate,metrics_funcs_to_calculate):
        if m_name in ["mse", "mae"]:
            val = metric(img, mask)
        elif m_name == "ssim":
            val = metric(img.to(torch.float32), mask.to(torch.float32))
        elif m_name == "psnr":
            val = metric((img*255).to(torch.uint8), (mask*255).to(torch.uint8))

        val = val.item() if not torch.isnan(val) else 0
        if m_name not in out_metrics:
            out_metrics[m_name] = []
        out_metrics[m_name].append(val)

    print("PSNR: {} - SSIM: {} - MAE: {}".format(out_metrics["psnr"][n],out_metrics["ssim"][n],out_metrics["mae"][n]))

print("Out metrics: {}".format(out_metrics))

for n, id_ in enumerate(ids):
    print("{} - PSNR: {} - SSIM: {} - MAE: {}".format(id_,out_metrics["psnr"][n],out_metrics["ssim"][n],out_metrics["mae"][n]))

print("Mean PSNR: {}".format(np.array(out_metrics["psnr"]).mean()))
print("Mean SSIM: {}".format(np.array(out_metrics["ssim"]).mean()))
print("Mean MAE: {}".format(np.array(out_metrics["mae"]).mean()))

print("FINISHED!")