import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Remap cityscape dataset classes (from 30 to 19)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-input_label_dir", "--input_label_dir", required=True, help="Directory to the folder where the original cityscape data labels reside")
parser.add_argument("-output_label_dir", "--output_label_dir", required=True, help="Output folder to store the remapped labels")
parser.add_argument("-BiaPy_dir", "--BiaPy_dir", required=True, help="BiaPy directory")
args = vars(parser.parse_args())

# python -u /data/dfranco/BiaPy_contrastive/biapy/utils/scripts/cityscape_remap_labels.py --BiaPy_dir /data/dfranco/BiaPy/ --input_label_dir /data/dfranco/datasets/cityscapes/val/label --output_label_dir /data/dfranco/datasets/cityscapes/val/label_remapped 

sys.path.insert(0, args['BiaPy_dir'])
from biapy.data.data_manipulation import save_tif, read_img_as_ndarray

print("Processing {} folder . . .".format(args['input_label_dir']))
img_ids = sorted(next(os.walk(args['input_label_dir']))[2])

for n, id_ in tqdm(enumerate(img_ids), total=len(img_ids)):
    gt_file_path = os.path.join(args['input_label_dir'], id_)
    print(f"FILE: {gt_file_path}")
    
    # Load image
    img = read_img_as_ndarray(gt_file_path)

    # Remap labels
    # 0: unlabeled, 1: road, 2: sidewalk, 3: building, 4: wall, 5: fence, 6: pole, 7: traffic light, 8: traffic sign,   
    # 9: vegetation, 10: terrain, 11: sky, 12: person, 13: rider, 14: car, 15: truck, 16: bus, 17: train, 18: motorcycle, 19: bicycle
    # Remapping to:
    # 0: unlabeled, 1: road, 2: sidewalk, 3: building, 4: wall, 5: fence, 6: pole, 7: traffic light, 8: traffic sign, 
    # 9: vegetation, 10: terrain, 11: sky, 12: person, 13: rider, 14: car, 15: truck, 16: bus, 17: -1 (train not used), 18: -1 (motorcycle not used),
    # 19: -1 (bicycle not used)

    id_to_trainid = {
        0: 255,    # unlabeled
        1: 255,    # ego vehicle
        2: 255,    # rectification border
        3: 255,    # out of roi
        4: 255,    # static
        5: 255,    # dynamic
        6: 255,    # ground
        7: 0,      # road
        8: 1,      # sidewalk
        9: 255,    # parking
        10: 255,   # rail track
        11: 2,     # building
        12: 3,     # wall
        13: 4,     # fence
        14: 255,   # guard rail
        15: 255,   # bridge
        16: 255,   # tunnel
        17: 5,     # pole
        18: 255,   # polegroup
        19: 6,     # traffic light
        20: 7,     # traffic sign
        21: 8,     # vegetation
        22: 9,     # terrain
        23: 10,    # sky
        24: 11,    # person
        25: 12,    # rider
        26: 13,    # car
        27: 14,    # truck
        28: 15,    # bus
        29: 255,   # caravan
        30: 255,   # trailer
        31: 16,    # train
        32: 17,    # motorcycle
        33: 18,    # bicycle
        -1: 255    # license plate (deprecated/ignored)
    }

    result = np.ones_like(img) * 255
    for k, v in id_to_trainid.items():
        result[img == k] = v
    
    save_tif(np.expand_dims(result,0), args['output_label_dir'], filenames=[id_], verbose=True)


print("Cityscape labels remapping completed.")
print("Remapped labels saved in: {}".format(args['output_label_dir']))
print("You can now use these remapped labels for training your models.")