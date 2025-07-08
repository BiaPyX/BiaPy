import csv
import sys
import numpy as np
from tqdm import tqdm
from skimage.feature import peak_local_max
from tqdm import tqdm

code_dir = "/home/user/BiaPy"
pred_file="/home/user/file.tif"
pred_csv_file="/home/user/file.csv"
gt_csv_file="/home/user/gt.csv"
v_size=(0.4,0.4,2)
ths_rel = np.linspace(0, 1, num=11)
tolerance=10

sys.path.insert(0, code_dir)
from biapy.engine.metrics import detection_metrics
from biapy.data.data_manipulation import imread

cfile = open(gt_csv_file)
csvreader = csv.reader(cfile, delimiter=',')
gt_header = []
gt_header = next(csvreader)
gt_coords = []
for row in csvreader:
    gt_coords.append([int(float(row[1])), int(float(row[2])), int(float(row[3]))])

img, _ = imread(pred_file)

d_precision = []
d_recall = []
d_f1 = []
for i in tqdm(range(len(ths_rel))):
    pred_coords = peak_local_max(img, threshold_abs=ths_rel[i], exclude_border=False)
    d_metrics = detection_metrics(gt_coords, pred_coords, tolerance=tolerance, voxel_size=v_size)
    d_precision.append(d_metrics[1])
    d_recall.append(d_metrics[3])
    d_f1.append(d_metrics[5])

print("THS: {}".format(ths_rel))
print("Precision: {}".format(d_precision))
print("Recall: {}".format(d_recall))
print("F1: {}".format(d_f1))


# Create the plots
import matplotlib.pyplot as plt

plt.plot(ths_rel, d_precision, label ="Precision")
plt.plot(ths_rel, d_recall, label="Recall")
plt.plot(ths_rel, d_f1, label="F1")
plt.legend()
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Score when varying peak threshold')
plt.savefig("score_vary_th.png")
plt.clf()

plt.plot(d_recall, d_precision, label ="Score")
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Recall vs Precision')
plt.savefig("recall_vs_precision.png")
plt.clf()

