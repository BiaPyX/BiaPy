import argparse
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


def detection_metrics(true, pred, tolerance=10, voxel_size=(1,1,1), verbose=False):
    """Calculate detection metrics based on
       Parameters
       ----------
       true : List of list
           List containing coordinates of ground truth points. E.g. ``[[5,3,2], [4,6,7]]``.
       pred : 4D Tensor
           List containing coordinates of predicted points. E.g. ``[[5,3,2], [4,6,7]]``.
       tolerance : optional, int
           Maximum distance to consider a point as a true positive.
       voxel_size : List of floats
           Physical resolution of each axis. E.g. ``(3.6,1.0,1.0)``.
       Returns
       -------
       metrics : List of strings
           List containing precision, accuracy and F1 between the predicted points and ground truth.
    """

    _true = np.array(true, dtype=np.float32)
    _pred = np.array(pred, dtype=np.float32)

    # Multiply each axis for the its real value
    for i in range(3):
        _true[:,i] *= voxel_size[i]
        _pred[:,i] *= voxel_size[i]

    # Create cost matrix
    distances = distance_matrix(_pred, _true)

    pred_ind, true_ind = linear_sum_assignment(distances)

    TP, FP, FN = 0, 0, 0
    for i in range(len(pred_ind)):
        if distances[pred_ind[i],true_ind[i]] < tolerance:
            TP += 1

    FN = len(_true) - TP
    FP = len(_pred) - TP

    try:
        precision = TP/(TP+FP)
    except:
        precision = 1
    try:
        recall = TP/(TP+FN)
    except:
        recall = 1
    try:
        F1 = 2*((precision*recall)/(precision+recall))
    except:
        F1 = 1

    if verbose:
    	print("Points in ground truth: {}, Points in prediction: {}".format(len(_true), len(_pred)))
    	print("True positives: {}, False positives: {}, False negatives: {}".format(TP, FP, FN))

    return ["Precision", precision, "Recall", recall, "F1", F1]
    
parser = argparse.ArgumentParser(description="Calculate detection metrics between prediction and ground truth CSV files",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("pred_csv", help="napari CSV file with predicted points")
parser.add_argument("gt_csv", help="napari CSV file with ground truth points")
parser.add_argument("tolerance", type=float, help="maximum distance to consider a point as a true positive")

parser.add_argument("-res0", "--resolution_axis0", type=float, default=1.0, help="Voxel size in physical units in axis 0")
parser.add_argument("-res1", "--resolution_axis1", type=float, default=1.0, help="Voxel size in physical units in axis 1")
parser.add_argument("-res2", "--resolution_axis2", type=float, default=1.0, help="Voxel size in physical units in axis 2")
parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")

args = vars(parser.parse_args())

pred_csv = args['pred_csv']
gt_csv = args['gt_csv']
tolerance = args['tolerance']

z_res = args['resolution_axis0']
y_res = args['resolution_axis1']
x_res = args['resolution_axis2']

verbose = args['verbose']

# read prediction CSV
df = pd.read_csv( pred_csv, index_col=False, usecols=['axis-0','axis-1','axis-2'] )


# read ground truth CSV
gt_df = pd.read_csv( gt_csv, index_col=False, usecols=['axis-0','axis-1','axis-2'] )

# extract points as list
pred_points = df.values.tolist()

gt_points = gt_df.values.tolist()

detection_metrics(gt_points, pred_points, tolerance=tolerance, voxel_size=(z_res, y_res, x_res), verbose=True)

