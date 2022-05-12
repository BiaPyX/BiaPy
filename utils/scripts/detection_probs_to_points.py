from scipy.spatial import cKDTree
import argparse
import pandas as pd
from skimage.io import imread
import numpy as np
from skimage.feature import peak_local_max
import time

def remove_close_points( point_list, radius ):
    """Remove all points from ``point_list`` that are at a ``radius``
       or less distance from each other.
       Code adapted from https://codereview.stackexchange.com/questions/196104/removing-neighbors-in-a-point-cloud

       Parameters
       ----------
       point_list : list
           List of 3D points. E.g. ``((0,0,0), (1,1,1)``.

       radius : float
           Radius from each point to decide what points to keep. E.g. ``10``.

       Returns
       -------
       value : list
           New list of points after removing those at a distance of ``radius``
           or less from each other.
    """
    mynumbers = [tuple(point) for point in point_list] 

    tree = cKDTree(mynumbers) # build k-dimensional tree
    
    pairs = tree.query_pairs( radius ) # find all pairs closer than radius
    
    neighbors = {} # create dictionary of neighbors

    for i,j in pairs: # iterate over all pairs
        if i not in neighbors:
            neighbors[i] = {j}
        else:
            neighbors[i].add(j)
        if j not in neighbors:
            neighbors[j] = {i}
        else:
            neighbors[j].add(i)
            
    positions = [i for i in range(0, len( point_list ))]

    keep = []
    discard = set()
    for node in positions:
        if node not in discard: # if node already in discard set: skip
            keep.append(node) # add node to keep list
            discard.update(neighbors.get(node,set())) # add node's neighbors to discard set

    # points to keep
    new_point_list = [ point_list[i] for i in keep]
    
    return new_point_list
    
parser = argparse.ArgumentParser(description="Calculate unique detection points from a probability image",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("input_image", help="probability image obtained from detection model (ex.: ./cells_001.tif)")
parser.add_argument("output_csv", help="output CSV file name (ex.: ./cells_001.csv)")
parser.add_argument("threshold", type=float, help="minimum probability of the peaks to be considered a detection (ex.: 0.5")

parser.add_argument("-r", "--radius", type=float, default=5.0, help="distance between points to be considered the same (in physical units)")
parser.add_argument("-res0", "--resolution_axis0", type=float, default=1.0, help="Voxel size in physical units in axis 0")
parser.add_argument("-res1", "--resolution_axis1", type=float, default=1.0, help="Voxel size in physical units in axis 1")
parser.add_argument("-res2", "--resolution_axis2", type=float, default=1.0, help="Voxel size in physical units in axis 2")
parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")

args = vars(parser.parse_args())

input_img = args['input_image']
out_csv = args['output_csv']
threshold = args['threshold']

radius = args['radius']
z_res = args['resolution_axis0']
x_res = args['resolution_axis1']
y_res = args['resolution_axis2']

verbose = args['verbose']

# read input image
img = imread( input_img )


start_time = time.time()

# calculate probability peaks
points = peak_local_max( img, threshold_abs=threshold, exclude_border=False )

if( verbose ):
    peak_time = time.time()
    print( 'Found local maxima in {} seconds'.format( round(peak_time - start_time, 3) ) )

# create data frame in Napari CSV format with calibration
z_axis = []
y_axis = []
x_axis = []
for i in range(0, len(points)):
    x_axis.append( points[i][0]*x_res )
    y_axis.append( points[i][1]*y_res )
    z_axis.append( points[i][2]*z_res )

df = pd.DataFrame(list(zip(z_axis,x_axis,y_axis)), columns =['axis-0', 'axis-1', 'axis-2'])

# extract calibrated points as list
points = df.values.tolist()

if verbose:
    print( '  Initial number of points: ' + str( len( points ) ) )

start_time = time.time()
# filter points
new_points = remove_close_points( points, radius )
if( verbose ):
    filter_time = time.time()
    print( 'Filtered closed points in {} seconds'.format( round(filter_time - start_time, 3) ) )
    
if verbose:
    print( '  Final number of points: ' + str( len( new_points ) ) )

# remove calibration
z_axis = []
y_axis = []
x_axis = []
for i in range(0, len(new_points)):
    z_axis.append( new_points[i][0]/z_res )
    x_axis.append( new_points[i][1]/x_res )
    y_axis.append( new_points[i][2]/y_res )

# create data frame in Napari CSV format
df_save = pd.DataFrame(list(zip(z_axis,x_axis,y_axis)), columns =['axis-0', 'axis-1', 'axis-2'])
# save to file
df_save.to_csv( out_csv, index=False )

if verbose:
    print( 'Filtered points saved in ' + out_csv )

