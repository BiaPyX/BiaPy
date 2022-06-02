from scipy.spatial import cKDTree
import argparse
import pandas as pd


def remove_close_points( point_list, radius ):
    """Remove all points from ``point_list`` that are at a ``radius``
       or less distance from each other.

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
    
parser = argparse.ArgumentParser(description="Filter points from napari CSV to remove those closer than a given radius",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("input_csv", help="napari CSV file with input points")
parser.add_argument("output", help="output CSV file name")
parser.add_argument("radius", type=float, help="distance between points to be considered the same")

parser.add_argument("-res0", "--resolution_axis0", type=float, default=1.0, help="Voxel size in physical units in axis 0")
parser.add_argument("-res1", "--resolution_axis1", type=float, default=1.0, help="Voxel size in physical units in axis 1")
parser.add_argument("-res2", "--resolution_axis2", type=float, default=1.0, help="Voxel size in physical units in axis 2")
parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")

args = vars(parser.parse_args())

input_csv = args['input_csv']
out_csv = args['output']
radius = args['radius']

z_res = args['resolution_axis0']
y_res = args['resolution_axis1']
x_res = args['resolution_axis2']

verbose = args['verbose']

# read input CSV
df = pd.read_csv( input_csv, index_col=False, usecols=['axis-0','axis-1','axis-2'] )

# apply calibration
df['axis-0'] = df['axis-0'].apply(lambda z: z * z_res)
df['axis-1'] = df['axis-1'].apply(lambda y: y * y_res)
df['axis-2'] = df['axis-2'].apply(lambda x: x * x_res)

# extract points as list
points = df.values.tolist()

if verbose:
    print( 'Initial number of points: ' + str( len( points ) ) )

# filter points
new_points = remove_close_points( points, radius )

if verbose:
    print( 'Final number of points: ' + str( len( new_points ) ) )

# remove calibration
z_axis = []
y_axis = []
x_axis = []
for i in range(0, len(new_points)):
    z_axis.append( round( new_points[i][0]/z_res ) )
    y_axis.append( round( new_points[i][1]/y_res ) )
    x_axis.append( round( new_points[i][2]/x_res ) )

# create data frame in Napari CSV format
df_save = pd.DataFrame(list(zip(z_axis,y_axis,x_axis)), columns =['axis-0', 'axis-1', 'axis-2'])
# save to file
df_save.to_csv( out_csv, index=False )

if verbose:
    print( 'Filtered points saved in ' + out_csv )

