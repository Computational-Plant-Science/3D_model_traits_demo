"""
Version: 1.5

Summary: visualize the skeleto of point cloud model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

    python3 skeleton_loader_vis.py -p ~/example/ -m1 test_skeleton.ply -m2 test.ply


argument:
("-p", "--path", required=True,    help="path to *.ply model file")
("-m", "--model", required=True,    help="file name")

"""
#!/usr/bin/env python

from mayavi import mlab
from tvtk.api import tvtk

# import the necessary packages
from plyfile import PlyData, PlyElement
import numpy as np 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from operator import itemgetter
import argparse

from scipy.spatial import KDTree

import os
import sys
import open3d as o3d
import copy



#import networkx as nx

import graph_tool.all as gt

import plotly.graph_objects as go

from matplotlib import pyplot as plt

from math import sqrt



# calculate length of a 3D path or curve
def path_length(X, Y, Z):

    n = len(X)
     
    lv = [sqrt((X[i]-X[i-1])**2 + (Y[i]-Y[i-1])**2 + (Z[i]-Z[i-1])**2) for i in range (1,n)]
    
    L = sum(lv)
    
    return L

'''
# distance between two points
def distance_pt(p0, p1):
    
    dist = np.linalg.norm(p0 - p1)
    
    return dist
'''
#find the closest points from a points sets to a fix point using Kdtree, O(log n) 
def closest_point(point_set, anchor_point):
    
    kdtree = KDTree(point_set)
    
    (d, i) = kdtree.query(anchor_point)
    
    #print("closest point:", point_set[i])
    
    return  i, point_set[i]

#colormap mapping
def get_cmap(n, name = 'Spectral'):
    """get the color mapping""" 
    #viridis, BrBG, hsv, copper
    #Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    #RGB color; the keyword argument name must be a standard mpl colormap name
    return plt.cm.get_cmap(name,n+1)

def visualize_skeleton(current_path, filename_skeleton, filename_ptcloud):
    
    model_skeleton = current_path + filename_skeleton
    print("Loading 3D skeleton file {}...\n".format(filename_skeleton))
    model_skeleton_name_base = os.path.splitext(model_skeleton)[0]
    
    #load the ply format skeleton file 
    try:
        with open(model_skeleton, 'rb') as f:
            plydata_skeleton = PlyData.read(f)
            num_vertex_skeleton = plydata_skeleton.elements[0].count
            N_edges_skeleton = len(plydata_skeleton['edge'].data['vertex_indices'])
            array_edges_skeleton = plydata_skeleton['edge'].data['vertex_indices']
            
            print("Ply data structure: \n")
            #print(plydata_skeleton)
            #print("\n")
            print("Number of 3D points in skeleton model: {0} \n".format(num_vertex_skeleton))
            print("Number of edges: {0} \n".format(N_edges_skeleton))

        
    except:
        print("Model skeleton file does not exist!")
        sys.exit(0)
    
    
    #Parse ply format skeleton file and Extract the data
    Data_array_skeleton = np.zeros((num_vertex_skeleton, 3))
    
    Data_array_skeleton[:,0] = plydata_skeleton['vertex'].data['x']
    Data_array_skeleton[:,1] = plydata_skeleton['vertex'].data['y']
    Data_array_skeleton[:,2] = plydata_skeleton['vertex'].data['z']
    
    X_skeleton = Data_array_skeleton[:,0]
    Y_skeleton = Data_array_skeleton[:,1]
    Z_skeleton = Data_array_skeleton[:,2]
    
    ####################################################################
    #mesh = trimesh.load(model_skeleton)
    
    
    '''
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(Data_array_skeleton)
    
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    
    # Build KDTree from point cloud for fast retrieval of nearest neighbors
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    print("Paint the 00th point red.")
    
    pcd.colors[0] = [1, 0, 0]
    
    print("Find its 200 nearest neighbors, paint blue.")
    
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[0], 100)
    
    np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
    
    print("Visualize the point cloud.")
    
    o3d.visualization.draw_geometries([pcd])
    
    #build octree, a tree data structure where each internal node has eight children.
    # fit to unit cube
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(num_vertex_skeleton, 3)))
    o3d.visualization.draw_geometries([pcd])

    print('octree division')
    octree = o3d.geometry.Octree(max_depth=4)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    o3d.visualization.draw_geometries([octree])

    print(octree.locate_leaf_node(pcd.points[243]))
    '''
    
    ####################################################################

    
    ###################################################################

    
    #Load ply point cloud file
    if not (filename_ptcloud is None):
        
        model_pcloud = current_path + filename_ptcloud
        
        print("Loading 3D point cloud {}...\n".format(filename_ptcloud))
        
        model_pcloud_name_base = os.path.splitext(model_pcloud)[0]
        
        pcd = o3d.io.read_point_cloud(model_pcloud)
        
        Data_array_pcloud = np.asarray(pcd.points)
        
        
        if pcd.has_colors():
            
            print("Render colored point cloud")
            
            pcd_color = np.asarray(pcd.colors)
            
            if len(pcd_color) > 0: 
                
                pcd_color = np.rint(pcd_color * 255.0)
            
            #pcd_color = tuple(map(tuple, pcd_color))
        else:
            
            print("Generate random color")
        
            pcd_color = np.random.randint(256, size = (len(Data_array_pcloud),3))
            
        #print(Data_array_pcloud.shape)
        
        #print(len(Data_array_pcloud))
        
        #print(pcd_color.shape)
        
        #print(type(pcd_color))
    
    
    
    
    #Skeleton Visualization pipeline
    ####################################################################
    # The number of points per line
    N = 2
    
    mlab.figure("Structure_graph", size = (800, 800), bgcolor = (0, 0, 0))
    mlab.clf()
    
    #pts = mlab.points3d(X_skeleton, Y_skeleton, Z_skeleton, mode = 'point', scale_factor = 0.5)
    
    #pts = mlab.points3d(X_skeleton[end_vlist], Y_skeleton[end_vlist], Z_skeleton[end_vlist], color = (1,1,1), mode = 'sphere', scale_factor = 0.03)
    
    #pts = mlab.points3d(X_skeleton[closet_pts_unique], Y_skeleton[closet_pts_unique], Z_skeleton[closet_pts_unique], color = (0,1,1), mode = 'sphere', scale_factor = 0.05)
    
    #pts = mlab.points3d(X_skeleton[end_vlist_offset], Y_skeleton[end_vlist_offset], Z_skeleton[end_vlist_offset], color=(1,0,0), mode = 'sphere', scale_factor = 0.05)
    

    #pts = mlab.points3d(Data_array_pcloud[:,0], Data_array_pcloud[:,1], Data_array_pcloud[:,2], mode = 'point')
    
    
    #mlab.show()
    #visualize point cloud model with color
    ####################################################################
    
    if not (filename_ptcloud is None):
        
        x, y, z = Data_array_pcloud[:,0], Data_array_pcloud[:,1], Data_array_pcloud[:,2] 
        
        pts = mlab.points3d(x,y,z, mode = 'point')
        
        sc = tvtk.UnsignedCharArray()
        
        sc.from_array(pcd_color)

        pts.mlab_source.dataset.point_data.scalars = sc
        
        pts.mlab_source.dataset.modified()
    
    
    #visualize skeleton model, edge, nodes
    ####################################################################
    x = list()
    y = list()
    z = list()
    s = list()
    connections = list()
    
    # The index of the current point in the total amount of points
    index = 0
    
    
    #N_edges_skeleton = 3698
    
    # Create each line one after the other in a loop
    for i in range(N_edges_skeleton):
    #for val in vlist_path:
        
        #i = int(val)
        #print("Edges {0} has nodes {1}, {2}\n".format(i, array_edges[i][0], array_edges[i][1]))
      
        x.append(X_skeleton[array_edges_skeleton[i][0]])
        y.append(Y_skeleton[array_edges_skeleton[i][0]])
        z.append(Z_skeleton[array_edges_skeleton[i][0]])
        
        x.append(X_skeleton[array_edges_skeleton[i][1]])
        y.append(Y_skeleton[array_edges_skeleton[i][1]])
        z.append(Z_skeleton[array_edges_skeleton[i][1]])
        
        # The scalar parameter for each line
        s.append(array_edges_skeleton[i])
        
        # This is the tricky part: in a line, each point is connected
        # to the one following it. We have to express this with the indices
        # of the final set of points once all lines have been combined
        # together, this is why we need to keep track of the total number of
        # points already created (index)
        #connections.append(np.vstack(array_edges[i]).T)
        
        connections.append(np.vstack(
                       [np.arange(index,   index + N - 1.5),
                        np.arange(index + 1, index + N - .5)]
                            ).T)
        index += 2
    
    
    # Now collapse all positions, scalars and connections in big arrays
    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    s = np.hstack(s)
    #connections = np.vstack(connections)

    # Create the points
    src = mlab.pipeline.scalar_scatter(x, y, z, s)

    # Connect them
    src.mlab_source.dataset.lines = connections
    src.update()

    # display the set of lines
    mlab.pipeline.surface(src, colormap = 'Accent', line_width = 5, opacity = 0.7)

    # And choose a nice view
    #mlab.view(33.6, 106, 5.5, [0, 0, .05])
    #mlab.roll(125)
    mlab.show()
    



if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m1", "--model_skeleton", required = False, help = "skeleton file name")
    ap.add_argument("-m2", "--model_pcloud", required = False, default = None, help = "point cloud model file name")
    args = vars(ap.parse_args())


    # setting input path to model file 
    current_path = args["path"]
    #current_path = os.path.join(current_path, '')
    
    folder_name = os.path.basename(os.path.dirname(current_path))
    
    ###################################################################
    # check file name input and default file name
    if args["model_skeleton"] is None:

        # search for file with default name
        filename_skeleton = current_path + folder_name + '_skeleton.ply'
        
        print(filename_skeleton)
        
        if os.path.isfile(filename_skeleton):
            print("Default skeleton file: {}\n".format(filename_skeleton))
            filename_skeleton = folder_name + '_skeleton.ply'
        else:
            print("Skeleton model is not found!\n")
            sys.exit()
    else:
        filename_skeleton = args["model_skeleton"]
    

    if args["model_pcloud"] is None:
        
        # search for file with default name
        filename_ptcloud = current_path + folder_name + '_aligned.ply'
        
        if os.path.isfile(filename_ptcloud):
            print("Default model file: {}\n".format(filename_ptcloud))
            filename_ptcloud = folder_name + '_aligned.ply'
        else:
            print("Aligned pointclod model is not found!\n")
            sys.exit()

    else:
        filename_ptcloud = args["model_pcloud"]
    
    
    #file_path = current_path + filename

    print ("results_folder: " + current_path)

    visualize_skeleton(current_path, filename_skeleton, filename_ptcloud)

 
