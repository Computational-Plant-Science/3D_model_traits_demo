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

#from mayavi import mlab
#from tvtk.api import tvtk

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
    
    # define the path to skeleton file
    model_skeleton = current_path + filename_skeleton
    print("Loading 3D skeleton file {}...\n".format(filename_skeleton))
    model_skeleton_name_base = os.path.splitext(model_skeleton)[0]
    
    #load the skeleton file in ply format 
    try:
        with open(model_skeleton, 'rb') as f:
            
            plydata_skeleton = PlyData.read(f)
            
            # get the number of points
            num_vertex_skeleton = plydata_skeleton.elements[0].count
            
            # get the number of vertices
            N_edges_skeleton = len(plydata_skeleton['edge'].data['vertex_indices'])
            
            # get the vertices array
            array_edges_skeleton = plydata_skeleton['edge'].data['vertex_indices']
            
            print("Ply data structure: \n")
            #print(plydata_skeleton)
            #print("\n")
            print("Number of 3D points in skeleton model: {0} \n".format(num_vertex_skeleton))
            print("Number of edges: {0} \n".format(N_edges_skeleton))

        
    except:
        print("Model skeleton file does not exist!")
        sys.exit(0)
    
    ####################################################################################3
    # Load ply point cloud file
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
            
            
    
    ######################################################################
    #Parse ply format skeleton file and Extract the points and edges
    Data_array_skeleton = np.zeros((num_vertex_skeleton, 3))
    
    # 
    Data_array_skeleton[:,0] = plydata_skeleton['vertex'].data['x']
    Data_array_skeleton[:,1] = plydata_skeleton['vertex'].data['y']
    Data_array_skeleton[:,2] = plydata_skeleton['vertex'].data['z']
    
    X_skeleton = Data_array_skeleton[:,0]
    Y_skeleton = Data_array_skeleton[:,1]
    Z_skeleton = Data_array_skeleton[:,2]
    
    
    # visualization 
    ####################################################################

    pcd_skeleton = o3d.geometry.PointCloud()
    
    pcd_skeleton.points = o3d.utility.Vector3dVector(Data_array_skeleton)
    
    pcd_skeleton.paint_uniform_color([0, 0, 1])
    
    #o3d.visualization.draw_geometries([pcd])
    
    points = Data_array_skeleton
    
    lines = array_edges_skeleton
    
    colors = [[1, 0, 0] for i in range(len(lines))]
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(line_set)
    vis.add_geometry(pcd_skeleton)
    #vis.add_geometry(pcd)
    vis.get_render_option().line_width = 15
    vis.get_render_option().point_size = 10
    vis.get_render_option().background_color = (0, 0, 0)
    vis.get_render_option().show_coordinate_frame = True
    vis.run()
    #vis.destroy_window()
    
    '''
    # Build KDTree from point cloud for fast retrieval of nearest neighbors
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_skeleton)
    
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

    print ("results_folder: {}\n".format(current_path))

    visualize_skeleton(current_path, filename_skeleton, filename_ptcloud)

 
