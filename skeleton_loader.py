"""
Version: 1.5

Summary: compute the cross section plane based on 3d model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 skeleton_loader.py -p ~/example/ -m1 test_skeleton.ply -m2 test.ply


argument:
("-p", "--path", required=True,    help="path to *.ply model file")
("-m", "--model", required=True,    help="file name")

"""
#!/usr/bin/env python



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

from mayavi import mlab
from tvtk.api import tvtk

#import networkx as nx

import graph_tool.all as gt

import plotly.graph_objects as go

from matplotlib import pyplot as plt

from math import sqrt

import itertools



#calculate length of a 3D path or curve
def path_length(X, Y, Z):

    n = len(X)
     
    lv = [sqrt((X[i]-X[i-1])**2 + (Y[i]-Y[i-1])**2 + (Z[i]-Z[i-1])**2) for i in range (1,n)]
    
    L = sum(lv)
    
    return L

#compute angle between two vectors(works for n-dimensional vector),
def dot_product_angle(v1,v2):

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        
        print("Zero magnitude vector!")
        
        return 0
        
    else:
        vector_dot_product = np.dot(v1,v2)
        
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        
        angle = np.degrees(arccos)
        
        if angle > 90:
            
            return (angle - 90)
        else:
            return angle
    

# SVD fiting lines to 3D points
def line_fiting_3D(data):

    # Calculate the mean of the points, i.e. the 'center' of the cloud
    datamean = data.mean(axis = 0)
    
    x_range = data[:,0].max() - data[:,0].min()
    y_range = data[:,1].max() - data[:,1].min()
    z_range = data[:,2].max() - data[:,2].min()
    
    
    data_range =(x_range, y_range, z_range)
    
    data_range_mean = sum(data_range) / len(data_range)
    
    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(data - datamean)

    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.

    # Now generate some points along this best fit line, for plotting.

    # use -7, 7 since the spread of the data is roughly 14
    # and we want it to have mean 0 (like the points we did
    # the svd on). Also, it's a straight line, so we only need 2 points.
    linepts = vv[0] * np.mgrid[-1*data_range_mean:data_range_mean:2j][:, np.newaxis]

    # shift by the mean to get the line in the right place
    linepts += datamean
    
    return linepts


# median-absolute-deviation (MAD) based outlier detection
def mad_based_outlier(points, thresh=3.5):
    
    if len(points.shape) == 1:
        
        points = points[:,None]
        
    median = np.median(points, axis=0)
    
    diff = np.sum((points - median)**2, axis=-1)
    
    diff = np.sqrt(diff)
    
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
    
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


# Skeleton visualization
def visualize_skeleton(current_path, filename_skeleton, filename_pcloud):
    
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
    '''
    filepath = current_path + 'edges_skeleton.txt'
    with open(filepath, 'w') as file_handler:
        for item in array_edges_skeleton.tolist():
            file_handler.write("{}\n".format(item))

    '''
    
    
    G_unordered = gt.Graph(directed = True)
    
    # assert directed graph
    #print(G.is_directed())
    
    nodes = G_unordered.add_vertex(num_vertex_skeleton)
    
    G_unordered.add_edge_list(array_edges_skeleton.tolist()) 
    
    #gt.graph_draw(G_unordered, vertex_text = G_unordered.vertex_index, output = current_path + "graph_view.pdf")
    
    
    # find all end vertex by fast iteration of all vertices
    end_vlist = []
    
    end_vlist_offset = []
    
    for v in G_unordered.iter_vertices():
        
        #print(G.vertex(v).out_degree(), G.vertex(v).in_degree())
        
        if G_unordered.vertex(v).out_degree() == 0 and G_unordered.vertex(v).in_degree() == 1:
        
            end_vlist.append(v)
            
            if (v+1) == num_vertex_skeleton:
                end_vlist_offset.append(v)
            else:
                end_vlist_offset.append(v+1)
            
    print("end_vlist = {} \n".format(end_vlist))
    print("end_vlist_offset = {} \n".format(end_vlist_offset))
    
    
    #test angle calculation
    #vector1 = [0,0,1]
    #vector2 = [-1,0,1]
    #print(dot_product_angle(vector1,vector2))
    
    
    #obtain all the sub branches edgeds and vetices
    sub_branch_list = []
    
    sub_branch_length_rec = []
    
    sub_branch_angle_rec = []
    
    sub_branch_start_rec = []
    
    sub_branch_end_rec = []
    
    #if len(end_vlist) == len(end_vlist_offset):
        
    for idx, v_end in enumerate(end_vlist):
        
        #print(idx, v_end)

        if idx == 0:
            v_list = [*range(0, int(end_vlist[idx])+1)]
        else:
            v_list = [*range(int(end_vlist[idx-1])+1, int(end_vlist[idx])+1)]
            
        sub_branch_list.append(v_list)
        
        int_v_list = [int(i) for i in v_list]
        
        sub_branch_length = path_length(X_skeleton[int_v_list], Y_skeleton[int_v_list], Z_skeleton[int_v_list])
        
        sub_branch_length_rec.append(sub_branch_length)
        
        start_v = [X_skeleton[int_v_list[0]], Y_skeleton[int_v_list[0]], Z_skeleton[int_v_list[0]]]
        
        end_v = [X_skeleton[int_v_list[0]] - X_skeleton[int_v_list[len(int_v_list)-1]], Y_skeleton[int_v_list[0]] - Y_skeleton[int_v_list[len(int_v_list)-1]], Z_skeleton[int_v_list[0]] - Z_skeleton[int_v_list[len(int_v_list)-1]]]
        
        angle_sub_branch = dot_product_angle(start_v, end_v)
        
        sub_branch_angle_rec.append(angle_sub_branch)
        
        sub_branch_start_rec.append(int_v_list[0])
        
        sub_branch_end_rec.append(int_v_list[len(int_v_list)-1])
        
        #print("angle_sub_branch = {}".format(angle_sub_branch))
        
    
    print("sub_branch_start_rec = {}\n".format(sub_branch_start_rec))
    
    '''
    k = int(len(sub_branch_length_rec) * 0.7)
    
    print(k)

    idx_dominant_branch = np.argsort(sub_branch_length_rec)[-k:]
    
    print(idx_dominant_branch)
    
    sub_branch_list_dominant = [sub_branch_list[index] for index in idx_dominant_branch] 
    
    vertex_dominant = sorted(list(itertools.chain(*sub_branch_list_dominant)))
    
    print(len(vertex_dominant))
    '''

        
                
    #print(len(sub_branch_list), len(end_vlist))
    
    #check individual child branches for each main branch

    v_closest_pair_rec = []
    
    closet_pts = []
    
    #find closest point set and connect graph edges
    for idx, (sub_branch, anchor_point) in enumerate(zip(sub_branch_list, end_vlist_offset)):
        
        anchor_point = (X_skeleton[end_vlist_offset[idx]], Y_skeleton[end_vlist_offset[idx]], Z_skeleton[end_vlist_offset[idx]])

        point_set = np.zeros((len(sub_branch_list[0]), 3))
        
        point_set[:,0] = X_skeleton[sub_branch_list[0]]
        point_set[:,1] = Y_skeleton[sub_branch_list[0]]
        point_set[:,2] = Z_skeleton[sub_branch_list[0]]
        
        (index_cp, value_cp) = closest_point(point_set, anchor_point)

        v_closest_pair = [index_cp, end_vlist_offset[idx]]

        dis_v_closest_pair = path_length(X_skeleton[v_closest_pair], Y_skeleton[v_closest_pair], Z_skeleton[v_closest_pair])
        
        #small numer threshold indicating close pair vetices
        if dis_v_closest_pair < 0.01:
            
            closet_pts.append(index_cp)
            
            #print("dis_v_closest_pair = {}".format(dis_v_closest_pair))
            v_closest_pair_rec.append(v_closest_pair)
            
            #print("closest point pair: {0}".format(v_closest_pair))
            
            #connect graph edges
            G_unordered.add_edge(index_cp, end_vlist_offset[idx])
            
    print("v_closest_pair_rec = {}\n".format(v_closest_pair_rec))
        
    #closet_pts_unique = list(set(closet_pts))
    
    closet_pts_unique = list((closet_pts))
    
    closet_pts_unique_sorted = sorted(closet_pts_unique)
    
    print("closet_pts_unique_sorted = {}\n".format(closet_pts_unique_sorted))
    
    # compute distance between adjacent vertices in closet_pts_unique_sorted
    X = X_skeleton[closet_pts_unique_sorted]
    Y = Y_skeleton[closet_pts_unique_sorted]
    Z = Z_skeleton[closet_pts_unique_sorted]
    
    dis_closet_pts = [sqrt((X[i]-X[i-1])**2 + (Y[i]-Y[i-1])**2 + (Z[i]-Z[i-1])**2) for i in range (1, len(X))]
    
    print("distance between closet_pts_unique = {}\n".format(dis_closet_pts))
    
    
    '''
    #find index of k smallest or biggest elements in list
    ####################################################
    k = int(len(dis_closet_pts) * 0.8)
    
    #print(k)
    
    k biggest
    idx_dominant_dis_closet_pts = np.argsort(dis_closet_pts)[-k:]
    
    #k smallest
    #idx_dominant_dis_closet_pts = np.argsort(dis_closet_pts)[:k]
    
    #print("idx_dominant_dis_closet_pts = {}".format(idx_dominant_dis_closet_pts))
    
    #print(idx_dominant_dis_closet_pts)
    
    dis_closet_pts_dominant = [closet_pts_unique_sorted[index] for index in idx_dominant_dis_closet_pts] 
    
    print("dis_closet_pts_dominant pairs = {}".format(dis_closet_pts_dominant))
    ####################################################
    '''
    
    #find outlier of cloest points distance list to combine close points
    #######################################################
    index_outlier = mad_based_outlier(np.asarray(dis_closet_pts), 1.5)
    
    #print("index_outlier = {}".format(index_outlier))
    
    index_outlier_loc = [i for i, x in enumerate(index_outlier) if x]
    
    print("index_outlier = {}".format(index_outlier_loc))
    
    closet_pts_unique_sorted_combined = [closet_pts_unique_sorted[index] for index in index_outlier_loc] 
    
    print("closet_pts_unique_sorted[index_outlier_loc] = {}\n".format(closet_pts_unique_sorted_combined))
    
    
    v_closest_pair_rec_selected = [v_closest_pair_rec[index] for index in index_outlier_loc] 
    
    print("v_closest_pair_rec_selected = {}\n".format(v_closest_pair_rec_selected))
    
    
    v_closest_start_selected = [v_closest_pair_rec[index][1] for index in index_outlier_loc] 
    
    print("v_closest_start_selected = {}\n".format(v_closest_start_selected))
    
    
    sub_branch_selected = [sub_branch_list[index+1] for index in index_outlier_loc]
    
    print("sub_branch_start_selected = {}\n".format(sub_branch_selected))
    
    
    
    
    
    
    
    
    # compute whorl distance based on distance between combined close points
    whorl_loc1_idx = [closet_pts_unique_sorted_combined[0], closet_pts_unique_sorted_combined[1]]
    
    whorl_dis_1 = path_length(X_skeleton[whorl_loc1_idx], Y_skeleton[whorl_loc1_idx], Z_skeleton[whorl_loc1_idx])
    
    whorl_loc2_idx = [closet_pts_unique_sorted_combined[1], closet_pts_unique_sorted_combined[2]]
    
    whorl_dis_2 = path_length(X_skeleton[whorl_loc2_idx], Y_skeleton[whorl_loc2_idx], Z_skeleton[whorl_loc2_idx])
    
    '''
    if whorl_dis_2 < whorl_dis_1*0.5:
    
        whorl_loc2_idx = [closet_pts_unique_sorted_combined[1], closet_pts_unique_sorted_combined[3]]
    
        whorl_dis_2 = path_length(X_skeleton[whorl_loc2_idx], Y_skeleton[whorl_loc2_idx], Z_skeleton[whorl_loc2_idx])
    '''
    print("whorl_dis_1 pair = {0} ,distance = {1}\n  whorl_dis_2 pair = {2}, distance = {3}\n".format(whorl_loc1_idx, whorl_dis_1,whorl_loc2_idx, whorl_dis_2))
    
    

    
    '''
    #define start and end vertex index
    start_v = 0
    end_v = 608
    
    
    #print(X_skeleton[start_v], Y_skeleton[start_v], Z_skeleton[start_v])
    
    # find shortest path in the graph between start and end vertices 
    vlist, elist = gt.shortest_path(G_unordered, G_unordered.vertex(start_v), G_unordered.vertex(end_v))
    
    vlist_path = [str(v) for v in vlist]
    
    elist_path = [str(e) for e in elist]
    
    #change format form str to int
    int_vlist_path = [int(i) for i in vlist_path]
    
    #print(int_vlist_path)
    
    if len(vlist_path) > 0: 
        
        print("Shortest path found in graph! \n")
        
        print("vlist_path = {} \n".format(int_vlist_path))
    
        curve_length = path_length(X_skeleton[int_vlist_path], Y_skeleton[int_vlist_path], Z_skeleton[int_vlist_path])
    
        print("curve_length = {} \n".format(curve_length))
    else:
        print("No shortest path found in graph...\n")
    
    '''
    ###################################################################

    
    #load ply point cloud file
    if not (filename_pcloud is None):
        
        model_pcloud = current_path + filename_pcloud
        
        print("Loading 3D point cloud {}...\n".format(filename_pcloud))
        
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
    
    #root of the skeleton
    pts = mlab.points3d(X_skeleton[0], Y_skeleton[0], Z_skeleton[0], color = (0.58, 0.29, 0), mode = 'sphere', scale_factor = 0.15)
    
    #pts = mlab.points3d(X_skeleton, Y_skeleton, Z_skeleton, mode = 'point', scale_factor = 0.5)
    
    #pts = mlab.points3d(X_skeleton[end_vlist], Y_skeleton[end_vlist], Z_skeleton[end_vlist], color = (1,1,1), mode = 'sphere', scale_factor = 0.03)
    
    pts = mlab.points3d(X_skeleton[closet_pts_unique], Y_skeleton[closet_pts_unique], Z_skeleton[closet_pts_unique], color = (0,1,1), mode = 'sphere', scale_factor = 0.05)
    
    pts = mlab.points3d(X_skeleton[closet_pts_unique_sorted_combined], Y_skeleton[closet_pts_unique_sorted_combined], Z_skeleton[closet_pts_unique_sorted_combined], color=(1,0,0), mode = 'sphere', scale_factor = 0.05)
    
    
    cmap = get_cmap(len(sub_branch_list))
    
    # loop draw all the sub branches
    for i, sub_branch in enumerate(sub_branch_selected):
        
        color_rgb = cmap(i)[:len(cmap(i))-1]
        
        pts = mlab.points3d(X_skeleton[sub_branch], Y_skeleton[sub_branch], Z_skeleton[sub_branch], color=color_rgb, mode = 'sphere', scale_factor = 0.05)

        
    
     
   
    '''
    for i, (end_val, x_e, y_e, z_e) in enumerate(zip(end_vlist, X_skeleton[end_vlist], Y_skeleton[end_vlist], Z_skeleton[end_vlist])):
        
        mlab.text3d(x_e, y_e, z_e, str(end_val), scale = (0.04, 0.04, 0.04))
    '''
    '''
    for i, (end_val, x_e, y_e, z_e) in enumerate(zip(end_vlist_offset, X_skeleton[end_vlist_offset], Y_skeleton[end_vlist_offset], Z_skeleton[end_vlist_offset])):
        
        mlab.text3d(x_e, y_e, z_e, str(end_val), scale = (0.04, 0.04, 0.04))
        
    '''
    
    for i, (end_val, x_e, y_e, z_e) in enumerate(zip(closet_pts_unique, X_skeleton[closet_pts_unique], Y_skeleton[closet_pts_unique], Z_skeleton[closet_pts_unique])):
        
        mlab.text3d(x_e, y_e, z_e, str(end_val), scale = (0.04, 0.04, 0.04))
    
    #pts = mlab.points3d(Data_array_pcloud[:,0], Data_array_pcloud[:,1], Data_array_pcloud[:,2], mode = 'point')
    
    
    #mlab.show()
    #visualize point cloud model with color
    ####################################################################
    
    if not (filename_pcloud is None):
        
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
    
    # Create each line one after the other in a loop
    for i in range(N_edges_skeleton):
    #for val in vlist_path:
        #if i in vertex_dominant:
        if True:
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
    #mlab.show()
    

    
    '''
    ###################################################################
    # visualize path
    #visualize skeleton model, edge, nodes
    ####################################################################
    if len(vlist_path) > 0:
    
        x = list()
        y = list()
        z = list()
        s = list()
        connections = list()
        
        # The index of the current point in the total amount of points
        index = 0
        
        # Create each line one after the other in a loop
        #for i in range(N_edges_skeleton):
        for val in vlist_path:
            
            i = int(val)
            #print("Edges {0} has nodes {1}, {2}\n".format(i, array_edges[i][0], array_edges[i][1]))
          
            x.append(X_skeleton[array_edges_skeleton[i][0]])
            y.append(Y_skeleton[array_edges_skeleton[i][0]])
            z.append(Z_skeleton[array_edges_skeleton[i][0]])
            
            x.append(X_skeleton[array_edges_skeleton[i][1]])
            y.append(Y_skeleton[array_edges_skeleton[i][1]])
            z.append(Z_skeleton[array_edges_skeleton[i][1]])
            
            # The scalar parameter for each line
            s.append(array_edges_skeleton[1])
            
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

        # The stripper filter cleans up connected lines
        #lines = mlab.pipeline.stripper(src)

        # display the set of lines
        mlab.pipeline.surface(src, colormap = 'winter', line_width = 10, opacity = 1.0)

    # And choose a nice view
    #mlab.view(33.6, 106, 5.5, [0, 0, .05])
    #mlab.roll(125)
    
    
    #mlab.savefig('skeleton_graph.x3d')
    
    mlab.show()
    
    #end of path visualization
    ##################################################################################
    '''
    
    mlab.show()
    
    '''
    #As before we use networkx to determine node positions. We want to do the same spring layout but in 3D
    spring_3D = nx.spring_layout(G, dim=3, seed=18)
    
    
    #we need to seperate the X,Y,Z coordinates for Plotly
    #x_nodes = [spring_3D[i][0] for i in range(Num_nodes)]# x-coordinates of nodes
    #y_nodes = [spring_3D[i][1] for i in range(Num_nodes)]# y-coordinates
    #z_nodes = [spring_3D[i][2] for i in range(Num_nodes)]# z-coordinates


    #We also need a list of edges to include in the plot
    edge_list = G.edges()

    #we  need to create lists that contain the starting and ending coordinates of each edge.
    x_edges=[]
    y_edges=[]
    z_edges=[]

    #need to fill these with all of the coordiates
    for edge in edge_list:
        #format: [beginning,ending,None]
        x_coords = [spring_3D[edge[0]][0],spring_3D[edge[1]][0],None]
        x_edges += x_coords

        y_coords = [spring_3D[edge[0]][1],spring_3D[edge[1]][1],None]
        y_edges += y_coords

        z_coords = [spring_3D[edge[0]][2],spring_3D[edge[1]][2],None]
        z_edges += z_coords

    #create a trace for the edges
    trace_edges = go.Scatter3d(x=x_edges,
                        y=y_edges,
                        z=z_edges,
                        mode='lines',
                        line=dict(color='black', width=2),
                        hoverinfo='none')
                    

    #we need to set the axis for the plot 
    axis = dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='')
            

    #also need to create the layout for our plot
    layout = go.Layout(title="Two Predicted Factions of Zachary's Karate Club",
                width=650,
                height=625,
                showlegend=False,
                scene=dict(xaxis=dict(axis),
                        yaxis=dict(axis),
                        zaxis=dict(axis),
                        ),
                margin=dict(t=100),
                hovermode='closest')
                
    #Include the traces we want to plot and create a figure
    data = [trace_edges]
    fig = go.Figure(data=data, layout=layout)

    fig.show()
    '''
    

 
    
    '''
    filepath = current_path + 'edges_skeleton_s.txt'
    with open(filepath, 'w') as file_handler:
        for item in s:
            file_handler.write("{}\n".format(item))
    
    
    '''


if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m1", "--model_skeleton", required = True, help = "skeleton file name")
    ap.add_argument("-m2", "--model_pcloud", required = False, default = None, help = "point cloud model file name")
    args = vars(ap.parse_args())


    # setting path to model file 
    current_path = args["path"]
    filename_skeleton = args["model_skeleton"]
    
    if args["model_pcloud"] is None:
        
        filename_pcloud = None
        
    else:
        
        filename_pcloud = args["model_pcloud"]
    
    #file_path = current_path + filename

    print ("results_folder: " + current_path)

    visualize_skeleton(current_path, filename_skeleton, filename_pcloud)

 
