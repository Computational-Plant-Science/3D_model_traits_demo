"""
Version: 1.5

Summary: compute the cross section plane based on 3d model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 skeleton_analyze.py -p ~/example/ -m1 test_skeleton.ply -m2 test.ply


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
from sklearn.cluster import KMeans
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
import math
import itertools

from tabulate import tabulate

import openpyxl
import csv


# save point cloud to open3d compatiable format
def format_converter(path, Data_array):
    '''
    model_file = current_path + model_name
    
    print("Converting file format for 3D point cloud model {}...\n".format(model_name))
    
    model_name_base = os.path.splitext(model_file)[0]
    
    
    # load the model file
    try:
        with open(model_file, 'rb') as f:
            plydata = PlyData.read(f)
            num_vertex = plydata.elements[0].count
            
            print("Ply data structure: \n")
            print(plydata)
            print("\n")
            print("Number of 3D points in current model: {0} \n".format(num_vertex))
        
    except:
        print("Model file does not exist!")
        sys.exit(0)
        
    
    #Parse the ply format file and Extract the data
    Data_array_ori = np.zeros((num_vertex, 3))
    
    Data_array_ori[:,0] = plydata['vertex'].data['x']
    Data_array_ori[:,1] = plydata['vertex'].data['y']
    Data_array_ori[:,2] = plydata['vertex'].data['z']
    
    #sort point cloud data based on Z values
    Data_array = np.asarray(sorted(Data_array_ori, key = itemgetter(2), reverse = False))
    '''
    '''
    #accquire data range
    min_x = Data_array[:, 0].min()
    max_x = Data_array[:, 0].max()
    min_y = Data_array[:, 1].min()
    max_y = Data_array[:, 1].max()
    min_z = Data_array[:, 2].min()
    max_z = Data_array[:, 2].max()
    
    range_data_x = max_x - min_x
    range_data_y = max_y - min_y
    range_data_z = max_z - min_z
    
    print (range_data_x, range_data_y, range_data_z)
    
    print(min_x,max_x)
    print(min_y,max_y)
    print(min_z,max_z)
    '''
    
    
    #Normalize data
    #min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1000000))
    
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,10000))

    point_normalized = min_max_scaler.fit_transform(Data_array)
    
    #point_normalized_scale = [i * 1 for i in point_normalized]
    # Pass xyz to Open3D.o3d.geometry.PointCloud 
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(point_normalized)
   
    #Save model file as ascii format in ply
    filename = path + 'converted.ply'
    
    #write out point cloud file
    o3d.io.write_point_cloud(filename, pcd, write_ascii = True)
    
    '''
    # check saved file
    if os.path.exists(filename):
        print("Converted 3d model was saved at {0}".format(filename))
        return True
    else:
        return False
        print("Model file converter failed !")
        sys.exit(0)
    '''

#calculate length of a 3D path or curve
def path_length(X, Y, Z):

    n = len(X)
     
    lv = [math.sqrt((X[i]-X[i-1])**2 + (Y[i]-Y[i-1])**2 + (Z[i]-Z[i-1])**2) for i in range (1,n)]
    
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
            
            return (180 - angle)
        else:
            return (angle)


#coordinates transformation from cartesian coords to sphere coord system
def cart2sph(x, y, z):
    
    hxy = np.hypot(x, y)
    
    r = np.hypot(hxy, z)
    
    elevation = np.arctan2(z, hxy)*180/math.pi
    
    azimuth = np.arctan2(y, x)*180/math.pi
    
    return r[2], azimuth[2], elevation[2]
    '''
    if azimuth > 90:
            angle = 180 - azimuth
        elif azimuth < 0:
            angle = 90 + azimuth
        else:
            angle = azimuth
    '''


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




# compute nearest neighbors of the anchor_pt_idx in point cloud by building KDTree
def get_neighbors(Data_array_pt, anchor_pt_idx, search_radius):
    
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(Data_array_pt)
    
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    #o3d.visualization.draw_geometries([pcd])
    
    # Build KDTree from point cloud for fast retrieval of nearest neighbors
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    #print("Paint the 00th point red.")
    
    pcd.colors[anchor_pt_idx] = [1, 0, 0]
    
    #print("Find its 50 nearest neighbors, paint blue.")
    
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[anchor_pt_idx], search_radius)
    
    #print("nearest neighbors = {}\n".format(sorted(np.asarray(idx[1:]))))

    return idx
    
           
    '''
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


# compute dimensions of point cloud and nearest neighbors by KDTree
def get_pt_parameter(Data_array_pt):
    
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(Data_array_pt)
    
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
   
    # get convex hull of a point cloud is the smallest convex set that contains all points.
    hull, _ = pcd.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    
    # get AxisAlignedBoundingBox
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (0, 1, 0)
    
    #Get the extent/length of the bounding box in x, y, and z dimension.
    aabb_extent = aabb.get_extent()
    
    aabb_extent_half = aabb.get_half_extent()
    
    # get OrientedBoundingBox
    obb = pcd.get_oriented_bounding_box()
    
    obb.color = (1, 0, 0)
    
    #visualize the convex hull as a red LineSet
    #o3d.visualization.draw_geometries([pcd, aabb, obb, hull_ls])
    
    pt_diameter_max = max(aabb_extent[0], aabb_extent[1])
    
    pt_diameter_min = max(aabb_extent_half[0], aabb_extent_half[1])
    
    pt_length = aabb_extent[2]
    
    return pt_diameter_max, pt_diameter_min, pt_length
    
    
    
#find the closest points from a points sets to a fix point using Kdtree, O(log n) 
def closest_point(point_set, anchor_point):
    
    kdtree = KDTree(point_set)
    
    (d, i) = kdtree.query(anchor_point)
    
    #print("closest point:", point_set[i])
    
    return  i, point_set[i]


#colormap mapping
def get_cmap(n, name = 'hsv'):
    """get the color mapping""" 
    #viridis, BrBG, hsv, copper, Spectral
    #Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    #RGB color; the keyword argument name must be a standard mpl colormap name
    return plt.cm.get_cmap(name,n+1)


#cluster 1D list using Kmeans
def cluster_1D(list_array, n_clusters):
    
    data = np.array(list_array)
     
    kmeans = KMeans(n_clusters).fit(data.reshape(-1,1))
    
    labels = kmeans.labels_
    
    #print(kmeans.cluster_centers_)
    
    return labels


def outlier_remove(data_list):

    #find index of k smallest or biggest elements in list
    ####################################################
    
    k = int(len(data_list) * 0.8)
    
    #print(k)
    
    #k biggest
    idx_dominant = np.argsort(data_list)[-k:]
    
    #k smallest
    #idx_dominant_dis_closest_pts = np.argsort(dis_closest_pts)[:k]
    
    #print("idx_dominant_dis_closest_pts = {}".format(idx_dominant_dis_closest_pts))
    
    #print(idx_dominant_dis_closest_pts)
    
    outlier_remove_list = [data_list[index] for index in idx_dominant] 
    
    #print("outlier_remove_list = {}".format(outlier_remove_list))
    
    return outlier_remove_list, idx_dominant
    ####################################################


# save point cloud data from numpy array as ply file, open3d compatiable format
def write_ply(path, data_numpy_array):
    
    data_range = 10000
    
    #Normalize data range for generate cross section level set scan
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0, data_range))

    point_normalized = min_max_scaler.fit_transform(data_numpy_array)
    
    #initialize pcd object for open3d 
    pcd = o3d.geometry.PointCloud()
     
    pcd.points = o3d.utility.Vector3dVector(point_normalized)
    
    #write out point cloud file
    o3d.io.write_point_cloud(path, pcd, write_ascii = True)
    
    
    # check saved file
    if os.path.exists(path):
        print("Converted 3d model was saved at {0}".format(path))
        return True
    else:
        return False
        print("Model file converter failed !")
        #sys.exit(0)
    

# Skeleton analysis
def analyze_skeleton(current_path, filename_skeleton, filename_pcloud):
    
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
    
    
    #build graph from skeleton data
    ####################################################################
   
    G_unordered = gt.Graph(directed = True)
    
    # assert directed graph
    #print(G.is_directed())
    
    nodes = G_unordered.add_vertex(num_vertex_skeleton)
    
    G_unordered.add_edge_list(array_edges_skeleton.tolist()) 
    
    #gt.graph_draw(G_unordered, vertex_text = G_unordered.vertex_index, output = current_path + "graph_view.pdf")
    
    
    # find all end vertices by fast iteration of all vertices
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
            
    #print("end_vlist = {} \n".format(end_vlist))
    #print("end_vlist_offset = {} \n".format(end_vlist_offset))
    

    #test angle calculation
    #vector1 = [0,0,1]
    # [1,0,0]
    #vector2 = [0-math.sqrt(2)/2, 0-math.sqrt(2)/2, 1]
    #print(dot_product_angle(vector1,vector2))
    #print(cart2sph(0-math.sqrt(2)/2, 0-math.sqrt(2)/2, 1)) 

    
    #obtain all the sub branches edgeds and vetices, start, end vetices
    sub_branch_list = []
    sub_branch_length_rec = []
    sub_branch_angle_rec = []
    sub_branch_start_rec = []
    sub_branch_end_rec = []
    sub_branch_projection_rec = []
    
    #if len(end_vlist) == len(end_vlist_offset):
        
    for idx, v_end in enumerate(end_vlist):
        
        #print(idx, v_end)
        #construct list of vertices in sub branches
        if idx == 0:
            v_list = [*range(0, int(end_vlist[idx])+1)]
        else:
            v_list = [*range(int(end_vlist[idx-1])+1, int(end_vlist[idx])+1)]
            
        # change type to interger 
        int_v_list = [int(i) for i in v_list]
        
        # current sub branch length
        sub_branch_length = path_length(X_skeleton[int_v_list], Y_skeleton[int_v_list], Z_skeleton[int_v_list])
        
        # current sub branch start and end points 
        start_v = [X_skeleton[int_v_list[0]], Y_skeleton[int_v_list[0]], Z_skeleton[int_v_list[0]]]
        
        end_v = [X_skeleton[int_v_list[0]] - X_skeleton[int_v_list[int(len(int_v_list)*0.7)]], Y_skeleton[int_v_list[0]] - Y_skeleton[int_v_list[int(len(int_v_list)*0.7)]], Z_skeleton[int_v_list[0]] - Z_skeleton[int_v_list[int(len(int_v_list)*0.7)]]]
        
        
        angle_sub_branch = dot_product_angle(start_v, end_v)
        
        
        p0 = np.array([X_skeleton[int_v_list[0]], Y_skeleton[int_v_list[0]], Z_skeleton[int_v_list[0]]])
        
        p1 = np.array([X_skeleton[int_v_list[0]], Y_skeleton[int_v_list[0]], Z_skeleton[int_v_list[-1]]])
        
        projection_radius = np.linalg.norm(p0 - p1)
        
        # save computed parameters for each branch
        sub_branch_list.append(v_list)
        
        sub_branch_length_rec.append(sub_branch_length)
        
        sub_branch_angle_rec.append(angle_sub_branch)
        
        sub_branch_start_rec.append(int_v_list[0])
        
        sub_branch_end_rec.append(int_v_list[-1])
        
        sub_branch_projection_rec.append(projection_radius)
        
    #sort branches according to the start vertex location(Z value)
    Z_loc = [Z_skeleton[index] for index in sub_branch_start_rec]
    
    sorted_idx_Z_loc = np.argsort(Z_loc)
    
    #print("Z_loc = {}\n".format(sorted_idx_Z_loc))
    
    #sort list according to sorted_idx_Z_loc 
    sub_branch_list[:] = [sub_branch_list[i] for i in sorted_idx_Z_loc] 
    sub_branch_length_rec[:] = [sub_branch_length_rec[i] for i in sorted_idx_Z_loc]
    sub_branch_angle_rec[:] = [sub_branch_angle_rec[i] for i in sorted_idx_Z_loc]
    sub_branch_start_rec[:] = [sub_branch_start_rec[i] for i in sorted_idx_Z_loc]
    sub_branch_end_rec[:] = [sub_branch_end_rec[i] for i in sorted_idx_Z_loc]

    print("sub_branch_length_rec = {}\n".format(sub_branch_length_rec[0:20]))
    
    
    # find dominant sub branches with longer length and depth values by clustering sub_branch_length_rec values
    ####################################################################
    labels_length_rec = cluster_1D(sub_branch_length_rec, n_clusters = 2)
    
    if labels_length_rec.tolist().index(0) == 0:
        dsf_length_divide_idx = labels_length_rec.tolist().index(1)
    else:
        dsf_length_divide_idx = labels_length_rec.tolist().index(0)
    
    print("divide_idx for sub_branch_length_rec = {}\n".format(dsf_length_divide_idx))
    
    
    #obtain parametres for dominant sub branches from index 'dsf_length_divide_idx'
    ####################################################################
    brace_length_list = sub_branch_length_rec[0:dsf_length_divide_idx]
    
    #print("brace_length_list = {}\n".format(brace_length_list))
    
    (outlier_remove_brace_length_list, idx_dominant) = outlier_remove(brace_length_list)
    
    #print("outlier_remove_brace_length_list = {}\n".format(idx_dominant))
    
    brace_angle_list = [sub_branch_angle_rec[index] for index in idx_dominant]
    
    projection_radius_list = [sub_branch_projection_rec[index] for index in idx_dominant]
    
    #print("brace_angle_list = {}\n".format(brace_angle_list))
    num_brace = dsf_length_divide_idx
    
    avg_brace_length = round(np.mean(outlier_remove_brace_length_list),2)
    
    avg_brace_angle = round(np.mean(brace_angle_list),2)
    
    avg_projection_radius = round(np.mean(projection_radius_list),2)
    
    print("num_brace = {} avg_brace_length = {}  avg_brace_angle = {}  avg_projection_radius = {}\n".format(num_brace, avg_brace_length, avg_brace_angle,avg_projection_radius))
    
    
    #find sub branch start vertices locations 
    sub_branch_start_rec_selected = sub_branch_start_rec[0:dsf_length_divide_idx]
    
    sub_branch_end_rec_selected = sub_branch_end_rec[0:dsf_length_divide_idx]
    
    #print("sub_branch_start_rec_selected = {}\n".format(sub_branch_start_rec_selected))
    
    sub_branch_start_Z = Z_skeleton[sub_branch_start_rec_selected]
    
    sub_branch_end_Z = Z_skeleton[sub_branch_end_rec_selected]
    
    #print("sub_branch_start_Z = {}\n".format(sub_branch_start_Z))
    
    
    #find location index from cluster label
    labels_start_Z = cluster_1D(sub_branch_start_Z, n_clusters = 2)
    
    if labels_start_Z.tolist().index(0) == 0:
        dsf_start_Z_divide_idx = labels_start_Z.tolist().index(1)
    else:
        dsf_start_Z_divide_idx = labels_start_Z.tolist().index(0)
    
    #print("dsf_start_Z_divide_idx for sub_branch_start_Z = {}\n".format(dsf_start_Z_divide_idx))
    
    
    #compute whorl distance based on distance between combined close points
    #whorl_dis_1 = abs(sub_branch_start_Z[0] - sub_branch_start_Z[dsf_start_Z_divide_idx])
    
    #whorl_dis_2 = abs(sub_branch_start_Z[dsf_start_Z_divide_idx] - sub_branch_start_Z[-1])
    
    #print("whorl_dis_1 = {}\n  whorl_dis_2 = {}\n".format(whorl_dis_1, whorl_dis_2))

    
    #find closest point pairs and connect close graph edges
    ####################################################################
    v_closest_pair_rec = []
    
    closest_pts = []
    
    #find closest point set and connect graph edges
    for idx, (sub_branch, anchor_point) in enumerate(zip(sub_branch_list, end_vlist_offset)):
        
        # start vertex of an edge
        anchor_point = (X_skeleton[end_vlist_offset[idx]], Y_skeleton[end_vlist_offset[idx]], Z_skeleton[end_vlist_offset[idx]])

        # curve of the edge in 3D
        point_set = np.zeros((len(sub_branch_list[0]), 3))
        
        point_set[:,0] = X_skeleton[sub_branch_list[0]]
        point_set[:,1] = Y_skeleton[sub_branch_list[0]]
        point_set[:,2] = Z_skeleton[sub_branch_list[0]]
        
        (index_cp, value_cp) = closest_point(point_set, anchor_point)

        v_closest_pair = [index_cp, end_vlist_offset[idx]]

        dis_v_closest_pair = path_length(X_skeleton[v_closest_pair], Y_skeleton[v_closest_pair], Z_skeleton[v_closest_pair])
        
        #small threshold indicating close pair vetices
        if dis_v_closest_pair < 0.01:
            
            closest_pts.append(index_cp)
            
            #print("dis_v_closest_pair = {}".format(dis_v_closest_pair))
            v_closest_pair_rec.append(v_closest_pair)
            
            #print("closest point pair: {0}".format(v_closest_pair))
            
            #connect graph edges
            G_unordered.add_edge(index_cp, end_vlist_offset[idx])
            
    print("v_closest_pair_rec = {}\n".format(v_closest_pair_rec))
    
    #get the unique values from the list
    #closest_pts_unique = list(set(closest_pts))
    
    #keep repeat values for correct indexing order
    closest_pts_unique = list((closest_pts))
    
    closest_pts_unique_sorted = sorted(closest_pts_unique)
    
    print("closest_pts_unique_sorted = {}\n".format(closest_pts_unique_sorted))

   
    #sort and combine adjacent connecting vertices in closest_pts  
    ####################################################################
    X = X_skeleton[closest_pts_unique_sorted]
    Y = Y_skeleton[closest_pts_unique_sorted]
    Z = Z_skeleton[closest_pts_unique_sorted]
    
    # compute distance between adjacent vertices in closest_pts_unique_sorted
    dis_closest_pts = [math.sqrt((X[i]-X[i-1])**2 + (Y[i]-Y[i-1])**2 + (Z[i]-Z[i-1])**2) for i in range (1, len(X))]
    
    print("distance between closest_pts_unique = {}\n".format(dis_closest_pts))
    
    
    #find outlier of closest points distance list to combine close points
    ####################################################################
    index_outlier = mad_based_outlier(np.asarray(dis_closest_pts),3.5)
    
    #print("index_outlier = {}".format(index_outlier))
    
    index_outlier_loc = [i for i, x in enumerate(index_outlier) if x]
    
    closest_pts_unique_sorted_combined = [closest_pts_unique_sorted[index] for index in index_outlier_loc]
    
    print("index_outlier = {}\n".format(index_outlier_loc))
    
    print("closest_pts_unique_sorted_combined = {}\n".format(closest_pts_unique_sorted_combined))
    
    #find Z locations of stem part
    Z_range_stem = (Z_skeleton[0], Z_skeleton[closest_pts_unique_sorted_combined[0]])
    
    #Z_range_crown = (Z_skeleton[closest_pts_unique_sorted_combined[0]], Z_skeleton[sub_branch_end_rec[dsf_length_divide_idx]])
    
    #Z_range_brace = (Z_skeleton[sub_branch_end_rec[dsf_length_divide_idx]], Z_skeleton[sub_branch_end_rec_selected[-1]])
    
    #compute whorl distance
    whorl_dis_1 = abs(Z_skeleton[closest_pts_unique_sorted_combined[0]] - Z_skeleton[sub_branch_end_rec[dsf_length_divide_idx]])
    
    whorl_dis_2 = abs(Z_skeleton[sub_branch_end_rec[dsf_length_divide_idx]] - Z_skeleton[sub_branch_end_rec_selected[-1]])
    
    print("whorl_dis_1 = {}\n  whorl_dis_2 = {}\n".format(whorl_dis_1, whorl_dis_2))
    
    
    #Z_range_crown = (sub_branch_start_Z[0], sub_branch_start_Z[dsf_start_Z_divide_idx])
    
    Z_range_crown = (Z_skeleton[closest_pts_unique_sorted_combined[0]], sub_branch_start_Z[-1])
    
    Z_range_brace = (sub_branch_start_Z[-1], sub_branch_end_Z[0])

    Z_range_brace_skeleton = (sub_branch_start_Z[dsf_start_Z_divide_idx], sub_branch_start_Z[-1])
    
    idx_brace_skeleton = np.where(np.logical_and(Z_skeleton[sub_branch_start_rec] >= Z_range_brace_skeleton[0], Z_skeleton[sub_branch_start_rec] <= Z_range_brace_skeleton[1]))
    
    print("idx_brace_skeleton = {}\n".format(idx_brace_skeleton))
    
    print("Z_range_crown = {}\n  Z_range_brace = {}\n".format(Z_range_crown, Z_range_brace))
    
    
    #find sub branches within Z_range_crown
    
    idx_crown = np.where(np.logical_and(Z_skeleton[sub_branch_start_rec] >= Z_range_crown[0], Z_skeleton[sub_branch_start_rec] <= Z_range_crown[1]))
    
    #convert tuple to array
    #idx_crown = idx_crown[0]
    
    print(idx_crown[0], idx_crown[0][0], idx_crown[0][-1])
    
    idx_brace = np.where(np.logical_and(Z_skeleton[sub_branch_start_rec] >= Z_range_brace[0], Z_skeleton[sub_branch_start_rec] <= Z_range_brace[1]))
    
    print(idx_brace[0], idx_brace[0][0], idx_brace[0][-1])
    
    #find sub branches within Z_range_brace
    
    
    #####################################################################
   
   
    
    '''
    #Search skeleton graph
    ####################################################################
    search_radius = 150
    
    neighbors_idx_rec = []
    
    # search neighbors of every vertex in closest_pts_unique_sorted_combined to find sub branches
    for idx, val in enumerate(closest_pts_unique_sorted_combined):
        
        anchor_pt_idx = int(val)
        
        idx = get_neighbors(Data_array_skeleton, anchor_pt_idx, search_radius)
    
        neighbors_idx = sorted(list(np.asarray(idx)))
        
        #find branches within near neighbors search range
        #print("neighbors_idx = {}\n".format(neighbors_idx))

        #find branches within near neighbors 
        neighbors_match = sorted(list(set(sub_branch_start_rec).intersection(set(neighbors_idx))))
        
        print("Found {} matches, neighbors_match = {}\n".format(len(neighbors_match), neighbors_match))
        
        neighbors_idx_rec.append(neighbors_match)
        
    ####################################################################
    
    
    v_closest_pair_rec_selected = [v_closest_pair_rec[index] for index in index_outlier_loc] 
    
    v_closest_start_selected = [v_closest_pair_rec[index][1] for index in index_outlier_loc]
    
    #print("v_closest_pair_rec_selected = {}\n".format(v_closest_pair_rec_selected))
    
    #print("v_closest_start_selected = {}\n".format(v_closest_start_selected))
    
    
    #sub_branch_selected = [sub_branch_list[index+1] for index in index_outlier_loc]
    
    index_level_selected = [int(index+1) for index in index_outlier_loc]
    
    print("index_level_selected = {}\n".format(index_level_selected))
    
    
    level_range_set = []
    
    for idx, val in enumerate(index_level_selected):
        
        if (idx+1) < len(index_level_selected): 
            
            range_idx = range(index_level_selected[idx], index_level_selected[idx+1])
        
            #print([*range_idx])
            
            level_range_set.append([*range_idx])


    #choose level set depth
    combined_level_range_set = level_range_set[0:2]
    
    combined_level_range_set = [item for sublist in combined_level_range_set for item in sublist]
    
    print("combined_level_range_set = {}\n".format(combined_level_range_set))
    
    #sub_branch_selected = [sub_branch_list[index] for index in combined_level_range_set]
    '''
   
    
    #convert skeleton data to KDTree using Open3D to search nearest neighbors
    #find branches within near neighbors search range
    ####################################################################
    '''
    anchor_pt_idx = 30
    
    search_radius = 150
    
    idx = get_neighbors(Data_array_skeleton, anchor_pt_idx, search_radius)
    
    neighbors_idx = sorted(list(np.asarray(idx)))
    
    print("neighbors_idx = {}\n".format(neighbors_idx))
    
    #find branches within near neighbors 
    neighbors_match = sorted(list(set(sub_branch_start_rec).intersection(set(neighbors_idx))))
    
    print("neighbors_match = {}\n".format(neighbors_match))
    
    
    
    level = 1
    
    neighbors_match_idx = [i for i, item in enumerate(sub_branch_start_rec) if item in neighbors_idx_rec[level]]
    
    #neighbors_match_idx = [int(i) for i in neighbors_match_idx]
    
    sub_branch_selected = [sub_branch_list[index] for index in sorted(neighbors_match_idx)]
    
    #print("neighbors_match_idx = {}\n".format(neighbors_match_idx))
    #print("sub_branch_selected = {}\n".format(len(sub_branch_selected)))
    
    num_1_order = len(sub_branch_selected)
    
    angle_1_order = [sub_branch_angle_rec[index] for index in sorted(neighbors_match_idx)]
    
    length_1_order = [sub_branch_length_rec[index] for index in sorted(neighbors_match_idx)]
    
    print("num_1_order = {0}\n  angle_1_order = {1}\n length_1_order = {2}\n".format(num_1_order, angle_1_order, length_1_order))
    
    '''
        

    #find shortest path between start and end vertex
    ####################################################################
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
    #initialize parameters
    pt_diameter_max=pt_diameter_min=pt_length=pt_eccentricity=pt_stem_diameter=0
        
    #load ply point cloud file
    if not (filename_pcloud is None):
        
        model_pcloud = current_path + filename_pcloud
        
        print("Loading 3D point cloud {}...\n".format(filename_pcloud))
        
        model_pcloud_name_base = os.path.splitext(model_pcloud)[0]
        
        pcd = o3d.io.read_point_cloud(model_pcloud)
        
        Data_array_pcloud = np.asarray(pcd.points)
       
        
        # sort points according to z value increasing order
        #Sorted_Data_array_pcloud = np.asarray(sorted(Data_array_pcloud, key = itemgetter(2), reverse = True))
        
        #compute dimensions of point cloud data
        (pt_diameter_max, pt_diameter_min, pt_length) = get_pt_parameter(Data_array_pcloud)
        
        print("pt_diameter_max = {} pt_diameter_min = {} pt_length = {}\n".format(pt_diameter_max,pt_diameter_min,pt_length))
        
        
        #print(Data_array_pcloud.shape)
        
        
        
        '''
        ################################################################
        
        print(idx_brace[0][0], idx_brace[0][-1])
        
        anchor_pt = (X_skeleton[25], Y_skeleton[25], Z_skeleton[25])
        
        pcd = o3d.geometry.PointCloud()
        
        pcd.points = o3d.utility.Vector3dVector(Data_array_pcloud)
        
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

        
        # Build KDTree from point cloud for fast retrieval of nearest neighbors
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        
        #print("Paint the 00th point red.")
        
        #pcd.colors[anchor_pt] = [1, 0, 0]
        
        search_radius = 150
        #print("Find its 50 nearest neighbors, paint blue.")
        
        [k, idx, _] = pcd_tree.search_knn_vector_3d(anchor_pt, search_radius)
        
        #print("nearest neighbors = {}\n".format(sorted(np.asarray(idx[1:]))))
        
        np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
        
        #o3d.visualization.draw_geometries([pcd])
        '''
        ################################################################
        
        
        #extract stem part from point cloud model
        idx_pt_Z_range_stem = np.where(np.logical_and(Data_array_pcloud[:,2] >= Z_range_stem[0], Data_array_pcloud[:,2] <= Z_range_stem[1]))
        
        Data_array_pcloud_Z_range_stem = Data_array_pcloud[idx_pt_Z_range_stem]
        

        idx_pt_Z_range_crown = np.where(np.logical_and(Data_array_pcloud[:,2] >= Z_range_crown[0], Data_array_pcloud[:,2] <= Z_range_crown[1]))
        
        Data_array_pcloud_Z_range_crown = Data_array_pcloud[idx_pt_Z_range_crown]
        
        
        idx_pt_Z_range_brace = np.where(np.logical_and(Data_array_pcloud[:,2] >= Z_range_brace[0], Data_array_pcloud[:,2] <= Z_range_brace[1]))
        
        Data_array_pcloud_Z_range_brace = Data_array_pcloud[idx_pt_Z_range_brace]
        
        
        ratio_stem = abs(Z_range_stem[0] - Z_range_stem[1])/pt_length
        ratio_crown = abs(Z_range_crown[0] - Z_range_crown[1])/pt_length
        ratio_brace = abs(Z_range_brace[0] - Z_range_brace[1])/pt_length
        
        print("ratio_stem = {} ratio_crown = {} ratio_brace = {}\n".format(ratio_stem,ratio_crown,ratio_brace))
        
        
        # save partital model for diameter measurement
        model_stem = (current_path + 'stem.ply')
        write_ply(model_stem, Data_array_pcloud_Z_range_stem)
        
        model_crown = (current_path + 'crown.ply')
        write_ply(model_crown, Data_array_pcloud_Z_range_crown)
        
        model_brace = (current_path + 'brace.ply')
        write_ply(model_brace, Data_array_pcloud_Z_range_brace)

        
        
        (pt_stem_diameter_max,pt_stem_diameter_min,pt_stem_length) = get_pt_parameter(Data_array_pcloud_Z_range_stem)
        
        print("pt_stem_diameter_max = {} pt_stem_diameter_min = {} pt_stem_length = {}\n".format(pt_stem_diameter_max,pt_stem_diameter_min,pt_stem_length))
        
        pt_stem_diameter = (pt_stem_diameter_max + pt_stem_diameter_min)*0.5
        
        pt_eccentricity = (pt_stem_diameter_min/pt_stem_diameter_max)
        
        
        if pcd.has_colors():
            
            print("Render colored point cloud\n")
            
            pcd_color = np.asarray(pcd.colors)
            
            if len(pcd_color) > 0: 
                
                pcd_color = np.rint(pcd_color * 255.0)
            
            #pcd_color = tuple(map(tuple, pcd_color))
        else:
            
            print("Generate random color\n")
        
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
    #pts = mlab.points3d(X_skeleton[0], Y_skeleton[0], Z_skeleton[0], color = (0.58, 0.29, 0), mode = 'sphere', scale_factor = 0.15)
    
    #pts = mlab.points3d(X_skeleton[neighbors_idx], Y_skeleton[neighbors_idx], Z_skeleton[neighbors_idx], mode = 'sphere', color=(0,0,1), scale_factor = 0.05)
    
    #pts = mlab.points3d(X_skeleton[end_vlist_offset], Y_skeleton[end_vlist_offset], Z_skeleton[end_vlist_offset], color = (1,1,1), mode = 'sphere', scale_factor = 0.03)
    
    pts = mlab.points3d(X_skeleton[sub_branch_start_rec_selected], Y_skeleton[sub_branch_start_rec_selected], Z_skeleton[sub_branch_start_rec_selected], color = (1,0,0), mode = 'sphere', scale_factor = 0.08)
    
    
   
    #cmap = get_cmap(len(sub_branch_list))
    
    cmap = get_cmap(dsf_length_divide_idx)
    
    #draw all the sub branches in loop 
    for i, (sub_branch, sub_branch_start, sub_branch_angle) in enumerate(zip(sub_branch_list, sub_branch_start_rec, sub_branch_angle_rec)):

        #if i < dsf_length_divide_idx:
        if i <= idx_brace_skeleton[0][-1] and i >= idx_brace_skeleton[0][0] :
            
            color_rgb = cmap(i)[:len(cmap(i))-1]
            
            pts = mlab.points3d(X_skeleton[sub_branch], Y_skeleton[sub_branch], Z_skeleton[sub_branch], color = color_rgb, mode = 'sphere', scale_factor = 0.05)
    
            mlab.text3d(X_skeleton[sub_branch_start], Y_skeleton[sub_branch_start], Z_skeleton[sub_branch_start]-0.05, str(i), color = color_rgb, scale = (0.04, 0.04, 0.04))
     
    
    
    
    #for i, (end_val, x_e, y_e, z_e) in enumerate(zip(closest_pts_unique_sorted_combined, X_skeleton[closest_pts_unique_sorted_combined], Y_skeleton[closest_pts_unique_sorted_combined], Z_skeleton[closest_pts_unique_sorted_combined])):
        
        #mlab.text3d(x_e, y_e, z_e, str(end_val), scale = (0.04, 0.04, 0.04))
    
    #pts = mlab.points3d(Data_array_pcloud[:,0], Data_array_pcloud[:,1], Data_array_pcloud[:,2], mode = 'point')
    
    

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
    mlab.show()
    

    return pt_diameter_max, pt_diameter_min, pt_length, pt_eccentricity, \
        pt_stem_diameter, num_brace, avg_brace_length, avg_brace_angle, avg_projection_radius, whorl_dis_1, whorl_dis_2


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

    (pt_diameter_max, pt_diameter_min, pt_length, pt_eccentricity, pt_stem_diameter, \
        num_brace, avg_brace_length, avg_brace_angle, avg_projection_radius, whorl_dis_1, whorl_dis_2) = analyze_skeleton(current_path, filename_skeleton, filename_pcloud)

    trait_sum = []
    
    trait_sum.append([pt_diameter_max, pt_diameter_min, pt_length, pt_eccentricity, \
        pt_stem_diameter, num_brace, avg_brace_length, avg_brace_angle, avg_projection_radius, whorl_dis_1, whorl_dis_2])
    
    #save reuslt file
    ####################################################################
    
    trait_file = (current_path + 'trait.xlsx')
    
    #trait_file_csv = (current_path + 'trait.csv')
    
    
    if os.path.isfile(trait_file):
        # update values
        #Open an xlsx for reading
        wb = openpyxl.load_workbook(trait_file)

        #Get the current Active Sheet
        sheet = wb.active
        
    else:
        # Keep presets
        wb = openpyxl.Workbook()
        sheet = wb.active

        sheet.cell(row = 1, column = 1).value = 'root system diameter max'
        sheet.cell(row = 1, column = 2).value = 'root system diameter min'
        sheet.cell(row = 1, column = 3).value = 'root system diameter'
        sheet.cell(row = 1, column = 4).value = 'root system eccentricity'
        sheet.cell(row = 1, column = 5).value = 'stem root diameter'
        sheet.cell(row = 1, column = 6).value = 'number of brace roots'
        sheet.cell(row = 1, column = 7).value = 'brace root length'
        sheet.cell(row = 1, column = 8).value = 'brace root angle'
        sheet.cell(row = 1, column = 9).value = 'root trace projection radius'
        sheet.cell(row = 1, column = 10).value = 'whorl distance 1'
        sheet.cell(row = 1, column = 11).value = 'whorl distance 2'
              
        
    for row in trait_sum:
        sheet.append(row)
   
   
    #save the csv file
    wb.save(trait_file)
    
    if os.path.exists(trait_file):
        
        print("Result file was saved")
    else:
        print("Error saving Result file")