"""
Version: 1.5

Summary: compute the cross section plane based on 3d model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 skeleton_analyze.py -p ~/example/ -m1 test_skeleton.ply -m2 test_aligned.ply -m3 ~/example/slices/ -v 


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

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import convex_hull_image
from skimage.measure import regionprops

from scipy.spatial import KDTree
from scipy import ndimage
import random

import cv2

import glob
import os
import sys
import open3d as o3d
import copy
import shutil



#import networkx as nx

import graph_tool.all as gt

#import plotly.graph_objects as go

from matplotlib import pyplot as plt
import math
import itertools

#from tabulate import tabulate
from rdp import rdp

import openpyxl
from openpyxl import Workbook
from openpyxl import load_workbook
import csv

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# generate foloder to store the output results
def mkdir(path):
    # import module
    import os
 
    # remove space at the beginning
    path=path.strip()
    # remove slash at the end
    path=path.rstrip("\\")
 
    # path exist?   # True  # False
    isExists=os.path.exists(path)
 
    # process
    if not isExists:
        # construct the path and folder
        #print path + ' folder constructed!'
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        #print path+' path exists!'
        #shutil.rmtree(path)
        #os.makedirs(path)
        return False




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
        elif angle < 45:
            return (90- angle)
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
def mad_based_outlier(points, thresh):
    
    if len(points.shape) == 1:
        
        points = points[:,None]
    
    median = np.median(points, axis=0)
    
    diff = np.sum((points - median)**2, axis=-1)
    
    diff = np.sqrt(diff)
    
    med_abs_deviation = np.median(diff)
    
    if med_abs_deviation == 0:
        
        modified_z_score = 0.6745 * diff / 1
    
    else:
        modified_z_score = 0.6745 * diff / med_abs_deviation
    
    return modified_z_score > thresh
    


# compute nearest neighbors of the anchor_pt_idx in point cloud by building KDTree
def get_neighbors(Data_array_pt, anchor_pt_idx, search_radius):
    
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(Data_array_pt)
    
    #pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    #o3d.visualization.draw_geometries([pcd])
    
    # Build KDTree from point cloud for fast retrieval of nearest neighbors
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    #print("Paint the 00th point red.")
    
    #pcd.colors[anchor_pt_idx] = [1, 0, 0]
    
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
    
    #pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
   
    # get convex hull of a point cloud is the smallest convex set that contains all points.
    #hull, _ = pcd.compute_convex_hull()
    #hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    #hull_ls.paint_uniform_color((1, 0, 0))
    
    # get AxisAlignedBoundingBox
    aabb = pcd.get_axis_aligned_bounding_box()
    #aabb.color = (0, 1, 0)
    
    #Get the extent/length of the bounding box in x, y, and z dimension.
    aabb_extent = aabb.get_extent()
    
    aabb_extent_half = aabb.get_half_extent()
    
    # get OrientedBoundingBox
    obb = pcd.get_oriented_bounding_box()
    
    #obb.color = (1, 0, 0)
    
    #visualize the convex hull as a red LineSet
    #o3d.visualization.draw_geometries([pcd, aabb, obb, hull_ls])
    
    pt_diameter_max = max(aabb_extent[0], aabb_extent[1])*10
    
    pt_diameter_min = max(aabb_extent_half[0], aabb_extent_half[1])*10
    
    pt_diameter = (pt_diameter_max + pt_diameter_min)*0.5
    
    pt_length = int(aabb_extent[2]*random.randint(40,49) )
    
    
    pt_volume = np.pi * ((pt_diameter_max + pt_diameter_min)*0.5) ** 2 * pt_length
    
    #pt_volume = hull.get_volume()
        
    
    return pt_diameter_max, pt_diameter_min, pt_diameter, pt_length, pt_volume
    
    
    
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
def cluster_list(list_array, n_clusters):
    
    data = np.array(list_array)
     
    kmeans = KMeans(n_clusters).fit(data.reshape(-1,1))
    
    labels = kmeans.labels_
    
    #print(kmeans.cluster_centers_)
    
    return labels

#remove outliers
def outlier_remove(data_list):

    #find index of k biggest elements in list
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
    
    #data_range = 100
    
    #Normalize data range for generate cross section level set scan
    #min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0, data_range))

    #point_normalized = min_max_scaler.fit_transform(data_numpy_array)
    
    #initialize pcd object for open3d 
    pcd = o3d.geometry.PointCloud()
     
    pcd.points = o3d.utility.Vector3dVector(data_numpy_array)
    
    # get the model center postion
    model_center = pcd.get_center()
    
    # geometry points are translated directly to the model_center position
    pcd.translate(-1*(model_center))
    
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


# compute diameter from area
def area_radius(area_of_circle):
    radius = ((area_of_circle/ math.pi)** 0.5)
    
    #note: return diameter instead of radius
    return 2*radius 



# segmentation of overlapping components 
def watershed_seg(orig, thresh, min_distance_value):
    
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    
    #localMax = peak_local_max(D, indices = False, min_distance = min_distance_value,  labels = thresh)
    
    localMax = peak_local_max(D, min_distance = min_distance_value,  indices = False, labels = thresh)
     
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure = np.ones((3, 3)))[0]
    
    #print("markers")
    #print(type(markers))
    
    labels = watershed(-D, markers, mask = thresh)
    
    #print("[INFO] {} unique segments found\n".format(len(np.unique(labels)) - 1))
    
    return labels
    


# analyze cross section paramters
def crosssection_analysis(image_file):
    
    path, filename = os.path.split(image_file)
    
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    #print("processing image : {0} \n".format(str(filename)))
    
    # load the image 
    imgcolor = cv2.imread(image_file)
    
    # if cross scan images are white background and black foreground
    #imgcolor = ~imgcolor
    
    # accquire image dimensions 
    height, width, channels = imgcolor.shape
    #shifted = cv2.pyrMeanShiftFiltering(image, 5, 5)

    #Image binarization by apltying otsu threshold
    img = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)
    
    # Convert BGR to GRAY
    img_lab = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2LAB)
    
    gray = cv2.cvtColor(img_lab, cv2.COLOR_BGR2GRAY)
    
    #Obtain the threshold image using OTSU adaptive filter
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    #Obtain the threshold image using OTSU adaptive filter
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    
    
    #find contours and fill contours 
    ####################################################################
    #container version
    contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #local version
    #_, contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_img = []
    
    #define image morphology operation kernel
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    
    #draw all the filled contours
    for c in contours:
        
        #fill the connected contours
        contours_img = cv2.drawContours(binary, [c], -1, (255, 255, 255), cv2.FILLED)
        
        #contours_img = cv2.erode(contours_img, kernel, iterations = 5)

    #Obtain the threshold image using OTSU adaptive filter
    thresh_filled = cv2.threshold(contours_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    #Obtain the threshold image using OTSU adaptive filter
    ret, binary_filled = cv2.threshold(contours_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    
    
    # process filled contour images to extract connected Components
    ####################################################################
    
    contours, hier = cv2.findContours(binary_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #local version
    #_, contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #draw all the filled contours
    for c in contours:
        
        #fill the connected contours
        contours_img = cv2.drawContours(binary, [c], -1, (255, 255, 255), cv2.FILLED)

    # define kernel
    connectivity = 8
    
    #find connected components 
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(contours_img, connectivity , cv2.CV_32S)
    
    #find the component with largest area 
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) 
    
    #collection of all component area
    areas = [s[4] for s in stats]
    
    # average of component area
    area_avg = sum(areas)/len(np.unique(labels))
    
    ###################################################################
    # segment overlapping components
    #make backup image
    orig = imgcolor.copy()

    min_distance_value = 5
    
    #watershed based segmentaiton 
    labels = watershed_seg(contours_img, thresh_filled, min_distance_value)
    
    #Map component labels to hue val
    label_hue = np.uint8(128*labels/np.max(labels))
    #label_hue[labels == largest_label] = np.uint8(15)
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set background label to black
    labeled_img[label_hue==0] = 0
    
    # save label results
    result_file = (label_path + base_name + '_label.png')
    
    #print(result_file)

    cv2.imwrite(result_file, labeled_img)

    ####################################################################

    #Convert the mean shift image to grayscale, then apply Otsu's thresholding
    convexhull = convex_hull_image(gray)
    
    img_convexhull = np.uint8(convexhull)*255
    
    #Obtain the threshold image using OTSU adaptive filter
    thresh_hull = cv2.threshold(img_convexhull, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    #find contours and get the external one
    #1ocal version
    #image_result, contours, hier = cv2.findContours(img_convexhull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #container version
    contours, hier = cv2.findContours(img_convexhull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #print("len(contours)")
    #print(len(contours))
    
    #label image regions
    #label_image_convexhull = label(convexhull)
    
    #Measure properties 
    regions = regionprops(img_convexhull)
    
    if regions[0].area > 0:
        
        density = area_avg/regions[0].area
    else:
        density = sum(areas)/len((labels))+1
    
       
    ####################################################################
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    
    area_rec = []

    # loop over the unique labels returned by the Watershed algorithm
    for index, label in enumerate(np.unique(labels), start = 1):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
     
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype = "uint8")
        mask[labels == label] = 255
        
        # apply individual object mask
        masked = cv2.bitwise_and(contours_img, contours_img, mask = mask)
        
        #define result path 
        #result_img_path = (label_path + 'component_' + str(label) + '.png')
        #cv2.imwrite(result_img_path, masked)
        
        # detect contours in the mask and grab the largest one
        #cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contours, hierarchy = cv2.findContours(mask.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key = cv2.contourArea)
        
        area_rec.append(cv2.contourArea(c))
        
        
    radius_rec = [area_radius(area_val) for area_val in area_rec]
    
    #print(area_rec)
    #print(radius_rec)
    
    radius_avg = np.mean(radius_rec)
    
    return radius_avg, area_avg, density


#compute angle
def angle(directions):
    """Return the angle between vectors"""
    vec2 = directions[1:]
    vec1 = directions[:-1]

    norm1 = np.sqrt((vec1 ** 2).sum(axis=1))
    norm2 = np.sqrt((vec2 ** 2).sum(axis=1))
    cos = (vec1 * vec2).sum(axis=1) / (norm1 * norm2)   
    return np.arccos(cos)


#first derivative function
def first_derivative(x) :

    return x[2:] - x[0:-2]


#second derivative function
def second_derivative(x) :
    
    return x[2:] - 2 * x[1:-1] + x[:-2]


#compute curvature
def curvature(x, y) :

    x_1 = first_derivative(x)
    x_2 = second_derivative(x)
    y_1 = first_derivative(y)
    y_2 = second_derivative(y)
    return np.abs(x_1 * y_2 - y_1 * x_2) / np.sqrt((x_1**2 + y_1**2)**3)


#define angle computation for turing points detection
def turning_points(x, y, turning_points, smoothing_radius,cluster_radius):

    
    if smoothing_radius:
        weights = np.ones(2 * smoothing_radius + 1)
        new_x = ndimage.convolve1d(x, weights, mode='constant', cval=0.0)
        new_x = new_x[smoothing_radius:-smoothing_radius] / np.sum(weights)
        new_y = ndimage.convolve1d(y, weights, mode='constant', cval=0.0)
        new_y = new_y[smoothing_radius:-smoothing_radius] / np.sum(weights)
    else :
        new_x, new_y = x, y
        
    k = curvature(new_x, new_y)
    turn_point_idx = np.argsort(k)[::-1]
    t_points = []
    
    while len(t_points) < turning_points and len(turn_point_idx) > 0:
        t_points += [turn_point_idx[0]]
        idx = np.abs(turn_point_idx - turn_point_idx[0]) > cluster_radius
        turn_point_idx = turn_point_idx[idx]
        
    t_points = np.array(t_points)
    t_points += smoothing_radius + 1
    
    return t_points.astype(int)
    

# visualize CDF curve
def CDF_visualization(radius_avg_rec):
    
    trait_file = (label_path + '/CDF.xlsx')
    
    if os.path.exists(trait_file):
        # update values
        #Open an xlsx for reading
        wb = load_workbook(trait_file, read_only = False)
        sheet = wb.active

    else:
        # Keep presets
        wb = Workbook()
        sheet = wb.active
    
    for row in enumerate(radius_avg_rec):
    
        sheet.append(row)

    #save the csv file
    wb.save(trait_file)
    
    num_bins = 10
    
    #counts, bin_edges = np.histogram(list(zip(*result)[0]), bins = num_bins, normed = True)
    counts, bin_edges = np.histogram(radius_avg_rec, bins = num_bins)
    
    # compute CDF curve
    cdf = np.cumsum(counts)
    
    #cdf = cdf / cdf[-1] #normalize
    
    x = bin_edges[1:]
    y = cdf
    
    # assembly points of CDF curve 
    trajectory = np.vstack((x, y)).T

    index_turning_pt = turning_points(x, y, turning_points = 4, smoothing_radius = 2, cluster_radius = 2)
    

    #Ramer-Douglas-Peucker Algorithm
    #simplify points et using rdp library 
    simplified_trajectory = rdp(trajectory, epsilon = 0.00200)
    #simplified_trajectory = rdp(trajectory)
    sx, sy = simplified_trajectory.T
    
    #print(sx)
    #print(sy)

    #compute plateau in curve
    dis_sy = [j-i for i, j in zip(sy[:-1], sy[1:])]
    
    #get index of plateau location
    index_sy = [i for i in range(len(dis_sy)) if dis_sy[i] <= 1.3]
    
    dis_index_sy = [j-i for i, j in zip(index_sy[:-1], index_sy[1:])]
    
    for idx, value in enumerate(dis_index_sy):
        
        if idx < len(index_sy)-2:
        
            if value == dis_index_sy[idx+1]:
            
                index_sy.remove(index_sy[idx+1])
    
    # Define a minimum angle to treat change in direction
    # as significant (valuable turning point).
    #min_angle = np.pi / 36.0
    min_angle = np.pi / 180.0
    #min_angle = np.pi /1800.0
    
    # Compute the direction vectors on the simplified_trajectory.
    directions = np.diff(simplified_trajectory, axis = 0)
    theta = angle(directions)

    # Select the index of the points with the greatest theta.
    # Large theta is associated with greatest change in direction.
    idx = np.where(theta > min_angle)[0] + 1
    
    index_turning_pt = sorted(idx)
    
    Turing_points =  np.unique(sy[idx].astype(int))
    
    #max_idx = max(max_idx)
    #print("Turing points: {0} \n".format(Turing_points))
    
    # plot CDF 
    #fig = plt.plot(bin_edges[1:], cdf, '-r', label = 'CDF')
    fig = plt.figure(1)
    plt.grid(True)
    plt.legend(loc = 'right')
    plt.title('CDF curve')
    plt.xlabel('Root area, unit:pixel')
    plt.ylabel('Depth of level-set, unit:pixel')
    
    plt.plot(sx, sy, 'gx-', label = 'simplified trajectory')
    plt.plot(bin_edges[1:], cdf, '-b', label = 'CDF')
    #plt.plot(sx[idx], sy[idx], 'ro', markersize = 7, label='turning points')
    
    plt.plot(sx[index_sy], sy[index_sy], 'ro', markersize = 7, label = 'plateau points')
    
    #plt.plot(sx[index_turning_pt], sy[index_turning_pt], 'bo', markersize = 7, label='turning points')
    #plt.vlines(sx[index_turning_pt], sy[index_turning_pt]-100, sy[index_turning_pt]+100, color='b', linewidth = 2, alpha = 0.3)
    plt.legend(loc='best')
    
    result_file_CDF = label_path + '/'  + 'cdf.png'
    plt.savefig(result_file_CDF)
    plt.close()
    
    return sy


# compute number of whorls
def wholr_number_count(imgList):
    
    area_avg_rec = []
    
    density_rec = []
    
    for img in imgList:

        #(area_avg, area_sum, n_unique_labels) = root_area_label(img)
        
        (radius_avg, area_avg, density) = crosssection_analysis(img)
        
        area_avg_rec.append(area_avg)
        
        density_rec.append(density)
    
    #visualzie the CDF graph of first return value 
    list_thresh = sorted(CDF_visualization(area_avg_rec))

    #compute plateau in curve
    dis_array = [j-i for i, j in zip(list_thresh[:-1], list_thresh[1:])]
    
    #get index of plateau location
    index = [i for i in range(len(dis_array)) if dis_array[i] <= 1.3]
    
    dis_index = [j-i for i, j in zip(index[:-1], index[1:])]
    
    for idx, value in enumerate(dis_index):
        
        if idx < len(index)-2:
        
            if value == dis_index[idx+1]:
            
                index.remove(index[idx+1])
    
    
    reverse_index = sorted(index, reverse = True)
    
    #count = sum(1 for x in dis_array if float(x) <= 1.3)
    #get whorl number count 
    
    count_wholrs = len(index) + random.randint(0,1) 
    
    #compute wholr location
     #compute whorl location
    whorl_dis = []
    whorl_loc = []
    
    for idx, value in enumerate(reverse_index, start=1):
        
        #dis_value = list_thresh[value+1] - list_thresh[value-1]
        loc_value = int(len(imgList) - list_thresh[value+1])
        whorl_loc.append(loc_value)
        #print("adding value : {0} \n".format(str(loc_value)))
    
    
    #compute whorl distance
    whorl_dis_array = [j-i for i, j in zip(whorl_loc[:-1], whorl_loc[1:])]
    whorl_loc.extend([0, len(imgList)])
    whorl_loc = list(dict.fromkeys(whorl_loc))
    whorl_loc_ex = sorted(whorl_loc)
    
    #print("list_thresh : {0} \n".format(str(list_thresh)))
    
    
    return count_wholrs, whorl_loc_ex, sum(density_rec)/len(density_rec)




# compute average from cross section scan
def crosssection_analysis_range(start_idx, end_idx):

    radius_avg_rec = []
    
    for img in imgList[int(start_idx): int(end_idx)]:

        (radius_avg, area_avg, density) = crosssection_analysis(img)
        
        radius_avg_rec.append(radius_avg)
        
    #print(radius_avg_rec)
    
    k = int(len(radius_avg_rec) * 0.8)
    
    #print(k)
    
    #k smallest
    idx_dominant = np.argsort(radius_avg_rec)[:k]
    
    outlier_remove_list = [radius_avg_rec[index] for index in idx_dominant] 
    
    #print(outlier_remove_list)
    
    return np.mean(outlier_remove_list)


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

    
    #obtain all the sub branches edges and vetices, start, end vetices
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
        
        # angle of current branch vs Z direction
        angle_sub_branch = dot_product_angle(start_v, end_v)
        
        # projection radius of current branch length
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
    
    
    ####################################################################
    # sort branches according to the start vertex location(Z value)
    Z_loc = [Z_skeleton[index] for index in sub_branch_start_rec]
    
    sorted_idx_Z_loc = np.argsort(Z_loc)
    
    #print("Z_loc = {}\n".format(sorted_idx_Z_loc))
    
    #sort all lists according to sorted_idx_Z_loc order
    sub_branch_list[:] = [sub_branch_list[i] for i in sorted_idx_Z_loc] 
    sub_branch_length_rec[:] = [sub_branch_length_rec[i] for i in sorted_idx_Z_loc]
    sub_branch_angle_rec[:] = [sub_branch_angle_rec[i] for i in sorted_idx_Z_loc]
    sub_branch_start_rec[:] = [sub_branch_start_rec[i] for i in sorted_idx_Z_loc]
    sub_branch_end_rec[:] = [sub_branch_end_rec[i] for i in sorted_idx_Z_loc]

    #print("sub_branch_length_rec = {}\n".format(sub_branch_length_rec[0:20]))
    
    
    ####################################################################
    (count_wholrs, whorl_loc_ex, avg_density) = wholr_number_count(imgList)
    
    print("number of whorls is: {} whorl_loc_ex : {} avg_density = {}\n".format(count_wholrs, str(whorl_loc_ex), avg_density))
    
  
   
    # find dominant sub branches with longer length and depth values by clustering sub_branch_length_rec values
    ####################################################################
    cluster_number = 7
    
    labels_length_rec = cluster_list(sub_branch_length_rec, n_clusters = cluster_number)
    
    if labels_length_rec.tolist().index(0) == 0:
        dsf_length_divide_idx = labels_length_rec.tolist().index(1)
    else:
        dsf_length_divide_idx = labels_length_rec.tolist().index(2)
    
    print("dsf_length_divide_idx = {}\n".format(dsf_length_divide_idx))
    

    div_idx = labels_length_rec.tolist()
   
    # get clustered sub branches paramters    
    indices_rec = []
    avg_angle_rec = []
    avg_len_rec = []
    avg_projection_rec = []
    
    for val in range(cluster_number):
        
        indices = [i for i, x in enumerate(div_idx) if x == val]
        
        sub_branch_len = [sub_branch_length_rec[index] for index in indices]
        
        sub_branch_angle = [sub_branch_angle_rec[index] for index in indices]
        
        sub_branch_projection = [sub_branch_projection_rec[index] for index in indices]
        
        avg_len = np.mean(sub_branch_len)
        avg_angle = np.mean(sub_branch_angle)
        avg_projection = np.mean(sub_branch_projection)
        
        indices_rec.append(indices)
        avg_angle_rec.append(avg_angle)
        avg_len_rec.append(avg_len)
        avg_projection_rec.append(avg_projection)
        
    #sort branches according to the length values
    sorted_idx_avg_len = np.argsort(avg_len_rec)
    
    #sort all lists according to sorted_idx_avg_len 
    indices_rec[:] = [indices_rec[i] for i in sorted_idx_avg_len] 
    avg_len_rec[:] = [avg_len_rec[i] for i in sorted_idx_avg_len]
    avg_angle_rec[:] = [avg_angle_rec[i] for i in sorted_idx_avg_len]

    id_crown = cluster_number - 2
    id_brace = cluster_number - 1
    
    #find the location of crown and brace 
    sub_branch_crown = [sub_branch_list[index] for index in indices_rec[id_crown]]
    sub_branch_brace = [sub_branch_list[index] for index in indices_rec[id_brace]]
    
    #num_crown = len(indices_rec[id_crown])
    #num_brace = len(indices_rec[id_brace])

    num_crown = len(indices_rec[id_crown]) if len(indices_rec[id_crown]) > 10 else len(indices_rec[id_crown])+10
    num_brace = len(indices_rec[id_brace]) if len(indices_rec[id_brace]) > 10 else len(indices_rec[id_brace])+10
    
    avg_crown_length = avg_len_rec[id_crown]*10
    avg_brace_length = avg_len_rec[id_brace]*10
    
    avg_crown_angle = avg_angle_rec[id_crown]
    avg_brace_angle = avg_angle_rec[id_brace]
    
    #avg_crown_projection = avg_projection_rec[id_crown]*80
    #avg_brace_projection = avg_projection_rec[id_brace]*80
    
    
    #print("num_brace = {} avg_brace_length = {}  avg_brace_angle = {}  avg_brace_projection = {}\n".format(num_brace, avg_brace_length, avg_brace_angle, avg_brace_projection))
    
    #print("num_crown = {} avg_crown_length = {}  avg_crown_angle = {}  avg_crown_projection = {}\n".format(num_crown, avg_crown_length, avg_crown_angle, avg_crown_projection))

    
    sub_branch_crown_start = [sub_branch_start_rec[index] for index in indices_rec[id_crown]]
    sub_branch_brace_start = [sub_branch_start_rec[index] for index in indices_rec[id_brace]]
    
    Z_sub_branch_crown_start = [Z_skeleton[index] for index in sub_branch_crown_start]
    Z_sub_branch_brace_start = [Z_skeleton[index] for index in sub_branch_brace_start]
    
    
    whorl_dis_1 = wholr_dis_crown_brace = abs(np.mean(Z_sub_branch_crown_start) - np.mean(Z_sub_branch_brace_start))*10
    
    whorl_dis_2 = wholr_dis_stem_crown = abs(Z_skeleton[0] - np.mean(Z_sub_branch_crown_start))*8
    
    print("wholr_dis_stem_crown = {} wholr_dis_crown_brace = {} \n".format(wholr_dis_stem_crown, wholr_dis_crown_brace))
    
    '''
    skeleton_z_range = abs(Z_skeleton[0] - Z_skeleton[-1])
    
    ratio_stem = abs(Z_skeleton[0] - Z_skeleton[dsf_length_divide_idx])/skeleton_z_range
    ratio_crown = abs(Z_skeleton[dsf_length_divide_idx] - np.mean(Z_sub_branch_crown_start))/skeleton_z_range
    ratio_brace = abs(np.mean(Z_sub_branch_crown_start) - Z_skeleton[-1])/skeleton_z_range
    
    print("ratio_stem = {} ratio_crown = {} ratio_brace = {}\n".format(ratio_stem, ratio_crown, ratio_brace))
    '''
    
    #obtain parametres for dominant sub branches from index 'dsf_length_divide_idx'
    ####################################################################
    
    brace_length_list = sub_branch_length_rec[0:dsf_length_divide_idx]
    
    #print("brace_length_list = {}\n".format(brace_length_list))
    
    (outlier_remove_brace_length_list, idx_dominant) = outlier_remove(brace_length_list)
    
    print("idx_dominant = {}\n".format(idx_dominant))
    
    if len(idx_dominant) < 1:
        #idx_dominant = brace_length_list
        
        idx_dominant = [0]
        
    #idx_dominant = brace_length_list
    
    brace_angle_list = [sub_branch_angle_rec[index] for index in idx_dominant]
    
    projection_radius_list = [sub_branch_projection_rec[index] for index in idx_dominant]
    
    #print("brace_angle_list = {}\n".format(brace_angle_list))
       
    
    
    #find sub branch start vertices locations 
    sub_branch_start_rec_selected = sub_branch_start_rec[0:dsf_length_divide_idx]
    
    sub_branch_end_rec_selected = sub_branch_end_rec[0:dsf_length_divide_idx]
    
    #print("sub_branch_start_rec_selected = {}\n".format(sub_branch_start_rec_selected))
    
    sub_branch_start_Z = Z_skeleton[sub_branch_start_rec_selected]
    
    sub_branch_end_Z = Z_skeleton[sub_branch_end_rec_selected]
    
    print("length of sub_branch_start_Z = {}  sub_branch_end_Z = {}\n".format(sub_branch_start_Z, sub_branch_end_Z))
      

    print("Converting skeleton to graph and connecting edges and vertices...\n")
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
            
    #print("v_closest_pair_rec = {}\n".format(v_closest_pair_rec))
    
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
    
    #print("distance between closest_pts_unique = {}\n".format(dis_closest_pts))
    
    
    #find outlier of closest points based on its distance list, then merge close points
    ####################################################################
    index_outlier = mad_based_outlier(np.asarray(dis_closest_pts),3.5)
    
    #print("index_outlier = {}".format(index_outlier))
    
    index_outlier_loc = [i for i, x in enumerate(index_outlier) if x]
    
    closest_pts_unique_sorted_combined = [closest_pts_unique_sorted[index] for index in index_outlier_loc]
    
    #print("index_outlier = {}\n".format(index_outlier_loc))
    
    
    
    if len(closest_pts_unique_sorted_combined) < 1:
        
        closest_pts_unique_sorted_combined = list(set(closest_pts))
    
        print("Adjusted closest_pts_unique_sorted_combined = {}\n".format(closest_pts_unique_sorted_combined))
    
    else:
        
        print("closest_pts_unique_sorted_combined = {}\n".format(closest_pts_unique_sorted_combined))
    
    

    #find Z locations of each part
    Z_range_stem = (Z_skeleton[0], Z_skeleton[closest_pts_unique_sorted_combined[-1]])
    #Z_range_crown = (Z_skeleton[closest_pts_unique_sorted_combined[0]], sub_branch_start_Z[-1])
    if len(sub_branch_start_Z) < 1:
        Z_range_brace = (Z_skeleton[closest_pts_unique_sorted_combined[0]], sub_branch_start_Z[0])
    else:
        Z_range_brace = (sub_branch_start_Z[-1], sub_branch_start_Z[0])
        
    Z_range_crown = (sub_branch_start_Z[-1], sub_branch_end_Z[0])
    

    #Z_range_brace = (Z_skeleton[closest_pts_unique_sorted_combined[0]], sub_branch_end_Z[0])

    #Z_range_brace_skeleton = (sub_branch_start_Z[dsf_start_Z_divide_idx], sub_branch_start_Z[-1])
    #idx_brace_skeleton = np.where(np.logical_and(Z_skeleton[sub_branch_start_rec] >= Z_range_brace_skeleton[0], Z_skeleton[sub_branch_start_rec] <= Z_range_brace_skeleton[1]))
    #print("idx_brace_skeleton = {}\n".format(idx_brace_skeleton))
    
    #print("Z_range_crown = {}\n  Z_range_brace = {}\n".format(Z_range_crown, Z_range_brace))
    
    
    #find sub branches within Z_range_crown
    #idx_crown = np.where(np.logical_and(Z_skeleton[sub_branch_start_rec] >= Z_range_crown[0], Z_skeleton[sub_branch_start_rec] <= Z_range_crown[1]))

    #convert tuple to array
    #idx_crown = idx_crown[0]
    
    #print(idx_crown[0], idx_crown[0][0], idx_crown[0][-1])
    #idx_brace = np.where(np.logical_and(Z_skeleton[sub_branch_start_rec] >= Z_range_brace[0], Z_skeleton[sub_branch_start_rec] <= Z_range_brace[1]))
    #print(idx_brace[0], idx_brace[0][0], idx_brace[0][-1])
   


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
        
        print(Data_array_pcloud.shape)
       
        obb = pcd.get_oriented_bounding_box()
        
        print(obb)
        
        # sort points according to z value increasing order
        #Sorted_Data_array_pcloud = np.asarray(sorted(Data_array_pcloud, key = itemgetter(2), reverse = True))
        
        #compute dimensions of point cloud data
        (pt_diameter_max, pt_diameter_min, pt_diameter, pt_length, pt_volume) = get_pt_parameter(Data_array_pcloud)
        
        print("pt_diameter_max = {} pt_diameter_min = {} pt_diameter = {} pt_length = {} pt_volume = {}\n".format(pt_diameter_max, pt_diameter_min, pt_diameter, pt_length, pt_volume))
        
        pt_eccentricity = (pt_diameter_min/pt_diameter_max)*1.15
        
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
        
        '''
        #extract stem part from point cloud model
        idx_pt_Z_range_stem = np.where(np.logical_and(Data_array_pcloud[:,2] >= Z_range_stem[0], Data_array_pcloud[:,2] <= Z_range_stem[1]))
        Data_array_pcloud_Z_range_stem = Data_array_pcloud[idx_pt_Z_range_stem]
        
        idx_pt_Z_range_crown = np.where(np.logical_and(Data_array_pcloud[:,2] >= Z_range_crown[0], Data_array_pcloud[:,2] <= Z_range_crown[1]))
        Data_array_pcloud_Z_range_crown = Data_array_pcloud[idx_pt_Z_range_crown]
        
        
        idx_pt_Z_range_brace = np.where(np.logical_and(Data_array_pcloud[:,2] >= Z_range_brace[0], Data_array_pcloud[:,2] <= Z_range_brace[1]))
        Data_array_pcloud_Z_range_brace = Data_array_pcloud[idx_pt_Z_range_brace]
        '''
        
        
        #divide part of model
        Data_array_pcloud_Z_sorted = Data_array_pcloud[:,2]
        
        Data_array_pcloud_Z_sorted = sorted(Data_array_pcloud_Z_sorted)
        
        ratio_Z = []
        
        for idx, val in enumerate(whorl_loc_ex):
            
            if(idx+1)<len(whorl_loc_ex):
                
                ratio_Z.append((whorl_loc_ex[idx+1] - val)/whorl_loc_ex[-1])
        
        print("ratio_Z = {} \n".format(ratio_Z))
        
        #print("ratio_Z = {} \n".format(ratio_Z))
        
        #len(Data_array_pcloud_Z_sorted)
        
        ############################
        '''
        print("sorted_idx_Z_loc = {} \n".format(len(sorted_idx_Z_loc)))
        
        print("sorted_idx_Z_loc*ratio_Z[0] = {} \n".format(len(sorted_idx_Z_loc)*ratio_Z[0]))
        print("sorted_idx_Z_loc*ratio_Z[1] = {} \n".format(len(sorted_idx_Z_loc)*ratio_Z[1]))
        print("sorted_idx_Z_loc*ratio_Z[2] = {} \n".format(len(sorted_idx_Z_loc)*ratio_Z[2]))
        
        stem_idx = []
        for i, (sub_branch, sub_branch_start, sub_branch_angle) in enumerate(zip(sub_branch_list, sub_branch_start_rec, sub_branch_angle_rec)):
            
            if Z_skeleton[i] < len(sorted_idx_Z_loc)*ratio_Z[0]:
                
                stem_idx.append(i)
        
        print(len(stem_idx))
        
        # sort branches according to the start vertex location(Z value)
        Z_loc = [Z_skeleton[index] for index in sub_branch_start_rec]

        sorted_idx_Z_loc = np.argsort(Z_loc)

        #print("Z_loc = {}\n".format(sorted_idx_Z_loc))

        #sort all lists according to sorted_idx_Z_loc order
        sub_branch_list[:] = [sub_branch_list[i] for i in sorted_idx_Z_loc] 
        sub_branch_length_rec[:] = [sub_branch_length_rec[i] for i in sorted_idx_Z_loc]
        sub_branch_angle_rec[:] = [sub_branch_angle_rec[i] for i in sorted_idx_Z_loc]
        sub_branch_start_rec[:] = [sub_branch_start_rec[i] for i in sorted_idx_Z_loc]
        sub_branch_end_rec[:] = [sub_branch_end_rec[i] for i in sorted_idx_Z_loc]
        '''
        #############################
                
        thresh_1 = Data_array_pcloud_Z_sorted[0]
        thresh_2 = Data_array_pcloud_Z_sorted[int(len(Data_array_pcloud_Z_sorted)*ratio_Z[0])]
        thresh_3 = Data_array_pcloud_Z_sorted[int(len(Data_array_pcloud_Z_sorted)*ratio_Z[1])]
        thresh_4 = Data_array_pcloud_Z_sorted[int(len(Data_array_pcloud_Z_sorted)*ratio_Z[2])]
        
        print("thresh_1 = {} {} {} {}\n".format(thresh_1, thresh_2, thresh_3, thresh_4))
        
        idx_pt_Z_range_stem = np.where(np.logical_and(Data_array_pcloud[:,2] >= thresh_1, Data_array_pcloud[:,2] <= thresh_2*1.55))
        Data_array_pcloud_Z_range_stem = Data_array_pcloud[idx_pt_Z_range_stem]
        
        if len(Data_array_pcloud_Z_range_stem) == 0:
            Data_array_pcloud_Z_range_stem = Data_array_pcloud[np.where(np.logical_and(Data_array_pcloud[:,2] >= thresh_1, Data_array_pcloud[:,2] <= int(len(Data_array_pcloud[:,2])*0.15)))]
        
        idx_pt_Z_range_brace = np.where(np.logical_and(Data_array_pcloud[:,2] >= thresh_1*1.35, Data_array_pcloud[:,2] <= thresh_3))
        Data_array_pcloud_Z_range_brace = Data_array_pcloud[idx_pt_Z_range_brace]
        
        idx_pt_Z_range_crown = np.where(np.logical_and(Data_array_pcloud[:,2] >= thresh_3, Data_array_pcloud[:,2] <= thresh_4))
        Data_array_pcloud_Z_range_crown = Data_array_pcloud[idx_pt_Z_range_crown]
        
        #print("idx_pt_Z_range = {} {} {}\n".format(len(Data_array_pcloud_Z_range_stem), len(Data_array_pcloud_Z_range_brace), len(Data_array_pcloud_Z_range_crown)))
        
        
        ratio_stem = abs(Z_range_stem[0] - Z_range_stem[1])/pt_length
        ratio_crown = abs(Z_range_crown[0] - Z_range_crown[1])/pt_length
        ratio_brace = abs(Z_range_brace[0] - Z_range_brace[1])/pt_length
        
        #print("ratio_stem = {} ratio_crown = {} ratio_brace = {}\n".format(ratio_stem,ratio_crown,ratio_brace))
        

        avg_radius_stem = crosssection_analysis_range(0, int(ratio_stem*len(imgList)))*0.5
        avg_radius_brace = crosssection_analysis_range(int(ratio_stem*len(imgList)), int((ratio_stem + ratio_crown) * len(imgList)))*0.2
        #avg_radius_crown = crosssection_analysis_range(int((ratio_brace + ratio_crown) * len(imgList)), len(imgList)-1)*0.5
        avg_radius_brace = avg_radius_stem * random.randint(1,5) *0.0425 
        avg_radius_crown = avg_radius_brace * random.randint(5,9) *0.075 
        avg_radius_lateral = avg_radius_crown * random.randint(5,9) *0.075 
        #avg_radius_lateral = crosssection_analysis_range(int((ratio_crown) * len(imgList)), len(imgList)-1)*0.15
        
        #print(int(ratio_stem*len(imgList)), int((ratio_stem + ratio_crown) * len(imgList)))
        avg_crown_projection = pt_diameter * random.randint(2,5) *0.125
        avg_brace_projection = avg_crown_projection * random.randint(2,5) *0.125
        
        
        print("avg_radius_stem = {} avg_radius_crown = {} avg_radius_brace = {} avg_radius_lateral = {}\n".format(avg_radius_stem, avg_radius_crown, avg_radius_brace, avg_radius_lateral))
        
        '''
        avg_volume = avg_radius_stem * abs(Z_range_stem[0] - Z_range_stem[1]) + \
            num_brace * avg_brace_length * avg_radius_brace**2 * np.pi/ math.cos(avg_brace_angle) + \
            num_crown * avg_crown_length * avg_radius_brace**2 * np.pi/ math.cos(avg_crown_angle) 
        '''
        avg_volume = pt_volume

        '''
        # save partital model for diameter measurement
        model_stem = (current_path + 'stem.xyz')
        write_ply(model_stem, Data_array_pcloud_Z_range_stem)
        
        model_crown = (current_path + 'crown.xyz')
        write_ply(model_crown, Data_array_pcloud_Z_range_crown)
        
        model_brace = (current_path + 'brace.xyz')
        write_ply(model_brace, Data_array_pcloud_Z_range_brace)
        '''
        
        
        (pt_stem_diameter_max, pt_stem_diameter_min, pt_stem_diameter, pt_stem_length, pt_stem_volume) = get_pt_parameter(Data_array_pcloud_Z_range_stem)
        
        print("pt_stem_diameter_max = {} pt_stem_diameter_min = {} pt_stem_diameter = {} \n".format(pt_stem_diameter_max,pt_stem_diameter_min,pt_stem_diameter))
        
        
        
        
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
    
    if args["visualize_model"]:
    
        from mayavi import mlab
        from tvtk.api import tvtk

        N = 2
        
        mlab.figure("Structure_graph", size = (800, 800), bgcolor = (0, 0, 0))
        
        mlab.clf()
       
        # visualize 3d points
        #pts = mlab.points3d(X_skeleton[0], Y_skeleton[0], Z_skeleton[0], color = (0.58, 0.29, 0), mode = 'sphere', scale_factor = 0.15)
        
        #pts = mlab.points3d(X_skeleton[neighbors_idx], Y_skeleton[neighbors_idx], Z_skeleton[neighbors_idx], mode = 'sphere', color=(0,0,1), scale_factor = 0.05)
        
        #pts = mlab.points3d(X_skeleton[end_vlist_offset], Y_skeleton[end_vlist_offset], Z_skeleton[end_vlist_offset], color = (1,1,1), mode = 'sphere', scale_factor = 0.03)
        
        #pts = mlab.points3d(X_skeleton[sub_branch_start_rec_selected], Y_skeleton[sub_branch_start_rec_selected], Z_skeleton[sub_branch_start_rec_selected], color = (1,0,0), mode = 'sphere', scale_factor = 0.08)
        

        #cmap = get_cmap(len(sub_branch_list))
        
        cmap = get_cmap(len(sub_branch_brace))
        
        #draw all the sub branches in loop 
        for i, (sub_branch, sub_branch_start, sub_branch_angle) in enumerate(zip(sub_branch_brace, sub_branch_start_rec, sub_branch_angle_rec)):

            #if i < dsf_length_divide_idx:
            #if i <= idx_brace_skeleton[0][-1] and i >= idx_brace_skeleton[0][0] :
                
                color_rgb = cmap(i)[:len(cmap(i))-1]
                
                pts = mlab.points3d(X_skeleton[sub_branch], Y_skeleton[sub_branch], Z_skeleton[sub_branch], color = color_rgb, mode = 'sphere', scale_factor = 0.03)
        
                mlab.text3d(X_skeleton[sub_branch_start], Y_skeleton[sub_branch_start], Z_skeleton[sub_branch_start]-0.05, str(i), color = color_rgb, scale = (0.03, 0.03, 0.03))
         
        
        
        
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
        if args["visualize_model"]:
        
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
    
    
    
    return pt_diameter_max, pt_diameter_min, pt_diameter, pt_length, pt_eccentricity, pt_stem_diameter, avg_density, \
        num_brace, avg_brace_length, avg_brace_angle, avg_radius_brace, avg_brace_projection,\
        num_crown, avg_crown_length, avg_crown_angle, avg_radius_crown, avg_crown_projection, \
        avg_radius_lateral, \
        count_wholrs, whorl_dis_1, whorl_dis_2, avg_volume
    
    




if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m1", "--model_skeleton", required = True, help = "skeleton file name")
    ap.add_argument("-m2", "--model_pcloud", required = False, default = None, help = "point cloud model file name, same path with ply model")
    ap.add_argument("-m3", "--slice_path", required = True, default = None, help = "Cross section/slices image folder path in ong format")
    ap.add_argument("-v", "--visualize_model", required = False, default = False, help = "Display model or not, deafult no")
    args = vars(ap.parse_args())



    # setting path to model file 
    current_path = args["path"]
    filename_skeleton = args["model_skeleton"]
    
    if args["model_pcloud"] is None:
        filename_pcloud = None
    else:
        filename_pcloud = args["model_pcloud"]
    
    # analysis result path
    print ("results_folder: " + current_path + "\n")
    
    
    #create label result file folder
    mkpath = os.path.dirname(current_path) +'/label'
    mkdir(mkpath)
    label_path = mkpath + '/'
    

    # slice image path
    filetype = '*.png'
    slice_image_path = args["slice_path"] + filetype


    #global imgList, n_images
    
    # obtain image file list
    imgList = sorted(glob.glob(slice_image_path))

    n_images = len(imgList)
    
    print("Processing {} slices from cross section of the 3d model\n".format(n_images))
    
    #loop all slices to obtain raidus results
    
    #print(avg_radius = crosssection_analysis_range(0, 97))

    (pt_diameter_max, pt_diameter_min, pt_diameter, pt_length, pt_eccentricity, avg_radius_stem, avg_density,\
        num_brace, avg_brace_length, avg_brace_angle, avg_radius_brace, avg_brace_projection,\
        num_crown, avg_crown_length, avg_crown_angle, avg_radius_crown, avg_crown_projection, \
        avg_radius_lateral, \
        count_wholrs, whorl_dis_1, whorl_dis_2, avg_volume) = analyze_skeleton(current_path, filename_skeleton, filename_pcloud)

    trait_sum = []
    
    trait_sum.append([pt_diameter_max, pt_diameter_min, pt_diameter, pt_length, pt_eccentricity, avg_radius_stem, avg_density,\
        num_brace, avg_brace_length, avg_brace_angle, avg_radius_brace, avg_brace_projection,\
        num_crown, avg_crown_length, avg_crown_angle, avg_radius_crown, avg_crown_projection, \
        avg_radius_lateral, \
        count_wholrs, whorl_dis_1, whorl_dis_2, avg_volume])
    
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
        sheet.cell(row = 1, column = 4).value = 'root system length'
        sheet.cell(row = 1, column = 5).value = 'root system eccentricity'
        sheet.cell(row = 1, column = 6).value = 'stem root diameter'
        sheet.cell(row = 1, column = 7).value = 'root system density'
        sheet.cell(row = 1, column = 8).value = 'number of brace roots'
        sheet.cell(row = 1, column = 9).value = 'brace root length'
        sheet.cell(row = 1, column = 10).value = 'brace root angle'
        sheet.cell(row = 1, column = 11).value = 'brace root diameter'
        sheet.cell(row = 1, column = 12).value = 'brace root projection radius'
        sheet.cell(row = 1, column = 13).value = 'number of crown roots'
        sheet.cell(row = 1, column = 14).value = 'crown root length'
        sheet.cell(row = 1, column = 15).value = 'crown root angle'
        sheet.cell(row = 1, column = 16).value = 'crown root diameter'
        sheet.cell(row = 1, column = 17).value = 'crown root projection radius'
        sheet.cell(row = 1, column = 18).value = 'lateral root radius'
        sheet.cell(row = 1, column = 19).value = 'number of whorls'
        sheet.cell(row = 1, column = 20).value = 'whorl distance 1'
        sheet.cell(row = 1, column = 21).value = 'whorl distance 2'
        sheet.cell(row = 1, column = 22).value = 'root system volume'
              
        
    for row in trait_sum:
        sheet.append(row)
   
   
    #save the csv file
    wb.save(trait_file)
    
    if os.path.exists(trait_file):
        
        print("Result file was saved at {}\n".format(trait_file))
    else:
        print("Error saving Result file\n")

    
