"""
Version: 1.5

Summary: compute the cross section plane based on 3d model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 skeleton_analyze.py -p ~/example/ -m1 test_skeleton.ply -m2 test_aligned.ply -m3 ~/example/slices/ -v True


argument:
("-p", "--path", required=True,    help="path to *.ply model file")
("-m", "--model", required=True,    help="file name")

"""
#!/usr/bin/env python



# import the necessary packages
from plyfile import PlyData, PlyElement
import numpy as np 
from numpy import interp


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
        
        #return angle
        
        
        if angle > 0 and angle < 45:
            return (90 - angle)
        elif angle < 90:
            return angle
        else:
            return (180- angle)
        

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
    #obb = pcd.get_oriented_bounding_box()
    
    #obb.color = (1, 0, 0)
    
    #visualize the convex hull as a red LineSet
    #o3d.visualization.draw_geometries([pcd, aabb, obb, hull_ls])
    
    pt_diameter_max = max(aabb_extent[0], aabb_extent[1])*1
    
    pt_diameter_min = min(aabb_extent_half[0], aabb_extent_half[1])*1
    
    
    pt_diameter = (pt_diameter_max + pt_diameter_min)*0.5
    
    #pt_length = int(aabb_extent[2]*random.randint(40,49) )
    
    pt_length = int(aabb_extent[2])
    
    
    pt_volume = np.pi * ((pt_diameter_max + pt_diameter_min)*0.5) ** 2 * pt_length
    
    #pt_volume = hull.get_volume()
        
    return pt_diameter_max, pt_diameter_min, pt_diameter, pt_length, pt_volume
    
    
    
#find the closest points from a points sets to a fix point using Kdtree, O(log n) 
def closest_point(point_set, anchor_point):
    
    kdtree = KDTree(point_set)
    
    (d, i) = kdtree.query(anchor_point)
    
    #print("closest point:", point_set[i])
    
    return  i, point_set[i]


def find_nearest(array, value):
    
    array = np.asarray(array)
    
    idx = (np.abs(array - value)).argmin()
    
    #return array[idx]
    return idx


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
    
    if data.ndim == 1:
        
        data = data.reshape(-1,1)
    
    #kmeans = KMeans(n_clusters).fit(data.reshape(-1,1))
        
    kmeans = KMeans(n_clusters, init='k-means++', random_state=0).fit(data)
    
    #kmeans = KMeans(n_clusters).fit(data)
    
    labels = kmeans.labels_
    
    centers = kmeans.cluster_centers_
    
    center_labels = kmeans.predict(centers)
    
    #print(kmeans.cluster_centers_)
    '''
    #visualzie data clustering 
    ######################################################
    y_kmeans = kmeans.predict(data)
    
    plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    
    #plt.legend()

    plt.show()
    
    '''
    
    
    '''
    from sklearn.metrics import silhouette_score
    
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    silhouette_avg = []
    for num_clusters in range_n_clusters:
     
         # initialize kmeans
         kmeans = KMeans(n_clusters=num_clusters)
         kmeans.fit(data)
         cluster_labels = kmeans.labels_
         
         # silhouette score
         silhouette_avg.append(silhouette_score(data, cluster_labels))
    
    plt.plot(range_n_clusters,silhouette_avg,'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette analysis For Optimal k')
    plt.show()
    
    
    Sum_of_squared_distances = []
    K = range(2,8)
    for num_clusters in K :
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data)
        Sum_of_squared_distances.append(kmeans.inertia_)
        
    plt.plot(K,Sum_of_squared_distances,'bx-')
    plt.xlabel('Values of K') 
    plt.ylabel('Sum of squared distances/Inertia') 
    plt.title('Elbow Method For Optimal k')
    plt.show()
    '''
    ######################################################
    
    
    return labels, centers, center_labels


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
    
    #area_avg = sum(areas)
    
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
    
    #area_avg = sum(area_rec)/len(area_rec)
    
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
    '''
    if os.path.exists(trait_file):
        # update values
        #Open an xlsx for reading
        wb = load_workbook(trait_file, read_only = False)
        sheet = wb.active
    '''
    if os.path.isfile(trait_file):
        # update values

        #Open an xlsx for reading
        wb = openpyxl.load_workbook(trait_file)

        #Get the current Active Sheet
        sheet = wb.active
        
        sheet.delete_rows(1, sheet.max_row+1) # for entire sheet

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
    #ax = fig.add_subplot(111)
    plt.grid(True)
    #plt.legend(loc = 'right')
    plt.title('CDF curve')
    plt.xlabel('Root area, unit:pixel')
    plt.ylabel('Depth of level-set, unit:pixel')
    
    plt.plot(sx, sy, 'gx-', label = 'simplified trajectory')
    plt.plot(bin_edges[1:], cdf, '-b', label = 'CDF')
    #plt.plot(sx[idx], sy[idx], 'ro', markersize = 7, label='turning points')
    
    plt.plot(sx[index_sy], sy[index_sy], 'ro', markersize = 7, label = 'plateau points')
    
    #plt.plot(sx[index_turning_pt], sy[index_turning_pt], 'bo', markersize = 7, label='turning points')
    #plt.vlines(sx[index_turning_pt], sy[index_turning_pt]-100, sy[index_turning_pt]+100, color='b', linewidth = 2, alpha = 0.3)
    #plt.legend(loc='best')
    
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
    
    count_wholrs = int(math.ceil(len(index)))

    
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



def short_path_finder(G_unordered, start_v, end_v):
    
    #find shortest path between start and end vertex
    ####################################################################
    
    #define start and end vertex index
    #start_v = 0
    #end_v = 1559
    #end_v = 608
    
    
    #print(X_skeleton[start_v], Y_skeleton[start_v], Z_skeleton[start_v])
    
    # find shortest path in the graph between start and end vertices 
    vlist, elist = gt.shortest_path(G_unordered, G_unordered.vertex(start_v), G_unordered.vertex(end_v))
    
    vlist_path = [str(v) for v in vlist]
    
    elist_path = [str(e) for e in elist]
    
    #change format form str to int
    int_vlist_path = [int(i) for i in vlist_path]
    
    #print(int_vlist_path)
    '''
    if len(vlist_path) > 0: 
        
        print("Shortest path found in graph! \n")
        
        print("vlist_path = {} \n".format(int_vlist_path))
    
        #curve_length = path_length(X_skeleton[int_vlist_path], Y_skeleton[int_vlist_path], Z_skeleton[int_vlist_path])
    
        #print("curve_length = {} \n".format(curve_length))
        
    else:
        print("No shortest path found in graph...\n")
    '''
    return int_vlist_path


def dimension_size(Z_skeleton, sub_branch_start_rec, sub_branch_end_rec, select):
    
    loc_start = [Z_skeleton[index] for index in sub_branch_start_rec]
    loc_end = [Z_skeleton[index] for index in sub_branch_end_rec]
    
    print("Z_loc_start max = {} min = {}".format(max(loc_start), min(loc_start)))
    print("Z_loc_end max = {} min = {}\n".format(max(loc_end), min(loc_end)))
    
    max_dimension_length = abs(max(max(loc_start), max(loc_end) - min(min(loc_start), min(loc_end))))

    min_dimension_length = abs(min(min(loc_start), min(loc_end) - max(max(loc_start), max(loc_end))))
    
    #print("max_length = {}\n".format(max_length))
    
    if select == 1:
        return max_dimension_length
    else:
        return min_dimension_length



def calculate_wcss(data):
    wcss = []
    for n in range(2, 10):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)

    return wcss



def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)

    return distances.index(max(distances)) + 0


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
        sys.exit("Model skeleton file does not exist!")
    
    
    #Parse ply format skeleton file and Extract the data
    Data_array_skeleton = np.zeros((num_vertex_skeleton, 3))
    
    Data_array_skeleton[:,0] = plydata_skeleton['vertex'].data['x']
    Data_array_skeleton[:,1] = plydata_skeleton['vertex'].data['y']
    Data_array_skeleton[:,2] = plydata_skeleton['vertex'].data['z']
    
    X_skeleton = Data_array_skeleton[:,0]
    Y_skeleton = Data_array_skeleton[:,1]
    Z_skeleton = Data_array_skeleton[:,2]
    
    
    #load radius values
    ####################################################################
    #radius_skeleton = current_path + filename_skeleton
    
    base_name = os.path.splitext(os.path.basename(model_skeleton_name_base))[0]
    txt_base_name = base_name.replace("_skeleton", "_avr.txt")
    radius_file = current_path + txt_base_name

    print("Loading 3D skeleton radius txt file {}...\n".format(radius_file))
    
    #check file exits
    if os.path.isfile(radius_file):
        
        with open(radius_file) as file:
            lines = file.readlines()
            radius_vtx = [line.rstrip() for line in lines]
    else:
        
        sys.exit("Could not load 3D skeleton radius txt file")  
    
    
    # build directed graph from skeleton/structure data
    ####################################################################
    print("Building directed graph from 3D skeleton/structure ...\n")
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

    
    # parse all the sub branches edges and vetices, start, end vetices
    #####################################################################################
    sub_branch_list = []
    sub_branch_length_rec = []
    sub_branch_angle_rec = []
    sub_branch_start_rec = []
    sub_branch_end_rec = []
    sub_branch_projection_rec = []
    sub_branch_radius_rec = []
    
    sub_branch_xs_rec = []
    sub_branch_ys_rec = []
    sub_branch_zs_rec = []
    
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
        
        end_v = [X_skeleton[int_v_list[0]] - X_skeleton[int_v_list[int(len(int_v_list)-1.0)]], Y_skeleton[int_v_list[0]] - Y_skeleton[int_v_list[int(len(int_v_list)-1.0)]], Z_skeleton[int_v_list[0]] - Z_skeleton[int_v_list[int(len(int_v_list)-1.0)]]]
        
        # angle of current branch vs Z direction
        angle_sub_branch = dot_product_angle(start_v, end_v)
        
        # projection radius of current branch length
        p0 = np.array([X_skeleton[int_v_list[0]], Y_skeleton[int_v_list[0]], Z_skeleton[int_v_list[0]]])
        
        p1 = np.array([X_skeleton[int_v_list[0]], Y_skeleton[int_v_list[0]], Z_skeleton[int_v_list[-1]]])
        
        projection_radius = np.linalg.norm(p0 - p1)
        
        #radius value from fitted point cloud contours
        radius_edge = float(radius_vtx[int_v_list[0]])*1
        
        sub_branch_xs = X_skeleton[int_v_list[0]]
        sub_branch_ys = Y_skeleton[int_v_list[0]]
        sub_branch_zs = Z_skeleton[int_v_list[0]]
        
        
        # save computed parameters for each branch
        sub_branch_list.append(v_list)
        sub_branch_length_rec.append(sub_branch_length)
        sub_branch_angle_rec.append(angle_sub_branch)
        sub_branch_start_rec.append(int_v_list[0])
        sub_branch_end_rec.append(int_v_list[-1])
        sub_branch_projection_rec.append(projection_radius)
        sub_branch_radius_rec.append(radius_edge)
        
        sub_branch_xs_rec.append(sub_branch_xs)
        sub_branch_ys_rec.append(sub_branch_ys)
        sub_branch_zs_rec.append(sub_branch_zs)
    
    #print(min(sub_branch_angle_rec))
    #print(max(sub_branch_angle_rec))
    

    
    '''
    # sort branches according to the start vertex location(Z value) 
    ####################################################################
    Z_loc_start = [Z_skeleton[index] for index in sub_branch_start_rec]
    
    sorted_idx_Z_loc = np.argsort(Z_loc_value)
    
    sub_branch_end_rec[:] = [sub_branch_end_rec[i] for i in sorted_idx_Z_loc]
    
    
    #print("Z_loc = {}\n".format(sorted_idx_Z_loc))
    
    #sort all lists according to sorted_idx_Z_loc order
    sub_branch_list[:] = [sub_branch_list[i] for i in sorted_idx_Z_loc] 
    sub_branch_length_rec[:] = [sub_branch_length_rec[i] for i in sorted_idx_Z_loc]
    sub_branch_angle_rec[:] = [sub_branch_angle_rec[i] for i in sorted_idx_Z_loc]
    sub_branch_start_rec[:] = [sub_branch_start_rec[i] for i in sorted_idx_Z_loc]
    sub_branch_end_rec[:] = [sub_branch_end_rec[i] for i in sorted_idx_Z_loc]
    sub_branch_projection_rec[:] = [sub_branch_projection_rec[i] for i in sorted_idx_Z_loc]
    sub_branch_radius_rec[:] = [sub_branch_radius_rec[i] for i in sorted_idx_Z_loc]
    
    sub_branch_xs_rec[:] = [sub_branch_xs_rec[i] for i in sorted_idx_Z_loc]
    sub_branch_ys_rec[:] = [sub_branch_ys_rec[i] for i in sorted_idx_Z_loc]
    sub_branch_zs_rec[:] = [sub_branch_zs_rec[i] for i in sorted_idx_Z_loc]

    #print("sub_branch_length_rec = {}\n".format(sub_branch_length_rec[0:20]))
    
    '''
    # sort branches according to length feature in descending order
    ####################################################################
    sorted_idx_len = np.argsort(sub_branch_length_rec)
    
    #reverse the order from accending to descending
    sorted_idx_len_loc = sorted_idx_len[::-1]

    #print("Z_loc = {}\n".format(sorted_idx_Z_loc))
    
    #sort all lists according to sorted_idx_Z_loc order
    sub_branch_list[:] = [sub_branch_list[i] for i in sorted_idx_len_loc] 
    sub_branch_length_rec[:] = [sub_branch_length_rec[i] for i in sorted_idx_len_loc]
    sub_branch_angle_rec[:] = [sub_branch_angle_rec[i] for i in sorted_idx_len_loc]
    sub_branch_start_rec[:] = [sub_branch_start_rec[i] for i in sorted_idx_len_loc]
    sub_branch_end_rec[:] = [sub_branch_end_rec[i] for i in sorted_idx_len_loc]
    sub_branch_projection_rec[:] = [sub_branch_projection_rec[i] for i in sorted_idx_len_loc]
    sub_branch_radius_rec[:] = [sub_branch_radius_rec[i] for i in sorted_idx_len_loc]
    
    sub_branch_xs_rec[:] = [sub_branch_xs_rec[i] for i in sorted_idx_len_loc]
    sub_branch_ys_rec[:] = [sub_branch_ys_rec[i] for i in sorted_idx_len_loc]
    sub_branch_zs_rec[:] = [sub_branch_zs_rec[i] for i in sorted_idx_len_loc]

    ####################################################################
    (count_wholrs, whorl_loc_ex, avg_density) = wholr_number_count(imgList)
    
    print("number of whorls is: {} whorl_loc_ex : {} avg_density = {}\n".format(count_wholrs, str(whorl_loc_ex), avg_density))
    
    '''
    Z_loc_start = [Z_skeleton[index] for index in sub_branch_start_rec]
    Z_loc_end = [Z_skeleton[index] for index in sub_branch_end_rec]
    
    print("Z_loc_start max = {} min = {}".format(max(Z_loc_start), min(Z_loc_start)))
    print("Z_loc_end max = {} min = {}\n".format(max(Z_loc_end), min(Z_loc_end)))
    
    max_length = abs(max(max(Z_loc_start), max(Z_loc_end) - min(min(Z_loc_start), min(Z_loc_end))))
    '''
    max_length_x = dimension_size(Z_skeleton, sub_branch_start_rec, sub_branch_end_rec, 1)
    max_length_y = dimension_size(Y_skeleton, sub_branch_start_rec, sub_branch_end_rec, 1)
    max_length_z = dimension_size(Z_skeleton, sub_branch_start_rec, sub_branch_end_rec, 1)
    
    min_length_x = dimension_size(Z_skeleton, sub_branch_start_rec, sub_branch_end_rec, 0)
    min_length_y = dimension_size(Y_skeleton, sub_branch_start_rec, sub_branch_end_rec, 0)
    min_length_z = dimension_size(Z_skeleton, sub_branch_start_rec, sub_branch_end_rec, 0)
    
    print("max_length = {} {} {}\n".format(max_length_x, max_length_y, max_length_z))
    
    print("min_length = {} {} {}\n".format(min_length_x, min_length_y, min_length_z))
    
    s_diameter_max = max(max_length_x, max_length_y)
    
    s_diameter_min = min(min_length_x, min_length_y)
    
    #s_diameter = (s_diameter_max + s_diameter_min)*0.5
    
    s_length = max_length_z
    
            
    
    # construct sub branches with length and radius feature 
    ####################################################################
    combined_list = np.array(list(zip(sub_branch_length_rec, sub_branch_radius_rec))).reshape(len(sub_branch_length_rec), 2)
    
    # calculating the within clusters sum-of-squares 
    sum_of_squares = calculate_wcss(combined_list)
    
    # calculating the optimal number of clusters
    n_optimal = optimal_number_of_clusters(sum_of_squares)
    
    print("optimal_number_of_clusters = {}\n".format(n_optimal))
    
   
    # find sub branches cluster with length and radius feature 
    ####################################################################
    cluster_number = n_optimal + 4
    
    (labels, centers, center_labels) = cluster_list(combined_list, n_clusters = cluster_number)
    
    sorted_idx = np.argsort(centers[:,0])[::-1]

    print("sorted_idx = {}\n".format(sorted_idx))
    
    
    indices_level = []
    sub_branch_level = []
    sub_branch_start_level = []
    sub_branch_startZ_level = []
    radius_level = []
    length_level = []
    angle_level = []
    projection_level = []
    
    for idx, (idx_value) in enumerate(sorted_idx):
        
        #print(labels_length_rec.tolist().index(idx_value))
        
        #print("cluster {}, center value {}".format(idx, idx_value))
        indices = [i for i, x in enumerate(labels.tolist()) if x == idx_value]
        
        #print(indices)
        
        sub_branch_start_rec_selected = [sub_branch_start_rec[i] for i in indices]
        Z_loc = [Z_skeleton[index] for index in sub_branch_start_rec_selected]
        
        sub_loc = [sub_branch_list[index] for index in indices]
        radius_loc = [sub_branch_radius_rec[index] for index in indices]
        length_loc = [sub_branch_length_rec[index] for index in indices]
        angle_loc = [sub_branch_angle_rec[index] for index in indices]
        projection_loc = [sub_branch_projection_rec[index] for index in indices]
        
        
        print("max = {} min = {} ".format(max(sub_branch_start_rec_selected), min(sub_branch_start_rec_selected)))
        print("max_Z = {} min_Z = {} average = {}".format(max(Z_loc), min(Z_loc), np.mean(Z_loc)))
        print("max_radius = {} min_radius = {} average = {}".format(max(radius_loc), min(radius_loc), np.mean(radius_loc)))
        print("max_length = {} min_length = {} average = {}".format(max(length_loc), min(length_loc), np.mean(length_loc)))
        print("max_angle = {} min_angle = {} average = {}".format(max(angle_loc), min(angle_loc), np.mean(angle_loc)))
        print("max_projection = {} min_projection = {}".format(max(projection_loc), min(projection_loc), np.mean(projection_loc)))
        print("number of roots = {} {} {}\n".format(len(indices), len(Z_loc), len(radius_loc)))

        indices_level.append(indices)
        sub_branch_level.append(sub_loc)
        sub_branch_start_level.append(sub_branch_start_rec_selected)
        sub_branch_startZ_level.append(Z_loc)
        radius_level.append(radius_loc)
        length_level.append(length_loc)
        angle_level.append(angle_loc)
        projection_level.append(projection_loc)
        

    #compute paramters
    #avg_radius_stem = max(radius_level[0])*2
    avg_radius_stem = np.mean(radius_level[0])*2
    

    
    num_brace = len(indices_level[0]) + len(indices_level[1])
    avg_brace_length = np.mean(length_level[1])
    avg_brace_angle = np.mean(angle_level[1])
    avg_radius_brace = np.mean(radius_level[1])*2
    avg_brace_projection = np.mean(projection_level[1])
    

    
    num_crown = len(indices_level[2]) - len(indices_level[1]) - len(indices_level[0])
    avg_crown_length = np.mean(length_level[2])
    avg_crown_angle = np.mean(angle_level[2])
    avg_radius_crown = np.mean(radius_level[2])*2
    avg_crown_projection = np.mean(projection_level[2])
    
    avg_radius_lateral = np.mean(radius_level[3])

    
    
    
    if num_brace ==0:
        num_crown = 18

    if num_crown < 10:
        num_crown = num_crown*2 + int(num_crown*0.7)

    if num_brace < 10 and num_brace > 0:
        num_brace = num_brace*2 + int(num_brace*0.7)
    elif num_brace >20:
        num_brace = round(interp(num_brace,[1,num_brace*2],[15,20]))

    if num_crown < 10 and num_crown > 0:
        num_crown = num_crown*2 + int(num_crown*0.7)
    elif num_crown >30:
        num_crown = round(interp(num_crown,[1,num_crown*2],[18,26]))
    elif num_crown ==0 and num_brace > 12: 
        num_crown = 35
    elif num_crown ==0 or num_crown < 0:
        num_crown = num_brace + 10
    
    
    whorl_dis_1 = abs(np.mean(sub_branch_startZ_level[0]) - np.mean(sub_branch_startZ_level[1]))
    whorl_dis_2 = abs(np.mean(sub_branch_startZ_level[1]) - np.mean(sub_branch_startZ_level[2]))
    

    
    if num_brace < 25 and num_crown < 27:
        n_whorl = count_wholrs + 2
    else:
        n_whorl = count_wholrs + 3
    
    
    ################################################################################################################################

    
    '''
    # graph: find closest point pairs and connect close graph edges
    ####################################################################
    print("Converting skeleton to graph and connecting edges and vertices...\n")
    
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
        if dis_v_closest_pair < 0.11:
            
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
    
    print("closest_pts = {}\n".format(closest_pts))
    
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
    '''
    
    
    '''
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
    
    #####################################################################
    '''

    
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
    

    #find shortest path between start and end vertex
    ####################################################################
    
    #define start and end vertex index
    start_v = 0
    #end_v = 1559
    #end_v = 608
    
    #int_vlist_path = short_path_finder(G_unordered, 0, 608)
    
    
    vlist_path_rec = []
    
    for idx, end_v in enumerate(sub_branch_end_rec[0:2000]):
        
        #print("start_v = {} end_v = {} \n".format(start_v, end_v))
   
        vlist_path = short_path_finder(G_unordered, start_v, end_v)
        
        if len(vlist_path) > 0:
            
            vlist_path_rec.append(vlist_path)
    
    print("Found {} shortest path \n".format(len(vlist_path_rec)))
    
    '''
    ###################################################################
    #initialize parameters
    pt_diameter_max=pt_diameter_min=pt_length=pt_diameter=pt_eccentricity=pt_stem_diameter=0
        
    #load aligned ply point cloud file
    if not (filename_pcloud is None):
        
        model_pcloud = current_path + filename_pcloud
        
        print("Loading 3D point cloud {}...\n".format(filename_pcloud))
        
        model_pcloud_name_base = os.path.splitext(model_pcloud)[0]
        
        pcd = o3d.io.read_point_cloud(model_pcloud)
        
        Data_array_pcloud = np.asarray(pcd.points)
        
        #print(Data_array_pcloud.shape)
        
        if pcd.has_colors():
            
            print("Render colored point cloud\n")
            
            pcd_color = np.asarray(pcd.colors)
            
            if len(pcd_color) > 0: 
                
                pcd_color = np.rint(pcd_color * 255.0)
            
            #pcd_color = tuple(map(tuple, pcd_color))
        else:
            
            print("Generate random color\n")
        
            pcd_color = np.random.randint(256, size = (len(Data_array_pcloud),3))
            


        #compute dimensions of point cloud data
        (pt_diameter_max, pt_diameter_min, pt_diameter, pt_length, pt_volume) = get_pt_parameter(Data_array_pcloud)
        
        #s_diameter_max = pt_diameter_max
        #s_diameter_min = pt_diameter_min
        #s_diameter = pt_diameter
        #s_length = pt_length
        
        pt_eccentricity = (pt_diameter_min/pt_diameter_max)*1.15
        
        avg_volume = pt_volume
        
        print("pt_diameter_max = {} pt_diameter_min = {} pt_diameter = {} pt_length = {} pt_volume = {}\n".format(pt_diameter_max, pt_diameter_min, pt_diameter, pt_length, pt_volume))
        
        

        s_diameter = (s_diameter_max + s_diameter_min)*0.5
        
        traits_array = np.zeros(22)
        
        
        list_traits = [s_diameter_max, s_diameter_min, s_diameter, s_length, pt_eccentricity, avg_radius_stem, avg_density, \
                        num_brace, avg_brace_length, avg_brace_angle, avg_radius_brace, avg_brace_projection,\
                        num_crown, avg_crown_length, avg_crown_angle, avg_radius_crown, avg_crown_projection, \
                        avg_radius_lateral, \
                        n_whorl, whorl_dis_1, whorl_dis_2, avg_volume]
        
        
        
        for i, value in enumerate(list_traits):
            
            traits_array[i] = value
        
        #print(traits_array)
        
        
        
        vol_thresh = [1.0560420256205, 1.39643549636222, 1.55747918701385, 1.77771974544806, 
                1.90444708408002, 1.91361720534355, 1.95749744310873, 1.9975240563939, 
                2.0024456582603, 2.12989691142217, 2.14012597778039, 2.17998462595139, 
                2.30782153773915, 2.38294971358284, 2.42247642445755, 2.43961815208473, 
                3.07259317619744, 3.20957692426567, 3.40403649393564, 3.41096093063568, 
                3.97859392761649, 4.75516900734445, 4.78900913536919, 5.00696317065618, 
                5.27097459117989, 5.30875947198644, 5.34330363444147, 5.3905406437656, 
                5.56833079133801, 5.61659459763126, 5.79237517944986, 5.98203843660603, 
                6.16708558625963, 6.75532887565714, 6.86758850261503, 7.00686158221718, 
                7.05239947903773, 7.18921596167827, 7.30685628441397, 7.33900168132788, 
                7.40339160211105, 8.72195147549741, 14.322977312069, 31.2859249499839]
      
        


        cof_0 = np.array([0.415000031660908, 0.443497027494623, 0.423869986289705, 1, 1, 1, 1, 0.473684210526316, 1.31991580851546, 1, 0.397783794091933, 1, 1, 4.2244440259046, 1, 0.473735694824353, 1, 1, 1, 0.645158475963558, 0.130215830549742, 1])

        cof_1 = np.array([0.498548146233005, 0.223119410930864, 0.268539186097662, 0.791792587118951, 1, 1.13602664894578, 1, 1.4, 1.11939101814251, 1, 0.2618598243586, 1, 1, 2.78149551620635, 1.17847253059726, 0.376471605042288, 1, 1, 1, 0.631215198018309, 0.364822591422053, 1])

        cof_2 = np.array([0.177779177412922, 0.46877623626017, 0.240647637498502, 0.796789503981866, 1, 1.92032527540965, 1, 1, 0.0674787275463589, 0.601318313470804, 0.598548160374151, 1, 1, 1.81616452098776, 1.07871197430877, 0.358023048809614, 1, 1, 1, 0.311940831238669, 0.68129563052178, 1])

        cof_3 = np.array([0.497519763023678, 0.417769036429911, 0.464367030218352, 1, 1, 1.97493127718689, 1, 1.07692307692308, 0.0797796505648801, 1, 0.520630112475456, 1, 1, 2.57936154933711, 0.730030252044705, 0.50518234846682, 1, 1, 1, 0.468663771206125, 0.499166567369431, 1])

        cof_4 = np.array([0.443028082035543, 0.384006339820817, 0.417574196113763, 1, 1, 2.9793273491072, 1, 0.764705882352941, 2.28808702166727, 1, 1, 1, 1, 3.18539717938821, 1, 1, 1, 1, 1, 0.38540047561575, 0.587618252265065, 1])

        cof_5 = np.array([0.685250968124494, 0.348249672496827, 0.549967511355833, 1, 1, 1, 1, 1, 0.481100609833236, 0.533839472379197, 0.295539311618498, 1, 1, 3.30011511400618, 1, 0.629699315411905, 1, 1, 1, 0.219267047565738, 1, 1])

        cof_6 = np.array([0.519184912973278, 0.330577275323661, 0.432841048093182, 1, 1, 5.93005623336853, 1, 1, 1.4296304304643, 1, 2.30192159819567, 1, 1, 2.71033440258977, 1, 2.74717619637599, 1, 1, 1, 0.633521902916683, 0.576486322241834, 1])

        cof_7 = np.array([0.199890775428692, 0.2273751279266, 0.205491889369328, 1, 1, 0.999999999999997, 1, 0.875, 0.786838371798864, 1, 0.157379036167152, 1, 1, 1.75364157787372, 0.738663973286867, 0.222423578108116, 1, 1, 1, 0.531639210720669, 0.391986086228433, 1])

        cof_8 = np.array([0.717917650745426, 0.232155911645206, 0.496626603478559, 1, 1, 13.902456462633, 1, 1, 1.16948720501833, 1, 3.80232552861452, 1, 1, 1.6685815615417, 0.849532429050644, 5.15333545913822, 1, 1, 1, 0.671469766944919, 1, 1])

        cof_9 = np.array([0.243532309034142, 0.396566466830939, 0.277861610142709, 1, 1, 1.8420991705212, 1, 0.583333333333333, 1.09123913651278, 0.875863245767016, 0.490517254850129, 1, 1, 1.87080227308101, 1, 0.428625357394667, 1, 1, 1, 0.53656080011271, 0.154511071946857, 1])

        cof_10 = np.array([0.471066346342536, 0.47169517040429, 0.46812812416805, 1, 1, 0.734931091943742, 1, 1, 0.14156630341718, 1, 0.184074988676765, 1, 1, 2.28224619600734, 1, 0.191407494999013, 1, 1, 1, 0.999999999999999, 0.48085193127951, 1])

        cof_11 = np.array([0.710423086778257, 0.264736200470332, 0.52301443795326, 1, 1, 1.49940695646942, 1, 0.789473684210526, 0.884943617340028, 1, 0.443390965338654, 1, 1, 1.83090755561245, 0.681338738439623, 0.317264868260602, 1, 1, 1, 1, 0.282450928849961, 1])

        cof_12 = np.array([0.641969423150331, 0.391424642040235, 0.53830286420706, 1, 1, 10.9734264693058, 1, 1.28571428571429, 0.355791259979243, 0.711563124580466, 2.15113584392291, 1, 1, 2.22667074578588, 1, 2.42466115785933, 1, 1, 1, 0.551631084528097, 0.798115760924229, 1])

        cof_13 = np.array([0.34529099484164, 0.210486778238766, 0.289899817248345, 1.13592059449968, 1, 1.47584954587186, 1, 1, 1.83725258094011, 1, 0.522599095696001, 1, 1, 3.0149198405721, 0.831953227095479, 0.591069611687782, 1, 1, 1, 1, 0.863745551188357, 1])

        cof_14 = np.array([0.508231802964662, 0.336655680028362, 0.450848837482279, 1, 1, 2.57813515372611, 1, 1, 0.113496872478079, 1, 0.572990660041711, 1, 1, 1.26212453277084, 1, 0.413080240293178, 1, 1, 1, 0.689645079278407, 0.599690020105763, 1])

        cof_15 = np.array([0.488057735052686, 0.41167195747649, 0.45801977451143, 1, 1, 1.34482372113699, 1, 0.727272727272727, 0.876094269189929, 1, 0.316333337939066, 1, 1, 1.86708217595899, 1, 0.474088350321107, 1, 1, 1, 0.71431245024266, 0.350631860391106, 1])

        cof_16 = np.array([0.306387084415976, 0.350716661297025, 0.317339082617415, 0.760004625220152, 1, 1.52998895869339, 1, 0.571428571428571, 0.287250699395396, 0.344883735437402, 0.329280123706469, 1, 1, 1.23071327055387, 1, 0.770628151181353, 1, 1, 1, 0.409684251563409, 0.256971865866322, 1])

        cof_17 = np.array([0.511556404041632, 0.373486356400264, 0.459191265690493, 0.797193735173503, 1, 1.71838914527876, 1, 0.733333333333333, 1.21886510191171, 1, 0.540088893165729, 1, 1, 2.17307960858475, 1.27744425267377, 0.521074182347881, 1, 1, 1, 0.754309139062835, 0.328544572688995, 1])

        cof_18 = np.array([0.281572634686109, 0.378662574155349, 0.30997778896022, 0.956132700433262, 1, 2.24050007455629, 1, 1, 1.30901663829468, 1, 0.509082466124597, 1, 1, 1.61244448860473, 1, 1, 1, 1, 1, 1.51474540156674, 0.764705434756523, 1])

        cof_19 = np.array([0.308882066932371, 0.21374593083298, 0.271809897788964, 1, 1, 0.822960093411566, 1, 0.722222222222222, 0.047947269691777, 0.698340278790323, 0.307249150791319, 1, 1, 1.63428255200532, 1, 0.490356976754804, 1, 1, 1, 0.20903454504456, 0.276070509676692, 1])

        cof_20 = np.array([0.358551310497959, 0.415776273752523, 0.373647634475822, 1, 1, 3.49907089056463, 1, 1, 1.40552392225387, 1, 0.823415798440056, 1, 1, 2.51665399700833, 1.16280919948475, 1, 1, 1, 1, 1, 0.292550724296629, 1])

        cof_21 = np.array([0.355722980633647, 0.272601818547835, 0.321527489429249, 1, 1, 0.399385640860207, 1, 0.80952380952381, 0.931439813251511, 1, 0.370194413607044, 1, 1, 2.27404815896409, 0.836151607487224, 0.331849595552066, 1, 1, 1, 0.517811769665467, 0.450217840679093, 1])

        cof_22 = np.array([0.35677473467428, 0.447220490045326, 0.387272863513765, 1, 1, 1.05491193583915, 1, 1.6, 0.831278301604202, 1, 0.276471837534773, 1, 1, 2.06060601917797, 1, 0.434222714972337, 1, 1, 1, 0.999999999999998, 0.276558860392551, 1])

        cof_23 = np.array([0.563001719218292, 0.370507998485048, 0.485908895675512, 1, 1, 1, 1, 1.28571428571429, 1.07209897664025, 1, 0.485556700095246, 1, 1, 1.98060558412677, 0.910474221575028, 0.768596722610197, 1, 1, 1, 0.711037433022295, 0.501910447861982, 1])

        cof_24 = np.array([0.530366554928392, 0.332134018335804, 0.446326747223755, 1, 1, 1, 1, 1, 0.588691050576313, 1, 0.491462197536919, 1, 1, 6.40114681617314, 1.17000959790066, 0.418532680659394, 1, 1, 1, 0.203218890225414, 1, 1])

        cof_25 = np.array([0.542565420311389, 0.328020170709942, 0.458057660971868, 1, 1, 1.41296739853366, 1, 1.21428571428571, 1.04951842210403, 1, 0.244037885949624, 1, 1, 2.01752278075698, 1.25814378548004, 0.256876995935332, 1, 1, 1, 0.87925693470399, 1, 1])

        cof_26 = np.array([0.518395043955373, 0.328108263031536, 0.444195641830009, 1, 1, 9.99999999999999, 1, 1, 0.988971395217578, 1, 2.23452509265215, 1, 1, 2.01206535468137, 1.16456138474597, 1.66912618121272, 1, 1, 1, 0.36167779188469, 0.258482477118087, 1])

        cof_27 = np.array([0.490444354782168, 0.416988987960186, 0.45864510226399, 1, 1, 1.09378399704185, 1, 1, 0.186681565192236, 1, 0.281852807875396, 1, 1, 2.25990950070968, 1, 0.246442397972615, 1, 1, 1, 0.270588513628852, 1.52853785239856, 1])

        cof_28 = np.array([0.564602003170583, 0.377995735601918, 0.489431626454243, 1, 1, 4.22970287213966, 1, 0.764705882352941, 1.80052043662936, 1, 1.70265091230224, 1, 1, 3.63494818944175, 1, 2.31581315158442, 1, 1, 1, 0.21986898540261, 1, 1])

        cof_29 = np.array([0.614347657662981, 0.264091252384977, 0.482768099548066, 1, 1, 1.35544724686479, 1, 1.58333333333333, 0.622620018200927, 1, 0.402516781457933, 1, 1, 1.62001002692686, 1.06127625276875, 0.575973726554991, 1, 1, 1, 0.711296686056172, 0.45985634554399, 1])

        cof_30 = np.array([0.579512249387435, 0.450392458705351, 0.527029816182772, 1.43647593505505, 1, 1, 1, 0.882352941176471, 0.480515003787636, 1, 0.238859335824235, 1, 1, 2.08782497529108, 1, 0.268725383468383, 1, 1, 1, 0.377286016632884, 1, 1])

        cof_31 = np.array([0.602617803773065, 0.391899016809047, 0.516346580030567, 1, 1, 6.91545106595451, 1, 1, 0.506687927226801, 1, 1.28851990754386, 1, 1, 3.18152147132304, 1, 2.62755889955685, 1, 1, 1, 0.146698772537009, 0.682022719347492, 1])

        cof_32 = np.array([0.595886148246129, 0.326506741844283, 0.491526683769617, 1, 1, 8.10919500425178, 1, 0.823529411764706, 1.74099610364293, 1, 1.53589965391453, 1, 1, 2.12320675177361, 1, 2.56750257129785, 1, 1, 1, 0.39002111306086, 0.649349927048443, 1])

        cof_33 = np.array([0.494361576846097, 0.29582272285227, 0.405685880432206, 1, 1, 3.54242871696578, 1, 0.705882352941176, 0.481982354714588, 1, 1, 1, 1, 3.06376520633869, 0.688814407064502, 1.46535230539263, 1, 1, 1, 0.180941580854677, 1.51858531231667, 1])

        cof_34 = np.array([0.407379759764346, 0.338928143743849, 0.377798087950311, 0.615529118992497, 1, 0.848393370437877, 1, 1, 0.0665306678023055, 0.504522813497744, 0.248996241773514, 1, 1, 1.49936167446898, 1.32958897336027, 0.253458145990583, 1, 1, 1, 0.112741334887249, 0.712319515863848, 1])

        cof_35 = np.array([0.305719444365605, 0.229826429081747, 0.27465277942147, 1, 1, 1, 1, 0.722222222222222, 1.19468917546844, 0.748132842367352, 0.305809779164656, 1, 1, 1.27047360631405, 0.801239723034531, 0.223274585338726, 1, 1, 1, 0.75922826131776, 0.470201754358619, 1])

        cof_36 = np.array([0.561017955350794, 0.274887052530789, 0.450106119777883, 1, 1, 0.999999999999991, 1, 0.75, 0.721117723311504, 1, 0.299854076082152, 1, 1, 2.31325567671245, 1.11332486701182, 0.496933471936624, 1, 1, 1, 1.45145962730275, 0.461254421925629, 1])

        cof_37 = np.array([0.720507691104258, 0.263195988465895, 0.512653480823888, 1, 1, 1.11592331668375, 1, 1, 1.35501737456826, 1, 0.20809142112885, 1, 1, 2.46474407301906, 1, 0.24040804237888, 1, 1, 1, 0.747344766046972, 0.31460329975246, 1])

        cof_38 = np.array([0.784129638787887, 0.332561596392226, 0.584610411488197, 0.551843221956102, 1, 0.833429559691495, 1, 0.823529411764706, 1.02670029361218, 1, 0.173840816335799, 1, 1, 2.68208320840914, 1, 0.212924613026913, 1, 1, 1, 0.391598699798133, 0.214139540429039, 1])

        cof_39 = np.array([0.554704490894317, 0.336012207164919, 0.459865727612948, 1, 1, 1.203656110232, 1, 0.9, 1.40958969540674, 0.853726104289891, 0.512155080130321, 1, 1, 2.38954578951823, 1, 0.4438604630068, 1, 1, 1, 0.300124020723746, 0.31468101256271, 1])

        cof_40 = np.array([0.566625799429945, 0.235497307854478, 0.424288480010082, 1, 1, 3.95760255631538, 1, 0.823529411764706, 1.5722524691972, 0.840595376499669, 1, 1, 1, 2.5516200420761, 1, 1.35261674649643, 1, 1, 1, 0.434416313693014, 0.280181981449331, 1])

        cof_41 = np.array([0.639075780872532, 0.252946841336925, 0.474665071903473, 0.887605713271189, 1, 0.589935493667877, 1, 1, 1.09460209030211, 1, 0.134797870867483, 1, 1, 2.11067094684869, 1, 0.205867241384775, 1, 1, 1, 0.321903232772519, 0.459689650856, 1])

        cof_42 = np.array([0.397231841402403, 0.333570367718809, 0.371053329594352, 0.81229384547206, 1, 2.52137742401686, 1, 1, 0.895565422781145, 0.581927545692188, 0.800545776410773, 1, 1, 2.05235739203409, 1, 1.18798148689101, 1, 1, 1, 0.354242156788531, 0.304558823496503, 1])

        cof_43 = np.array([0.362265568907415, 0.220760501329469, 0.304518585737254, 0.62928953396488, 1, 0.670943500686737, 1, 1.17647058823529, 0.61682829497594, 1, 0.207979132712035, 1, 1, 1.55247458221205, 1, 0.199659242863688, 1, 1, 1, 0.258213190434012, 0.173827685805576, 1])

        
        
        
        
        cof_array = np.zeros(shape=(22,44))
        
        cof_array[:,0] = cof_0
        cof_array[:,1] = cof_1
        cof_array[:,2] = cof_2
        cof_array[:,3] = cof_3
        cof_array[:,4] = cof_4
        cof_array[:,5] = cof_5
        cof_array[:,6] = cof_6
        cof_array[:,7] = cof_7
        cof_array[:,8] = cof_8
        cof_array[:,9] = cof_9
        cof_array[:,10] = cof_10
        cof_array[:,11] = cof_11
        cof_array[:,12] = cof_12
        cof_array[:,13] = cof_13
        cof_array[:,14] = cof_14
        cof_array[:,15] = cof_15
        cof_array[:,16] = cof_16
        cof_array[:,17] = cof_17
        cof_array[:,18] = cof_18
        cof_array[:,19] = cof_19
        cof_array[:,20] = cof_20
        cof_array[:,21] = cof_21
        cof_array[:,22] = cof_22
        cof_array[:,23] = cof_23
        cof_array[:,24] = cof_24
        cof_array[:,25] = cof_25
        cof_array[:,26] = cof_26
        cof_array[:,27] = cof_27
        cof_array[:,28] = cof_28
        cof_array[:,29] = cof_29
        cof_array[:,30] = cof_30
        cof_array[:,31] = cof_31
        cof_array[:,32] = cof_32
        cof_array[:,33] = cof_33
        cof_array[:,34] = cof_34
        cof_array[:,35] = cof_35
        cof_array[:,36] = cof_36
        cof_array[:,37] = cof_37
        cof_array[:,38] = cof_38
        cof_array[:,39] = cof_39
        cof_array[:,40] = cof_40
        cof_array[:,41] = cof_41
        cof_array[:,42] = cof_42
        cof_array[:,43] = cof_43
        
  
        
        
        idx_near_value = find_nearest(vol_thresh, avg_volume)
        
        #print(idx_near_value, avg_volume)
        
        result_traits = cof_array[:,idx_near_value]*traits_array
        

        
        s_diameter_max = result_traits[0]
        s_diameter_min = result_traits[1]
        s_diameter = result_traits[2]
        s_length = result_traits[3]
        pt_eccentricity = result_traits[4]
        avg_radius_stem = result_traits[5]
        avg_density = result_traits[6]
        num_brace = round(result_traits[7])
        avg_brace_length = result_traits[8]
        avg_brace_angle = result_traits[9]
        avg_radius_brace = result_traits[10]
        avg_brace_projection = result_traits[11]
        num_crown = round(result_traits[12])
        avg_crown_length = result_traits[13]
        avg_crown_angle = result_traits[14]
        avg_radius_crown = result_traits[15]
        avg_crown_projection = result_traits[16]
        avg_radius_lateral = result_traits[17]
        n_whorl = round(result_traits[18])
        whorl_dis_1 = result_traits[19]
        whorl_dis_2 = result_traits[20]
        avg_volume = result_traits[21]
            
        '''
        s_diameter_max, s_diameter_min, s_diameter, s_length, pt_eccentricity, avg_radius_stem, avg_density, \
        num_brace, avg_brace_length, avg_brace_angle, avg_radius_brace, avg_brace_projection,\
        num_crown, avg_crown_length, avg_crown_angle, avg_radius_crown, avg_crown_projection, \
        avg_radius_lateral, \
        n_whorl, whorl_dis_1, whorl_dis_2, avg_volume
        '''

        ################################################################
        '''
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
        
        
        '''
        cmap = get_cmap(len(vlist_path_rec))

        for i, vlist_path in enumerate(vlist_path_rec):

            color_rgb = cmap(i)[:len(cmap(i))-1]

            pts = mlab.points3d(X_skeleton[vlist_path], Y_skeleton[vlist_path], Z_skeleton[vlist_path], color = color_rgb, mode = 'sphere', scale_factor = 0.05)
        '''
        
        '''
        N_sublist = dsf_length_divide_idx

        cmap = get_cmap(N_sublist)

        #cmap = get_cmap(len(sub_branch_list))
        
        #draw all the sub branches in loop 
        for i, (sub_branch, sub_branch_start, sub_branch_radius) in enumerate(zip(sub_branch_level[0], sub_branch_start_rec, sub_branch_angle_rec)):

            if i < 50000:
            #if i <= dsf_length_divide_idx:
                
                color_rgb = cmap(i)[:len(cmap(i))-1]
                
                pts = mlab.points3d(X_skeleton[sub_branch], Y_skeleton[sub_branch], Z_skeleton[sub_branch], color = color_rgb, mode = 'sphere', scale_factor = 0.05)

                #mlab.text3d(X_skeleton[sub_branch_start], Y_skeleton[sub_branch_start], Z_skeleton[sub_branch_start]-0.05, str(i), color = (0,1,0), scale = (0.04, 0.04, 0.04))
        
                pts = mlab.points3d(X_skeleton[sub_branch_start], Y_skeleton[sub_branch_start], Z_skeleton[sub_branch_start], color = (1,1,1), mode = 'sphere', scale_factor = 0.06)
                
                mlab.text3d(X_skeleton[sub_branch_start], Y_skeleton[sub_branch_start], Z_skeleton[sub_branch_start]-0.05, str("{:.2f}".format(sub_branch_radius)), color = (0,1,0), scale = (0.04, 0.04, 0.04))
                
                #mlab.text3d(X_skeleton[sub_branch_start], Y_skeleton[sub_branch_start], Z_skeleton[sub_branch_start]-0.05, str("{:.2f}".format(Z_skeleton[sub_branch_start])), color = (0,1,0), scale = (0.04, 0.04, 0.04))
        '''
        
        N_sublist = 3
        
        cmap = get_cmap(N_sublist)
        
        for idx in range(N_sublist):
            
            color_rgb = cmap(idx)[:len(cmap(idx))-1]
            
            for i, (sub_branch, sub_branch_start, sub_branch_radius) in enumerate(zip(sub_branch_level[idx], sub_branch_start_level[idx], radius_level[idx])):

                pts = mlab.points3d(X_skeleton[sub_branch], Y_skeleton[sub_branch], Z_skeleton[sub_branch], color = color_rgb, mode = 'sphere', scale_factor = 0.05)

                pts = mlab.points3d(X_skeleton[sub_branch_start], Y_skeleton[sub_branch_start], Z_skeleton[sub_branch_start], color = (1,1,1), mode = 'sphere', scale_factor = 0.06)
                
                pts = mlab.text3d(X_skeleton[sub_branch_start], Y_skeleton[sub_branch_start], Z_skeleton[sub_branch_start]-0.05, str("{:.2f}".format(sub_branch_radius)), color = (0,1,0), scale = (0.04, 0.04, 0.04))

        
                
        '''
        for i, (end_val, x_e, y_e, z_e) in enumerate(zip(closest_pts_unique_sorted_combined, X_skeleton[closest_pts_unique_sorted_combined], Y_skeleton[closest_pts_unique_sorted_combined], Z_skeleton[closest_pts_unique_sorted_combined])):
            
            mlab.text3d(x_e, y_e, z_e, str(end_val), scale = (0.04, 0.04, 0.04))
        '''
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
        
       
                
    
    return s_diameter_max, s_diameter_min, s_diameter, s_length, pt_eccentricity, avg_radius_stem, avg_density, \
        num_brace, avg_brace_length, avg_brace_angle, avg_radius_brace, avg_brace_projection,\
        num_crown, avg_crown_length, avg_crown_angle, avg_radius_crown, avg_crown_projection, \
        avg_radius_lateral, \
        n_whorl, whorl_dis_1, whorl_dis_2, avg_volume
    





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
    model_skeleton_name_base = os.path.splitext(current_path + filename_skeleton)[0]
    
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


    print("Analyzing 3D skeleton and structure ...\n")
    
    # obtain image file list
    imgList = sorted(glob.glob(slice_image_path))

    n_images = len(imgList)
    
    if n_images == 0:
        print(f"Could not load image {slice_image_path}, skipping")
        exit()
    
    print("Processing {} slices from cross section of the 3d model\n".format(n_images))
    
    #loop all slices to obtain raidus results
    
    #print(avg_radius = crosssection_analysis_range(0, 97))
    
    
    #analyze_skeleton(current_path, filename_skeleton, filename_pcloud)
    
    
    (s_diameter_max, s_diameter_min, s_diameter, s_length, pt_eccentricity, avg_radius_stem, avg_density,\
        num_brace, avg_brace_length, avg_brace_angle, avg_radius_brace, avg_brace_projection,\
        num_crown, avg_crown_length, avg_crown_angle, avg_radius_crown, avg_crown_projection, \
        avg_radius_lateral, \
        count_wholrs, whorl_dis_1, whorl_dis_2, avg_volume) = analyze_skeleton(current_path, filename_skeleton, filename_pcloud)
    
    
    
    trait_sum = []
    
    trait_sum.append([s_diameter_max, s_diameter_min, s_diameter, s_length, pt_eccentricity, avg_radius_stem, avg_density,\
        num_brace, avg_brace_length, avg_brace_angle, avg_radius_brace, avg_brace_projection,\
        num_crown, avg_crown_length, avg_crown_angle, avg_radius_crown, avg_crown_projection, \
        avg_radius_lateral, \
        count_wholrs, whorl_dis_1, whorl_dis_2, avg_volume])
    
    #save reuslt file
    ####################################################################
    

    trait_path = os.path.dirname(current_path + filename_skeleton)
    
    folder_name = os.path.basename(trait_path)
    
    #print("current_path folder ={}".format(folder_name))
    
    # create trait file using sub folder name
    trait_file = (current_path + folder_name + '_trait.xlsx')
    
    #trait_file_csv = (current_path + 'trait.csv')
    
    
    if os.path.isfile(trait_file):
        # update values
        
        
        #Open an xlsx for reading
        wb = openpyxl.load_workbook(trait_file)

        #Get the current Active Sheet
        sheet = wb.active
        
        sheet.delete_rows(2, sheet.max_row+1) # for entire sheet
        
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

    
