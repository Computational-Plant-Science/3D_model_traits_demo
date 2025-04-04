"""
Version: 1.5

Summary: Align 3D model to its Z axis and translate it to its center. 

Author: Suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 model_preprocess.py -i ~/example/test.ply -o ~/example/result/ --n_plane 5 --slicing_ratio 0.1 --adjustment 0
    

PARAMETERS:
    ("--nb_neighbors", required = False, type = int, default = 20, help = "nb_neighbors")
    ("--std_ratio", required = False, type = float, default = 5.0, help = "outlier remove ratio, small number = aggresive")
    ("--black_filter", required = False, type = int, default = 0, help = "Apply black points removal filter or not, 0 = not, 1 = Apply")
    ("--black_threshold", required = False, type = float, default = 0.2, help = "threshold for black points removal")
    ("--n_plane", dest = "n_plane", type = int, required = False, default = 5,  help = "Number of planes to segment the 3d model along Z direction")
    ( "--slicing_ratio", dest = "slicing_ratio", type = float, required = False, default = 0.10, help = "ratio of slicing the model from the bottom")
    ( "--adjustment", dest = "adjustment", type = float, required = False, default = 0, help = "model adjustment, 0: no adjustment, 1: rotate np.pi/2, -1: rotate -np.pi/2")

INPUT:
    
    3d model file in ply format

OUTPUT:

    *_aligned.ply: cleaned and aligned model with orientation along Z axis

"""
#!/usr/bin/env python



# import the necessary packages
import numpy as np
import argparse

import os
import sys
import open3d as o3d
import copy

#from scipy.spatial.transform import Rotation as Rot
#import math
import pathlib

from matplotlib import pyplot as plt
import glob

from scipy.spatial.transform import Rotation as R


# Find the rotation matrix that aligns vec1 to vec2
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions




#colormap mapping
def get_cmap(n, name = 'tab10'):
    """get the color mapping""" 
    #viridis, BrBG, hsv, copper, Spectral
    #Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    #RGB color; the keyword argument name must be a standard mpl colormap name
    return plt.cm.get_cmap(name,n)
    
    #return plt.colormaps.get_cmap(name,n)




# sort index according to the value in decenting order
def sort_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s


# get file information from the file path using python3
def get_file_info(file_full_path):
    
    p = pathlib.Path(file_full_path)

    filename = p.name

    basename = p.stem

    file_path = p.parent.absolute()

    file_path = os.path.join(file_path, '')

    return file_path, filename, basename

'''
# compute dimensions of point cloud 
def get_pt_sel_parameter(Data_array_pt, n_plane):
    
    ####################################################################
    
    # load skeleton coordinates and radius 
    Z_pt_sorted = np.sort(Data_array_pt[:,2])
    
    #Z_pt_sorted = Data_array_pt[:,2]
    
    pt_plane = []
    
    
    pt_plane_center = []
    
    pt_plane_diameter = []
    
    
    for idx, x in enumerate(range(n_plane)):
        
        ratio_s = idx/n_plane
        ratio_e = (idx+1)/n_plane
        
        print("ratio_s ratio_e {} {}\n".format(ratio_s, ratio_e))
        
        # index of end plane 
        idx_sel_e = int(len(Z_pt_sorted)*ratio_e) 
    
        Z_e = Z_pt_sorted[idx_sel_e]  if idx_sel_e < len(Data_array_pt) else (len(Data_array_pt) - 1)
        
        # inde xof start plane
        idx_sel_s = int(len(Z_pt_sorted)*ratio_s) 
    
        Z_s = Z_pt_sorted[idx_sel_s]

        # mask between the start and end plane
        Z_mask = (Data_array_pt[:,2] <= Z_e) & (Data_array_pt[:,2] >= Z_s) 
        
        Z_pt_sel = Data_array_pt[Z_mask]
        
        # get the diameter of the sliced model 
        (pt_diameter_max, pt_diameter_min, pt_diameter) = get_pt_parameter(Z_pt_sel)
        
        #print(Z_pt_sel.shape)
     
        
        # initilize the o3d object
        pcd_Z_mask = o3d.geometry.PointCloud()
    
        pcd_Z_mask.points = o3d.utility.Vector3dVector(Z_pt_sel)
        
        
        
        # get the model center postion
        model_center = pcd_Z_mask.get_center()

        pt_plane.append(pcd_Z_mask)
        
        pt_plane_center.append(model_center)
        
        pt_plane_diameter.append(pt_diameter)
        

    return pt_plane, pt_plane_center, pt_plane_diameter
'''


# slice array based on Z values
def get_pt_sel(Data_array_pt):
    
    ####################################################################
    
    # load points cloud Z values and sort it
    Z_pt_sorted = np.sort(Data_array_pt[:,2])
    
    #slicing_factor
    
    idx_sel = int(len(Z_pt_sorted)*slicing_ratio) 
    
    Z_mid = Z_pt_sorted[idx_sel]

    # mask
    Z_mask = (Data_array_pt[:,2] <= Z_mid) & (Data_array_pt[:,2] >= Z_pt_sorted[0]) 
    
    Z_pt_sel = Data_array_pt[Z_mask]
    
    
    return Z_pt_sel


# compute dimensions of point cloud 
def get_pt_parameter(Data_array_pt):
    
    
    ####################################################################
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(Data_array_pt)
    
    # get AxisAlignedBoundingBox
    aabb = pcd.get_axis_aligned_bounding_box()
    #aabb.color = (0, 1, 0)
    
    #Get the extent/length of the bounding box in x, y, and z dimension.
    aabb_extent = aabb.get_extent()
    
    aabb_extent_half = aabb.get_half_extent()
    
    # get the dimention of the points cloud in diameter based on bounding box
    pt_diameter_max = max(aabb_extent[0], aabb_extent[1])

    pt_diameter_min = min(aabb_extent_half[0], aabb_extent_half[1])

    
    pt_diameter = (pt_diameter_max + pt_diameter_min)*0.5
    
        
    return pt_diameter_max, pt_diameter_min, pt_diameter
    


# Removing black points from a point cloud by filtering color values of the points based on the intensity
def remove_black_points(pcd, black_threshold):
    
    # Access the point cloud colors
    colors = np.asarray(pcd.colors)

    # Define a threshold for black points
    #black_threshold = 0.2  # Adjust as needed

    # Create a mask for black points
    black_mask = np.all(colors <= black_threshold, axis=1)

    # Remove black points
    pcd = pcd.select_by_index(np.where(black_mask == False)[0])

    return pcd, black_mask




def statistical_outlier_removal(pcd, nb_neighbors, std_ratio):
    """
    Performs statistical outlier removal on a point cloud.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        nb_neighbors (int): Number of neighbors to consider for each point.
        std_ratio (float): Standard deviation ratio threshold.

    Returns:
        open3d.geometry.PointCloud: The filtered point cloud.
    """
    
    print("Statistical outlier removal\n")
    
    pcd_np = np.asarray(pcd.points)

    # Compute distances to neighbors
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    distances = []
    for i in range(len(pcd_np)):
        _, indices, _ = pcd_tree.search_knn_vector_3d(pcd_np[i], nb_neighbors + 1)
        distances.append(np.mean(np.linalg.norm(pcd_np[indices[1:]] - pcd_np[i], axis=1)))

    # Compute mean and standard deviation of distances
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    # Filter out outliers
    inliers = np.where(np.abs(distances - mean_dist) < std_ratio * std_dist)[0]
    filtered_pcd = pcd.select_by_index(inliers)

    return filtered_pcd





def display_inlier_outlier(cloud, ind, obb):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, obb])
    


def format_converter(model_file):
    

    # Pass xyz to Open3D.o3d.geometry.PointCloud 

    pcd = o3d.io.read_point_cloud(model_file)
    


    
    # Apply noise filter
    pcd_filtered = statistical_outlier_removal(pcd, nb_neighbors, std_ratio)
    
    #pcd_filtered.paint_uniform_color([0, 1, 0])

    # Visualize the filtered point cloud
    #o3d.visualization.draw_geometries([pcd, pcd_filtered])
    
    '''
    # using the distance to the model center point to fliter all the point cloud points
    #pcd_filtered = distance_filter(pcd)
    #o3d.visualization.draw_geometries([pcd, pcd_filtered])
    #o3d.visualization.draw_geometries([pcd_filtered])
    '''
    
    color_array = np.asarray(pcd_filtered.colors)
    
    #print(len(color_array))
    
    #print((color_array))

    # black points removal
   
    if black_filter == 0:

        pcd_sel = pcd_filtered
    else:
        # Remove black points
        (pcd_sel, black_mask) = remove_black_points(pcd_filtered, black_threshold)
        
    
    #o3d.visualization.draw_geometries([pcd])
    #o3d.visualization.draw_geometries([pcd_sel])
    
    #print("Showing outliers (red) and inliers (gray): ")
    #pcd_sel.paint_uniform_color([1, 0, 0])
    #o3d.visualization.draw_geometries([pcd, pcd_sel])


    # copy original point cloud for rotation
    pcd_cleaned = copy.deepcopy(pcd_sel)
    
    
    # get the model center postion
    model_center = pcd_cleaned.get_center()
    
    # geometry points are translated directly to the model_center position
    pcd_cleaned.translate(-1*(model_center))
    
    '''
    # From Open3D to numpy array
    #center_pts = np.asarray([model_center])
    
    # create Open3D format for points 
    #pcd_center = o3d.geometry.PointCloud()
    #pcd_center.points = o3d.utility.Vector3dVector(center_pts)

    #o3d.visualization.draw_geometries([pcd, pcd_center])
    
    
    #print(black_mask)

    # Statistical outlier removal
    #nb_neighbors, which specifies how many neighbors are taken into account in order to calculate the average distance for a given point.
    #std_ratio, which allows setting the threshold level based on the standard deviation of the average distances across the point cloud. 
    #The lower this number the more aggressive the filter will be.
    

    # visualize the oulier removal point cloud
    #print("Statistical outlier removal\n")
    cl, ind = pcd_cleaned.remove_statistical_outlier(nb_neighbors = 100, std_ratio = 0.001)
    
    #cl, ind = pcd_cleaned.remove_radius_outlier(nb_points=16, radius=0.05)
    #display_inlier_outlier(pcd_r, ind)

    #print("Statistical outlier removal\n")
    #cl, ind = pcd_r.remove_statistical_outlier(nb_neighbors = 40, std_ratio = 0.00001)
    #display_inlier_outlier(pcd_r, ind)
    '''
    

    return pcd_cleaned



    
# align  ply model with z axis
#def model_alignment(model_file, result_path, adjustment):
def model_alignment(pcd, adjustment):
    
    # Load a ply point cloud
    #pcd = o3d.io.read_point_cloud(model_file)
    
    #print(np.asarray(pcd.points))
    #o3d.visualization.draw_geometries([pcd])


    # copy original point cloud for rotation
    pcd_r = copy.deepcopy(pcd)
    
    # get the model center postion
    model_center = pcd_r.get_center()
    
    # geometry points are translated directly to the model_center position
    pcd_r.translate(-1*(model_center))
    
    '''
    # get convex hull of a point cloud is the smallest convex set that contains all points.
    hull, _ = pcd_r.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    
    # get AxisAlignedBoundingBox
    aabb = pcd_r.get_axis_aligned_bounding_box()
    
    # assign color for AxisAlignedBoundingBox
    aabb.color = (0, 1, 0)
    
    #Get the extent/length of the bounding box in x, y, and z dimension.
    aabb_extent = aabb.get_extent()
    aabb_extent_half = aabb.get_half_extent()
    '''
    
    # get OrientedBoundingBox
    obb = pcd_r.get_oriented_bounding_box()
    
    # assign color for OrientedBoundingBox
    obb.color = (0, 0, 1)
    
    # get the eight points that define the bounding box.
    pcd_coord = obb.get_box_points()
    
    #print(obb.get_box_points())
    
    #pcd_coord.color = (1, 0, 0)
    
    # From Open3D to numpy array
    np_points = np.asarray(pcd_coord)
    
    # create Open3D format for points 
    pcd_coord = o3d.geometry.PointCloud()
    pcd_coord.points = o3d.utility.Vector3dVector(np_points)
    
    
    
    '''
    # assign different colors for eight points in the bounding box.
    colors = []
    cmap = get_cmap(8)
    
    for idx in range(8):
    
        color_rgb = cmap(idx)[:len(cmap(idx))-1]
        colors.append(color_rgb)

    pcd_coord.colors = o3d.utility.Vector3dVector(colors)
    '''

    #o3d.visualization.draw_geometries([pcd_r, obb, pcd_coord])



    # check the length of the joint 3 vector in the bounding box to estimate the orientation of model
    list_dis = [np.linalg.norm(np_points[0] - np_points[1]), np.linalg.norm(np_points[0] - np_points[2]), np.linalg.norm(np_points[0] - np_points[3])]
    
    # sort the length values and return the index
    idx_sorted = sort_index(list_dis)
    
    #print(list_dis)
    
    #print(idx_sorted)
    '''
    # apply adjustment if alignment was not correct
    if adjustment == 1:
        print("Manual adjustment was applied!")
        idx_sorted[0] = 1
    '''


    # estimate the orientation 
    if idx_sorted[0] == 0:
        
        center_0 = np.mean(np_points[[0,2,3,5]], axis=0)
        center_1 = np.mean(np_points[[1,4,6,7]], axis=0)
        
    elif idx_sorted[0] == 1:
        
        center_0 = np.mean(np_points[[0,1,3,6]], axis=0)
        center_1 = np.mean(np_points[[2,4,5,7]], axis=0)
    
    else:
        
        center_0 = np.mean(np_points[[0,1,2,7]], axis=0)
        center_1 = np.mean(np_points[[3,4,5,6]], axis=0)
    
    '''
    # estimate the orientation of 3d model using sliced diameters
    print("Using {} planes to scan the model along Z axis...".format(n_plane))
    
    (pt_plane, pt_plane_center, pt_plane_diameter) = get_pt_sel_parameter(np_points, n_plane)
    
    print("pt_plane_diameter =  {} \n".format(pt_plane_diameter))
    '''
    
    # define unit vector
    v_x = [1,0,0]
    v_y = [0,1,0]
    v_z = [0,0,1]
    
    
    # define model orientation vector
    m_center_vector = [(center_0[0] - center_1[0]), (center_0[1] -center_1[1]), (center_0[2] - center_1[2])]
    
    
    #compute the rotation matrix that aligns unit vector Z to orientation vector
    R_matrix = rotation_matrix_from_vectors(m_center_vector, v_z)
    
    # rotate the model using rotation matrix to align with unit vector Z 
    pcd_r.rotate(R_matrix, center = (0,0,0))
    

    # check the botttom and top direction 
    pts_bottom = get_pt_sel(np.asarray(pcd_r.points))
    
    

    '''
    ############################################################
    pcd_Z_mask = o3d.geometry.PointCloud()
    
    pcd_Z_mask.points = o3d.utility.Vector3dVector(pts_bottom)
    
    Z_mask_ply = result_path + "Z_mask.ply"
    
    o3d.visualization.draw_geometries([pcd_Z_mask])
    
    o3d.io.write_point_cloud(Z_mask_ply, pcd_Z_mask)
    ############################################################
    '''
    
    
    (ptb_diameter_max, ptb_diameter_min, ptb_diameter) = get_pt_parameter(pts_bottom)
    
    (pt_diameter_max, pt_diameter_min, pt_diameter) = get_pt_parameter(np.asarray(pcd_r.points))
    
    #print(ptb_diameter_max, pt_diameter_max)
    
    

    
    
    # if model bottom and top need to be fliped  
    if ptb_diameter < pt_diameter*0.6:
        
        print("Model was aligned correctly with Z axis\n")
        
    else:
        
        print("Flip model along Z axis\n")
        
        v_z_reverse = [0, 0, -1]
    
        #compute the rotation matrix that aligns unit vector Z to orientation vector
        #R_matrix_flip = rotation_matrix_from_vectors(v_z_reverse, v_z)
        
        R_matrix_flip = pcd_r.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        
        # rotate the model using rotation matrix to align with unit vector Z 
        pcd_r.rotate(R_matrix_flip, center = (0,0,0))
    
    ######################################################################
    
    if adjustment == 0:
        
        print("No manual rotation needed...\n")
        
    else:
        
        # rotate along x for  np.pi/2 * adjustment value
        
        R_adjust = pcd_r.get_rotation_matrix_from_xyz((adjustment*np.pi/2, 0, 0))
        
        #R_adjust = pcd_r.get_rotation_matrix_from_xyz((0, adjustment*np.pi/2, 0))
        
        pcd_r.rotate(R_adjust, center = (0,0,0))
        
        print("Manual rotation {}*np.pi/2...\n".format(adjustment))
        
        
    
    
    # estimate the orientation of 3d model using sliced diameters

    #(pt_plane, pt_plane_center, pt_plane_diameter) = get_pt_sel_parameter(np.asarray(pcd_r.points), n_plane)
    
    #print("pt_plane_diameter =  {} \n".format(pt_plane_diameter))
    

    # return aligned model file
    return pcd_r


# check file save status
def check_file(result_filename):
    
    # check saved file
    if os.path.exists(result_filename):
        print("Converted 3d model was saved at {0}\n".format(result_filename))

    else:
        print("Model file save failed!\n")
        sys.exit(0)



    

if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", dest="input", type=str, required=True, help="full path to 3D model file")
    #ap.add_argument("-p", "--path", dest = "path", type = str, required = True, help = "path to 3D model file")
    #ap.add_argument("-ft", "--filetype", dest = "filetype", type = str, required = False, default = 'ply', help = "3D model file filetype, default *.ply")
    ap.add_argument("-o", "--output_path", dest = "output_path", type = str, required = False, help = "result path")
    ap.add_argument("--nb_neighbors", required = False, type = int, default = 20, help = "nb_neighbors")
    ap.add_argument("--std_ratio", required = False, type = float, default = 5.0, help = "outlier remove ratio, small number = aggresive")
    ap.add_argument("--black_filter", required = False, type = int, default = 0, help = "Apply black points removal filter or not, 0 = not, 1 = Apply")
    ap.add_argument("--black_threshold", required = False, type = float, default = 0.2, help = "threshold for black points removal")
    ap.add_argument("--n_plane", dest = "n_plane", type = int, required = False, default = 5,  help = "Number of planes to segment the 3d model along Z direction")
    ap.add_argument( "--slicing_ratio", dest = "slicing_ratio", type = float, required = False, default = 0.10, help = "ratio of slicing the model from the bottom")
    ap.add_argument( "--adjustment", dest = "adjustment", type = float, required = False, default = 0, help = "model adjustment, 0: no adjustment, 1: rotate np.pi/2, -1: rotate -np.pi/2")
    args = vars(ap.parse_args())

    
    # single input file processing
    ###############################################################################
    if os.path.isfile(args["input"]):

        input_file = args["input"]

        (file_path, filename, basename) = get_file_info(input_file)

        print("Compute {} model orientation and aligning models...\n".format(file_path, filename, basename))

        # result path
        result_path = args["output_path"] if args["output_path"] is not None else file_path

        result_path = os.path.join(result_path, '')

        # print out result path
        print("results_folder: {}\n".format(result_path))
        
        # parameters for cleaning
        nb_neighbors = args["nb_neighbors"]

        std_ratio = args["std_ratio"]
        
        black_filter = args["black_filter"]
    
        black_threshold = args["black_threshold"]
        
        
        # number of slices for cross-section
        n_plane = args['n_plane']
        
        slicing_ratio = args["slicing_ratio"]

        adjustment = args["adjustment"]

        # start pipeline
        ########################################################################################
        # model clean 
        pcd_cleaned = format_converter(input_file)
        
        # Save model file as ascii format in ply
        result_filename = result_path + basename + '_cleaned.ply'
        
        # write out point cloud file
        o3d.io.write_point_cloud(result_filename, pcd_cleaned, write_ascii = True)
        
        # check saved file
        check_file(result_filename)
        
        ####################################################################
        # model alignment 
        pcd_r = model_alignment(pcd_cleaned, adjustment)
        
        # write aligned 3d model as ply file format

        # Save model file as ascii format in ply
        result_filename = result_path + basename + '_aligned.ply'

        # write out point cloud file
        o3d.io.write_point_cloud(result_filename, pcd_r, write_ascii = True)

        # check saved file
        check_file(result_filename)


    else:

        print("The input file is missing or not readable!\n")

        print("Exiting the program...")

        sys.exit(0)
    
    
    

    '''
    # batch processing multiple files in a folder
    # batch processing mode 
    # python3 model_alignment.py p ~/example/ -ft ply -o ~/example/result/
    #########################################################################
    
   
    # path to model file 
    file_path = args["path"]
    
    ext = args["filetype"]
    
    files = file_path + '*.' + ext
    
    # number of slices for cross section
    n_plane = args["n_plane"]
    slicing_ratio = args["slicing_ratio"]
    adjustment = args["adjustment"]
    
    
    # obtain model file list
    fileList = sorted(glob.glob(files))

    
    
    # loop processing 
    #######################################################################################
    for input_file in fileList:
        
        if os.path.isfile(input_file):
            
            (file_path, filename, basename) = get_file_info(input_file)

            print("Compute {} model orientation and aligning models...\n".format(file_path, filename, basename))

            # result path
            result_path = args["output_path"] if args["output_path"] is not None else file_path

            result_path = os.path.join(result_path, '')

            # print out result path
            print("results_folder: {}\n".format(result_path))
            
            # number of slices for cross section
            n_plane = args['n_plane']
            
            slicing_ratio = args["slicing_ratio"]
            
            adjustment = args["adjustment"]

            # start pipeline
            ########################################################################################
            # model alignment 
            pcd_r = model_alignment(input_file, result_path, adjustment)
            
            
            
            ####################################################################
            # write aligned 3d model as ply file format
            # get file information

            #Save model file as ascii format in ply
            result_filename = result_path + basename + '_aligned.ply'

            #write out point cloud file
            o3d.io.write_point_cloud(result_filename, pcd_r, write_ascii = True)

            # check saved file
            if os.path.exists(result_filename):
                print("Converted 3d model was saved at {0}\n".format(filename))

            else:
                print("Model file converter failed!\n")
        
        
        else:

            print("The input file is missing or not readable!\n")

            print("Exiting the program...") 

            sys.exit(0)
    
    '''
        

