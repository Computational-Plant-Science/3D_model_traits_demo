"""
Version: 1.5

Summary: Automatic 3D model cleaning using Statistical outlier removal 

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

    python3 model_process.py -i ~/example/test.ply -o ~/example/result/ 


argument:
    ("-i", "--input", dest="input", type=str, required=True, help="full path to 3D model file")
    ("-o", "--output_path", dest = "output_path", type = str, required = False, help = "result path")
    ("--nb_neighbors", required = False, type = int, default = 20, help = "nb_neighbors")
    ("--std_ratio", required = False, type = float, default = 5.0, help = "outlier remove ratio, small number = aggresive")
    ("--black_filter", required = False, type = int, default = 0, help = "Apply black points removal filter or not, 0 = not, 1 = Apply")
    ("--black_threshold", required = False, type = float, default = 0.2, help = "threshold for black points removal")


output:

    *_cleaned.ply: cleaned 3d model 

"""
#!/usr/bin/env python



# import the necessary packages
import numpy as np 
import argparse

import os
import sys
import open3d as o3d
import copy
import pathlib
from scipy.spatial.transform import Rotation as Rot
from sklearn.neighbors import KDTree

from scipy.spatial import cKDTree



import math



def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


#compute angle between two vectors(works for n-dimensional vector),
def dot_product_angle(v1,v2):

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        
        print("Zero magnitude vector!")
        
        return 0
        
    else:
        vector_dot_product = np.dot(v1,v2)
        
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        
        if np.degrees(arccos) > 90:
            
            angle = np.degrees(arccos) - 90
        else:
            
            angle = np.degrees(arccos)
    
    return angle
        


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


# get file information from the file path using python3
def get_file_info(file_full_path):
    
    p = pathlib.Path(file_full_path)

    filename = p.name

    basename = p.stem

    file_path = p.parent.absolute()

    file_path = os.path.join(file_path, '')

    return file_path, filename, basename



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

'''
# using the distance to the model center point to fliter all the point cloud points
def distance_filter(pcd):
    
    model_center = pcd.get_center()
    
    pcd_points = np.asarray(pcd.points)
    
    # set up distance threshold value
    radius = 1.5

    # Calculate distances to center, set new points
    distances = np.linalg.norm(pcd_points - model_center, axis = 1)
    
    index_mask = np.where(pcd_points[distances <= radius])[0]
    
    # Remove black points
    pcd_filtered = pcd.select_by_index(index_mask)
    
    #pcd_filtered.paint_uniform_color([0, 1, 0])
    
    return pcd_filtered



# Calculates point cloud density using K-nearest neighbors.
def calculate_density_knn(pcd, k):
    """Calculates point cloud density using K-nearest neighbors."""

    # Build a KDTree for efficient neighbor search
    kdtree = KDTree(np.asarray(pcd.points))
    

    # Calculate density for each point
    densities = []
    for point in np.asarray(pcd.points):
        distances, indices = kdtree.query([point], k=k+1)  # +1 to include the point itself
        radius = np.max(distances[0][1:])  # Exclude the point itself
        volume = 4/3 * np.pi * radius ** 3
        density = k / volume
        densities.append(density)
    
    return densities
'''


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








if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", dest="input", type=str, required=True, help="full path to 3D model file")
    ap.add_argument("-o", "--output_path", dest = "output_path", type = str, required = False, help = "result path")
    ap.add_argument("--nb_neighbors", required = False, type = int, default = 20, help = "nb_neighbors")
    ap.add_argument("--std_ratio", required = False, type = float, default = 5.0, help = "outlier remove ratio, small number = aggresive")
    ap.add_argument("--black_filter", required = False, type = int, default = 0, help = "Apply black points removal filter or not, 0 = not, 1 = Apply")
    ap.add_argument("--black_threshold", required = False, type = float, default = 0.2, help = "threshold for black points removal")
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
        
        # parameters
        nb_neighbors = args["nb_neighbors"]

        std_ratio = args["std_ratio"]
        
        black_filter = args["black_filter"]
    
        black_threshold = args["black_threshold"]

        # start pipeline
        ########################################################################################
        # model alignment 
        pcd_cleaned = format_converter(input_file)
        
        
        
        ####################################################################
        # write aligned 3d model as ply file format
        # get file information

        #Save model file as ascii format in ply
        result_filename = result_path + basename + '_cleaned.ply'

        #write out point cloud file
        o3d.io.write_point_cloud(result_filename, pcd_cleaned, write_ascii = True)

        # check saved file
        if os.path.exists(result_filename):
            print("Converted 3d model was saved at {0}\n".format(result_filename))

        else:
            print("Model file converter failed!\n")
            sys.exit(0)
        
        

    else:

        print("The input file is missing or not readable!\n")

        print("Exiting the program...")

        sys.exit(0)
