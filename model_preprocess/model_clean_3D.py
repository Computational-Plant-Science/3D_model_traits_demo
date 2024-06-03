"""
Version: 1.5

Summary: alignment 3d model to Z axis and translate it to its center.

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 model_clean_3D.py -p ~/example/ -m test.ply -r 0.1


argument:
("-p", "--path", required=True,    help="path to *.ply model file")
("-m", "--model", required=True,    help="file name")


output:
*.xyz: xyz format file only has 3D coordinates of points 
*_aligned.ply: aligned model with only 3D coordinates of points 

"""
#!/usr/bin/env python



# import the necessary packages
import numpy as np 
import argparse

import os
import sys
import open3d as o3d
import copy

from scipy.spatial.transform import Rotation as Rot
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


def format_converter(current_path, model_name):
    
    model_file = current_path + model_name
    
    if os.path.isfile(model_file):
        print("Converting file format for 3D point cloud model {}...\n".format(model_name))
    else:
        print("File not exist")
        sys.exit()
    
        
    abs_path = os.path.abspath(model_file)
    filename, file_extension = os.path.splitext(abs_path)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
     
    # Pass xyz to Open3D.o3d.geometry.PointCloud 

    pcd = o3d.io.read_point_cloud(model_file)
    
    
    #filename = current_path + base_name + '_binary.ply'
    
    #o3d.io.write_point_cloud(filename, pcd)
    
    #pcd = o3d.io.read_point_cloud(filename)
    
    #print(np.asarray(pcd.points))
    
    #visualize the original point cloud
    #o3d.visualization.draw_geometries([pcd])
    
    Data_array = np.asarray(pcd.points)
    
    color_array = np.asarray(pcd.colors)
    
    #print(len(color_array))
    
    
    
    #color_array[:,2] = 0.24
    
    #pcd.colors = o3d.utility.Vector3dVector(color_array)
    
    #o3d.visualization.draw_geometries([pcd])
    
    #pcd.points = o3d.utility.Vector3dVector(points)

    # threshold data
    
    if len(color_array) == 0:
        
        pcd_sel = pcd
    else:
        pcd_sel = pcd.select_by_index(np.where(color_array[:, 2] > ratio)[0])
    
    #o3d.visualization.draw_geometries([pcd])
    #o3d.visualization.draw_geometries([pcd_sel])


    # copy original point cloud for rotation
    pcd_r = copy.deepcopy(pcd_sel)
    
    
    # get the model center postion
    model_center = pcd_r.get_center()
    
    # geometry points are translated directly to the model_center position
    pcd_r.translate(-1*(model_center))
    
    
    # Statistical oulier removal
    #nb_neighbors, which specifies how many neighbors are taken into account in order to calculate the average distance for a given point.
    #std_ratio, which allows setting the threshold level based on the standard deviation of the average distances across the point cloud. 
    #The lower this number the more aggressive the filter will be.
    

    # visualize the oulier removal point cloud
    print("Statistical oulier removal\n")
    cl, ind = pcd_r.remove_statistical_outlier(nb_neighbors = 100, std_ratio = 0.00001)
    #display_inlier_outlier(pcd_r, ind)
    
    

    
    #print("Statistical oulier removal\n")
    #cl, ind = pcd_r.remove_statistical_outlier(nb_neighbors = 40, std_ratio = 0.00001)
    #display_inlier_outlier(pcd_r, ind)
   
    
    ####################################################################
    
    #Save model file as ascii format in ply
    filename = current_path + base_name + '_cleaned.ply'
    
    for i in range(5):
        #write out point cloud file
        o3d.io.write_point_cloud(filename, pcd_r, write_ascii = True)
    
 
    
    # check saved file
    if os.path.exists(filename):
        print("Converted 3d model was saved at {0}\n".format(filename))
        return True
    else:
        return False
        print("Model file converter failed!\n")
        sys.exit(0)
    

if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m", "--model", required = True, help = "model file name")
    ap.add_argument("-r", "--ratio", required = False, type = float, default = 0.1, help = "outlier remove ratio")
    ap.add_argument("-t", "--test", required = False, default = False, help = "if using test setup")
    args = vars(ap.parse_args())


    # setting path to model file 
    current_path = args["path"]
    filename = args["model"]
    ratio = args["ratio"]
    
    file_path = current_path + filename
    
    #rotation_angle = args["angle"]

    #print ("results_folder: " + current_path)

    format_converter(current_path, filename)

 
