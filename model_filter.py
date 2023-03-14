"""
Version: 1.0

Summary: fliter 3d model based on radius size and ratio.

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 model_filter.py -p ~/example/ -m test.ply -fr 0.1 -fd 100


argument:
("-p", "--path", required=True,    help="path to *.ply model file")
("-m", "--model", required=True,    help="file name")


output:
*.xyz: xyz format file only has 3D coordinates of points 

"""
#!/usr/bin/env python



# import the necessary packages
import numpy as np 
import argparse

import os
import sys
import open3d as o3d
import copy

import math
import pathlib


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def format_converter(current_path, model_name):
    
    model_file = current_path + model_name
    
    if os.path.isfile(model_file):
        print("Filtering 3D point cloud model {}...\n".format(model_name))
    else:
        print("File not exist")
        sys.exit()
    
        
    abs_path = os.path.abspath(model_file)
    filename, file_extension = os.path.splitext(abs_path)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
     
    # Pass xyz to Open3D.o3d.geometry.PointCloud 
    pcd = o3d.io.read_point_cloud(model_file)
    

    #visualize the original point cloud
    #o3d.visualization.draw_geometries([pcd])
    

    # copy original point cloud for rotation
    pcd_r = copy.deepcopy(pcd)
    
    
    # get the model center postion
    model_center = pcd_r.get_center()
    
    # geometry points are translated directly to the model_center position
    pcd_r.translate(-1*(model_center))
    
    print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_pcd = pcd_r.voxel_down_sample(voxel_size=0.015)


    # Statistical oulier removal
    #nb_neighbors, which specifies how many neighbors are taken into account in order to calculate the average distance for a given point.
    #std_ratio, which allows setting the threshold level based on the standard deviation of the average distances across the point cloud. 
    #The lower this number the more aggressive the filter will be.
    
    
    # visualize the oulier removal point cloud
    print("Statistical oulier removal\n")
    #cl, ind = pcd_r.remove_statistical_outlier(nb_neighbors = filter_radius, std_ratio = filter_ratio)
    #display_inlier_outlier(pcd_r, ind)
    
    print("Radius oulier removal")
    cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.00005)
    #display_inlier_outlier(voxel_down_pcd, ind)
    ####################################################################
    
    #Save model file as ascii format in ply
    filename = current_path + base_name + '_filtered.xyz'
    
    #write out point cloud file
    o3d.io.write_point_cloud(filename, voxel_down_pcd, write_ascii = True)
    
 
    
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
    ap.add_argument("-m", "--model", required = False, help = "model file name")
    ap.add_argument("-fr", "--filter_ratio", required = False, type = float, default = 5, help = "filter ratio, The lower this number the more aggressive the filter will be")
    ap.add_argument("-fd", "--filter_radius", required = False, type = int, default = 100, help = "number of neighbors are to calculate the average distance for a given point")
    ap.add_argument("-t", "--test", required = False, default = False, help = "if using test setup")
    args = vars(ap.parse_args())


    # setting path to model file 
    current_path = args["path"]
    filename = args["model"]
   
    filter_radius = args["filter_radius"]
    filter_ratio = args["filter_ratio"]
    
    if args["model"] is None:
        
        filename = pathlib.PurePath(current_path).name + ".xyz"
        
        print("Default file name is {}".format(filename))
    
    else:
        
        filename = args["model"]
    
    file_path = current_path + filename
    


    format_converter(current_path, filename)

 
