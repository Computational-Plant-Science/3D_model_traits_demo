"""
Version: 1.0

Summary: fliter 3d model based on radius size and ratio.

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

    python3 model_filter.py -p ~/example/ -fr 0.1 -fd 100


argument:
    ("-p", "--path", required=True,    help="path to *.ply model file")
    ("-m", "--model", required=True,    help="file name")
    ("-fr", "--filter_ratio", required = False, type = float, default = 5, help = "filter ratio, The lower this number the more aggressive the filter will be")
    ("-fd", "--filter_radius", required = False, type = int, default = 100, help = "number of neighbors are to calculate the average distance for a given point")


output:
    filtered 3D model file in ply format
"""
#!/usr/bin/env python



# import the necessary packages
import numpy as np 
import argparse
import glob
import os
import sys
import open3d as o3d
import copy

import math
import pathlib

import psutil
import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from contextlib import closing


# generate folder to store the output results
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
        print (path+' path exists!')
        return False




# visualization of 3d models
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])




# ofilter the ply model using lier removal 
def model_filter(model_file):
    
    
    
    if os.path.isfile(model_file):
        print("Filtering 3D point cloud model {}...\n".format(model_file))
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
    #pcd_r = copy.deepcopy(pcd)
    
    
    # get the model center postion
    #model_center = pcd_r.get_center()
    
    # geometry points are translated directly to the model_center position
    #pcd_r.translate(-1*(model_center))
    
    #print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.0015)


    # Statistical oulier removal
    #nb_neighbors, which specifies how many neighbors are taken into account in order to calculate the average distance for a given point.
    #std_ratio, which allows setting the threshold level based on the standard deviation of the average distances across the point cloud. 
    #The lower this number the more aggressive the filter will be.
    

    # visualize the oulier removal point cloud
    print("Statistical oulier removal\n")
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors = filter_radius, std_ratio = filter_ratio)
    #display_inlier_outlier(voxel_down_pcd, ind)
    
    #print("Radius oulier removal")
    #cl, ind = pcd.remove_radius_outlier(nb_points = filter_radius, radius = filter_ratio)
    #display_inlier_outlier(pcd, ind)
    ####################################################################
    
    #Save model file as ascii format in ply
    filename = save_path + base_name + '_filtered.' + ext
    
    #write out point cloud file
    #o3d.io.write_point_cloud(filename, pcd)
    
    o3d.io.write_point_cloud(filename, voxel_down_pcd, write_ascii = True)
    

    # check saved file
    if os.path.exists(filename):
        print("Filtered 3D model was saved at {0}\n".format(filename))
        return True
    else:
        return False
        print("Model filter failed!\n")
        sys.exit(0)
    




if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-ft", "--filetype", required = False,  default = 'ply',  help = "3D model filetype")
    ap.add_argument("-fr", "--filter_ratio", required = False, type = float, default = 0.01, help = "filter ratio, The lower this number the more aggressive the filter will be")
    ap.add_argument("-fd", "--filter_radius", required = False, type = int, default = 100, help = "number of neighbors are to calculate the average distance for a given point")
    ap.add_argument("-t", "--test", required = False, default = False, help = "if using test setup")
    args = vars(ap.parse_args())


    # setting path to model file 
    current_path = args["path"]
    
    ext = args['filetype']
   
    filter_radius = args["filter_radius"]
    filter_ratio = args["filter_ratio"]
    
    # create result folder 
    mkpath = os.path.dirname(current_path) + '/result'
    mkdir(mkpath)
    save_path = mkpath + '/'
    

    # path to all ply file 
    filetype = '*.' + ext
    model_file_path = current_path + "/" + filetype
    
    # accquire ply file list
    model_List = sorted(glob.glob(model_file_path))
    
    #print(model_List)
    
    # number of all ply files
    n_models = len(model_List)
    
    result_list = []
    
    
    
    # loop execute
    ###############################################################################
    
    for m_id, model_file in enumerate(model_List):
        

        result_list.append(model_filter(model_file))
        
    
    
    # parallel processing
    ########################################################################
    '''
    # get cpu number for parallel processing
    agents = psutil.cpu_count()-2
    #agents = multiprocessing.cpu_count() 
    #agents = 8
    
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(model_filter, model_List)
        pool.terminate()
    '''

 
