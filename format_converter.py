"""
Version: 1.5

Summary: Align the center of the point cloud model and rotate it aligned with Z axis. 

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 format_converter.py -p ~/example/ -m test.ply


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

import os
import sys
import open3d as o3d
import copy

#from mayavi import mlab

import networkx as nx


def format_converter(current_path, model_name):
    
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
    
    #o3d.visualization.draw_geometries([pcd])
    
    
    '''
    #load point cloud using open3d loader
    pcd = o3d.io.read_point_cloud(model_file)
    
    
    #print(np.asarray(pcd.points))
    
    #visualize the original point cloud
    o3d.visualization.draw_geometries([pcd])
    
    Data_array = np.asarray(pcd.points)
    '''
    
    
    # copy original point cloud for rotation
    pcd_r = copy.deepcopy(pcd)
    
    # define rotation matrix
    R = pcd.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
    #R = pcd.get_rotation_matrix_from_xyz((0, -np.pi/2, 0))
    
    # Apply rotation transformation to copied point cloud data
    pcd_r.rotate(R, center = (0,0,0))
    
    # get the model center postion
    model_center = pcd_r.get_center()
    
    # geometry points are translated directly to the model_center position
    pcd_r.translate(-1*(model_center))
    
    # Visualize rotated point cloud 
    #o3d.visualization.draw_geometries([pcd, pcd_r])
    
   
    
    #Save model file as ascii format in ply
    filename = current_path + 'converted.ply'
    
    #write out point cloud file
    o3d.io.write_point_cloud(filename, pcd, write_ascii = True)
    
    
    #mesh = o3d.io.read_point_cloud(filename)
    
    #print(mesh)

    #Save modelfilea as ascii format in xyz
    #filename = model_name_base + '.xyz'
    #o3d.io.write_point_cloud(filename, pcd_r, write_ascii = True)
    
    
    # check saved file
    if os.path.exists(filename):
        print("Converted 3d model was saved at {0}".format(filename))
        return True
    else:
        return False
        print("Model file converter failed !")
        sys.exit(0)
    
if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m", "--model", required = True, help = "model file name")
    args = vars(ap.parse_args())


    # setting path to model file 
    current_path = args["path"]
    filename = args["model"]
    file_path = current_path + filename

    print ("results_folder: " + current_path)

    format_converter(current_path, filename)

 
