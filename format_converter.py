"""
Version: 1.5

Summary: compute the cross section plane based on 3d model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 format_converter.py -p /home/suxingliu/model-scan/model-data/ -m surface.ply 


argument:
("-p", "--path", required=True,    help="path to *.ply model file")
("-m", "--model", required=True,    help="file name")
("-i", "--interval", required=True,    type = int, help="intervals along sweeping plane")
("-d", "--direction", required=True,    type = int, help="direction of sweeping plane, X=0, Y=1, Z=2")
("-r", "--reverse", required=True,    type = int, help="Reverse model top_down, 1 for Ture, 0 for False")

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
import open3d as o3d



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

    
    # load the model file
    try:
        with open(file_path, 'rb') as f:
            plydata = PlyData.read(f)
            num_vertex = plydata.elements[0].count
            
            print("Ply data structure: \n")
            print(plydata)
            print("Number of 3D points in current model: {0} \n".format(num_vertex))
        
    except:
        print("Model file not exist!")
        sys.exit(0)
        
    
    #Parse the ply format file and Extract the data
    Data_array_ori = np.zeros((num_vertex, 3))
    
    Data_array_ori[:,0] = plydata['vertex'].data['x']
    Data_array_ori[:,1] = plydata['vertex'].data['y']
    Data_array_ori[:,2] = plydata['vertex'].data['z']
    
    Data_array = np.asarray(sorted(Data_array_ori, key = itemgetter(2), reverse = False))
   
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
    
    
    #print "range_data_x, range_data_y"
    print (range_data_x, range_data_y, range_data_z)
    
    print(min_x,max_x)
    print(min_y,max_y)
    print(min_z,max_z)
    
    #Normalize data
    
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1000))

    point_normalized = min_max_scaler.fit_transform(Data_array)
    
    point_normalized_scale = [i * 1 for i in point_normalized]
   
    
    # Pass xyz to Open3D.o3d.geometry.PointCloud and save it in ascii format ply file
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(point_normalized_scale)
    
    #Save images as jpeg format
    filename = current_path + 'converted.ply'
    
    o3d.io.write_point_cloud(filename, pcd, write_ascii = True)
    
    
