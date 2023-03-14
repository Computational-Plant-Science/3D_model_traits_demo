"""
Version: 1.5

Summary: compute the cross section plane based on 3d model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 model_range.py -p ~/example/ -m test.ply


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
import random
import glob

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
        
        
        
def format_converter(model_file):
    
    path, filename = os.path.split(model_file)
    
    model_name_base = os.path.splitext(os.path.basename(filename))[0]
    
    print("Parsing {} file format for level set scanning ...\n".format(model_name_base))
    
    '''
    # load the model file
    try:
        with open(model_file, 'rb') as f:
            plydata = PlyData.read(f)
            num_vertex = plydata.elements[0].count
            
            print("Ply data structure: \n")
            print(plydata)
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
    
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(Data_array)
    
    o3d.visualization.draw_geometries([pcd])
    
    pcd.colors = 
    
    
    abs_path = os.path.abspath(model_file)
    filename, file_extension = os.path.splitext(abs_path)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    '''
     
    # Pass xyz to Open3D.o3d.geometry.PointCloud 

    pcd = o3d.io.read_point_cloud(model_file)
    
    
    #visualize the original point cloud
    #o3d.visualization.draw_geometries([pcd])
    
    Data_array = np.asarray(pcd.points)
    
    color_array = np.asarray(pcd.colors)
    
    
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
    
    
    data_range = 10000
    
    #Normalize data
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,data_range))
    
    #min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1*data_range, data_range))

    point_normalized = min_max_scaler.fit_transform(Data_array)
    
    #point_normalized_scale = [i * 1 for i in point_normalized]
   
    # Pass xyz to Open3D.o3d.geometry.PointCloud 
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(point_normalized)
    '''
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
    '''
    #Voxel downsampling uses a regular voxel grid to create a uniformly downsampled point cloud from an input point cloud
    #print("Downsample the point cloud with a voxel of 0.05")
    
    factor = random.randint(5, 10)/1000
    
    downpcd = pcd.voxel_down_sample(voxel_size=factor)
    #o3d.visualization.draw_geometries([downpcd])
    
    #cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    #create label result file folder
    mkpath = os.path.dirname(current_path) +'/result/' + model_name_base + "/"
    mkdir(mkpath)
    result_path = mkpath + '/'
    
    #Save model file as ascii format in ply
    filename = result_path + model_name_base + '.ply'
    
    #print(filename)
    
    #write out point cloud file
    #o3d.io.write_point_cloud(filename, ind, write_ascii = True)
    
    o3d.io.write_point_cloud(filename, downpcd)
    
    
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
    #ap.add_argument("-m", "--model", required = True, help = "model file name")
    ap.add_argument("-ft", "--filetype", required = True,    help = "Image filetype")
    args = vars(ap.parse_args())


    # setting path to model file 
    current_path = args["path"]
    #filename = args["model"]
    #file_path = current_path + filename
    
    

    
    ext = args['filetype']
    
    #accquire image file list
    filetype = '*.' + ext
    ply_file_path = current_path + filetype
    
    #accquire image file list
    plyList = sorted(glob.glob(ply_file_path))
    
    print ("plyList: {}\n".format(plyList))

   
    ####################################################################################
    #loop execute to get all traits
    for ply_file in plyList:
        
        format_converter(ply_file)

        

 
