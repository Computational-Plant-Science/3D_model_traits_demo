"""
Version: 1.5

Summary: render the cross section with moving slicing plane of a 3d model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 pt_scan_engine.py -p /home/suxingliu/ply_data/ -m surface.ply -i 5 -de X 

python3 pt_scan_engine.py -p /home/suxingliu/ply_data/ -m test.ply -i 5 -de Y


arguments:
("-p", "--path", required=True,    help="path to *.ply model file")
("-m", "--model", required=True,    help="file name")
("-i", "--interval", required=True,    type = int, help="intervals along sweeping plane")
("-de", "--direction", required = False, default = 'X',   help = "direction of sweeping plane, X, Y, Z")

"""
#!/usr/bin/env python
import sys
import os
import argparse
import numpy as np 

from sklearn import preprocessing
from operator import itemgetter
import open3d as o3d

from plyfile import PlyData, PlyElement
from pointcloud.render.pointcloudrenderengine import PointCloudRenderEngine


def format_converter(current_path, model_name):
    
    model_file = current_path + model_name
    
    print("Converting file format for 3D point cloud model {}...\n".format(model_name))
    
    # load the model file
    try:
        with open(model_file, 'rb') as f:
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
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1000))

    point_normalized = min_max_scaler.fit_transform(Data_array)
    
    #point_normalized_scale = [i * 1 for i in point_normalized]
   
    # Pass xyz to Open3D.o3d.geometry.PointCloud 
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(point_normalized)
    
    #Save modelfilea as ascii format 
    filename = current_path + 'converted.ply'
    
    o3d.io.write_point_cloud(filename, pcd, write_ascii = True)
    
    if os.path.exists(filename):
        return True
    else:
        return False
        print("Model file converter failed !")
        sys.exit(0)
    

    
def mkdir(path):
    """Create result folder"""
    
    # remove space at the beginning
    path = path.strip()
    # remove slash at the end
    path = path.rstrip("\\")
 
    # path exist?   # True  # False
    isExists = os.path.exists(path)
 
    # process
    if not isExists:
        # construct the path and folder
        #print (path + ' folder constructed!\n')
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        print (path +' path exists!\n')
        return False


if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m", "--model", required = True, help = "model file name")
    ap.add_argument("-i", "--interval", required = False, default = '1',  type = int, help= "intervals along sweeping plane")
    ap.add_argument("-de", "--direction", required = False, default = 'X',   help = "direction of sweeping plane, X, Y, Z")
    args = vars(ap.parse_args())
    
    # setting path to model file 
    current_path = args["path"]
    filename = args["model"]
    file_path = current_path + filename
    

    #make the folder to store the results
    mkpath = current_path + "cross_section_scan"
    mkdir(mkpath)
    save_path = mkpath + '/'
    print ("results_folder: " + save_path + "\n")
    
    if format_converter(current_path,filename):
        
        converted_model = current_path + 'converted.ply'
        
        renderEngine = PointCloudRenderEngine(converted_model, save_path, args["direction"], args["interval"])
        
        renderEngine.render()
