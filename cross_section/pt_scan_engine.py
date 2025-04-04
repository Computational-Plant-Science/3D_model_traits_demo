"""
Version: 1.5

Summary: render the cross section with moving slicing plane of a 3d model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 pt_scan_engine.py -p ~/ply_data/ -m surface.ply -i 5 -de X 

python3 pt_scan_engine.py -p ~/ply_data/ -m converted.ply -i 5 -de Z


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
    ap.add_argument("-m", "--model", required = False, default = 'converted.ply', help = "model file name")
    ap.add_argument("-i", "--interval", required = False, default = '1',  type = int, help = "intervals along sweeping plane")
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
    
    if os.path.exists(file_path):
        
        converted_model = file_path
        
        renderEngine = PointCloudRenderEngine(converted_model, save_path, args["direction"], args["interval"])
        
        renderEngine.render()
