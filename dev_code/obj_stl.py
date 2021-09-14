"""
Version: 1.5

Summary: Align the center of the point cloud model and rotate it aligned with Z axis. 

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 obj_stl.py -p ~/example/ -m test.obj


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
    
    mesh = o3d.io.read_triangle_mesh(model_file)
    
    print(mesh)
    
    stl_mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)
    
    stl_output = model_name_base + '.stl'
    
    print(stl_output)
    
    o3d.io.write_triangle_mesh(stl_output, stl_mesh)
    

    # check saved file
    if os.path.exists(stl_output):
        print("Converted 3d model was saved at {0}".format(stl_output))
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

 
