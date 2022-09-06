"""
Version: 1.0

Summary: visualization of two sets of rotation vectors

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 sphere_vis_rotation.py -p ~/example/pt_cloud/tiny/ -f1 tiny_quaternion.csv -f2 test_quaternion.csv


"""
from mayavi import mlab
from tvtk.api import tvtk

import argparse
import pandas as pd

import numpy as np 




def visualization_rotation_vector(rotVec_rec_1,rotVec_rec_2):
    
    
    ###############################################################################
    # Display a semi-transparent sphere

    mlab.figure("sphere_representation_rotation_vector: B101 v.s. Pa762", size = (800, 800), bgcolor = (0, 0, 0))

    # use a sphere Glyph, through the points3d mlab function, rather than
    # building the mesh ourselves, because it gives a better transparent
    # rendering.
    sphere = mlab.points3d(0, 0, 0, scale_mode='none',
                            scale_factor=2,
                            color=(0.67, 0.77, 0.93),
                            resolution=50,
                            opacity=0.7,
                            name='Sphere')

    # These parameters, as well as the color, where tweaked through the GUI,
    # with the record mode to produce lines of code usable in a script.
    sphere.actor.property.specular = 0.45
    sphere.actor.property.specular_power = 5
    
    # Backface culling is necessary for more a beautiful transparent rendering.
    sphere.actor.property.backface_culling = True


    # visualize rotation vectors
    for idx, Vec in enumerate(rotVec_rec_1):

        mlab.quiver3d(0,0,0, Vec[0], Vec[1], Vec[2], color = (1, 0, 0)) 

    for idx, Vec in enumerate(rotVec_rec_2):

        mlab.quiver3d(0,0,0, Vec[0], Vec[1], Vec[2], color = (0, 1, 0)) 

        
    mlab.show()
    
    
    
if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-f1", "--file1", required = True, help = "file1 name")
    ap.add_argument("-f2", "--file2", required = False, help = "file1 name")
    args = vars(ap.parse_args())

    
    # setting path to data file 
    current_path = args["path"]
    file1 = args["file1"]
    file2 = args["file2"]
    
    file1_full_path = current_path + file1
    file2_full_path = current_path + file2
    
    #Read rotation vector data from csv
    data1 = pd.read_csv(file1_full_path)
    data2 = pd.read_csv(file2_full_path)
    
    #construct data array
    rotVec_rec_1 = np.vstack((data1['rotVec_rec_0'],data1['rotVec_rec_1'],data1['rotVec_rec_2'])).T
    rotVec_rec_2 = np.vstack((data2['rotVec_rec_0'],data2['rotVec_rec_1'],data2['rotVec_rec_2'])).T
    
    #visualize rotation vectors
    visualization_rotation_vector(rotVec_rec_1,rotVec_rec_2)
    
        
