"""
Version: 1.5

Summary: compute the cross section plane based on 3d model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 skeleton_loader.py -p ~/example/ -m1 test_skeleton.ply -m2 test.ply


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

from mayavi import mlab

import networkx as nx


def visualize_skeleton(current_path, filename_skeleton, filename_pcloud):
    
    model_skeleton = current_path + filename_skeleton
    print("Loading 3D skeleton file {}...\n".format(filename_skeleton))
    model_skeleton_name_base = os.path.splitext(model_skeleton)[0]
    
    # load the model file
    try:
        with open(model_skeleton, 'rb') as f:
            plydata_skeleton = PlyData.read(f)
            num_vertex_skeleton = plydata_skeleton.elements[0].count
            N_edges_skeleton = len(plydata_skeleton['edge'].data['vertex_indices'])
            array_edges_skeleton = plydata_skeleton['edge'].data['vertex_indices']
            
            print("Ply data structure:")
            #print(plydata_skeleton)
            print("Number of 3D points in skeleton model: {0} \n".format(num_vertex_skeleton))
            print("Number of edges: {0} \n".format(N_edges_skeleton))
        
    except:
        print("Model skeleton file does not exist!")
        sys.exit(0)
    
    
    model_pcloud = current_path + filename_pcloud
    print("Loading 3D point cloud {}...\n".format(filename_pcloud))
    model_pcloud_name_base = os.path.splitext(model_pcloud)[0]
    
    '''
    # load the model file
    try:
        with open(model_pcloud, 'rb') as f:
            plydata_pcloud = PlyData.read(f)
            num_vertex_pcloud = plydata_pcloud.elements[0].count
            
            print("Ply data structure:")
            #print(plydata_pcloud)
            print("Number of 3D points in point cloud model: {0} \n".format(num_vertex_pcloud))
        
    except:
        print("Model pcloud file does not exist!")
        sys.exit(0)
    '''
    pcd = o3d.io.read_point_cloud(model_pcloud)
    
    Data_array_pcloud = np.asarray(pcd.points)
    
    #print(Data_array_pcloud.shape)
    
    
    #Parse the ply format file and Extract the data
    Data_array_skeleton = np.zeros((num_vertex_skeleton, 3))
    
    Data_array_skeleton[:,0] = plydata_skeleton['vertex'].data['x']
    Data_array_skeleton[:,1] = plydata_skeleton['vertex'].data['y']
    Data_array_skeleton[:,2] = plydata_skeleton['vertex'].data['z']
    
    X_skeleton = Data_array_skeleton[:,0]
    Y_skeleton = Data_array_skeleton[:,1]
    Z_skeleton = Data_array_skeleton[:,2]
    
    '''
    #Parse the ply format file and Extract the data
    Data_array_pcloud = np.zeros((num_vertex_pcloud, 3))
    
    Data_array_pcloud[:,0] = plydata_pcloud['vertex'].data['x']
    Data_array_pcloud[:,1] = plydata_pcloud['vertex'].data['y']
    Data_array_pcloud[:,2] = plydata_pcloud['vertex'].data['z']
    '''
    
    
    # The number of points per line
    N = 2
    
    mlab.figure(1, size=(800, 800), bgcolor=(0, 0, 0))
    mlab.clf()
    
    #pts = mlab.points3d(X_skeleton, Y_skeleton, Z_skeleton, mode = 'point')
    
    pts = mlab.points3d(Data_array_pcloud[:,0], Data_array_pcloud[:,1], Data_array_pcloud[:,2], mode = 'point')
    
    x = list()
    y = list()
    z = list()
    s = list()
    connections = list()
    
    # The index of the current point in the total amount of points
    index = 0
    
    for i in range(N_edges_skeleton):
        
        #print("Edges {0} has nodes {1}, {2}\n".format(i, array_edges[i][0], array_edges[i][1]))
        
        x.append(X_skeleton[array_edges_skeleton[i][0]])
        y.append(Y_skeleton[array_edges_skeleton[i][0]])
        z.append(Z_skeleton[array_edges_skeleton[i][0]])
        
        x.append(X_skeleton[array_edges_skeleton[i][1]])
        y.append(Y_skeleton[array_edges_skeleton[i][1]])
        z.append(Z_skeleton[array_edges_skeleton[i][1]])
        
        # The scalar parameter for each line
        s.append(array_edges_skeleton[i])
        
        # This is the tricky part: in a line, each point is connected
        # to the one following it. We have to express this with the indices
        # of the final set of points once all lines have been combined
        # together, this is why we need to keep track of the total number of
        # points already created (index)
        #connections.append(np.vstack(array_edges[i]).T)
        
        connections.append(np.vstack(
                       [np.arange(index,   index + N - 1.5),
                        np.arange(index + 1, index + N - .5)]
                            ).T)
        index += 2
    
    
    # Now collapse all positions, scalars and connections in big arrays
    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    s = np.hstack(s)
    #connections = np.vstack(connections)

    # Create the points
    src = mlab.pipeline.scalar_scatter(x, y, z, s)

    # Connect them
    src.mlab_source.dataset.lines = connections
    src.update()

    # The stripper filter cleans up connected lines
    #lines = mlab.pipeline.stripper(src)

    # display the set of lines
    mlab.pipeline.surface(src, colormap = 'Accent', line_width = 10, opacity = 0.8)

    # And choose a nice view
    #mlab.view(33.6, 106, 5.5, [0, 0, .05])
    #mlab.roll(125)
    mlab.show()
    
    
     





if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m1", "--model_skeleton", required = True, help = "skeleton file name")
    ap.add_argument("-m2", "--model_pcloud", required = True, help = "point cloud model file name")
    args = vars(ap.parse_args())


    # setting path to model file 
    current_path = args["path"]
    filename_skeleton = args["model_skeleton"]
    filename_pcloud = args["model_pcloud"]
    
    #file_path = current_path + filename

    print ("results_folder: " + current_path)

    visualize_skeleton(current_path, filename_skeleton, filename_pcloud)

 
