"""
Version: 1.5

Summary: visualize the skeleto of point cloud model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

    python3 skeleton_loader_vis.py -p ~/example/ -m1 test_skeleton.ply -m2 test.ply


argument:
("-p", "--path", required=True,    help="path to *.ply model file")
("-m", "--model", required=True,    help="file name")

"""
#!/usr/bin/env python

#from mayavi import mlab
#from tvtk.api import tvtk

# import the necessary packages
from plyfile import PlyData, PlyElement
import numpy as np 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from operator import itemgetter
import argparse

from scipy.spatial import KDTree

import os
import sys
import open3d as o3d
import copy



#import networkx as nx

import graph_tool.all as gt

import plotly.graph_objects as go

from matplotlib import pyplot as plt

from math import sqrt



# calculate length of a 3D path or curve
def path_length(X, Y, Z):

    n = len(X)
     
    lv = [sqrt((X[i]-X[i-1])**2 + (Y[i]-Y[i-1])**2 + (Z[i]-Z[i-1])**2) for i in range (1,n)]
    
    L = sum(lv)
    
    return L


#compute angle between two vectors(works for n-dimensional vector),
def dot_product_angle(v1,v2):

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        
        print("Zero magnitude vector!")
        
        return 0
        
    else:
        vector_dot_product = np.dot(v1,v2)
        
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        
        angle = np.degrees(arccos)
        
        #return angle
        
        
        if angle > 0 and angle < 45:
            return (90 - angle)
        elif angle < 90:
            return angle
        else:
            return (180- angle)


'''
# distance between two points
def distance_pt(p0, p1):
    
    dist = np.linalg.norm(p0 - p1)
    
    return dist
'''




#find the closest points from a points sets to a fix point using Kdtree, O(log n) 
def closest_point(point_set, anchor_point):
    
    kdtree = KDTree(point_set)
    
    (d, i) = kdtree.query(anchor_point)
    
    #print("closest point:", point_set[i])
    
    return  i, point_set[i]



#colormap mapping
def get_cmap(n, name = 'Spectral'):
    """get the color mapping""" 
    #viridis, BrBG, hsv, copper
    #Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    #RGB color; the keyword argument name must be a standard mpl colormap name
    return plt.cm.get_cmap(name,n+1)


def dimension_size(Z_skeleton, sub_branch_start_rec, sub_branch_end_rec, select):
    
    loc_start = [Z_skeleton[index] for index in sub_branch_start_rec]
    loc_end = [Z_skeleton[index] for index in sub_branch_end_rec]
    
    print("Z_loc_start max = {} min = {}".format(max(loc_start), min(loc_start)))
    print("Z_loc_end max = {} min = {}\n".format(max(loc_end), min(loc_end)))
    
    max_dimension_length = abs(max(max(loc_start), max(loc_end) - min(min(loc_start), min(loc_end))))

    min_dimension_length = abs(min(min(loc_start), min(loc_end) - max(max(loc_start), max(loc_end))))
    
    #print("max_length = {}\n".format(max_length))
    
    if select == 1:
        return max_dimension_length
    else:
        return min_dimension_length


def visualize_skeleton(current_path, filename_skeleton, filename_ptcloud):
    
    # define the path to skeleton file
    model_skeleton = current_path + filename_skeleton
    print("Loading 3D skeleton file {}...\n".format(filename_skeleton))
    model_skeleton_name_base = os.path.splitext(model_skeleton)[0]
    
    #load the skeleton file in ply format 
    try:
        with open(model_skeleton, 'rb') as f:
            
            plydata_skeleton = PlyData.read(f)
            
            # get the number of points
            num_vertex_skeleton = plydata_skeleton.elements[0].count
            
            # get the number of vertices
            N_edges_skeleton = len(plydata_skeleton['edge'].data['vertex_indices'])
            
            # get the vertices array
            array_edges_skeleton = plydata_skeleton['edge'].data['vertex_indices']
            
            print("Ply data structure: \n")
            #print(plydata_skeleton)
            #print("\n")
            print("Number of 3D points in skeleton model: {0} \n".format(num_vertex_skeleton))
            print("Number of edges: {0} \n".format(N_edges_skeleton))

        
    except:
        print("Model skeleton file does not exist!")
        sys.exit(0)
    
    ####################################################################################3
    # Load ply point cloud file
    if not (filename_ptcloud is None):
        
        model_pcloud = current_path + filename_ptcloud
        
        print("Loading 3D point cloud {}...\n".format(filename_ptcloud))
        
        model_pcloud_name_base = os.path.splitext(model_pcloud)[0]
        
        pcd = o3d.io.read_point_cloud(model_pcloud)
        
        Data_array_pcloud = np.asarray(pcd.points)
        
        
        if pcd.has_colors():
            
            print("Render colored point cloud")
            
            pcd_color = np.asarray(pcd.colors)
            
            '''
            if len(pcd_color) > 0: 
                
                pcd_color = np.rint(pcd_color * 255.0)
            '''
            
            #pcd_color = tuple(map(tuple, pcd_color))
        else:
            
            print("Generate random color")
        
            pcd_color = np.random.randint(256, size = (len(Data_array_pcloud),3))
            
            
    
    ######################################################################
    #Parse ply format skeleton file and Extract the points and edges
    Data_array_skeleton = np.zeros((num_vertex_skeleton, 3))
    
     
    Data_array_skeleton[:,0] = plydata_skeleton['vertex'].data['x']
    Data_array_skeleton[:,1] = plydata_skeleton['vertex'].data['y']
    Data_array_skeleton[:,2] = plydata_skeleton['vertex'].data['z']
    #Data_array_skeleton[:,3] = plydata_skeleton['vertex'].data['radius']
    
    X_skeleton = Data_array_skeleton[:,0]
    Y_skeleton = Data_array_skeleton[:,1]
    Z_skeleton = Data_array_skeleton[:,2]
    #radius_vtx = Data_array_skeleton[:,3]
    
    '''
    # build directed graph from skeleton/structure data
    ####################################################################
    print("Building directed graph from 3D skeleton/structure ...\n")
    G_unordered = gt.Graph(directed = True)
    
    # assert directed graph
    #print(G.is_directed())
    
    nodes = G_unordered.add_vertex(num_vertex_skeleton)
    
    G_unordered.add_edge_list(array_edges_skeleton.tolist()) 
    
    #gt.graph_draw(G_unordered, vertex_text = G_unordered.vertex_index, output = current_path + "graph_view.pdf")
    
    
    # find all end vertices by fast iteration of all vertices
    end_vlist = []
    
    end_vlist_offset = []
    
    for v in G_unordered.iter_vertices():
        
        #print(G.vertex(v).out_degree(), G.vertex(v).in_degree())
        
        if G_unordered.vertex(v).out_degree() == 0 and G_unordered.vertex(v).in_degree() == 1:
        
            end_vlist.append(v)
            
            if (v+1) == num_vertex_skeleton:
                end_vlist_offset.append(v)
            else:
                end_vlist_offset.append(v+1)
            
    #print("end_vlist = {} \n".format(end_vlist))
    #print("end_vlist_offset = {} \n".format(end_vlist_offset))
    
    
    
    # parse all the sub branches edges and vetices, start, end vetices
    #####################################################################################
    sub_branch_list = []
    sub_branch_length_rec = []
    
    #sub_branch_angle_rec = []
    #sub_branch_start_rec = []
    #sub_branch_end_rec = []
   
    #sub_branch_projection_rec = []
    #sub_branch_radius_rec = []
    
    #sub_branch_xs_rec = []
    #sub_branch_ys_rec = []
    #sub_branch_zs_rec = []
    
        
    for idx, v_end in enumerate(end_vlist):
        
        #print(idx, v_end)
        #construct list of vertices in sub branches
        if idx == 0:
            v_list = [*range(0, int(end_vlist[idx])+1)]
        else:
            v_list = [*range(int(end_vlist[idx-1])+1, int(end_vlist[idx])+1)]
            
        # change type to interger 
        int_v_list = [int(i) for i in v_list]
        
        # current sub branch length
        sub_branch_length = path_length(X_skeleton[int_v_list], Y_skeleton[int_v_list], Z_skeleton[int_v_list])
        
        # current sub branch start and end points 
        #start_v = [X_skeleton[int_v_list[0]], Y_skeleton[int_v_list[0]], Z_skeleton[int_v_list[0]]]
        
        #end_v = [X_skeleton[int_v_list[0]] - X_skeleton[int_v_list[int(len(int_v_list)-1.0)]], Y_skeleton[int_v_list[0]] - Y_skeleton[int_v_list[int(len(int_v_list)-1.0)]], Z_skeleton[int_v_list[0]] - Z_skeleton[int_v_list[int(len(int_v_list)-1.0)]]]
        
        # angle of current branch vs Z direction
        #angle_sub_branch = dot_product_angle(start_v, end_v)
        
        # projection radius of current branch length
        #p0 = np.array([X_skeleton[int_v_list[0]], Y_skeleton[int_v_list[0]], Z_skeleton[int_v_list[0]]])
        
        #p1 = np.array([X_skeleton[int_v_list[0]], Y_skeleton[int_v_list[0]], Z_skeleton[int_v_list[-1]]])
        
        #projection_radius = np.linalg.norm(p0 - p1)
        
        #radius value from fitted point cloud contours
        #radius_edge = float(radius_vtx[int_v_list[0]])*1
        
        #sub_branch_xs = X_skeleton[int_v_list[0]]
        #sub_branch_ys = Y_skeleton[int_v_list[0]]
        #sub_branch_zs = Z_skeleton[int_v_list[0]]
        
        
        # save computed parameters for each branch
        sub_branch_list.append(v_list)
        sub_branch_length_rec.append(sub_branch_length)
        
        
        #sub_branch_angle_rec.append(angle_sub_branch)
        
        #sub_branch_start_rec.append(int_v_list[0])
        #sub_branch_end_rec.append(int_v_list[-1])
        
        #sub_branch_projection_rec.append(projection_radius)
        #sub_branch_radius_rec.append(radius_edge)
        
        #sub_branch_xs_rec.append(sub_branch_xs)
        #sub_branch_ys_rec.append(sub_branch_ys)
        #sub_branch_zs_rec.append(sub_branch_zs)
        
    
    
    # sort branches according to length feature in descending order
    ####################################################################
    sorted_idx_len = np.argsort(sub_branch_length_rec)
    
    #reverse the order from accending to descending
    sorted_idx_len_loc = sorted_idx_len[::-1]

    #print("Z_loc = {}\n".format(sorted_idx_Z_loc))
    
    #sort all lists according to sorted_idx_Z_loc order
    sub_branch_list[:] = [sub_branch_list[i] for i in sorted_idx_len_loc] 
    sub_branch_length_rec[:] = [sub_branch_length_rec[i] for i in sorted_idx_len_loc]
    
    #sub_branch_angle_rec[:] = [sub_branch_angle_rec[i] for i in sorted_idx_len_loc]
    #sub_branch_start_rec[:] = [sub_branch_start_rec[i] for i in sorted_idx_len_loc]
    #sub_branch_end_rec[:] = [sub_branch_end_rec[i] for i in sorted_idx_len_loc]
    #sub_branch_projection_rec[:] = [sub_branch_projection_rec[i] for i in sorted_idx_len_loc]
    #sub_branch_radius_rec[:] = [sub_branch_radius_rec[i] for i in sorted_idx_len_loc]
    
    #sub_branch_xs_rec[:] = [sub_branch_xs_rec[i] for i in sorted_idx_len_loc]
    #sub_branch_ys_rec[:] = [sub_branch_ys_rec[i] for i in sorted_idx_len_loc]
    #sub_branch_zs_rec[:] = [sub_branch_zs_rec[i] for i in sorted_idx_len_loc]
    
    ####################################################################

    
    
    Z_loc_start = [Z_skeleton[index] for index in sub_branch_start_rec]
    Z_loc_end = [Z_skeleton[index] for index in sub_branch_end_rec]
    
    print("Z_loc_start max = {} min = {}".format(max(Z_loc_start), min(Z_loc_start)))
    print("Z_loc_end max = {} min = {}\n".format(max(Z_loc_end), min(Z_loc_end)))
    
    max_length = abs(max(max(Z_loc_start), max(Z_loc_end) - min(min(Z_loc_start), min(Z_loc_end))))
    
    max_length_x = dimension_size(Z_skeleton, sub_branch_start_rec, sub_branch_end_rec, 1)
    max_length_y = dimension_size(Y_skeleton, sub_branch_start_rec, sub_branch_end_rec, 1)
    max_length_z = dimension_size(Z_skeleton, sub_branch_start_rec, sub_branch_end_rec, 1)
    
    min_length_x = dimension_size(Z_skeleton, sub_branch_start_rec, sub_branch_end_rec, 0)
    min_length_y = dimension_size(Y_skeleton, sub_branch_start_rec, sub_branch_end_rec, 0)
    min_length_z = dimension_size(Z_skeleton, sub_branch_start_rec, sub_branch_end_rec, 0)
    
    print("max_length = {} {} {}\n".format(max_length_x, max_length_y, max_length_z))
    
    print("min_length = {} {} {}\n".format(min_length_x, min_length_y, min_length_z))
    '''
    
    
    
    
    sel_points = Data_array_skeleton
    
    sel_skeletons = array_edges_skeleton
    
    #sel_points = Data_array_skeleton[sub_branch_list[0]]
    
    #sel_skeletons = array_edges_skeleton[sub_branch_list[0]]
    
    # visualization 
    ####################################################################

    pcd_skeleton = o3d.geometry.PointCloud()
    
    pcd_skeleton.points = o3d.utility.Vector3dVector(sel_points)
    
    #pcd_skeleton.points = o3d.utility.Vector3dVector(sel_points)
    
    pcd_skeleton.paint_uniform_color([0, 0, 1])
    
    #o3d.visualization.draw_geometries([pcd])
    
    points = sel_points
    
    lines = sel_skeletons
    
    colors = [[1, 0, 0] for i in range(len(lines))]
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    
    #o3d.visualization.draw_geometries([line_set])
    
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(line_set)
    vis.add_geometry(pcd_skeleton)
    vis.add_geometry(pcd)
    vis.get_render_option().line_width = 1
    vis.get_render_option().point_size = 1
    vis.get_render_option().background_color = (0, 0, 0)
    vis.get_render_option().show_coordinate_frame = True
    
    vis.run()
    #vis.destroy_window()
    
    '''
    print("Let's draw a box using o3d.geometry.LineSet.")
    points = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    ]
    lines = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [4, 5],
    [4, 6],
    [5, 7],
    [6, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    point_cloud2 = o3d.geometry.PointCloud()
    point_cloud2.points = o3d.utility.Vector3dVector(points)
    point_cloud2.paint_uniform_color([0, 1, 0])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(line_set)
    vis.add_geometry(point_cloud2)
    vis.get_render_option().line_width = 25
    vis.get_render_option().point_size = 20
    vis.run()
    
    '''
    
    
    '''
    # Build KDTree from point cloud for fast retrieval of nearest neighbors
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_skeleton)
    
    print("Paint the 00th point red.")
    
    pcd.colors[0] = [1, 0, 0]
    
    print("Find its 200 nearest neighbors, paint blue.")
    
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[0], 100)
    
    np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
    
    print("Visualize the point cloud.")
    
    o3d.visualization.draw_geometries([pcd])
    
    
    #build octree, a tree data structure where each internal node has eight children.
    # fit to unit cube
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(num_vertex_skeleton, 3)))
    o3d.visualization.draw_geometries([pcd])

    print('octree division')
    octree = o3d.geometry.Octree(max_depth=4)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    o3d.visualization.draw_geometries([octree])

    print(octree.locate_leaf_node(pcd.points[243]))
    '''
    
    




if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m1", "--model_skeleton", required = False, help = "skeleton file name")
    ap.add_argument("-m2", "--model_pcloud", required = False, default = None, help = "point cloud model file name")
    args = vars(ap.parse_args())


    # setting input path to model file 
    current_path = args["path"]
    #current_path = os.path.join(current_path, '')
    
    folder_name = os.path.basename(os.path.dirname(current_path))
    
    ###################################################################
    # check file name input and default file name
    if args["model_skeleton"] is None:

        # search for file with default name
        filename_skeleton = current_path + folder_name + '_skeleton.ply'
        
        print(filename_skeleton)
        
        if os.path.isfile(filename_skeleton):
            print("Default skeleton file: {}\n".format(filename_skeleton))
            filename_skeleton = folder_name + '_skeleton.ply'
        else:
            print("Skeleton model is not found!\n")
            sys.exit()
    else:
        filename_skeleton = args["model_skeleton"]
    

    if args["model_pcloud"] is None:
        
        # search for file with default name
        filename_ptcloud = current_path + folder_name + '_aligned.ply'
        
        if os.path.isfile(filename_ptcloud):
            print("Default model file: {}\n".format(filename_ptcloud))
            filename_ptcloud = folder_name + '_aligned.ply'
        else:
            print("Aligned pointclod model is not found!\n")
            sys.exit()

    else:
        filename_ptcloud = args["model_pcloud"]
    
    
    #file_path = current_path + filename

    print ("results_folder: {}\n".format(current_path))

    visualize_skeleton(current_path, filename_skeleton, filename_ptcloud)

 
