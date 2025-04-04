"""
Version: 1.5

Summary: Align 3D model to its Z axis and translate it to its center. 

Author: Suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 model_alignment.py -i ~/example/test.ply -o ~/example/result/ --n_plane 5 --slicing_ratio 0.1 --adjustment 0
    

PARAMETERS:
    ("-i", "--input", dest="input", required=True, type=str, help="full path to 3D model file")
    ("-o", "--output_path", dest = "output_path", type = str, required = False, help = "result path")
    ("--n_plane", dest = "n_plane", type = int, required = False, default = 5,  help = "Number of planes to segment the 3d model along Z direction")
    ("--slicing_ratio", dest = "slicing_ratio", type = float, required = False, default = 0.10, help = "ratio of slicing the model from the bottom")
    ( "--adjustment", dest = "adjustment", type = int, required = False, default = 0, help = "model adjustment, 0: no adjustment, 1: rotate np.pi/2, -1: rotate -np.pi/2")

INPUT:
    
    3d model file in ply format

OUTPUT:

    *_aligned.ply: aligned model with only 3D coordinates of points 

"""
#!/usr/bin/env python



# import the necessary packages
import numpy as np
import argparse

import os
import sys
import open3d as o3d
import copy

#from scipy.spatial.transform import Rotation as Rot
#import math
import pathlib

from matplotlib import pyplot as plt
import glob

from scipy.spatial.transform import Rotation as R


# Find the rotation matrix that aligns vec1 to vec2
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions




#colormap mapping
def get_cmap(n, name = 'tab10'):
    """get the color mapping""" 
    #viridis, BrBG, hsv, copper, Spectral
    #Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    #RGB color; the keyword argument name must be a standard mpl colormap name
    return plt.cm.get_cmap(name,n)
    
    #return plt.colormaps.get_cmap(name,n)




# sort index according to the value in decenting order
def sort_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s


# get file information from the file path using python3
def get_file_info(file_full_path):
    
    p = pathlib.Path(file_full_path)

    filename = p.name

    basename = p.stem

    file_path = p.parent.absolute()

    file_path = os.path.join(file_path, '')

    return file_path, filename, basename


# compute dimensions of point cloud 
def get_pt_sel_parameter(Data_array_pt, n_plane):
    
    ####################################################################
    
    # load skeleton coordinates and radius 
    Z_pt_sorted = np.sort(Data_array_pt[:,2])
    
    #Z_pt_sorted = Data_array_pt[:,2]
    
    pt_plane = []
    
    
    pt_plane_center = []
    
    pt_plane_diameter = []
    
    
    for idx, x in enumerate(range(n_plane)):
        
        ratio_s = idx/n_plane
        ratio_e = (idx+1)/n_plane
        
        print("ratio_s ratio_e {} {}\n".format(ratio_s, ratio_e))
        
        # index of end plane 
        idx_sel_e = int(len(Z_pt_sorted)*ratio_e) 
    
        Z_e = Z_pt_sorted[idx_sel_e]  if idx_sel_e < len(Data_array_pt) else (len(Data_array_pt) - 1)
        
        # inde xof start plane
        idx_sel_s = int(len(Z_pt_sorted)*ratio_s) 
    
        Z_s = Z_pt_sorted[idx_sel_s]

        # mask between the start and end plane
        Z_mask = (Data_array_pt[:,2] <= Z_e) & (Data_array_pt[:,2] >= Z_s) 
        
        Z_pt_sel = Data_array_pt[Z_mask]
        
        # get the diameter of the sliced model 
        (pt_diameter_max, pt_diameter_min, pt_diameter) = get_pt_parameter(Z_pt_sel)
        
        #print(Z_pt_sel.shape)
     
        
        # initilize the o3d object
        pcd_Z_mask = o3d.geometry.PointCloud()
    
        pcd_Z_mask.points = o3d.utility.Vector3dVector(Z_pt_sel)
        
        
        
        # get the model center postion
        model_center = pcd_Z_mask.get_center()

        pt_plane.append(pcd_Z_mask)
        
        pt_plane_center.append(model_center)
        
        pt_plane_diameter.append(pt_diameter)
        

    return pt_plane, pt_plane_center, pt_plane_diameter



# slice array based on Z values
def get_pt_sel(Data_array_pt):
    
    ####################################################################
    
    # load points cloud Z values and sort it
    Z_pt_sorted = np.sort(Data_array_pt[:,2])
    
    #slicing_factor
    
    idx_sel = int(len(Z_pt_sorted)*slicing_ratio) 
    
    Z_mid = Z_pt_sorted[idx_sel]

    # mask
    Z_mask = (Data_array_pt[:,2] <= Z_mid) & (Data_array_pt[:,2] >= Z_pt_sorted[0]) 
    
    Z_pt_sel = Data_array_pt[Z_mask]
    
    
    return Z_pt_sel


# compute dimensions of point cloud 
def get_pt_parameter(Data_array_pt):
    
    
    ####################################################################
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(Data_array_pt)
    
    # get AxisAlignedBoundingBox
    aabb = pcd.get_axis_aligned_bounding_box()
    #aabb.color = (0, 1, 0)
    
    #Get the extent/length of the bounding box in x, y, and z dimension.
    aabb_extent = aabb.get_extent()
    
    aabb_extent_half = aabb.get_half_extent()
    
    # get the dimention of the points cloud in diameter based on bounding box
    pt_diameter_max = max(aabb_extent[0], aabb_extent[1])

    pt_diameter_min = min(aabb_extent_half[0], aabb_extent_half[1])

    
    pt_diameter = (pt_diameter_max + pt_diameter_min)*0.5
    
        
    return pt_diameter_max, pt_diameter_min, pt_diameter
    


def display_inlier_outlier(cloud, ind, obb):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, obb])
    

    
# align  ply model with z axis
def model_alignment(model_file, result_path, adjustment):
    
    
    # Load a ply point cloud
    pcd = o3d.io.read_point_cloud(model_file)
    
    #print(np.asarray(pcd.points))
    #o3d.visualization.draw_geometries([pcd])


    # copy original point cloud for rotation
    pcd_r = copy.deepcopy(pcd)
    
    # get the model center postion
    model_center = pcd_r.get_center()
    
    # geometry points are translated directly to the model_center position
    pcd_r.translate(-1*(model_center))
    
    '''
    # get convex hull of a point cloud is the smallest convex set that contains all points.
    hull, _ = pcd_r.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    
    # get AxisAlignedBoundingBox
    aabb = pcd_r.get_axis_aligned_bounding_box()
    
    # assign color for AxisAlignedBoundingBox
    aabb.color = (0, 1, 0)
    
    #Get the extent/length of the bounding box in x, y, and z dimension.
    aabb_extent = aabb.get_extent()
    aabb_extent_half = aabb.get_half_extent()
    '''
    
    # get OrientedBoundingBox
    obb = pcd_r.get_oriented_bounding_box()
    
    # assign color for OrientedBoundingBox
    obb.color = (0, 0, 1)
    
    # get the eight points that define the bounding box.
    pcd_coord = obb.get_box_points()
    
    #print(obb.get_box_points())
    
    #pcd_coord.color = (1, 0, 0)
    
    # From Open3D to numpy array
    np_points = np.asarray(pcd_coord)
    
    # create Open3D format for points 
    pcd_coord = o3d.geometry.PointCloud()
    pcd_coord.points = o3d.utility.Vector3dVector(np_points)
    
    
    
    '''
    # assign different colors for eight points in the bounding box.
    colors = []
    cmap = get_cmap(8)
    
    for idx in range(8):
    
        color_rgb = cmap(idx)[:len(cmap(idx))-1]
        colors.append(color_rgb)

    pcd_coord.colors = o3d.utility.Vector3dVector(colors)
    '''

    #o3d.visualization.draw_geometries([pcd_r, obb, pcd_coord])



    # check the length of the joint 3 vector in the bounding box to estimate the orientation of model
    list_dis = [np.linalg.norm(np_points[0] - np_points[1]), np.linalg.norm(np_points[0] - np_points[2]), np.linalg.norm(np_points[0] - np_points[3])]
    
    # sort the length values and return the index
    idx_sorted = sort_index(list_dis)
    
    #print(list_dis)
    
    #print(idx_sorted)
    '''
    # apply adjustment if alignment was not correct
    if adjustment == 1:
        print("Manual adjustment was applied!")
        idx_sorted[0] = 1
    '''


    # estimate the orientation 
    if idx_sorted[0] == 0:
        
        center_0 = np.mean(np_points[[0,2,3,5]], axis=0)
        center_1 = np.mean(np_points[[1,4,6,7]], axis=0)
        
    elif idx_sorted[0] == 1:
        
        center_0 = np.mean(np_points[[0,1,3,6]], axis=0)
        center_1 = np.mean(np_points[[2,4,5,7]], axis=0)
    
    else:
        
        center_0 = np.mean(np_points[[0,1,2,7]], axis=0)
        center_1 = np.mean(np_points[[3,4,5,6]], axis=0)
    
    '''
    # estimate the orientation of 3d model using sliced diameters
    print("Using {} planes to scan the model along Z axis...".format(n_plane))
    
    (pt_plane, pt_plane_center, pt_plane_diameter) = get_pt_sel_parameter(np_points, n_plane)
    
    print("pt_plane_diameter =  {} \n".format(pt_plane_diameter))
    '''
    
    # define unit vector
    v_x = [1,0,0]
    v_y = [0,1,0]
    v_z = [0,0,1]
    
    
    # define model orientation vector
    m_center_vector = [(center_0[0] - center_1[0]), (center_0[1] -center_1[1]), (center_0[2] - center_1[2])]
    
    
    #compute the rotation matrix that aligns unit vector Z to orientation vector
    R_matrix = rotation_matrix_from_vectors(m_center_vector, v_z)
    
    # rotate the model using rotation matrix to align with unit vector Z 
    pcd_r.rotate(R_matrix, center = (0,0,0))
    

    # check the botttom and top direction 
    pts_bottom = get_pt_sel(np.asarray(pcd_r.points))
    
    

    '''
    ############################################################
    pcd_Z_mask = o3d.geometry.PointCloud()
    
    pcd_Z_mask.points = o3d.utility.Vector3dVector(pts_bottom)
    
    Z_mask_ply = result_path + "Z_mask.ply"
    
    o3d.visualization.draw_geometries([pcd_Z_mask])
    
    o3d.io.write_point_cloud(Z_mask_ply, pcd_Z_mask)
    ############################################################
    '''
    
    
    (ptb_diameter_max, ptb_diameter_min, ptb_diameter) = get_pt_parameter(pts_bottom)
    
    (pt_diameter_max, pt_diameter_min, pt_diameter) = get_pt_parameter(np.asarray(pcd_r.points))
    
    #print(ptb_diameter_max, pt_diameter_max)
    
    

    
    
    # if model bottom and top need to be fliped  
    if ptb_diameter < pt_diameter*0.6:
        
        print("Model was aligned correctly with Z axis\n")
        
    else:
        
        print("Flip model along Z axis\n")
        
        v_z_reverse = [0, 0, -1]
    
        #compute the rotation matrix that aligns unit vector Z to orientation vector
        #R_matrix_flip = rotation_matrix_from_vectors(v_z_reverse, v_z)
        
        R_matrix_flip = pcd_r.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        
        # rotate the model using rotation matrix to align with unit vector Z 
        pcd_r.rotate(R_matrix_flip, center = (0,0,0))
    
    ######################################################################
    
    if adjustment == 0:
        
        print("No manual rotation needed...\n")
        
    else:
        
        # rotate along x for  np.pi/2 * adjustment value
        
        R_adjust = pcd_r.get_rotation_matrix_from_xyz((adjustment*np.pi/2, 0, 0))
        
        #R_adjust = pcd_r.get_rotation_matrix_from_xyz((0, adjustment*np.pi/2, 0))
        
        pcd_r.rotate(R_adjust, center = (0,0,0))
        
        print("Manual rotation {}*np.pi/2...\n".format(adjustment))
        
        
    
    
    # estimate the orientation of 3d model using sliced diameters

    #(pt_plane, pt_plane_center, pt_plane_diameter) = get_pt_sel_parameter(np.asarray(pcd_r.points), n_plane)
    
    #print("pt_plane_diameter =  {} \n".format(pt_plane_diameter))
    
    
    
    '''
    # visualize the result
    ###################################################################
    filter_plane_center = []
    
    for (pt_sel, model_center, dia_value) in zip(pt_plane, pt_plane_center, pt_plane_diameter):
        
        
        pt_sel_filter = copy.deepcopy(pt_sel)
        
        # get OrientedBoundingBox
        #obb = pt_sel.get_oriented_bounding_box()
        
        # assign color for OrientedBoundingBox
        #obb.color = (0, 0, 1)
        
        # get the eight points that define the bounding box.
        #pcd_coord = obb.get_box_points()
        
        # get the model center postion

        points = np.asarray(pt_sel_filter.points)

        # Sphere center and radius
        radius = dia_value*0.5
        
        print("radius =  {} \n".format(radius))

        # Calculate distances to center, set new points
        distances = np.linalg.norm(points - model_center, axis=1)
        pt_sel_filter.points = o3d.utility.Vector3dVector(points[distances <= radius])
        
        pt_sel_filter.paint_uniform_color([1, 0, 0])
        pt_sel.paint_uniform_color([0.8, 0.8, 0.8])

        #display_inlier_outlier(pt_sel, ind, obb)
        
        obb = pt_sel_filter.get_oriented_bounding_box()
        
        # assign color for OrientedBoundingBox
        obb.color = (0, 0, 1)
        
        print("obb.R =  {} \n".format(obb.R))
        
        #rotation_array = obb.R
        
        #print(type(rotation_array))
        
        #print(rotation_array.shape)
        

        rotation_array = obb.R.tolist()

        r = R.from_matrix(rotation_array)
        
        orientation_angle = r.as_euler('xyz', degrees=True)
                   
        print("orientation_angle =  {} \n".format(orientation_angle))
        
        #axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.5, origin = model_center)
        
        #o3d.visualization.draw_geometries([pt_sel, pt_sel_filter, obb, axis])
        
        # get the model center postion
       
        filter_plane_center.append(pt_sel_filter.get_center())
    
    
    
    #print("length of pt_plane_center =  {} \n".format(len(pt_plane_center)))
    
    ####################################################################
    filter_center_points = np.vstack(filter_plane_center)
    
    filter_center_line = []
    
    for i in range(n_plane):
        
        if i+1 < n_plane:
            filter_center_line.append([i, i+1])
    
    
    print(filter_center_line)
    
    #plane_center_line = [[0, 1], [1, 2], [2, 3], [3, 4]]
    
    #print(np_points)
    
    colors_filter = [[1, 0, 0] for i in range(n_plane-1)]
    
    # assign different colors for eight points in the bounding box.
    #colors = []
    #cmap = get_cmap(n_plane-1)
    
    #for idx in range(n_plane-1):
    
        #color_rgb = cmap(idx)[:len(cmap(idx))-1]
        #colors.append(color_rgb)
    
    
    lines_filter_set = o3d.geometry.LineSet()
    lines_filter_set.points = o3d.utility.Vector3dVector(filter_center_points)
    lines_filter_set.lines = o3d.utility.Vector2iVector(filter_center_line)
    lines_filter_set.colors = o3d.utility.Vector3dVector(colors_filter)
    
    
    
    #####################################################################
    plane_center_points = np.vstack(pt_plane_center)
    
    plane_center_line = []
    
    for i in range(n_plane):
        
        if i+1 < n_plane:
            plane_center_line.append([i, i+1])
    
    
    print(plane_center_line)
    
    #plane_center_line = [[0, 1], [1, 2], [2, 3], [3, 4]]
    
    #print(np_points)
    
    #colors = [[0, 1, 0] for i in range(len(plane_center_line))]
    
    # assign different colors for eight points in the bounding box.
    colors = []
    cmap = get_cmap(n_plane-1)
    
    for idx in range(n_plane-1):
    
        color_rgb = cmap(idx)[:len(cmap(idx))-1]
        colors.append(color_rgb)

    
    lines_plane_set = o3d.geometry.LineSet()
    lines_plane_set.points = o3d.utility.Vector3dVector(plane_center_points)
    lines_plane_set.lines = o3d.utility.Vector2iVector(plane_center_line)
    lines_plane_set.colors = o3d.utility.Vector3dVector(colors)
    
    
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(lines_plane_set)
    vis.add_geometry(lines_filter_set)
    vis.add_geometry(pcd_r)
    vis.get_render_option().line_width = 5
    vis.get_render_option().point_size = 1
    vis.run()
    
    '''
    
    '''
    ##########################################################################################
    # visualize the bounding box and center lines
    # get the model center
    center = obb.get_center()
    
    center_pts = []
    
    center_pts.append(center_0)
    center_pts.append(center_1)
    
    center_pts = np.asarray(center_pts)
    
    print(center_pts)
    
    
    
    # define a LineSet with a set of points and a set of edges (pairs of point indices)
    lines = [[0,1],[0,2],[0,3],
            [3,5],[3,6],
            [4,5],[4,6],[4,7],
            [7,1],[7,2],
            [5,2],[6,1]]

    colors = [[1, 0, 0] for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    center_line = [[0,1]]

    colors = [[1, 0, 0] for i in range(len(center_line))]
    
    line_set_center = o3d.geometry.LineSet()
    line_set_center.points = o3d.utility.Vector3dVector(center_pts)
    line_set_center.lines = o3d.utility.Vector2iVector(center_line)
    line_set_center.colors = o3d.utility.Vector3dVector(colors)
    
    

    
    # Creating a mesh of the XYZ axes Cartesian coordinates frame.
    # This mesh will show the directions in which the X, Y & Z-axes point,
    # and can be overlaid on the 3D mesh to visualize its orientation in the Euclidean space.
    # X-axis : Red arrow
    # Y-axis : Green arrow
    # Z-axis : Blue arrow
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.5, origin=[0, 0, 0])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(line_set)
    vis.add_geometry(line_set_center)
    vis.add_geometry(pcd_coord)
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    vis.get_render_option().line_width = 5
    vis.get_render_option().point_size = 10
    vis.run()
    #vis.destroy_window()
    ###################################################################################################
    '''
    

    # return aligned model file
    return pcd_r


    

if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", dest="input", type=str, required=True, help="full path to 3D model file")
    #ap.add_argument("-p", "--path", dest = "path", type = str, required = True, help = "path to 3D model file")
    #ap.add_argument("-ft", "--filetype", dest = "filetype", type = str, required = False, default = 'ply', help = "3D model file filetype, default *.ply")
    ap.add_argument("-o", "--output_path", dest = "output_path", type = str, required = False, help = "result path")
    ap.add_argument("--n_plane", dest = "n_plane", type = int, required = False, default = 5,  help = "Number of planes to segment the 3d model along Z direction")
    ap.add_argument( "--slicing_ratio", dest = "slicing_ratio", type = float, required = False, default = 0.10, help = "ratio of slicing the model from the bottom")
    ap.add_argument( "--adjustment", dest = "adjustment", type = float, required = False, default = 0, help = "model adjustment, 0: no adjustment, 1: rotate np.pi/2, -1: rotate -np.pi/2")
    args = vars(ap.parse_args())

    
    # single input file processing
    ###############################################################################
    if os.path.isfile(args["input"]):

        input_file = args["input"]

        (file_path, filename, basename) = get_file_info(input_file)

        print("Compute {} model orientation and aligning models...\n".format(file_path, filename, basename))

        # result path
        result_path = args["output_path"] if args["output_path"] is not None else file_path

        result_path = os.path.join(result_path, '')

        # print out result path
        print("results_folder: {}\n".format(result_path))
        
        # number of slices for cross-section
        n_plane = args['n_plane']
        
        slicing_ratio = args["slicing_ratio"]

        adjustment = args["adjustment"]

        # start pipeline
        ########################################################################################
        # model alignment 
        pcd_r = model_alignment(input_file, result_path, adjustment)
        
        
        
        ####################################################################
        # write aligned 3d model as ply file format
        # get file information

        #Save model file as ascii format in ply
        result_filename = result_path + basename + '_aligned.ply'

        #write out point cloud file
        o3d.io.write_point_cloud(result_filename, pcd_r, write_ascii = True)

        # check saved file
        if os.path.exists(result_filename):
            print("Converted 3d model was saved at {0}\n".format(result_filename))

        else:
            print("Model file converter failed!\n")


    else:

        print("The input file is missing or not readable!\n")

        print("Exiting the program...")

        sys.exit(0)
    
    
    

    '''
    # batch processing multiple files in a folder
    # batch processing mode 
    # python3 model_alignment.py p ~/example/ -ft ply -o ~/example/result/
    #########################################################################
    
   
    # path to model file 
    file_path = args["path"]
    
    ext = args["filetype"]
    
    files = file_path + '*.' + ext
    
    # number of slices for cross section
    n_plane = args["n_plane"]
    slicing_ratio = args["slicing_ratio"]
    adjustment = args["adjustment"]
    
    
    # obtain model file list
    fileList = sorted(glob.glob(files))

    
    
    # loop processing 
    #######################################################################################
    for input_file in fileList:
        
        if os.path.isfile(input_file):
            
            (file_path, filename, basename) = get_file_info(input_file)

            print("Compute {} model orientation and aligning models...\n".format(file_path, filename, basename))

            # result path
            result_path = args["output_path"] if args["output_path"] is not None else file_path

            result_path = os.path.join(result_path, '')

            # print out result path
            print("results_folder: {}\n".format(result_path))
            
            # number of slices for cross section
            n_plane = args['n_plane']
            
            slicing_ratio = args["slicing_ratio"]
            
            adjustment = args["adjustment"]

            # start pipeline
            ########################################################################################
            # model alignment 
            pcd_r = model_alignment(input_file, result_path, adjustment)
            
            
            
            ####################################################################
            # write aligned 3d model as ply file format
            # get file information

            #Save model file as ascii format in ply
            result_filename = result_path + basename + '_aligned.ply'

            #write out point cloud file
            o3d.io.write_point_cloud(result_filename, pcd_r, write_ascii = True)

            # check saved file
            if os.path.exists(result_filename):
                print("Converted 3d model was saved at {0}\n".format(filename))

            else:
                print("Model file converter failed!\n")
        
        
        else:

            print("The input file is missing or not readable!\n")

            print("Exiting the program...") 

            sys.exit(0)
    
    '''
        

