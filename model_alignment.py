"""
Version: 1.5

Summary: compute the cross section plane based on 3d model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 model_alignment.py -p ~/example/ -m test.ply


argument:
("-p", "--path", required=True,    help="path to *.ply model file")
("-m", "--model", required=True,    help="file name")

"""
#!/usr/bin/env python



# import the necessary packages
import numpy as np 
import argparse

import os
import sys
import open3d as o3d
import copy



def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])



def format_converter(current_path, model_name):
    
    model_file = current_path + model_name
    
    if os.path.isfile(model_file):
        print("Converting file format for 3D point cloud model {}...\n".format(model_name))
    else:
        print("File not exist")
        sys.exit()
    
        
    abs_path = os.path.abspath(model_file)
    filename, file_extension = os.path.splitext(abs_path)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
     
    # Pass xyz to Open3D.o3d.geometry.PointCloud 

    pcd = o3d.io.read_point_cloud(model_file)
    
    
    #print(np.asarray(pcd.points))
    
    #visualize the original point cloud
    #o3d.visualization.draw_geometries([pcd])
    
    Data_array = np.asarray(pcd.points)
    
    
    #Normalize data

    
    # copy original point cloud for rotation
    pcd_r = copy.deepcopy(pcd)
    
    # define rotation matrix
    R = pcd.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
    
    #R = pcd.get_rotation_matrix_from_xyz((0, np.pi/4, 0))
    
    #R = pcd.get_rotation_matrix_from_xyz((0, -np.pi/2, 0))
    
    # Apply rotation transformation to copied point cloud data
    pcd_r.rotate(R, center = (0,0,0))
    
    # get the model center postion
    model_center = pcd_r.get_center()
    
    # geometry points are translated directly to the model_center position
    pcd_r.translate(-1*(model_center))
    
    # Statistical oulier removal
    #nb_neighbors, which specifies how many neighbors are taken into account in order to calculate the average distance for a given point.
    #std_ratio, which allows setting the threshold level based on the standard deviation of the average distances across the point cloud. 
    #The lower this number the more aggressive the filter will be.
    
    '''
    # visualize the oulier removal point cloud
    print("Statistical oulier removal")
    cl, ind = pcd_r.remove_statistical_outlier(nb_neighbors = 40, std_ratio = 0.001)
    display_inlier_outlier(pcd_r, ind)
    
    #builds a KDTree from point cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_r)
    
    #paint the 1500th point red.
    pcd_r.colors[1500] = [1, 0, 0]
    
    #Find its 200 nearest neighbors, and paint them blue.
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd_r.points[1500], 200)
    np.asarray(pcd_r.colors)[idx[1:], :] = [0, 0, 1]
    
    print("Find its neighbors with distance less than 0.2, and paint them green.")
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd_r.points[1500], 0.2)
    np.asarray(pcd_r.colors)[idx[1:], :] = [0, 1, 0]


    # Visualize rotated point cloud 
    o3d.visualization.draw_geometries([pcd_r])

    # get convex hull of a point cloud is the smallest convex set that contains all points.
    hull, _ = pcd_r.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    #visualize the convex hull as a red LineSet
    #o3d.visualization.draw_geometries([pcd_r, hull_ls])
    
    # get AxisAlignedBoundingBox
    aabb = pcd_r.get_axis_aligned_bounding_box()
    aabb.color = (0, 1, 0)
    aabb_height = aabb.get_extent()
    
    print(aabb_height)
    
    # get OrientedBoundingBox
    obb = pcd_r.get_oriented_bounding_box()
    obb.color = (1, 0, 0)
    o3d.visualization.draw_geometries([pcd_r, aabb, hull_ls])
    
    
    #Voxel downsampling uses a regular voxel grid to create a uniformly downsampled point cloud from an input point cloud
    #print("Downsample the point cloud with a voxel of 0.05")
    #downpcd = pcd_r.voxel_down_sample(voxel_size=0.5)
    #o3d.visualization.draw_geometries([downpcd])
    '''
    
    #Save model file as ascii format in ply
    filename = current_path + base_name + '_aligned.ply'
    
    #write out point cloud file
    o3d.io.write_point_cloud(filename, pcd_r, write_ascii = True)
    
    #Save modelfilea as ascii format in xyz
    filename = current_path + base_name + '.xyz'
    o3d.io.write_point_cloud(filename, pcd_r, write_ascii = True)
    
    
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

 
