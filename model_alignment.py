"""
Version: 1.5

Summary: alignment 3d model to Z axis and translate it to its center.

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 model_alignment.py -p ~/example/ -m test.ply -r 0.1


argument:
("-p", "--path", required=True,    help="path to *.ply model file")
("-m", "--model", required=True,    help="file name")


output:
*.xyz: xyz format file only has 3D coordinates of points 
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

from scipy.spatial.transform import Rotation as Rot
import math



def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


#compute angle between two vectors(works for n-dimensional vector),
def dot_product_angle(v1,v2):

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        
        print("Zero magnitude vector!")
        
        return 0
        
    else:
        vector_dot_product = np.dot(v1,v2)
        
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        
        if np.degrees(arccos) > 90:
            
            angle = np.degrees(arccos) - 90
        else:
            
            angle = np.degrees(arccos)
    
    return angle
        


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
    
    color_array = np.asarray(pcd.colors)
    
    #print(color_array.shape)
    
    #color_array[:,2] = 0.24
    
    #pcd.colors = o3d.utility.Vector3dVector(color_array)
    
    #o3d.visualization.draw_geometries([pcd])
    
    #pcd.points = o3d.utility.Vector3dVector(points)

    # threshold data

    pcd_sel = pcd.select_by_index(np.where(color_array[:, 2] > ratio)[0])
    
    #o3d.visualization.draw_geometries([pcd])
    #o3d.visualization.draw_geometries([pcd_sel])


    # copy original point cloud for rotation
    pcd_r = copy.deepcopy(pcd_sel)
    
    
    # get the model center postion
    model_center = pcd_r.get_center()
    
    # geometry points are translated directly to the model_center position
    pcd_r.translate(-1*(model_center))
    
    
    aabb = pcd_r.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)

    # get OrientedBoundingBox
    obb = pcd_r.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    
    #o3d.visualization.draw_geometries([pcd_r, obb])
    
    #print("Oriented Bounding Box center is: {}\n".format(obb.center))
    
    
    #compute convex hull of a point cloud, the smallest convex set that contains all points
    
    hull, _ = pcd_r.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    #o3d.visualization.draw_geometries([pcd_r, hull_ls, aabb, obb, o3d.geometry.TriangleMesh.create_coordinate_frame()])
    
    #print(hull.get_volume())
    
    #obb.R @ np.array([1,0,0]).reshape(3,1)
    original_rotation = copy.deepcopy(obb.R)
    
    print("original_rotation is: {}\n".format(original_rotation))
    
    #from scipy.spatial.transform import Rotation as Rot

    #q = R.from_matrix(original_rotation)
    
    #print(q)
    
    obb_x = obb.R @ np.array([1,0,0])
    obb_y = obb.R @ np.array([0,1,0])
    obb_z = obb.R @ np.array([0,0,1])

    
    #print("obb_x, obb_y, obb_z is: {} {} {}\n".format(obb_x, obb_y, obb_z))
   
    #define unit vector
    v_x = [1,0,0]
    v_y = [0,1,0]
    v_z = [0,0,1]
    
    
    #Find the rotation matrix that aligns vec1 to vec2
    R_matrix = rotation_matrix_from_vectors(obb_z, v_z)
    
    
    
    #compute rotation angle
    angle_x = dot_product_angle(obb_x, v_x)
    angle_y = dot_product_angle(obb_y, v_y)
    angle_z = dot_product_angle(obb_z, v_z)
    
    print("angle_x = {0} angle_y = {1} angle_z = {2}\n".format(angle_x, angle_y, angle_z))
    
    # test setup
    #R = pcd_r.get_rotation_matrix_from_xyz((-1* math.radians(angle_x), 0, 0))
    
    #R = pcd_r.get_rotation_matrix_from_xyz((0, 1* math.radians(angle_y), 0))
    
    # copy original point cloud for rotation
    #pcd_r_ori = copy.deepcopy(pcd_r)
    
    #R = pcd_r.get_rotation_matrix_from_xyz((-1* math.radians(angle_x), -1* math.radians(angle_y), 0))
    
    
   
    
    #r = Rot.from_euler('xyz', [-1*angle_x, -1*angle_y, -1*angle_z], degrees=True)
    
    # Apply rotation transformation to copied point cloud data
    #print(r.as_quat())
    
    #R = pcd_r.get_rotation_matrix_from_quaternion((r.as_quat()))
    
    # Apply rotation transformation to copied point cloud data
    #R_matrix = rotation_matrix_from_vectors(obb_x, v_x)
    
    pcd_r.rotate(R_matrix, center = (0,0,0))
    

    R = pcd.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
    
    pcd_r.rotate(R, center = (0,0,0))
    
    
    
    '''
    r = Rot.from_euler('xyz', [90, 0, 0], degrees=True)
    
    print(r.as_quat())
    
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    
    mesh_r = copy.deepcopy(mesh)
    
    #R = mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, np.pi/4))
    
    R = mesh.get_rotation_matrix_from_quaternion((r.as_quat()))
    
    mesh_r.rotate(R, center=(0,0,0))
    
    o3d.visualization.draw_geometries([mesh_r])
    
    #o3d.visualization.draw_geometries([pcd_r, o3d.geometry.TriangleMesh.create_coordinate_frame()])
    '''
    
    
    # test setup
    #R = pcd.get_rotation_matrix_from_quaternion((-np.pi/2, 0, 0))


    # define rotation matrix test setup
    #R = pcd.get_rotation_matrix_from_xyz((-np.pi/2, 0, np.pi*angle_z/90))
    
    
    #R = pcd.get_rotation_matrix_from_xyz((-np.pi/2, -np.pi/2 + np.pi/8, 0))
    
    # normal setup
    #R = pcd.get_rotation_matrix_from_xyz((0, -np.pi/2 - 1*np.pi/8, 0))
    '''
    # test setup
    R = pcd.get_rotation_matrix_from_xyz((0, -np.pi* (-90 + angle_z)/180, 0))
        
    # Apply rotation transformation to copied point cloud data
    pcd_r.rotate(R, center = (0,0,0))
    
    # get the model center postion
    model_center = pcd_r.get_center()
    
    # geometry points are translated directly to the model_center position
    pcd_r.translate(-1*(model_center))
    
    '''
    # Statistical oulier removal
    #nb_neighbors, which specifies how many neighbors are taken into account in order to calculate the average distance for a given point.
    #std_ratio, which allows setting the threshold level based on the standard deviation of the average distances across the point cloud. 
    #The lower this number the more aggressive the filter will be.
    
    
    # visualize the oulier removal point cloud
    #print("Statistical oulier removal\n")
    #cl, ind = pcd_r.remove_statistical_outlier(nb_neighbors = 100, std_ratio = 0.00001)
    #display_inlier_outlier(pcd_r, ind)
    
    
    '''
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
    
    #print("Statistical oulier removal\n")
    #cl, ind = pcd_r.remove_statistical_outlier(nb_neighbors = 40, std_ratio = 0.00001)
    #display_inlier_outlier(pcd_r, ind)
   
    
    ####################################################################
    
    #Save model file as ascii format in ply
    filename = current_path + base_name + '_aligned.ply'
    
    #write out point cloud file
    o3d.io.write_point_cloud(filename, pcd_r, write_ascii = True)
    
    #Save modelfilea as ascii format in xyz
    filename = current_path + base_name + '.xyz'
    o3d.io.write_point_cloud(filename, pcd_r, write_ascii = True)
    
    
    # check saved file
    if os.path.exists(filename):
        print("Converted 3d model was saved at {0}\n".format(filename))
        return True
    else:
        return False
        print("Model file converter failed!\n")
        sys.exit(0)
    

if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m", "--model", required = True, help = "model file name")
    ap.add_argument("-a", "--angle", required = False, default = -90, help = "rotation_angle")
    ap.add_argument("-r", "--ratio", required = False, type = float, default = 0.1, help = "outlier remove ratio")
    args = vars(ap.parse_args())


    # setting path to model file 
    current_path = args["path"]
    filename = args["model"]
    ratio = args["ratio"]
    file_path = current_path + filename
    
    #rotation_angle = args["angle"]

    #print ("results_folder: " + current_path)

    format_converter(current_path, filename)

 
