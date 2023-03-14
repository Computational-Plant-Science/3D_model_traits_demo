"""
Version: 1.5

Summary: alignment 3d model to Z axis and translate it to its center.

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 model_alignment_syn.py -p ~/example/ -m test.ply


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

import pathlib

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
        
        if np.degrees(arccos) > 180:
            
            angle = np.degrees(arccos) - 180
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


def rotation_angle_matrix(pcd_r):
    
    
    aabb = pcd_r.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)

    # get OrientedBoundingBox
    obb = pcd_r.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    
    #o3d.visualization.draw_geometries([pcd_r, obb])
    
    #print("Oriented Bounding Box center is: {}\n".format(obb.center))
    
    
    #compute convex hull of a point cloud, the smallest convex set that contains all points
    
    #hull, _ = pcd_r.compute_convex_hull()
    #hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    #hull_ls.paint_uniform_color((1, 0, 0))
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
    
    return R_matrix, angle_x, angle_y, angle_z
    


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
    
    
    #filename = current_path + base_name + '_binary.ply'
    
    #o3d.io.write_point_cloud(filename, pcd)
    
    #pcd = o3d.io.read_point_cloud(filename)
    
    #print(np.asarray(pcd.points))
    
    #visualize the original point cloud
    #o3d.visualization.draw_geometries([pcd])
    
    Data_array = np.asarray(pcd.points)
    
    color_array = np.asarray(pcd.colors)
    
    #print(len(color_array))
    
    
    
    #color_array[:,2] = 0.24
    
    #pcd.colors = o3d.utility.Vector3dVector(color_array)
    
    #o3d.visualization.draw_geometries([pcd])
    
    #pcd.points = o3d.utility.Vector3dVector(points)

    # threshold data
    
    if len(color_array) == 0:
        
        pcd_sel = pcd
    else:
        pcd_sel = pcd.select_by_index(np.where(color_array[:, 2] > ratio)[0])
    
    #o3d.visualization.draw_geometries([pcd])
    #o3d.visualization.draw_geometries([pcd_sel])


    # copy original point cloud for rotation
    pcd_r = copy.deepcopy(pcd_sel)
    
    
    # get the model center postion
    model_center = pcd_r.get_center()
    
    # geometry points are translated directly to the model_center position
    pcd_r.translate(-1*(model_center))
    
    ################################################################

    (R_matrix, angle_x, angle_y, angle_z) = rotation_angle_matrix(pcd_r)
    
    pcd_r.rotate(R_matrix, center = (0,0,0))
    
    
    print("angle_x = {0} angle_y = {1} angle_z = {2}\n".format(angle_x, angle_y, angle_z))
    
    
    '''
    R = pcd.get_rotation_matrix_from_xyz((angle_x*np.pi/180, 0, 0))
        
    pcd_r.rotate(R, center = (0,0,0))
    
    
    R = pcd.get_rotation_matrix_from_xyz((0, angle_y*np.pi/180 - np.pi/2, 0))
        
    pcd_r.rotate(R, center = (0,0,0))
    
    
    R = pcd.get_rotation_matrix_from_xyz((0, 0, angle_z*np.pi/180 ))
        
    pcd_r.rotate(R, center = (0,0,0))
    '''
    
    
    if int(args["test"]) == 1:
        
        print("test mode was on: {}\n".format(args["test"]))
        
        R = pcd.get_rotation_matrix_from_xyz((-np.pi/2 + np.pi/8, 0, 0))
        
        pcd_r.rotate(R, center = (0,0,0))
        
    else:
        
        print("test mode was off: {}\n".format(args["test"]))

        R = pcd.get_rotation_matrix_from_xyz((0, rotation_angle, 0))
        
        pcd_r.rotate(R, center = (0,0,0))
            
        ###################################################################
        #check aligned model
        
        (R_matrix_al, angle_x_al, angle_y_al, angle_z_al) = rotation_angle_matrix(pcd_r)
        
        
        print("angle_x_al = {0} angle_y_al = {1} angle_z_al = {2}\n".format(angle_x_al, angle_y_al, angle_z_al))
        
        
        if angle_y_al < 45:
            
            if angle_y_al < 2:
                print("A")
                R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al) , 0, 0))
                
            elif angle_y_al < 7:
                print("B0")
                R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al) , 0, 0))
           
            elif angle_y_al < 12:
                print("B1")
                R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al) - np.pi, 0, 0))
            
            elif angle_y_al < 15:
                print("B2")
                R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al), 0, 0))
            
            elif angle_y_al < 20:
                print("C0")
                R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al), 0, 0))
                
            elif angle_y_al < 30:
                print("C1")
                R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al) - np.pi/4, 0, 0))
            
            elif angle_y_al < 40:
                print("C2")
                R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al) , 0, 0))
                
            elif angle_y_al < 45:
                print("C3")
                R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al) , 0, 0))
            else:
                print("D")
                R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al), 0, 0))
            
        elif angle_y_al < 90:
            
            #if abs(angle_y_al - 90) < abs(180 - angle_y_al):

            if abs(angle_y_al - 45) < abs(90 - angle_y_al):
                
                if abs(angle_z - 90) < abs(180 - angle_z):
                    
                    if abs(angle_z - 90)  < 5:
                        print("E")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi/2 , 0, 0))
                    elif abs(angle_z - 90)  < 35:
                        print("F")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi/2, 0, 0))
                    elif abs(angle_z - 90)  < 45:
                        print("G")
                        #R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi, -np.pi/2 + np.pi/4, -np.pi/2 + np.pi/4))
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) + np.pi/2 + np.pi/4, 0, 0))
                    else:
                        print("H")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi, 0, 0))
                else:
                    #print("&&&& abs(angle_z - 90) = {} \n".format(abs(angle_z - 90)))

                    if angle_y_al < 50:
                        print("I")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) + np.pi , 0, 0))
                    elif angle_y_al < 55:
                        print("J")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi , 0, 0))
                    elif angle_y_al < 70:
                        print("K")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi - np.pi/4, 0, 0))
                    else:
                        print("L")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi/2, 0, 0))
            else:
                if abs(angle_z - 90) < abs(180 - angle_z):
                    
                    if abs(angle_z - 90) < 5 :
                        print("M")
                        R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al) - 0*np.pi/2 , 0, 0))
                    elif abs(angle_z - 90) < 10 :
                        print("N")
                        R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al), 0, 0))
                    elif abs(angle_z - 90) < 15 :
                        print("O")
                        R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al) + np.pi/8, 0, 0))
                    else:
                        print("P")
                        R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al) - np.pi + np.pi/4, 0, 0))
                else:
                    if abs(angle_z - 90) > 60 :
                        print("Q")
                        #R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al)  - np.pi/2 - np.pi/4, 0, 0))
                        R = pcd.get_rotation_matrix_from_xyz((1*math.radians(angle_y_al)  - np.pi/2 - np.pi/4 + np.pi, 0, 0))
                    else:
                        print("R")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) , 0, 0))
            
        elif angle_y_al < 135 :
            
            if abs(angle_y_al - 90) < abs(180 - angle_y_al):
                
                if abs(angle_z - 90) < abs(180 - angle_z):

                    if abs(angle_z - 90) < 6:
                        print("S")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) + np.pi, 0, 0))

                    elif abs(angle_z - 90) < 45:
                        print("T")
                        #R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) + np.pi/2, -np.pi/2 + np.pi/4, -np.pi/2 + np.pi/4))
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) + 0*np.pi/2 - np.pi , 0, 0))
                    
                    else:
                        print("U")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) + np.pi/2 , 0, 0))
                        
                else:
                    
                    if abs(angle_z - 90) < 55:
                        print("V")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) + np.pi + np.pi/4, 0, 0))
                        
                    elif abs(angle_z - 90) < 70:
                        print("W")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi/2, 0, 0))
                    else:
                        print("X")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) + np.pi, 0, 0))
                    
            else:
                print("Y")
                R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi , 0, 0))

        else:
            
            if abs(angle_y_al - 90) < abs(180 - angle_y_al):
                
                print("Z")
            
                R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi/2 - np.pi/2, 0, 0))
            
            else:
                
                if abs(angle_z - 90) > abs(180 - angle_z):
                    
                    
                    R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi/2, 0, 0))
                    
                    if abs(180 - angle_z) < 15:
                        
                        print("AA")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi, 0, 0))
                    
                    elif abs(180 - angle_z) < 24:
                        print("BB")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi/2 + np.pi/4, 0, 0))
                    
                    elif abs(180 - angle_z) < 30:
                        
                        print("CC")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) + np.pi/2 , 0, 0))
                    
                    elif abs(180 - angle_z) < 40:
                        
                        print("DD")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) , 0, 0))
                        
                    else:
                        print("EE")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi/2 + np.pi/4, 0, 0))
                else:
                    
                    if abs(180 - angle_z) < 55:
                        print("FF")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi/2 - np.pi/2, 0, 0))
                    
                    elif abs(180 - angle_z) < 75:
                        print("GG")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) + np.pi, 0, 0))
                    
                    elif abs(180 - angle_z) < 95:
                        print("HH")
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi, 0, 0))
                    
                    elif abs(180 - angle_z) < 170:
                        print("II")
                        #R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi - np.pi/4, 0, 0))
                        
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) - np.pi, 0, 0))
                    else:
                        print("JJ")
                        #print("abs(180 - angle_z) = {} \n".format(abs(180 - angle_z)))
                        R = pcd.get_rotation_matrix_from_xyz((-1*math.radians(angle_y_al) + np.pi/2 + np.pi/4, 0, 0))
             
       
        pcd_r.rotate(R, center = (0,0,0))
        
        
    
    ###################################################################
    
    
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
    ap.add_argument("-m", "--model", required = False, help = "model file name")
    ap.add_argument("-a", "--angle", required = False, type = int, default = 1, help = "rotation_angle")
    ap.add_argument("-r", "--ratio", required = False, type = float, default = 0.01, help = "outlier remove ratio")
    ap.add_argument("-t", "--test", required = False, type = int, default = 0, help = "if using test setup")
    args = vars(ap.parse_args())


    # setting path to model file 
    current_path = args["path"]
    
    rotation_angle = args["angle"]*np.pi/2
    ratio = args["ratio"]
    
    
    #path = pathlib.PurePath(current_path)
    
    if args["model"] is None:
        
        filename = pathlib.PurePath(current_path).name + ".ply"
        
        print("Default file name is {}".format(filename))
    
    else:
        
        filename = args["model"]
    
    
    
    file_path = current_path + filename
    
    #rotation_angle = args["angle"]

    #print ("results_folder: " + current_path)

    format_converter(current_path, filename)

 
