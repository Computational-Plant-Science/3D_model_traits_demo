"""
Version: 1.0

Summary: Slice 3d model(STL format) into cross section and rasterize a path2d object into a boolean image

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 section_scan.py -p ~/example/ -m test.stl


argument:
("-p", "--path", required=True,    help="path to *.stl model file")
("-m", "--model", required=True,    help="file name")

"""

import trimesh
import numpy as np
from shapely.geometry import LineString

import matplotlib.pylab as plt
import argparse

import os
import sys
import copy
import shutil

import matplotlib
matplotlib.use('agg')


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
        shutil.rmtree(path)
        os.makedirs(path)
        return False
        

def heights_generator(mesh, step, axis):
    """helper function to create heights appropriate for use in
    section_multiplane and mesh_multiplane
    
    Parameters
    _______________
    mesh : mesh object used in multiplane operations
    step : float, desired step size for sections
    axis: integer, 0, 1, or 2, for selecting axis of bounds for heights
    
    Returns
    ______________
    heights : array of heights suitable for use in multiplane methods
    """
    #levels = np.arange(start = mesh.bounds[0][axis], stop = mesh.bounds[1][axis] + step, step = step)
    #heights = levels + abs(mesh.bounds[0][axis])
    
    #heights = levels + (mesh.bounds[0][axis])
    
    z_extents = mesh.bounds[:,axis]
    
    z_levels  = np.arange(*z_extents, step)
    
    #return heights
    
    return z_levels
    


def slice_rasterize(mesh):

    #print("Current model: {0} \n".format(mesh))
    
    print("Current model centroid {0} \n".format(mesh.centroid))
    
    '''
    # facets are groups of coplanar adjacent faces
    # set each facet to a random color
    # colors are 8 bit RGBA by default (n, 4) np.uint8
    for facet in mesh.facets:
        mesh.visual.face_colors[facet] = trimesh.visual.random_color()

    # preview mesh in an opengl window if you installed pyglet with pip
    mesh.show()
    '''
    

    # Take a bunch of parallel slices,
    # slice the mesh into evenly spaced chunks along z
    # this takes the (2,3) bounding box and slices it into [minz, maxz]
    heights = heights_generator(mesh, 0.125, 2)
    
    print(heights)
    
    print(mesh.bounds)
    
    print("Scanning model along Z depth axis with {0} parallel planes...\n".format(len(heights)))

    #print(mesh.bounds[0][2])
    
    #print(mesh.bounds[0])
    
    # construct a set of scanning plane origins
    #plane_origin_array = np.array([mesh.bounds[0],]*len(heights))
    
    plane_origin_array = np.array([mesh.centroid,]*len(heights))
    
    #plane_origin_array = repeat(mesh.centroid[newaxis,:], 3, 0)
   
    plane_origin_array[:,2] = heights
    
    #print(plane_origin_array[0])
    
    fig = plt.figure()
    
    #plt.axes().set_aspect('equal', 'datalim')
    
    plt.axes().set_aspect('equal')

    
    offset_x = abs(mesh.bounds[0][0] - mesh.bounds[1][0]) * 0.15
    offset_y = abs(mesh.bounds[0][1] - mesh.bounds[1][1]) * 0.15
    
    for idx, value in enumerate(plane_origin_array):
     
     
        #print("plane {0} is {1} \n".format(idx))
        # get a single cross section of the mesh
        slice_3d = mesh.section(plane_origin = plane_origin_array[idx], plane_normal = [0,0,1])
        
        print("plane {0} is {1} \n".format(idx, slice_3d))

        
        # the section will be in the original mesh frame
        #slice.show()
        if not (slice_3d is None):
            # we can move the 3D curve to a Path2D object easily
            slice_2D, to_3D = slice_3d.to_planar()
            
            #slice_2D.show()
            #(slice_2D + slice_2D.medial_axis()).show()
            
            #Save 2d slice as image file
            filename = save_path + 'slice_' + str(idx).zfill(4) + '.png'
            
            #for p in slice_2D.polygons_full:
            for p in slice_2D.polygons_closed:
                if not (p is None):
                    plt.plot(*(p.exterior.xy),'k')
                    for r in p.interiors:
                        plt.plot(*zip(*r.coords), 'b')
            
            #plt.axes().set_aspect('equal')
            
            plt.xlim([mesh.bounds[0][0] - offset_x, mesh.bounds[1][0] + offset_x])
            plt.ylim([mesh.bounds[0][1] - offset_y, mesh.bounds[1][1] + offset_y])
            
            plt.axis('off')
                    
            plt.savefig(filename)
            
            plt.cla()
        
    plt.close()
    
    
    
    '''
    figure = plt.figure()
    plt.axes().set_aspect('equal', 'datalim')
    
    
    # hardcode a format for each entity type
    eformat = {'Line0': {'color': 'g', 'linewidth': 1},
           'Line1': {'color': 'y', 'linewidth': 1},
           'Arc0': {'color': 'r', 'linewidth': 1},
           'Arc1': {'color': 'b', 'linewidth': 1},
           'Bezier0': {'color': 'k', 'linewidth': 1},
           'Bezier1': {'color': 'k', 'linewidth': 1},
           'BSpline0': {'color': 'm', 'linewidth': 1},
           'BSpline1': {'color': 'm', 'linewidth': 1}}
    
    
    for idx, value in enumerate(plane_origin_array):
    
        # get a single cross section of the mesh
        slice_3d = mesh.section(plane_origin = plane_origin_array[idx], plane_normal = [0,0,1])
        
        print("plane {0} is {1} \n".format(idx, slice_3d))
        
        if slice_3d:
           # we can move the 3D curve to a Path2D object easily
            slice_2D, to_3D = slice_3d.to_planar()
            
            for entity in slice_2D.entities:
                # if the entity has it's own plot method use it
                if hasattr(entity, 'plot'):
                    entity.plot(slice_2D.vertices)
                    continue
                # otherwise plot the discrete curve
                discrete = entity.discrete(slice_2D.vertices)
                
                
                # a unique key for entities
                e_key = entity.__class__.__name__ + str(int(entity.closed))
                
                
                fmt = eformat[e_key].copy()
                if hasattr(entity, 'color'):
                    # if entity has specified color use it
                    fmt['color'] = entity.color
                plt.plot(*discrete.T, **fmt)
                
                plt.plot(*discrete.T)
                plt.xlim([-2, 2])
                plt.ylim([-2, 2])
                
                plt.axis('off')
                
                #Save 2d slice as image file
                filename = save_path + 'slice_' + str(idx).zfill(4) + '.png'
                plt.savefig(filename)
                
          
    
    plt.close()
    '''
    


if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.stl model file")
    ap.add_argument("-m", "--model", required = True, help = "model file name")
    args = vars(ap.parse_args())


    # setting path to model file 
    current_path = args["path"]
    filename = args["model"]
    file_path = current_path + filename
    
    global save_path

    #make the folder to store the results
    mkpath = current_path + "cross_section_scan"
    mkdir(mkpath)
    save_path = mkpath + '/'
    print ("results_folder: " + save_path + "\n")


    # load the mesh from filename
    # file objects are also supported
    mesh = trimesh.load_mesh(file_path)
    
    print(mesh.bounds)
    
    #Save model
    scaled_mesh_file = save_path + 'scaled.stl'
            
    scaled_mesh = mesh.apply_scale(10.0)
    
    scaled_mesh.vertices -= scaled_mesh.center_mass
    
    scaled_mesh.export(scaled_mesh_file)
    
    print(mesh.bounds)
    
    #mesh.show()
    
    #slice_rasterize(mesh)











