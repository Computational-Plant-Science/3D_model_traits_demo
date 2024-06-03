"""
Version: 1.0

Summary: visualization of two sets of rotation vectors

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

    python3 sphere_vis_downsample.py -p ~/example/quaternion/species_comp/ -v 1


Input:
    *.xlsx

header:
    file_name    Ratio of cluster    quaternion_a    quaternion_b    quaternion_c    quaternion_d    rotVec_rec_0    rotVec_rec_1    rotVec_rec_2    genotype    genotype_label


"""

import glob
from pathlib import Path


from mayavi import mlab
from tvtk.api import tvtk

import argparse
import pandas as pd

import numpy as np 
import plotly
import plotly.graph_objs as go
import plotly.express as px

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import math

import plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff

import random

from scipy import linalg 
from scipy.spatial.transform import Rotation as R

from sklearn.preprocessing import normalize

from statistics import mean 





def euler_to_rotMat(yaw, pitch, roll):
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]])
    Ry_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx_roll = np.array([
        [1,            0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]])
    # R = RzRyRx
    rotMat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
    
    return rotMat


# RPY/Euler angles to Rotation Vector
def euler_to_rotVec(yaw, pitch, roll):

    # compute the rotation matrix
    Rmat = euler_to_rotMat(yaw, pitch, roll)
    
    theta = math.acos(((Rmat[0, 0] + Rmat[1, 1] + Rmat[2, 2]) - 1) / 2)
    sin_theta = math.sin(theta)
    if sin_theta == 0:
        rx, ry, rz = 0.0, 0.0, 0.0
    else:
        multi = 1 / (2 * math.sin(theta))
        rx = multi * (Rmat[2, 1] - Rmat[1, 2]) * theta
        ry = multi * (Rmat[0, 2] - Rmat[2, 0]) * theta
        rz = multi * (Rmat[1, 0] - Rmat[0, 1]) * theta
    return rx, ry, rz


# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation
def averageQuaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = np.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    #return np.real(eigenVectors[:,0])
    return np.ravel(eigenVectors[:,0])
    

def averageVectors(avg_quaternion):
    
    avg_quaternion = avg_quaternion.flatten()

    # get Rotation matrix from quaternion
    rot = R.from_quat(avg_quaternion)

    # get the rotation vector
    avg_rotVec = rot.as_rotvec()
    
    return avg_rotVec



def cMap(x):
    #whatever logic you want for colors
    return [random.random() for i in x]
    
    
    
    
def visualization_rotation_vector(rotVec_rec, data_q_arr, genotype_sub):
    
  
    #####################################################################
    #group quaternion values and rotation vectors by genotypes
    avg_rotVec_list = []
    
    genotype_unique = np.unique(genotype_sub)
    
    #print(genotype_unique)
    

    for idx, genoype_value  in enumerate(genotype_unique):
        
        #print("genotype_ID = {}, genoype_value = {}".format(idx, genoype_value))
        
        index_sel = np.where(genotype_sub == genotype_unique[idx])[0]
    
        print(data_q_arr[index_sel])
        
        # use eigenvalues to compute average of quaternions, The quaternions input are arranged as (w,x,y,z),
        avg_quaternion = averageQuaternions(data_q_arr[index_sel])

        # use components averaging to compute average of quaternions, The quaternions input are arranged as (w,x,y,z),
        #avg_quaternion = ((sum_quaternion.sum(axis=0))/len(vlist_path)).flatten()

        #the signs of the output quaternion can be reversed, since q and -q describe the same orientation
        #avg_quaternion = np.absolute(avg_quaternion)

        avg_quaternion = avg_quaternion.flatten()
        
        avg_rotVec = averageVectors(avg_quaternion)
        
        avg_rotVec_list.append(avg_rotVec)
        
        
   
    print("{} genotypes in total, average roration vectors = {}\n".format(len(genotype_unique), avg_rotVec_list))
    
    ###############################################################################
    # Display a semi-transparent sphere

    mlab.figure("sphere_representation_rotation_vector", size = (800, 800), bgcolor = (0, 0, 0))
    
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
    
    #cmap = matplotlib.cm.get_cmap('viridis')
    
    '''
    # Plot the equator and the tropiques
    theta = np.linspace(0, 2 * np.pi, 100)
    for angle in (- np.pi / 6, 0, np.pi / 6):
        x = np.cos(theta) * np.cos(angle)
        y = np.sin(theta) * np.cos(angle)
        z = np.ones_like(theta) * np.sin(angle)

    mlab.plot3d(x, y, z, color=(1, 1, 1), opacity=0.2, tube_radius=None)
    '''
    
    
    
    cm = plt.get_cmap('turbo')
    
    # Primitives
    #N = len(np.unique(genotype_sub)) 
    
    N = len(rotVec_rec)
    
    print(N)

    # Key point: set an integer for each point
    scalars = genotype_sub

    # Define color table (including alpha), which must be uint8 and [0,255]
    colors = (np.random.random((N, 4))*255).astype(np.uint8)
    colors[:,-1] = 255 # No transparency

    
    zeros = np.zeros(N)
    
    
    #rotVec_rec = np.absolute(rotVec_rec)

    # draw all the rotation vectors in pipeline
    #mlab.quiver3d( 0,0,0, Vec_arr[0], Vec_arr[1], Vec_arr[2], color = current_color)
    sphere = mlab.quiver3d(zeros,zeros,zeros, rotVec_rec[:,0], rotVec_rec[:,1], rotVec_rec[:,2], scalars=scalars, mode = '2ddash')
    
    # Color by scalar
    sphere.glyph.color_mode = 'color_by_scalar' 

    # Set look-up table and redraw
    sphere.module_manager.scalar_lut_manager.lut.table = colors
    
    
    
    
    for idx, avg_rotVec  in enumerate(avg_rotVec_list):
        
        sphere = mlab.quiver3d(0,0,0, avg_rotVec[0], avg_rotVec[1], avg_rotVec[2], color = (1, 0, 0), line_width = 15, mode = '2darrow')
        
    
    '''
    for idx, (Vec_arr, genoype_value)  in enumerate(zip(rotVec_rec, genotype_sub)):
        
        #print(type(genoype_value.item()))
            
        if genoype_value == "LowN":
            current_color = (1, 0, 0)
        else:
            current_color = (0, 1, 0)
            
        #mlab.quiver3d( 0,0,0, Vec_arr[0], Vec_arr[1], Vec_arr[2], color = current_color)
        #pts = mlab.quiver3d(0,0,0, Vec_arr[0], Vec_arr[1], Vec_arr[2], scalars=scalars)
        
        magnitude = linalg.norm(Vec_arr)
        
        #print("genoype_value = {} Vector_length = {}".format(genoype_value, magnitude))
    '''
        
    #rotVec_obj = (current_path + 'rotation_vector_sphere.obj')
    
    #mlab.savefig(rotVec_obj)
    
    mlab.show()
    
    
    
    
    
    
    
if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-ft", "--filetype", required = False, default = 'xlsx', help = "file type")
    ap.add_argument("-v", "--visulize", required = False, type= int, default = 0, help = "Visualize rotation vector or not")
    args = vars(ap.parse_args())

    
    ###################################################################
    
    current_path = args["path"]
    
    file_path = current_path + '*.' + args['filetype']

    # get the absolute path of all Excel files 
    ExcelFiles_list = glob.glob(file_path)
    
    

    
    ####################################################################
    # loop over the list of excel files
    data_q = []
    
    data_v = []
    
    genotype_sub = []
    
    #sample_rate = 100
    
    for f in ExcelFiles_list:
        
        filename = Path(f).name
       
        print("Processing file '{}'...\n".format(filename))
        
        # read the csv file
        data = pd.read_excel(f)
        ###############################################################
        #get downsampled rotation vectors
        #rotVec = np.vstack((data['rotVec_rec_0'],data['rotVec_rec_1'],data['rotVec_rec_2'])).T
        
        cols_vec = ['rotVec_rec_0','rotVec_rec_1','rotVec_rec_2']
        data_v = data[cols_vec].values.tolist()
        

        # downsample along coloum direction, every 10th
        #data_v = np.asarray(data_v)[::sample_rate,:]
        
        data_v_arr = np.asarray(data_v)

        ################################################################
        #get downsampled quarterunion values
        cols_q = ['quaternion_a','quaternion_b','quaternion_c', 'quaternion_d']
        data_q = data[cols_q].values.tolist()
        
        # downsample along coloum direction, every 10th
        #data_q = quarterunionVec[::sample_rate,:]
        #data_q = np.asarray(data_q)[::sample_rate,:]
        
        data_q_arr = np.asarray(data_q)
        
    
        ################################################################
        #get downsampled genotype values
        genotype_sub = data['genotype_label'].values.tolist()

        # downsample along coloum direction, every 10th
        #genotype_sub = genotype_v[::sample_rate,:]
        #genotype_sub = np.asarray(genotype_sub)[::sample_rate]
        
        genotype_sub_arr = np.asarray(genotype_sub)
        

        

    ####################################################################
    # filter small vectors 
    '''
    print(len(data_v))
    
    vector_length = []
    
    for idx, Vec_arr  in enumerate(data_v):
        
        magnitude = linalg.norm(Vec_arr)
        
        vector_length.append(magnitude)
        
        
    avg_vec_len = mean(vector_length)
    
    indices_keep = [idx for idx, value in enumerate(vector_length) if value >= mean(vector_length)*1.2]
    
    print(indices_keep)
    
    data_v_sel = data_v[indices_keep, :]
    
    genotype_sub_sel = genotype_sub[indices_keep]
    
    '''
    

    
    ###########################################################################
    #indices_HighN = np.where(genotype_sub == 'HighN')
    
    #genotype_label[indices_HighN] == 255
    
    data_v_sel = data_v_arr
    genotype_sub_sel = genotype_sub_arr
    
    #print(genotype_sub_sel)
    
    #visualize rotation vectors
    #visualization_rotation_vector(np.asarray(normalized_data_v), genotype_sub)
    
    if args['visulize'] == 1:
        
        visualization_rotation_vector(np.asarray(data_v_sel), np.asarray(data_q_arr), genotype_sub_sel)
    

    ###########################################################################################3
    #print(ExcelFiles_list[0])
    # merge all excel files read all Excel files at once
    df = pd.concat(pd.read_excel(excelFile) for excelFile in ExcelFiles_list)
    
    n_colors = len(np.unique(genotype_sub))
    
    if n_colors < 2:
        markercolor = px.colors.sample_colorscale("turbo", [n/(n_colors) for n in range(n_colors)])
    else:
        markercolor = px.colors.sample_colorscale("turbo", [n/(n_colors -1) for n in range(n_colors)])
    
    
    fig = px.scatter_3d(df, x='quaternion_b', y='quaternion_c', z='quaternion_d', color='genotype',   size_max = 20, opacity = 1.0)

    #fig = px.scatter_3d(df, x='quaternion_b', y='quaternion_c', z='quaternion_d', color='genotype',  color_discrete_sequence=markercolor,  symbol='genotype',  size = 'label', size_max = 20, opacity = 1.0)

    fig.update_traces(marker_size = 4)
    
    
    #fig.add_scatter3d(x=data2['quaternion_b'], y=data2['quaternion_c'], z=data2['quaternion_d'])

    #fig.add_scatter3d(df, x='quaternion_b', y='quaternion_c', z='quaternion_d', color='genotype',  color_discrete_sequence = markercolor)

    #fig.update_traces(marker_size = 10)
    
    
    quaternion_4D = (current_path + 'avg_quaternion_4D.html')
    
    plotly.offline.plot(fig, auto_open=False, filename=quaternion_4D)

    
    
    
    ######################################################################
    fig = px.scatter_3d(df, x='rotVec_rec_0', y='rotVec_rec_1', z='rotVec_rec_2', color='genotype', size_max = 20, opacity = 1.0)
    
    fig.update_traces(marker_size = 4)
    
    quaternion_4D = (current_path + 'avg_rotation_vector.html')
    
    plotly.offline.plot(fig, auto_open=False, filename=quaternion_4D)
    
    



    

