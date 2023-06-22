"""
Version: 1.0

Summary: visualization of two sets of rotation vectors

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

    python3 compare_distribution.py -p ~/example/quaternion/species_comp/ -v 1 -tq


Input:
    *.xlsx

header:
    file_name    Ratio of cluster    quaternion_a    quaternion_b    quaternion_c    quaternion_d    rotVec_rec_0    rotVec_rec_1    rotVec_rec_2    genotype    genotype_label
    'quaternion_Mahalanobis','quaternion_p','rotVec_Mahalanobis', 'rotVec_p'


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


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import math

import plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff

import random

from scipy import linalg 
from scipy.spatial.transform import Rotation as R

from scipy.stats import kstest

from sklearn.preprocessing import normalize

from statistics import mean 

import seaborn as sns



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
    
    
    
    
def visualization_rotation_vector(rotVec_rec, data_q_arr, genotype_label):
    
  
    #####################################################################
    #group quaternion values and rotation vectors by genotypes
    avg_rotVec_list = []
    
    genotype_unique = np.unique(genotype_label)
    
    #print(genotype_unique)
    

    for idx, genoype_value  in enumerate(genotype_unique):
        
        #print("genotype_ID = {}, genoype_value = {}".format(idx, genoype_value))
        
        index_sel = np.where(genotype_label == genotype_unique[idx])[0]
    
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
    #N = len(np.unique(genotype_label)) 
    
    N = len(rotVec_rec)
    
    print(N)

    # Key point: set an integer for each point
    scalars = genotype_label

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
    for idx, (Vec_arr, genoype_value)  in enumerate(zip(rotVec_rec, genotype_label)):
        
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
    ap.add_argument("-v", "--visualize", required = False, type= int, default = 0, help = "Visualize rotation vector or not")
    ap.add_argument("-tq", "--type_quaternion", required = False, type = int, default = 0, help = "analyze quaternion type, average_quaternion=0, composition_quaternion=1, diff_quaternion=2, distance_quaternion=3")

    args = vars(ap.parse_args())

    
    ###################################################################
    
    current_path = args["path"]
    
    file_path = current_path + '*.' + args['filetype']
    
    type_quaternion = args["type_quaternion"]

    # get the absolute path of all Excel files 
    ExcelFiles_list = glob.glob(file_path)
    
    

    
    ####################################################################
    # loop over the list of excel files
    data_q = []
    
    data_v = []
    
    genotype_label = []
    
    #sample_rate = 100
    
    for f in ExcelFiles_list:
        
        filename = Path(f).name
        
        base_name = filename.replace(".xlsx", "")
       
        print("Processing file '{}'...\n".format(filename))
        
        # read the csv file
        df = pd.read_excel(f)
        ###############################################################
        #get downsampled rotation vectors
        #rotVec = np.vstack((data['rotVec_rec_0'],data['rotVec_rec_1'],data['rotVec_rec_2'])).T
        
        if type_quaternion == 0:
            cols_vec = ['rotVec_avg_0','rotVec_avg_1','rotVec_avg_2']
        elif type_quaternion == 1:
            cols_vec = ['rotVec_composition_0','rotVec_composition_1','rotVec_composition_2']
        elif type_quaternion == 2:
            cols_vec = ['rotVec_diff_0','rotVec_diff_1','rotVec_diff_2']
        elif type_quaternion == 3:
            cols_vec = ['rotVec_avg_0','rotVec_avg_1','rotVec_avg_2']
        

        data_v = df[cols_vec].values.tolist()
        

        # downsample along coloum direction, every 10th
        #data_v = np.asarray(data_v)[::sample_rate,:]
        
        data_v_arr = np.asarray(data_v)

        ################################################################
        #get quarterunion values
        
        if type_quaternion == 0:
            cols_q = ['quaternion_a','quaternion_b','quaternion_c', 'quaternion_d']
        elif type_quaternion == 1:
            cols_q = ['composition_quaternion_a','composition_quaternion_b','composition_quaternion_c', 'composition_quaternion_d']
        elif type_quaternion == 2:
            cols_q = ['diff_quaternion_a','diff_quaternion_b','diff_quaternion_c', 'diff_quaternion_d']
        elif type_quaternion == 3:
            cols_q = ['distance_absolute','distance_intrinsic', 'distance_symmetrized']
        
        data_q = df[cols_q].values.tolist()
        
        # downsample along coloum direction, every 10th
        #data_q = quarterunionVec[::sample_rate,:]
        #data_q = np.asarray(data_q)[::sample_rate,:]
        
        data_q_arr = np.asarray(data_q)
        
    
        ################################################################
        #get downsampled genotype values
        genotype_label = df['genotype_label'].values.tolist()

        # downsample along coloum direction, every 10th
        #genotype_label = genotype_v[::sample_rate,:]
        #genotype_label = np.asarray(genotype_label)[::sample_rate]
        
        genotype_label_arr = np.asarray(genotype_label)
        
        
        
        ################################################################
        genotype_type = df['genotype'].values.tolist()
        
        genotype_type_arr = np.asarray(genotype_label)
        
        genotype_unique = list(set(genotype_type))
        
        print("Genotypes are {} , {}\n".format(genotype_unique[0], genotype_unique[1]))
        
        ################################################################
        #get downsampled quarterunion values
        cols_ma = ['quaternion_Mahalanobis','quaternion_p','rotVec_Mahalanobis', 'rotVec_p']
        data_ma = df[cols_ma].values.tolist()
        
        data_ma_arr = np.asarray(data_ma)
        
        
        #The center of the box represents the median while the borders represent the first (Q1) and third quartile (Q3), respectively. 
        #sns.boxplot(data = data, x = 'quaternion_Mahalanobis', y = 'genotype')
        #plt.title("Boxplot")

        #histogram groups the data into equally wide bins and plots the number of observations within each bin
        #sns.histplot(data = df, y = 'quaternion_Mahalanobis', hue = 'genotype', bins=50)
        #plt.title("Histogram")


        
        #Density Histogram
        fig = plt.plot(figsize =(10, 7), tight_layout = True)
        sns.histplot(data = df, x = 'quaternion_Mahalanobis', hue = 'genotype', bins=50, stat='density', common_norm=False)
        plt.title("Density Histogram")
        
        result_path = current_path + base_name + '_Density_his.png'
        print("result file was saved as {}\n".format(result_path))
        plt.savefig(result_path, bbox_inches='tight', dpi=1000)
        plt.close()
        
        
        #Kernel Density
        fig = plt.plot(figsize =(10, 7), tight_layout = True)
        sns.kdeplot(x='quaternion_Mahalanobis', data=df, hue='genotype', common_norm=False)
        plt.title("Kernel Density Function")
        result_path = current_path + base_name + '_Kernel_Density.png'
        print("result file was saved as {}\n".format(result_path))
        plt.savefig(result_path, bbox_inches='tight', dpi=1000)
        plt.close()
        
        #sns.histplot(x='quaternion_Mahalanobis', data=df, hue='genotype', bins=len(df), stat="density", element="step", fill=False, cumulative=True, common_norm=False);
        #plt.title("Cumulative distribution function")


        
        
        
        q_m = df['quaternion_Mahalanobis'].values
        
        
        
        q_type_1 = df.loc[df.genotype==genotype_unique[0], 'quaternion_Mahalanobis'].values
        q_type_2 = df.loc[df.genotype==genotype_unique[1], 'quaternion_Mahalanobis'].values
        
        stat, p_value = kstest(q_type_1, q_type_2)
        print(f" Kolmogorov-Smirnov Test: statistic={stat:.4f}, p-value={p_value:.4f}")

        #################################################################
        #qq plot
        #q stands for quantile. The Q-Q plot plots the quantiles of the two distributions against each other. 
        #If the distributions are the same, we should get a 45-degree line.
        
        df_pct = pd.DataFrame()
        df_pct[genotype_unique[0]] = np.percentile(q_type_1, range(100))
        df_pct[genotype_unique[1]] = np.percentile(q_type_2, range(100))
        
        
        fig = plt.plot(figsize =(10, 7), tight_layout = True)
        plt.scatter(x=genotype_unique[0], y=genotype_unique[1], data=df_pct, label='Actual fit');
        sns.lineplot(x=genotype_unique[0], y=genotype_unique[0], data=df_pct, color='r', label='Line of perfect fit')
        plt.xlabel('Quantile of quaternion_Mahalanobis' + genotype_unique[0])
        plt.ylabel('Quantile of quaternion_Mahalanobis' + genotype_unique[1])
        plt.legend()
        plt.title("QQ plot")
        
        result_path = current_path + base_name + '_QQ_plot.png'
        print("result file was saved as {}\n".format(result_path))
        plt.savefig(result_path, bbox_inches='tight', dpi=1000)
        plt.close()
        
        ###############################################################
        #Kolmogorov-Smirnov Test
        # If the p-value is below 5%: we reject the null hypothesis that the two distributions are the same, with 95% confidence.
        
        df_ks = pd.DataFrame()
        
        df_ks['quaternion_Mahalanobis'] = np.sort(df['quaternion_Mahalanobis'].unique())
        
        
        df_ks[genotype_unique[0]] = df_ks['quaternion_Mahalanobis'].apply(lambda x: np.mean(q_type_1<=x))
        
        df_ks[genotype_unique[1]] = df_ks['quaternion_Mahalanobis'].apply(lambda x: np.mean(q_type_2<=x))
        
        print(df_ks.head())
        
        
        k = np.argmax( np.abs(df_ks[genotype_unique[0]] - df_ks[genotype_unique[1]]))
        ks_stat = np.abs(df_ks[genotype_unique[1]][k] - df_ks[genotype_unique[0]][k])
        
        y = (df_ks[genotype_unique[1]][k] + df_ks[genotype_unique[0]][k])/2
        
        
        # Creating histogram
        fig = plt.plot(figsize =(10, 7), tight_layout = True)
        
        plt.plot('quaternion_Mahalanobis', genotype_unique[0], data=df_ks, label=genotype_unique[0])
        plt.plot('quaternion_Mahalanobis', genotype_unique[1], data=df_ks, label=genotype_unique[1])
        plt.errorbar(x=df_ks['quaternion_Mahalanobis'][k], y=y, yerr=ks_stat/2, color='k', capsize=5, mew=3, label=f"Test statistic: {ks_stat:.4f}, p_value {p_value:.4f}")
        plt.legend(loc='center right')
        plt.title("Kolmogorov-Smirnov Test")
        
        result_path = current_path + base_name + '_Kolmogorov_Smirnov_Test.png'
        
        print("result file was saved as {}\n".format(result_path))
        
        plt.savefig(result_path, bbox_inches='tight', dpi=1000)
        plt.close()
        
        
    
    '''
    q_ma_list = []
    
    genotype_unique = np.unique(genotype_label_arr)
    
    genotype_unique_list = genotype_unique.tolist()
    
    print(type(genotype_unique))
     
    for idx, genoype_value  in enumerate(genotype_unique):
        
        print("genotype_ID = {}, genoype_value = {}".format(idx, genoype_value))
        
        index_sel = np.where(genotype_label_arr == genotype_unique[idx])[0]

        ma_arr = data_ma_arr[index_sel][:,0]
        
        #print(index_sel)
        
        #print(ma_arr)
        
        q_ma_list.append(ma_arr)
            
    
    
    diff_list = [0] * abs(len(q_ma_list[0]) - len(q_ma_list[1]))

    extend_list = []

    if len(q_ma_list[0]) > len(q_ma_list[1]):
        extend_list.extend(q_ma_list[1])
        
    else:
        extend_list.extend(q_ma_list[0])
    
    extend_list.extend(diff_list)
    
    
    print(len(extend_list), len(q_ma_list[0]), len(q_ma_list[1]))


    df = pd.DataFrame({'maize': q_ma_list[0], 'bean': extend_list, }, columns=['maize', 'bean'])

    df['maize'].hist()
    
    df['bean'].hist()


    #df4 = pd.DataFrame({'maize': q_ma_list[0], 'bean': q_ma_list[1], }, columns=['maize', 'bean'])

    #fig = df.plot(kind='scatter', alpha=0.5)
    
    plt.show()
    '''
            

        
    
    
    

    
    
    ###########################################################################
    #indices_HighN = np.where(genotype_label == 'HighN')
    
    #genotype_label[indices_HighN] == 255
    
    data_v_sel = data_v_arr
    genotype_label_sel = genotype_label_arr
    
    #print(genotype_label_sel)
    
    #visualize rotation vectors
    #visualization_rotation_vector(np.asarray(normalized_data_v), genotype_label)
    
    if args['visualize'] == 1:
        
        visualization_rotation_vector(np.asarray(data_v_sel), np.asarray(data_q_arr), genotype_label_sel)
    

    ###########################################################################################3
    #print(ExcelFiles_list[0])
    # merge all excel files read all Excel files at once
    df = pd.concat(pd.read_excel(excelFile) for excelFile in ExcelFiles_list)
    
    n_colors = len(np.unique(genotype_label))
    
    if n_colors < 2:
        markercolor = px.colors.sample_colorscale("turbo", [n/(n_colors) for n in range(n_colors)])
    else:
        markercolor = px.colors.sample_colorscale("turbo", [n/(n_colors -1) for n in range(n_colors)])
    
    
    if type_quaternion == 0:

        X_col_q = 'quaternion_b'
        Y_col_q = 'quaternion_c'
        Z_col_q = 'quaternion_d'
        
        X_col_v = 'rotVec_avg_0'
        Y_col_v = 'rotVec_avg_1'
        Z_col_v = 'rotVec_avg_2'
        
        quaternion_4D = (current_path + 'avg_quaternion_4D.html')
        
    elif type_quaternion == 1:
        
        X_col_q = 'composition_quaternion_b'
        Y_col_q = 'composition_quaternion_c'
        Z_col_q = 'composition_quaternion_d'
        
        X_col_v = 'rotVec_composition_0'
        Y_col_v = 'rotVec_composition_1'
        Z_col_v = 'rotVec_composition_2'
        
        quaternion_4D = (current_path + 'composition_quaternion_4D.html')
        
    elif type_quaternion == 2:
        
        X_col_q = 'diff_quaternion_b'
        Y_col_q = 'diff_quaternion_c'
        Z_col_q = 'diff_quaternion_d'

        X_col_v = 'rotVec_diff_0'
        Y_col_v = 'rotVec_diff_1'
        Z_col_v = 'rotVec_diff_2'
        
        quaternion_4D = (current_path + 'diff_quaternion_4D.html')
        

        

    
    fig = px.scatter_3d(df, x = X_col_q, y = Y_col_q, z = Z_col_q, color = 'genotype',   size_max = 20, opacity = 1.0)

    #fig = px.scatter_3d(df, x='quaternion_b', y='quaternion_c', z='quaternion_d', color='genotype',  color_discrete_sequence=markercolor,  symbol='genotype',  size = 'label', size_max = 20, opacity = 1.0)

    fig.update_traces(marker_size = 4)
    
    
    #fig.add_scatter3d(x=data2['quaternion_b'], y=data2['quaternion_c'], z=data2['quaternion_d'])

    #fig.add_scatter3d(df, x='quaternion_b', y='quaternion_c', z='quaternion_d', color='genotype',  color_discrete_sequence = markercolor)

    #fig.update_traces(marker_size = 10)
    
   
    plotly.offline.plot(fig, auto_open=False, filename=quaternion_4D)

    
    
    
    ######################################################################
    fig = px.scatter_3d(df, x = X_col_v, y = Y_col_v, z = Z_col_v, color='genotype', size_max = 20, opacity = 1.0)
    
    fig.update_traces(marker_size = 4)
    
    
    
    plotly.offline.plot(fig, auto_open=False, filename=quaternion_4D)
    
    



    

