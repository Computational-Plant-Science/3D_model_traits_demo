"""
Version: 1.0

Summary: visualization of two sets of rotation vectors

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

    python3 sphere_vis_downsample.py -p ~/example/model_COLMAP_results_quarterunion/quaternion_summary/ 



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


import plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff

import random

from scipy import linalg 

from sklearn.preprocessing import normalize

from statistics import mean 



def cMap(x):
    #whatever logic you want for colors
    return [random.random() for i in x]
    
    
    
    
def visualization_rotation_vector(rotVec_rec, genotype_sub):
    
  
    
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
    
    
    NUM_COLORS = 2
    
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
    
    
    #mlab.quiver3d( 0,0,0, Vec_arr[0], Vec_arr[1], Vec_arr[2], color = current_color)
    pts = mlab.quiver3d(zeros,zeros,zeros, rotVec_rec[:,0], rotVec_rec[:,1], rotVec_rec[:,2], scalars=scalars)
    
    # Color by scalar
    pts.glyph.color_mode = 'color_by_scalar' 

    # Set look-up table and redraw
    pts.module_manager.scalar_lut_manager.lut.table = colors
        
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
        
        data_v = np.asarray(data_v)

        ################################################################
        #get downsampled quarterunion values
        cols_q = ['quaternion_a','quaternion_b','quaternion_c', 'quaternion_d']
        data_q = data[cols_q].values.tolist()
        
        # downsample along coloum direction, every 10th
        #data_q = quarterunionVec[::sample_rate,:]
        #data_q = np.asarray(data_q)[::sample_rate,:]
        
        data_q = np.asarray(data_q)
        
    
        ################################################################
        #get downsampled genotype values
        genotype_sub = data['genotype_label'].values.tolist()

        # downsample along coloum direction, every 10th
        #genotype_sub = genotype_v[::sample_rate,:]
        #genotype_sub = np.asarray(genotype_sub)[::sample_rate]
        
        genotype_sub = np.asarray(genotype_sub)
        

        
        
    
    ####################################################################
    # filter small vectors 
    
    '''
    vector_length = []
    
    for idx, Vec_arr  in enumerate(data_v):
        
        magnitude = linalg.norm(Vec_arr)
        
        vector_length.append(magnitude)
        
        
    avg_vec_len = mean(vector_length)
    
    indices_keep = [idx for idx, value in enumerate(vector_length) if value >= mean(vector_length)*1.2]
     
    data_v_sel = data_v[indices_keep, :]
    
    genotype_sub_sel = genotype_sub[indices_keep]
    '''
    
    
    '''
    genotype_label = np.zeros((len(genotype_sub), 4))
    
    
    genotype_sub = genotype_sub.tolist() 
    
    #print(type(genotype_sub))
    #print(len(genotype_sub))
    
    indices_LowN = np.where(genotype_sub == )
    
    print(indices_LowN)
    
    print(genotype_label[indices_LowN,:])
    '''
    
    
    #indices_HighN = np.where(genotype_sub == 'HighN')
    
    #genotype_label[indices_HighN] == 255
    
    
    
    
    #colors = (np.random.random((N, 4))*255).astype(np.uint8)
    

    data_v_sel = data_v
    genotype_sub_sel = genotype_sub
    
    #print(genotype_sub_sel)
    
    #visualize rotation vectors
    #visualization_rotation_vector(np.asarray(normalized_data_v), genotype_sub)
    
    if args['visulize'] == 1:
        
        visualization_rotation_vector(np.asarray(data_v_sel), genotype_sub_sel)
    
    
    
    '''
    ###################################################################
    data1['x'] = 0
    data1['y'] = 0
    data1['z'] = 0
    
    fig = go.Figure(data = go.Cone(
        x=data1['x'],
        y=data1['y'],
        z=data1['z'],
        u=data1['rotVec_rec_0'],
        v=data1['rotVec_rec_1'],
        w=data1['rotVec_rec_2'],
        colorscale='Viridis',
        sizemode="absolute",
        sizeref=4))

    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
                                 camera_eye=dict(x=1.2, y=1.2, z=0.6)))

    fig.show()
    '''
    ####################################################################
    
    ####################################################################
    #Multi-dimension plots in ploty, color represents quaternion_a

    #Set marker properties
    #markercolor = data1['Ratio of cluster']
    
    #markercolor = dict(color = data1['label'])
    
 
    
 
    
    
    '''
    fig = go.Figure()
    
    #Make Plotly figure
    fig = go.Scatter3d(x=data1['quaternion_b'],
                    y=data1['quaternion_c'],
                    z=data1['quaternion_d'],
                    marker=dict(color=data1['label'],
                                opacity=1,
                                reversescale=True,
                                colorscale='Viridis',
                                colorbar=dict(thickness=10),
                                size=5),
                    line=dict (width=0.02),
                    mode='markers')
    

    
 
    #Make Plot.ly Layout
    mylayout = go.Layout(scene=dict(xaxis=dict( title="quaternion_b"),
                                yaxis=dict( title="quaternion_c"),
                                zaxis=dict(title="quaternion_d")),)
    

    quaternion_4D = (current_path + 'avg_quaternion_4D.html')
    
    #Plot and save html
    plotly.offline.plot({"data": [fig],
                     "layout": mylayout},
                     auto_open=False,
                     filename=quaternion_4D)
                     
    '''
    #fig = go.Scatter3d(data1, x = data1['quaternion_b'], y = data1['quaternion_c'], z = data1['quaternion_d'], color = data1['file_name'])
    
    
    ###########################################################################################3
    #print(ExcelFiles_list[0])
    # merge all excel files read all Excel files at once
    df = pd.concat(pd.read_excel(excelFile) for excelFile in ExcelFiles_list)
    
    #n_colors = len(ExcelFiles_list)
    
    n_colors = 2
    
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
    
    

