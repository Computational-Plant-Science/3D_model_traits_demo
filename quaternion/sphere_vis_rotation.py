"""
Version: 1.0

Summary: visualization of two sets of rotation vectors

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

    python3 sphere_vis_rotation.py -p ~/example/model_COLMAP_results_quarterunion/quaternion_summary/ -f1 all_genotype.xlsx -f2 avg_12_genotype.xlsx



"""
from mayavi import mlab
from tvtk.api import tvtk

import argparse
import pandas as pd

import numpy as np 
import plotly
import plotly.graph_objs as go
import plotly.express as px

import matplotlib

import plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff


#def visualization_rotation_vector(rotVec_rec_1,rotVec_rec_2):
    
def visualization_rotation_vector(rotVec_rec_1):
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

    cmap = matplotlib.cm.get_cmap('Spectral')
    
    # visualize rotation vectors
    for idx, Vec in enumerate(rotVec_rec_1):

        #mlab.quiver3d(0,0,0, Vec[0], Vec[1], Vec[2], color = (1, 0, 0)) 
        current_color = cmap(idx/12)[:-1]
        
        mlab.quiver3d( 0,0,0, Vec[0], Vec[1], Vec[2], color = current_color)

    mlab.show()
    
    
    
    
    
if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-f1", "--file1", required = True, help = "file1 name")
    ap.add_argument("-f2", "--file2", required = False, help = "file1 name")
    args = vars(ap.parse_args())

    
    # setting path to data file 
    current_path = args["path"]
    file1 = args["file1"]
    #file2 = args["file2"]
    
    file1_full_path = current_path + file1
    #file2_full_path = current_path + file2
    
    #Read rotation vector data from csv
    data1 = pd.read_excel(file1_full_path)
    
    #data2 = pd.read_excel(file2_full_path)
    
    #construct data array
    #rotVec_rec_1 = np.vstack((data1['rotVec_rec_0'],data1['rotVec_rec_1'],data1['rotVec_rec_2'])).T
    #rotVec_rec_2 = np.vstack((data2['rotVec_rec_0'],data2['rotVec_rec_1'],data2['rotVec_rec_2'])).T
    
    #visualize rotation vectors
    #visualization_rotation_vector(rotVec_rec_1,rotVec_rec_2)
    
    #visualization_rotation_vector(rotVec_rec_1)
    
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
    
 
    
    n_colors = 12
    
    markercolor = px.colors.sample_colorscale("turbo", [n/(n_colors -1) for n in range(n_colors)])
    
    
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

    
    fig = px.scatter_3d(data1, x='quaternion_b', y='quaternion_c', z='quaternion_d', color='genotype',  color_discrete_sequence=markercolor,  symbol='genotype', size = 'label', size_max = 20, opacity = 1.0)

    #fig = px.scatter_3d(data1, x='quaternion_b', y='quaternion_c', z='quaternion_d', color='genotype',  color_discrete_sequence=markercolor,  symbol='genotype',  size = 'label', size_max = 20, opacity = 1.0)

    fig.update_traces(marker_size = 4)
    
    
    #fig.add_scatter3d(x=data2['quaternion_b'], y=data2['quaternion_c'], z=data2['quaternion_d'])

    #fig = px.scatter_3d(data1, x='quaternion_b', y='quaternion_c', z='quaternion_d', color='genotype',  color_discrete_sequence = markercolor)

    #fig.update_traces(marker_size = 10)
    
    
    quaternion_4D = (current_path + 'avg_quaternion_4D.html')
    
    plotly.offline.plot(fig, auto_open=False, filename=quaternion_4D)

    
    
    
    ######################################################################
    fig = px.scatter_3d(data1, x='rotVec_rec_0', y='rotVec_rec_1', z='rotVec_rec_2', color='genotype', color_discrete_sequence = markercolor, symbol='genotype', size = 'label', size_max = 20, opacity = 1.0)
    
    fig.update_traces(marker_size = 4)
    
    quaternion_4D = (current_path + 'avg_rotation_vector.html')
    
    plotly.offline.plot(fig, auto_open=False, filename=quaternion_4D)
    
    

