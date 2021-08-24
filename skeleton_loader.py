"""
Version: 1.5

Summary: compute the cross section plane based on 3d model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 skeleton_loader.py -p ~/example/ -m1 test_skeleton.ply -m2 test.ply


argument:
("-p", "--path", required=True,    help="path to *.ply model file")
("-m", "--model", required=True,    help="file name")

"""
#!/usr/bin/env python



# import the necessary packages
from plyfile import PlyData, PlyElement
import numpy as np 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from operator import itemgetter
import argparse

import os
import sys
import open3d as o3d
import copy

from mayavi import mlab
from tvtk.api import tvtk

import networkx as nx

import graph_tool.all as gt

import plotly.graph_objects as go

#from matplotlib import pyplot as plt

from math import sqrt



# calculate length of a 3D path or curve
def path_length(X, Y, Z):

    n = len(X)
     
    lv = [sqrt((X[i]-X[i-1])**2 + (Y[i]-Y[i-1])**2 + (Z[i]-Z[i-1])**2) for i in range (1,n)]
    
    L = sum(lv)
    
    return L

'''
# distance between two points
def distance_pt(p0, p1):
    
    dist = np.linalg.norm(p0 - p1)
    
    return dist
'''

def visualize_skeleton(current_path, filename_skeleton, filename_pcloud):
    
    model_skeleton = current_path + filename_skeleton
    print("Loading 3D skeleton file {}...\n".format(filename_skeleton))
    model_skeleton_name_base = os.path.splitext(model_skeleton)[0]
    
    #load the ply format skeleton file 
    try:
        with open(model_skeleton, 'rb') as f:
            plydata_skeleton = PlyData.read(f)
            num_vertex_skeleton = plydata_skeleton.elements[0].count
            N_edges_skeleton = len(plydata_skeleton['edge'].data['vertex_indices'])
            array_edges_skeleton = plydata_skeleton['edge'].data['vertex_indices']
            
            print("Ply data structure: \n")
            #print(plydata_skeleton)
            #print("\n")
            print("Number of 3D points in skeleton model: {0} \n".format(num_vertex_skeleton))
            print("Number of edges: {0} \n".format(N_edges_skeleton))

        
    except:
        print("Model skeleton file does not exist!")
        sys.exit(0)
    
    
    #Parse ply format skeleton file and Extract the data
    Data_array_skeleton = np.zeros((num_vertex_skeleton, 3))
    
    Data_array_skeleton[:,0] = plydata_skeleton['vertex'].data['x']
    Data_array_skeleton[:,1] = plydata_skeleton['vertex'].data['y']
    Data_array_skeleton[:,2] = plydata_skeleton['vertex'].data['z']
    
    X_skeleton = Data_array_skeleton[:,0]
    Y_skeleton = Data_array_skeleton[:,1]
    Z_skeleton = Data_array_skeleton[:,2]
    
    ####################################################################
    #mesh = trimesh.load(model_skeleton)
    
    
    '''
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(Data_array_skeleton)
    
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    
    # Build KDTree from point cloud for fast retrieval of nearest neighbors
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
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
    
    ####################################################################
    '''
    filepath = current_path + 'edges_skeleton.txt'
    with open(filepath, 'w') as file_handler:
        for item in array_edges_skeleton.tolist():
            file_handler.write("{}\n".format(item))

    '''
    
    
    G_unordered = gt.Graph(directed = True)
    
    # assert directed graph
    #print(G.is_directed())
    
    nodes = G_unordered.add_vertex(num_vertex_skeleton)
    
    G_unordered.add_edge_list(array_edges_skeleton.tolist()) 
    
    #gt.graph_draw(G_unordered, vertex_text = G_unordered.vertex_index, output = current_path + "graph_view.pdf")
    
    
    # find all end vertex by fast iteration of all vertices
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
            
    print("end_vlist = {} \n".format(end_vlist))
    
    print("end_vlist_offset = {} \n".format(end_vlist_offset))
    
    
    #define start and end vertex index
    start_v = 0
    end_v = 221
    
    # find shortest path in the graph between start and end vertices 
    vlist, elist = gt.shortest_path(G_unordered, G_unordered.vertex(start_v), G_unordered.vertex(end_v))
    
    vlist_path = [str(v) for v in vlist]
    
    elist_path = [str(e) for e in elist]
    
    index_vlist_path = [int(i) for i in vlist_path]
    
    print("vlist_path = {} \n".format(type(index_vlist_path[0])))
    
    curve_length = path_length(X_skeleton[index_vlist_path], Y_skeleton[index_vlist_path], Z_skeleton[index_vlist_path])
    
    print("curve_length = {} \n".format(curve_length))
    
    ###################################################################

    
    #Load ply point cloud file
    if not (filename_pcloud is None):
        
        model_pcloud = current_path + filename_pcloud
        
        print("Loading 3D point cloud {}...\n".format(filename_pcloud))
        
        model_pcloud_name_base = os.path.splitext(model_pcloud)[0]
        
        pcd = o3d.io.read_point_cloud(model_pcloud)
        
        Data_array_pcloud = np.asarray(pcd.points)
        
        
        if pcd.has_colors():
            
            print("Render colored point cloud")
            
            pcd_color = np.asarray(pcd.colors)
            
            if len(pcd_color) > 0: 
                
                pcd_color = np.rint(pcd_color * 255.0)
            
            #pcd_color = tuple(map(tuple, pcd_color))
        else:
            
            print("Generate random color")
        
            pcd_color = np.random.randint(256, size = (len(Data_array_pcloud),3))
            
        #print(Data_array_pcloud.shape)
        
        #print(len(Data_array_pcloud))
        
        #print(pcd_color.shape)
        
        #print(type(pcd_color))
    
    
    
    #Skeleton Visualization pipeline
    ####################################################################
    # The number of points per line
    N = 2
    
    mlab.figure("Structure_graph", size = (800, 800), bgcolor = (0, 0, 0))
    mlab.clf()
    
    #pts = mlab.points3d(X_skeleton, Y_skeleton, Z_skeleton, mode = 'point', scale_factor = 0.5)
    
    pts = mlab.points3d(X_skeleton[end_vlist], Y_skeleton[end_vlist], Z_skeleton[end_vlist], color=(0,0,1), mode = 'sphere', scale_factor = 0.05)
    
    pts = mlab.points3d(X_skeleton[end_vlist_offset], Y_skeleton[end_vlist_offset], Z_skeleton[end_vlist_offset], color=(1,0,0), mode = 'sphere', scale_factor = 0.05)
    
    for i, (end_val, x_e, y_e, z_e) in enumerate(zip(end_vlist, X_skeleton[end_vlist], Y_skeleton[end_vlist], Z_skeleton[end_vlist])):
        
        mlab.text3d(x_e, y_e, z_e, str(end_val), scale = (0.04, 0.04, 0.04))
    
    for i, (end_val, x_e, y_e, z_e) in enumerate(zip(end_vlist_offset, X_skeleton[end_vlist_offset], Y_skeleton[end_vlist_offset], Z_skeleton[end_vlist_offset])):
        
        mlab.text3d(x_e, y_e, z_e, str(end_val), scale = (0.04, 0.04, 0.04))
        
    #pts = mlab.points3d(Data_array_pcloud[:,0], Data_array_pcloud[:,1], Data_array_pcloud[:,2], mode = 'point')
    
    #visualize point cloud model with color
    ####################################################################
    
    if not (filename_pcloud is None):
        
        x, y, z = Data_array_pcloud[:,0], Data_array_pcloud[:,1], Data_array_pcloud[:,2] 
        
        pts = mlab.points3d(x,y,z, mode = 'point')
        
        sc = tvtk.UnsignedCharArray()
        
        sc.from_array(pcd_color)

        pts.mlab_source.dataset.point_data.scalars = sc
        
        pts.mlab_source.dataset.modified()
    
    
    #visualize skeleton model, edge, nodes
    ####################################################################
    x = list()
    y = list()
    z = list()
    s = list()
    connections = list()
    
    # The index of the current point in the total amount of points
    index = 0
    
    # Create each line one after the other in a loop
    for i in range(N_edges_skeleton):
    #for val in vlist_path:
        
        #i = int(val)
        #print("Edges {0} has nodes {1}, {2}\n".format(i, array_edges[i][0], array_edges[i][1]))
      
        x.append(X_skeleton[array_edges_skeleton[i][0]])
        y.append(Y_skeleton[array_edges_skeleton[i][0]])
        z.append(Z_skeleton[array_edges_skeleton[i][0]])
        
        x.append(X_skeleton[array_edges_skeleton[i][1]])
        y.append(Y_skeleton[array_edges_skeleton[i][1]])
        z.append(Z_skeleton[array_edges_skeleton[i][1]])
        
        # The scalar parameter for each line
        s.append(array_edges_skeleton[i])
        
        # This is the tricky part: in a line, each point is connected
        # to the one following it. We have to express this with the indices
        # of the final set of points once all lines have been combined
        # together, this is why we need to keep track of the total number of
        # points already created (index)
        #connections.append(np.vstack(array_edges[i]).T)
        
        connections.append(np.vstack(
                       [np.arange(index,   index + N - 1.5),
                        np.arange(index + 1, index + N - .5)]
                            ).T)
        index += 2
    
    
    # Now collapse all positions, scalars and connections in big arrays
    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    s = np.hstack(s)
    #connections = np.vstack(connections)

    # Create the points
    src = mlab.pipeline.scalar_scatter(x, y, z, s)

    # Connect them
    src.mlab_source.dataset.lines = connections
    src.update()

    # display the set of lines
    mlab.pipeline.surface(src, colormap = 'Accent', line_width = 5, opacity = 0.7)

    # And choose a nice view
    #mlab.view(33.6, 106, 5.5, [0, 0, .05])
    #mlab.roll(125)
    
    
    '''
    filepath = current_path + 'edges_skeleton_s.txt'
    with open(filepath, 'w') as file_handler:
        for item in s:
            file_handler.write("{}\n".format(item))
    '''
    
   
    
    
    #########################
    '''
    #netweorkx graph
    G = nx.Graph()
    
    #G.add_nodes_from()
    
    G.add_edges_from(array_edges_skeleton)
    
    tree = nx.bfs_tree(G, 0)
    
    #nx.draw(G, with_labels=True, font_weight='bold')
    
    #plt.show()  
    path_list = nx.shortest_path(G, source=0, target=221)
    
    print("path_list = {}".format(path_list))
    '''
    '''
    filepath = current_path + 'edges_skeleton_connections.txt'
    with open(filepath, 'w') as file_handler:
        for item in connections:
            file_handler.write("{}\n".format(item))
    '''
    '''
    # Directed by default​
    G = gt.Graph(directed = True)
    
    # assert directed graph
    #print(G.is_directed())
    
    nodes = G.add_vertex(num_vertex_skeleton)
    
    
    for i in range(N_edges_skeleton):
        
        G.add_edge(s[i], s[i+1])
    
    #gt.graph_draw(G, vertex_text = G.vertex_index, output = "graph_view.pdf")
    
    
    # find all end vertex by fast iteration of all vertices
    end_vlist = []
    
    for v in G.iter_vertices():
        
        #print(G.vertex(v).out_degree(), G.vertex(v).in_degree())
        
        if G.vertex(v).out_degree() == 2 and G.vertex(v).in_degree() == 1:
        
            end_vlist.append(v)
            
    print("end_vlist = {}".format(end_vlist))
    
    
    start_v = 0
    end_v = 221
    '''
    
    '''    
    for path in gt.all_paths(G, start_v, end_v):

        print("path = {}".format(path))
    '''
    
    '''
    for path in gt.all_shortest_paths(G, 0, 243):

        print(path)
    
    
    # find shortest path in the graph between start and end vertices 
    vlist, elist = gt.shortest_path(G, G.vertex(start_v), G.vertex(end_v))
    
    vlist_path = [str(v) for v in vlist]
    
    elist_path = [str(e) for e in elist]
    
    #print(vlist_path)
     
    
    n_paths = gt.count_shortest_paths(G, G.vertex(start_v), G.vertex(end_v))
    
    print(n_paths)
    
    '''
    
    
    ###################################################################
    # visualize path
    #visualize skeleton model, edge, nodes
    ####################################################################
    x = list()
    y = list()
    z = list()
    s = list()
    connections = list()
    
    # The index of the current point in the total amount of points
    index = 0
    
    # Create each line one after the other in a loop
    #for i in range(N_edges_skeleton):
    for val in vlist_path:
        
        i = int(val)
        #print("Edges {0} has nodes {1}, {2}\n".format(i, array_edges[i][0], array_edges[i][1]))
      
        x.append(X_skeleton[array_edges_skeleton[i][0]])
        y.append(Y_skeleton[array_edges_skeleton[i][0]])
        z.append(Z_skeleton[array_edges_skeleton[i][0]])
        
        x.append(X_skeleton[array_edges_skeleton[i][1]])
        y.append(Y_skeleton[array_edges_skeleton[i][1]])
        z.append(Z_skeleton[array_edges_skeleton[i][1]])
        
        # The scalar parameter for each line
        s.append(array_edges_skeleton[1])
        
        # This is the tricky part: in a line, each point is connected
        # to the one following it. We have to express this with the indices
        # of the final set of points once all lines have been combined
        # together, this is why we need to keep track of the total number of
        # points already created (index)
        #connections.append(np.vstack(array_edges[i]).T)
        
        connections.append(np.vstack(
                       [np.arange(index,   index + N - 1.5),
                        np.arange(index + 1, index + N - .5)]
                            ).T)
        index += 2
    
    
    # Now collapse all positions, scalars and connections in big arrays
    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    s = np.hstack(s)
    #connections = np.vstack(connections)

    # Create the points
    src = mlab.pipeline.scalar_scatter(x, y, z, s)

    # Connect them
    src.mlab_source.dataset.lines = connections
    src.update()

    # The stripper filter cleans up connected lines
    #lines = mlab.pipeline.stripper(src)

    # display the set of lines
    mlab.pipeline.surface(src, colormap = 'Accent', line_width = 10, opacity = 1.0)

    # And choose a nice view
    #mlab.view(33.6, 106, 5.5, [0, 0, .05])
    #mlab.roll(125)
    
    
    #mlab.savefig('skeleton_graph.x3d')
    
    
    
    
    mlab.show()
    
    
    
    '''
    #As before we use networkx to determine node positions. We want to do the same spring layout but in 3D
    spring_3D = nx.spring_layout(G, dim=3, seed=18)
    
    
    #we need to seperate the X,Y,Z coordinates for Plotly
    #x_nodes = [spring_3D[i][0] for i in range(Num_nodes)]# x-coordinates of nodes
    #y_nodes = [spring_3D[i][1] for i in range(Num_nodes)]# y-coordinates
    #z_nodes = [spring_3D[i][2] for i in range(Num_nodes)]# z-coordinates


    #We also need a list of edges to include in the plot
    edge_list = G.edges()

    #we  need to create lists that contain the starting and ending coordinates of each edge.
    x_edges=[]
    y_edges=[]
    z_edges=[]

    #need to fill these with all of the coordiates
    for edge in edge_list:
        #format: [beginning,ending,None]
        x_coords = [spring_3D[edge[0]][0],spring_3D[edge[1]][0],None]
        x_edges += x_coords

        y_coords = [spring_3D[edge[0]][1],spring_3D[edge[1]][1],None]
        y_edges += y_coords

        z_coords = [spring_3D[edge[0]][2],spring_3D[edge[1]][2],None]
        z_edges += z_coords

    #create a trace for the edges
    trace_edges = go.Scatter3d(x=x_edges,
                        y=y_edges,
                        z=z_edges,
                        mode='lines',
                        line=dict(color='black', width=2),
                        hoverinfo='none')
                    

    #we need to set the axis for the plot 
    axis = dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='')
            

    #also need to create the layout for our plot
    layout = go.Layout(title="Two Predicted Factions of Zachary's Karate Club",
                width=650,
                height=625,
                showlegend=False,
                scene=dict(xaxis=dict(axis),
                        yaxis=dict(axis),
                        zaxis=dict(axis),
                        ),
                margin=dict(t=100),
                hovermode='closest')
                
    #Include the traces we want to plot and create a figure
    data = [trace_edges]
    fig = go.Figure(data=data, layout=layout)

    fig.show()
    '''
    


    '''
    # reorder nodes from 0,len(G)-1
    G_mayavi = nx.convert_node_labels_to_integers(G)
    # 3d spring layout
    pos = nx.spring_layout(G_mayavi, dim=3, seed=1001)
    # numpy array of x,y,z positions in sorted node order
    xyz = np.array([pos[v] for v in sorted(G_mayavi)])
    # scalar colors
    scalars = np.array(list(G_mayavi.nodes())) + 5

    mlab.figure()

    pts = mlab.points3d(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    scalars,
    scale_factor=0.1,
    scale_mode="none",
    colormap="Blues",
    resolution=20,
    )

    pts.mlab_source.dataset.lines = np.array(list(G_mayavi.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=0.01)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
    mlab.orientation_axes()
    mlab.show()
    '''


if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m1", "--model_skeleton", required = True, help = "skeleton file name")
    ap.add_argument("-m2", "--model_pcloud", required = False, default = None, help = "point cloud model file name")
    args = vars(ap.parse_args())


    # setting path to model file 
    current_path = args["path"]
    filename_skeleton = args["model_skeleton"]
    
    if args["model_pcloud"] is None:
        
        filename_pcloud = None
        
    else:
        
        filename_pcloud = args["model_pcloud"]
    
    #file_path = current_path + filename

    print ("results_folder: " + current_path)

    visualize_skeleton(current_path, filename_skeleton, filename_pcloud)

 
