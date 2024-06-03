
import networkx as nx
import collections
#import itertools as IT
from scipy.ndimage import label
from mayavi import mlab
import numpy as np
#import sys
import pdb
import math

from collections import Counter, defaultdict

from scipy.spatial import distance
from numpy import random

import progressbar
from time import sleep


def get_nhood(img):
    """
    calculates the neighborhood of all voxel, needed to create graph out of skel image
    inspired by [Skel2Graph](https://github.com/phi-max/skel2graph3d-matlab) from Philip Kollmannsberger.
    Parameters
    ----------
    img : ndarray
        Binary input image.  Needs to be padded with zeros
    
    Returns
    -------
    nhood : neighborhood of all voxels that are True
    nhi : the indices of all 27 neighbors, incase they are true
    inds : of all True voxels (raveled)
    """
    print('calculating neighborhood')
    
    #check if padded
    assert img.dtype==np.bool
    assert img[0,:,:].max()==0
    assert img[-1,:,:].max()==0
    assert img[:,0,:].max()==0
    assert img[:,-1,:].max()==0
    assert img[:,:,0].max()==0
    assert img[:,:,-1].max()==0

    inds=np.arange(img.size)
    inds=inds[img.ravel()]
    
    x, y, z = np.unravel_index(inds, img.shape)  # unraveled indices
    # space for all neighbors
    nhi = np.zeros((inds.shape[0], 27), dtype=np.uint32)
    
    s0, s1, s2 = np.array(img.shape) -2
    assert s0 * s1 * s2 < np.iinfo(nhi.dtype).max, 'the array is too big, fix datatype of nhi'
    
    for xx in range(0, 3):
        for yy in range(0, 3):
            for zz in range(0, 3):
                n = img[xx:(s0 + xx), yy:(s1 + yy), zz:(s2 + zz)]  # shifted image
                
                w = np.lib.ravel_multi_index(np.array([xx, yy, zz]), (3, 3, 3))
                
                nhi[:, w] = (np.lib.ravel_multi_index(np.array([xx + x - 1, yy + y - 1, zz + z - 1]), np.array(img.shape)) * n[img[1:-1, 1:-1, 1:-1]])

                
    # calculate number of neighbors (+1)
    nhood = np.sum(nhi > 0, axis=-1, dtype=np.uint8)
   
    print('done')
    return nhood, nhi, inds


def skel2graph(skel,  origin=[0, 0, 0], spacing=[1, 1, 1], pad=True):
    """
    converts skeletonized image into networkx structure with coordinates at node and edge attributes.
    This algorithmn is based on the matlab algorithmn of [Philip Kollmannsberger](https://github.com/phi-max/skel2graph3d-matlab) and was written to analyze the denspacingitic cell networks.
nodevoxels (neighborhood >3) which are not seperated by edge voxels (neighborhood = 2) will become one node with coordinates 
    Parameters
    ----------
    skel : ndarray (bool)
        Binary skeleton image.  
    origin : list, tuple
        coordinates of the corner of the 3D image
    spacing : list, tuple
        spacing of the voxels. important to give length of the edges a meanigful interpretation
    pad : bool
        should the skeleton be padded first. If you are unsure if the skeleton was padded before, you should enable this option
    
    
    Returns
    -------
    G : networkx.Graph
        
        """
    
    print("Creating graph...")
    
    assert skel.dtype==np.bool
    if pad:
        img = np.pad(skel, pad_width=1, mode='constant')
        
        origin -= np.array(spacing)
    else:
        img = skel
    
    G = nx.MultiGraph()
    w, l, h = np.shape(img)

    # get neighborhood
    nhood, nhi, inds = get_nhood(img)
 
    # edge voxesl are those with 2 neighbors
    e_v = nhood == 3
    e_inds = inds[e_v] #position

    # all others are node voxels
    n_v = (nhood != 3) 
    n_inds = inds[n_v] #position
    
    # what are the two neighbors
    e_nh_idx = nhi[e_v]
    e_nh_idx[:, 13] = 0
    e_nb = np.sort(e_nh_idx)[:, -2:]#find the two neighbors
    #free memory    
    del nhi    
    del e_nh_idx

    # get clusters of node voxels
    nodes = np.zeros_like(img, dtype=bool)
    nodes[np.lib.unravel_index(inds[n_v], [w, l, h])] = 1
    indnode, num = label(nodes, structure=np.ones((3, 3, 3)))
    #make collection of all nodevoxel in cluster
    node_inds_all = collections.defaultdict(list)
    if num < 2**16:
        indnode = indnode.astype(np.uint16)
    try: #this might produce memory error
        inspacingavel = indnode.ravel()
        for idx, val in zip(np.arange(nodes.size, dtype=np.uint64), inspacingavel) :
            #stop
            if not val == 0:
                node_inds_all[val].append(idx)

        del indnode
    except:  # do it in juncks
        
        inspacingavel = indnode.ravel()
        del indnode
        steps = np.linspace(
            0, inspacingavel.size, inspacingavel.size / 10000, dtype=np.uint64)
        for i in range(len(steps) - 1):
            for idx, val in zip(np.arange(steps[i], steps[i + 1], dtype=np.uint64), inspacingavel[steps[i]:steps[i + 1]]):
                if not val == 0:
                    node_inds_all[val].append(idx)
    
    
    N = len(node_inds_all.keys())
    
    print("Processing graph nodes...")
    #progress bar display
    bar = progressbar.ProgressBar(maxval = N)
    
    bar.start()
    
    for i in node_inds_all.keys():

        #sys.stdout.write("\rpercentage: " + str(100 * i / N) + "%")
        #sys.stdout.write("\rpercentage: " + '{:.2%}'.format(i / N))
        #sys.stdout.flush()
        
        node_inds = node_inds_all[i]
        i -= 1  # to have n_v starting at 0
        x, y, z = np.lib.unravel_index(node_inds, nodes.shape)
        
        bar.update(i+1)
        sleep(0.1)
        
        xmean = x.mean()
        ymean = y.mean()
        zmean = z.mean()
       
        if len(x) > 1:
            # node concids out of more than one voxel (only important for
            # debugging?)
            mn = True
        else:
            mn = False
        G.add_node(i, **{'x': xmean * spacing[0] + origin[0], 'y': ymean * spacing[1] + origin[1], 'z': zmean * spacing[
                   2] + origin[2], 'multinode': mn,  'idx': node_inds})
        # find all canal vox in nb of all node idx
 
    print(' nodes done, track edges')
    
    #bar.finish()
    
    # del node_inds_all
    nodex = nx.get_node_attributes(G, 'x')
    nodey = nx.get_node_attributes(G, 'y')
    nodez = nx.get_node_attributes(G, 'z')
    edge_inds = []

    # now, find all edges
    
  
    
    e_ravel = e_nb.ravel()  # might make it faster
    node_list = G.nodes()
    N=len(G.nodes())
    alledges=[]
    
    print("Processing graph edges...")
    #progress bar display
    bar = progressbar.ProgressBar(maxval = N)
    
    bar.start()
    
    for i, nd in G.nodes(data=True):
        
        #sys.stdout.write("\rpercentage: " + str(100 * i / N) + "%")
        #sys.stdout.write("\rpercentage: " +  '{:.2%}'.format(i / N))
        #sys.stdout.flush()
        
        
        # if i ==46:
        # pdb.set_trace()
        node_inds = nd['idx']
        if nhood[inds==node_inds[0]]==1:
            continue

        bar.update(i+1)
        sleep(0.1)
        
        # find all edge voxels which are neighbors of node i
        nbs = np.in1d(e_ravel, node_inds).reshape(e_nb.shape)
        e_n, pos = np.where(nbs)
        
        # iterate through edge untill node is reached
        for m, idx in enumerate(e_n):
            #pdb.set_trace()
            edge_inds = [e_inds[idx]]
            test = e_nb[idx, int(pos[m] == 0)]
            #pdb.set_trace()
            while test in e_inds:
                edge_inds.append(test)
                newcan = e_nb[e_inds == test]
                otest = test

                test = newcan[newcan != edge_inds[-2]][0]
            # pdb.set_trace()
            if test in n_inds:
               
                n2 = inspacingavel[test] - 1
                
                # to avoid starting the edge from the oposite side
                e_nb[e_inds == edge_inds[-1]] *= 0
                assert  n2 in node_list
                
                x, y, z = np.lib.unravel_index(np.array(edge_inds).astype(np.int64), img.shape)
                
                x_ = np.r_[nodex[i], x * spacing[0] + origin[0], nodex[n2]]
                y_ = np.r_[nodey[i], y * spacing[1] + origin[1], nodey[n2]]
                z_ = np.r_[nodez[i], z * spacing[2] + origin[2], nodez[n2]]
                alledges.append((i, n2, {'x': x_, 'y': y_, 'z': z_, 'weight': get_length(x_, y_, z_), 'n1': i, 'n2': n2}))
    
            
    G.add_edges_from(alledges)
    print(' edges done')
    
    #bar.finish()
    

    return G




def get_r(inds, shape, origin=[0, 0, 0], spacing=[1, 1, 1], order='C'):
    """
    get position of voxels 
    Parameters
    ----------
    inds : ndarray
        Binary skeleton image.  
    origin : list, tuple
        coordinates of the corner of the 3D image
    spacing : list, tuple
        spacing of the voxels. important to give length of the edges a meanigful interpretation
    order : bool
        see np.lib.unravel_index documentation
    
    
    Returns
    -------
        r : list of ndarrays
            position of voxels

        """
    return map(np.add, origin, map(np.multiply, spacing, np.lib.unravel_index(inds, shape, order=order)))


def get_cms(inds, shape, r0=[0, 0, 0], dr=[1, 1, 1], order='C'):
    """
    get mean position of voxels
      
    Parameters
    ----------
    inds : ndarray
        Binary skeleton image.  
    origin : list, tuple
        coordinates of the corner of the 3D image
    spacing : list, tuple
        spacing of the voxels. important to give length of the edges a meanigful interpretation
    order : bool
        see np.lib.unravel_index documentation
    
    
    Returns
    -------
        r : tuple 
            mean position of voxels

        """
    x_, y_, z_ = get_r(inds, shape, r0, dr, order=order)
    return x_.mean(), y_.mean(), z_.mean()


def get_length(x_, y_, z_):
    """
    calculate length of an edge
    Parameters:
    -----------
    x_ : list
        x coordinate 
    y_ : list
        y coordinate 
    z_ : list
        z coordinate

    Returns
    -------
    length : float
        length calculated as sum od eucledian distances between voxels in list
        

    """
    if len(x_) > 1:
        return np.sum(np.sqrt((np.array(x_[1:]) - np.array(x_[:-1]))**2 + (np.array(y_[1:]) - np.array(y_[:-1]))**2 + (np.array(z_[1:]) - np.array(z_[:-1]))**2))
    else:
        return 0


def duplicates(lst):

    cnt = Counter(lst)
    
    return [key for key in cnt.keys() if cnt[key]> 1]


def indices(lst, items= None):
    """return index of repeat value in a list"""
    items, ind = set(lst) if items is None else items, defaultdict(list)
    
    for i, v in enumerate(lst):
        
        if v in items: ind[v].append(i)
        
    return ind


def asSpherical(x, y, z):
    """coordinates transormation from cartesian coords to sphere coord system"""
    #eg. defaultdict(<class 'list'>, {10: [10, 11], 11: [12, 13], 12: [14, 15, 16, 17, 18]})
    r = math.sqrt(x*x + y*y + z*z)
    if r > 0 :
        elevation = math.acos(z/r)*180/math.pi #to degrees
        
        azimuth = np.arctan2(y,x)*180/math.pi
    else:
        elevation = 0
        azimuth = 0
        
    return r, elevation, azimuth



def trace_angle(x, y, z):
    """compute the angle of each trace in 3D space"""
    '''
    cx = x[0] - x[len(x)-1]
    cy = y[0] - y[len(y)-1]
    cz = z[0] - z[len(z)-1]

    (r,theta,phi) = asSpherical(cx, cy, cz)
    '''
    (r,theta,phi) = asSpherical(x, y, z)
    
    return r, theta, phi

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    if np.linalg.norm(vector) > 0:
        return vector / np.linalg.norm(vector)
    else:
        return vector 

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2':
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/math.pi
    

def lineseg_dist(p, a, b):
    """ Returns the distance the distance from point p to line segment [a,b]. p, a and b are np.arrays. 
    """
    if np.all(a - b):
        return np.linalg.norm(p - a, axis=0)
    
    if np.array_equal(b, a):
        return 0
    else:
        
        # normalized tangent vector
        d = np.divide(b - a, np.linalg.norm(b - a))

        # signed parallel distance components
        s = np.dot(a - p, d)
        t = np.dot(p - b, d)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, 0])

        # perpendicular distance component
        c = np.cross(p - a, d)

        return np.hypot(h, np.linalg.norm(c))


# Function to Detection Outlier on one-dimentional datasets.
def find_anomalies(data):
    
    # Set upper and lower limit to 3 standard deviation
    data_std = np.std(data, axis=0)
    data_mean = np.mean(data, axis=0)
    anomaly_cut_off = data_std * 2
    
    #lower_limit  = data_mean - anomaly_cut_off 
    #upper_limit = data_mean + anomaly_cut_off
    
    upper_limit = np.percentile(data, 75)
    lower_limit = np.percentile(data, 25)
    
    return lower_limit,upper_limit


def plot_graph( G,
               node_size = 1, node_color = (1, 1, 1), scale_node_keyword = None,
               node_color_keyword = None, tube_radius = 1.4, edge_color = (1, 1, 1),
               edge_color_keyword = None, edge_radius_keyword = None,
                edge_colormap = 'BrBG', **kwargs):
                    
    """ 3D plot of a 3D  network, this function uses a list of coordinates to visualize a network which might represent a 3D skeleton
 For both, edges and nodes the size and the color can be used to visualize a parameter of the attribute dictionary, 
 for edges this needs to be either a a number per edge, or a sequence with the length equal to the length of coordinates

        Parameters
        ----------
        G : networkx.Graph
            nodes and edges must have coordinates stored in attribute dictionary named 'x','y','z'
        node_size : float 
            size of spheres 
        node_color : tuple
            color of sphears
        edge_color : tuple
            color of tubes

        scale_node_keyword : string or None 
            if None, constant sizes are used, otherwise 
                the nodes are scaled with float given in G.node[i][scale_node_keyword] could also be 'degree'
                
        node_color_keyword: string or None
            if None is given node spheres have the same color, otherwise the float value of G.node[i][node_color_keyword] is used in cobination with a defauld colormap
            
        edge_color_keyword : string or None
            if None use edgecolor, otherwise G[i][j][edge_color_keyword] is used to colorecode this value or list of values
            
        edge_radius_keyword : string or None
            if None use edge_radius, otherwise Ge[i][j][edge_radius_keyword] is used to vary the radius according to  this value or list of values
            
        Returns
        -------
        tube_surf : tvtk actor
            actor of the edges
            
        pts : tvtk actor
            actor of the nodes
    """ 
    print('plot graph')
    from tvtk.api import tvtk
    cell_array = tvtk.CellArray()
   
    # initialize coordinates of edge segments
    xstart = []
    ystart = []
    zstart = []
    xstop = []
    ystop = []
    zstop = []
    edge_r = [] #for radius
    edge_c = [] #for color
    edgecount = 0
       
    lines = []
    # initialize coordinates of nodes
    xn = []
    yn = []
    zn = []
  
    edge_node_n1 = []
    edge_node_n2 = []
    edge_angle = []
    edge_length = []
    edge_projection = []


    for node in G.nodes():
        xn.append(G.nodes[node]['x'])
        yn.append(G.nodes[node]['y'])
        zn.append(G.nodes[node]['z'])
       
    # add edge segments
    i = 0 #number of added segments
    for n1, n2, edgedict in G.edges(data=True):

        edgecount += 1
        
        xstart.extend(edgedict['x'])
        xstop.extend(edgedict['x'])
        ystart.extend(edgedict['y'])
        ystop.extend(edgedict['y'])
        zstart.extend(edgedict['z'])
        zstop.extend(edgedict['z'])
        
        
        #how many segments in line?
        l = len(edgedict['x'])
        line = list(range(i, i + l))
        line.insert(0, l)
        lines.extend(line)
        i += l
                
        #print("edgedict[edge_color_keyword] is {0} \n:".format(edgedict[edge_color_keyword]))
        
        #ori
        #edgedict_temp = edgedict[edge_color_keyword]
        
        #color by location
        edgedict_temp = np.ones(len(edgedict[edge_color_keyword]))*np.mean(edgedict[edge_color_keyword])
        
        #gradient color
        #edgedict_temp = np.arange(len(edgedict[edge_color_keyword]))
        
        #same color
        #edgedict_temp = np.ones(len(edgedict[edge_color_keyword]))
        
        #color
        if edge_color_keyword is None:
            edge_c += [1] * (l) 

        elif edge_color_keyword not in edgedict.keys():
            edge_c += [1] * (l) 
            
        else:
            
            if len(edgedict[edge_color_keyword]) == 1:
                #edge_c.extend([edgedict[edge_color_keyword]] * (l))
                edge_c.extend([edgedict_temp] * (l))
            
            else:
                assert len(edgedict[edge_color_keyword]) == len(edgedict['x'])
                #edge_c.extend(edgedict[edge_color_keyword].tolist())
                edge_c.extend(edgedict_temp.tolist())
                
        #print("edge_c after is {0} \n:".format(edge_c))

        #radius
        if edge_radius_keyword in edgedict.keys():
            if np.size(edgedict[edge_radius_keyword]) == 1:
                edge_r.extend([edgedict[edge_radius_keyword]] * (l))
            else:
                assert len(edgedict[edge_radius_keyword]) == len(edgedict['x'])
                edge_r.extend(edgedict[edge_radius_keyword])
        else:
            edge_r += [tube_radius] * (l)
            
        
        #coord_n1 = (xn[n1], yn[n1], zn[n1])
        #coord_n2 = (xn[n2], yn[n2], zn[n2])
        
        #compute angle for each edge
        vector_n1_n2 = (xn[n2]-xn[n1] , yn[n2]-yn[n1], zn[n2]-zn[n1])
        
        (r_data, theta_data, phi_data) = trace_angle(xn[n2]-xn[n1] , yn[n2]-yn[n1], zn[n2]-zn[n1])
        
        reference_vector = (1, 0, 0)
        
        reference_vector_angle = angle_between(vector_n1_n2, reference_vector)
        
        #print("Angle is {0}:".format(reference_vector_angle))
        #print("r_data = {0}, theta_data = {1}, phi_data = {2}:\n".format(r_data, theta_data, phi_data))

        #record every node and edge properities 
        edge_node_n1.append(n1)
        edge_node_n2.append(n2)
       
        edge_angle.append((reference_vector_angle))
        edge_length.append(abs(edgedict['weight']))
        edge_projection.append(r_data)

    lower_limit, upper_limit = find_anomalies(np.asarray(edge_length))
    
    print("edge_count is {0} \n:".format(edgecount))
    print("edge_length is {0} \n:".format(edge_length))
    print("lower_limit upper_limit is {0} {1}:\n".format(lower_limit,upper_limit))
    
    #print("lines is {0} :\n".format(len(lines)))
    
            
    #pdb.set_trace()
    #pot network
    fig = mlab.gcf()
    fig = mlab.figure(1, bgcolor = (1,1,1))
    disable_render = fig.scene.disable_render
    fig.scene.disable_render = True

    # ------------------------------------------------------------------------
    # render edges
    #    
    edge_c = np.array(edge_c)
    if np.isinf(edge_c).any():
        edge_c[np.isinf(edge_c)] = np.nan
        
    xv = np.array([xstart])  
    yv = np.array([ystart])  
    zv = np.array([zstart])  
    
    # pdb.set_trace()
    edges_src = mlab.pipeline.scalar_scatter(xv.ravel(), yv.ravel(), zv.ravel(), edge_c.ravel())

    edges_src.mlab_source.dataset.point_data.add_array(np.abs(np.array(edge_r).ravel()))
    
    edges_src.mlab_source.dataset.point_data.get_array(1).name = 'radius'

    cell_array.set_cells(edgecount, lines)
    
    edges_src.mlab_source.dataset.lines = cell_array

    edges_src.mlab_source.update()
    
    
    if tube_radius is not None:
        #edges_src.mlab_source.dataset.point_data.vectors = np.abs(
        #        np.array([edge_r,edge_r,edge_r]).T)
        edges_src.mlab_source.dataset.point_data.add_array(np.abs(np.array(edge_r).ravel()))
        
        edges_src.mlab_source.dataset.point_data.get_array(1).name = 'radius'
        
        tubes = mlab.pipeline.tube(mlab. pipeline.set_active_attribute(edges_src, point_scalars='radius'), tube_radius = tube_radius,)
        
        #tubes.filter.radius=edge_r
        tubes.filter.vary_radius = 'vary_radius_by_scalar'
    else:
        tubes = edges_src
    
    tube_surf = mlab.pipeline.surface(mlab.pipeline.set_active_attribute(tubes, point_scalars='scalars'), colormap = edge_colormap, color = edge_color, **kwargs)
    
    tubes.children[0].point_scalars_name='scalars'
    
    if edge_color_keyword is None:
        tube_surf.actor.mapper.scalar_visibility = False
    else:
        tube_surf.actor.mapper.scalar_visibility = True
    #pdb.set_trace()    
    if not node_size:
        return tube_surf, None
    
    
    # ------------------------------------------------------------------------
    # Plot the nodes
    if node_size is not None:
        nodes = G.nodes()
        L = len(G.nodes())
        
        if node_color_keyword is None and scale_node_keyword is None:
            
            nodes = G.nodes()
            L = len(G.nodes())

            s = np.ones(len(xn))
            # pdb.set_trace()
            pts = mlab.points3d(xn, yn, zn, s, scale_factor=node_size,
                                # colormap=node_colormap,
                                resolution=16,
                                color=node_color)

        else:
         
            if  scale_node_keyword is not None:
               if scale_node_keyword == 'degree':
                    dic = G.degree(nodes)
                    s = []
                    for n in nodes:
                            s.append(dic[n])
               else:
                    s= list(nx.get_node_attributes(G,scale_node_keyword).values() )
            else:
                s = np.ones(len(xn)) * node_size   
                      
            if node_color_keyword is not None:
                node_color_scalar = list(nx.get_node_attributes(
                    G, node_color_keyword ).values())
            else:
                node_color_scalar = len(xn) * [1]

            pts = mlab.quiver3d(np.array(xn).ravel(), np.array(yn).ravel(), np.array(zn).ravel(),s,s,s, scalars=np.array(node_color_scalar).ravel(), mode = 'sphere', scale_factor = node_size,resolution=16)
            
            pts.glyph.glyph_source.glyph_position = 'center'
            
            pts.glyph.color_mode = 'color_by_scalar'
            
            #pts.glyph.scale_mode = 'scale_by_vector'
    
    
    # ------------------------------------------------------------------------
    # Plot the edge index at the first node location
    
    '''
    print("edge_node_n1: {0}\n".format(edge_node_n1))
    print("edge_node_n2: {0}\n".format(edge_node_n2))
    print("length: {0} {1}\n".format(len(edge_angle),len(edge_node_n1)))
    '''
    
    
    # ------------------------------------------------------------------------
    # Select edges from shared nodes
    duplicate_node = duplicates(edge_node_n1)
    
    idx = indices(edge_node_n1, duplicate_node)
    
    #print("duplicate_node: {0}\n".format(duplicate_node))
    #print("duplicate index: {0} \n".format(idx))
    
    #connect node index in edge_node_n1 list
    connect_code_idx = []
    
    angle_thresh = 100
    
    dis_thresh = (lower_limit + upper_limit)*0.8

    for i in range(len(idx)):
        
        #print("Index = {0}:".format(i))
        #print("Value = {0}:".format(idx[duplicate_node[i]]))
    
        angle_rec = []
        
        for element in idx[duplicate_node[i]]:
            
            #print("element = {0}:".format(element))

            angle_rec.append(edge_angle[element])
            
            #print(edge_node_n1[element])
            #print(edge_node_n2[element])
            
        angle_diff = np.diff(angle_rec)
        
        #print("angle_rec =  {0}".format(angle_rec))
        #print("angle_diff = {0}".format(angle_diff))
        
        
        for j in range(len(angle_diff)):
            
            if angle_diff[j] < angle_thresh:
                
                '''
                print(angle_rec[j],angle_rec[j+1])
                print("Index of duplicated: {0} \n".format(idx[duplicate_node[i]][j]))
                print("Index of duplicated: {0} \n".format(idx[duplicate_node[i]][j+1]))
                print("Index of duplicated: {0} {1} \n".format(j, j+1))
                '''
                connect_code_idx.append(idx[duplicate_node[i]][j])
                #connect_code.append(idx[duplicate_node[i]][j+1])

    
    #connect_node_unique = list(set(connect_code))
    #connect_node_unique = connect_code
    
    print("connect_node: {0}\n".format(connect_code_idx))

    edge_node_n1_select = [j for i, j in enumerate(edge_node_n1) if i not in connect_code_idx]
    edge_node_n2_select = [j for i, j in enumerate(edge_node_n2) if i not in connect_code_idx]
    edge_angle_select = [j for i, j in enumerate(edge_angle) if i not in connect_code_idx]
    edge_length_select = [j for i, j in enumerate(edge_length) if i not in connect_code_idx]
    
    '''
    print("edge_node_n1_select: {0}\n".format(edge_node_n1_select))
    print("edge_node_n2_select: {0}\n".format(edge_node_n2_select))
    print("edge_angle_select: {0}\n".format(edge_angle_select))
    print("edge_length_select: {0}\n".format(edge_length_select))
    
    print("edge_node_n1_select length: {0}\n".format(len(edge_node_n2_select)))
    '''
    # ------------------------------------------------------------------------
    # Select edges with threshold from distance between edges
    #    
    
    close_edge_nd = []
    
    for i in range(len(edge_node_n1_select)):
        
        idx_n1 = edge_node_n1_select[i]
        idx_n2 = edge_node_n2_select[i]
        
        #n1_coord = np.array([xn[idx_n1], yn[idx_n1], zn[idx_n1]])
        #n2_coord = np.array([xn[idx_n2], yn[idx_n2], zn[idx_n2]])

        n12_mid = np.mean([np.array([xn[idx_n1], yn[idx_n1], zn[idx_n1]]), np.array([xn[idx_n2], yn[idx_n2], zn[idx_n2]])], axis = 0)
        
        for j in range(len(edge_node_n1_select)):
            
            if j == i :
                
                dis_P2line = 10000
                #print("self_dis = {0}\n".format(dis_P2line))
           
            else: 
                
                n1_coord_line = np.array([xn[edge_node_n1_select[j]], yn[edge_node_n1_select[j]], zn[edge_node_n1_select[j]]])
                n2_coord_line = np.array([xn[edge_node_n2_select[j]], yn[edge_node_n2_select[j]], zn[edge_node_n2_select[j]]])
                
                dis_P2line = lineseg_dist(n12_mid, n1_coord_line, n2_coord_line)
        
                #print("n1_coord: {0}\n".format(n1_coord_line))
                #print("n2_coord: {0}\n".format(n2_coord_line))
                #print("n12_mid: {0}\n".format(n12_mid))
                #print("distance {0} {1} = {2}\n".format(i, j, dis_P2line))
                try:
                    dif_angle = abs(edge_angle_select[i] - edge_angle_select[j])
                except (IndexError, ValueError):
                    dif_angle = 100
                
                try:
                    if (dis_P2line < dis_thresh) and (dif_angle) < angle_thresh:
                    #if (dis_P2line < dis_thresh) or (edge_length_select[i] < dis_thresh):
                        close_edge_nd.append(i)
                except (IndexError, ValueError):
                    dif_angle = 100
    
    close_edge_nd = np.unique(close_edge_nd)
    
    print("close_edge_nd index = {0}\n".format(close_edge_nd))

    #combined_nd = connect_code_idx + close_edge_nd 
    
    edge_node_n1_select_final = [j for i, j in enumerate(edge_node_n1_select) if i not in close_edge_nd]
    edge_node_n2_select_final = [j for i, j in enumerate(edge_node_n2_select) if i not in close_edge_nd]
    edge_angle_select_final = [j for i, j in enumerate(edge_angle_select) if i not in close_edge_nd]
    edge_length_select_final = [j for i, j in enumerate(edge_length_select) if i not in close_edge_nd]
    
    '''
    print("edge_node_n1_select_final: {0}\n".format(edge_node_n1_select_final))
    print("edge_node_n2_select final: {0}\n".format(edge_node_n2_select_final))
    '''
    
    angle_select = []
    length_select = []
    projection_select = []
    
    #render seletced nodes with text index                
    for i in range(len(edge_node_n1_select_final)):
        
        idx = edge_node_n1_select_final[i]
        
        idx_2 = edge_node_n2_select_final[i]
        
        if i in duplicate_node:
            #pts = mlab.text3d(xn[idx], yn[idx], zn[idx], str(i), scale=(2, 2, 2), color=(1, 0.0, 0.0))
            
            print("mlab.text3d")
            #pts = mlab.plot3d([xn[idx], xn[idx_2]], [yn[idx], yn[idx_2]], [zn[idx], zn[idx_2]], color = (1, 0.0, 0.0), tube_radius = 0.4)
            
        else:
            #pts = mlab.text3d(xn[idx], yn[idx], zn[idx], str(i), scale=(2, 2, 2))
            
            print("mlab.text3d")
            #pts = mlab.plot3d([xn[idx], xn[idx_2]], [yn[idx], yn[idx_2]], [zn[idx], zn[idx_2]], color = (0, 0.0, 1.0), tube_radius = 0.4)
        
        
        
        '''
        try:
            pts = mlab.text3d(xn[idx], yn[idx], zn[idx], str(edge_length_select_final[i]), scale=(2, 2, 2), color=(1, 0.0, 0.0))
        except (IndexError, ValueError):
            pass 
        '''
      
        try:
            angle_select.append(edge_angle[i])
            length_select.append(edge_length[i])
            projection_select.append(edge_projection[i])
        except IndexError:
            pass
 
    angle_select = [0 if math.isnan(x) else x for x in angle_select]
 
    #print("angle_select = {0}\n ".format(angle_select))
    #print("length_select = {0}\n ".format(length_select))
    #print("projection_select = {0}\n ".format(projection_select))

    
    print('graph rendering finished')
    fig.scene.disable_render = disable_render
    

    
    #return tube_surf, pts, edge_node_n1_select_final, edge_node_n2_select_final, edge_angle_select_final, edge_length_select_final, projection_select
    
    return edge_node_n1_select_final, edge_node_n2_select_final, edge_angle_select_final, edge_length_select_final, projection_select
    


