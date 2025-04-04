# -*- coding: utf-8 -*-
"""
Version: 1.0
Summary: 
This is a Python implementation of the algorithms introduced in the paper

  Márquez-Neila, P., Baumela, L., Álvarez, L., "A morphological approach
  to curvature-based evolution of curves and surfaces". IEEE Transactions
  on Pattern Analysis and Machine Intelligence (PAMI), 2013.

This implementation is intended to be as brief, understandable and self-contained
as possible. It does not include any enhancement to make it fast or efficient.

Any practical implementation of this algorithm should work only over the
neighbor pixels of the 0.5-levelset, not over all the embedding function,
and perhaps should feature multi-threading or GPU capabilities.

The classes MorphGAC and MorphACWE provide most of the functionality of this
module. 
Author: suxing liu
Author-email: suxingliu@gmail.com

USAGE

import morphsnakes

"""



from itertools import cycle
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, gaussian_gradient_magnitude
                        
import cv2
from skimage import img_as_ubyte

class fcycle(object):
    
    def __init__(self, iterable):
        """Call functions from the iterable each time it is called."""
        self.funcs = cycle(iterable)
    
    def __call__(self, *args, **kwargs):
        f = next(self.funcs)
        return f(*args, **kwargs)
    

# SI and IS operators for 2D and 3D.
_P2 = [np.eye(3), np.array([[0,1,0]]*3), np.flipud(np.eye(3)), np.rot90([[0,1,0]]*3)]
_P3 = [np.zeros((3,3,3)) for i in range(9)]

_P3[0][:,:,1] = 1
_P3[1][:,1,:] = 1
_P3[2][1,:,:] = 1
_P3[3][:,[0,1,2],[0,1,2]] = 1
_P3[4][:,[0,1,2],[2,1,0]] = 1
_P3[5][[0,1,2],:,[0,1,2]] = 1
_P3[6][[0,1,2],:,[2,1,0]] = 1
_P3[7][[0,1,2],[0,1,2],:] = 1
_P3[8][[0,1,2],[2,1,0],:] = 1

_aux = np.zeros((0))
def SI(u):
    """SI operator."""
    global _aux
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions (should be 2 or 3)")
    
    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P),) + u.shape)
    
    for _aux_i, P_i in zip(_aux, P):
        _aux_i[:] = binary_erosion(u, P_i)
    
    return _aux.max(0)

def IS(u):
    """IS operator."""
    global _aux
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions (should be 2 or 3)")
    
    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P),) + u.shape)
    
    for _aux_i, P_i in zip(_aux, P):
        _aux_i[:] = binary_dilation(u, P_i)
    
    return _aux.min(0)

# SIoIS operator.
SIoIS = lambda u: SI(IS(u))
ISoSI = lambda u: IS(SI(u))
curvop = fcycle([SIoIS, ISoSI])

# Stopping factors (function g(I) in the paper).
def gborders(img, alpha=1.0, sigma=1.0):
    """Stopping criterion for image borders."""
    # The norm of the gradient.
    gradnorm = gaussian_gradient_magnitude(img, sigma, mode='constant')
    return 1.0/np.sqrt(1.0 + alpha*gradnorm)

def glines(img, sigma=1.0):
    """Stopping criterion for image black lines."""
    return gaussian_filter(img, sigma)

class MorphACWE(object):
    """Morphological ACWE based on the Chan-Vese energy functional."""
    
    def __init__(self, data, smoothing=1, lambda1=1, lambda2=1):
        """Create a Morphological ACWE solver.
        
        Parameters
        ----------
        data : ndarray
            The image data.
        smoothing : scalar
            The number of repetitions of the smoothing step (the
            curv operator) in each iteration. In other terms,
            this is the strength of the smoothing. This is the
            parameter µ.
        lambda1, lambda2 : scalars
            Relative importance of the inside pixels (lambda1)
            against the outside pixels (lambda2).
        """
        self._u = None
        self.smoothing = smoothing
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        self.data = data
    
    def set_levelset(self, u):
        self._u = np.double(u)
        self._u[u>0] = 1
        self._u[u<=0] = 0
    
    levelset = property(lambda self: self._u,
                        set_levelset,
                        doc="The level set embedding function (u).")
    
    def step(self):
        """Perform a single step of the morphological Chan-Vese evolution."""
        # Assign attributes to local variables for convenience.
        u = self._u
        
        if u is None:
            raise ValueError("the levelset function is not set (use set_levelset)")
        
        data = self.data
        
        # Determine c0 and c1.
        inside = u>0
        outside = u<=0
        c0 = data[outside].sum() / float(outside.sum())
        c1 = data[inside].sum() / float(inside.sum())
        
        # Image attachment.
        dres = np.array(np.gradient(u))
        abs_dres = np.abs(dres).sum(0)
        #aux = abs_dres * (c0 - c1) * (c0 + c1 - 2*data)
        aux = abs_dres * (self.lambda1*(data - c1)**2 - self.lambda2*(data - c0)**2)
        
        res = np.copy(u)
        res[aux < 0] = 1
        res[aux > 0] = 0
        
        # Smoothing.
        for i in range(self.smoothing):
            res = curvop(res)
        
        self._u = res
    
    def run(self, iterations):
        """Run several iterations of the morphological Chan-Vese method."""
        for i in range(iterations):
            self.step()
    

class MorphGAC(object):
    """Morphological GAC based on the Geodesic Active Contours."""
    
    def __init__(self, data, smoothing=1, threshold=0, balloon=0):
        """Create a Morphological GAC solver.
        
        Parameters
        ----------
        data : array-like
            The stopping criterion g(I). See functions gborders and glines.
        smoothing : scalar
            The number of repetitions of the smoothing step in each
            iteration. This is the parameter µ.
        threshold : scalar
            The threshold that determines which areas are affected
            by the morphological balloon. This is the parameter θ.
        balloon : scalar
            The strength of the morphological balloon. This is the parameter ν.
        """
        self._u = None
        self._v = balloon
        self._theta = threshold
        self.smoothing = smoothing
        
        self.set_data(data)
    
    def set_levelset(self, u):
        self._u = np.double(u)
        self._u[u>0] = 1
        self._u[u<=0] = 0
    
    def set_balloon(self, v):
        self._v = v
        self._update_mask()
    
    def set_threshold(self, theta):
        self._theta = theta
        self._update_mask()
    
    def set_data(self, data):
        self._data = data
        self._ddata = np.gradient(data)
        self._update_mask()
        # The structure element for binary dilation and erosion.
        self.structure = np.ones((3,)*np.ndim(data))
    
    def _update_mask(self):
        """Pre-compute masks for speed."""
        self._threshold_mask = self._data > self._theta
        self._threshold_mask_v = self._data > self._theta/np.abs(self._v)
    
    levelset = property(lambda self: self._u,
                        set_levelset,
                        doc="The level set embedding function (u).")
    data = property(lambda self: self._data,
                        set_data,
                        doc="The data that controls the snake evolution (the image or g(I)).")
    balloon = property(lambda self: self._v,
                        set_balloon,
                        doc="The morphological balloon parameter (ν (nu, not v)).")
    threshold = property(lambda self: self._theta,
                        set_threshold,
                        doc="The threshold value (θ).")
    
    def step(self):
        """Perform a single step of the morphological snake evolution."""
        # Assign attributes to local variables for convenience.
        u = self._u
        gI = self._data
        dgI = self._ddata
        theta = self._theta
        v = self._v
        
        if u is None:
            raise ValueError("the levelset is not set (use set_levelset)")
        
        res = np.copy(u)
        
        # Balloon.
        if v > 0:
            aux = binary_dilation(u, self.structure)
        elif v < 0:
            aux = binary_erosion(u, self.structure)
        if v!= 0:
            res[self._threshold_mask_v] = aux[self._threshold_mask_v]
        
        # Image attachment.
        aux = np.zeros_like(res)
        dres = np.gradient(res)
        for el1, el2 in zip(dgI, dres):
            aux += el1*el2
        res[aux > 0] = 1
        res[aux < 0] = 0
        
        # Smoothing.
        for i in range(self.smoothing):
            res = curvop(res)
        
        self._u = res
    
    def run(self, iterations):
        """Run several iterations of the morphological snakes method."""
        for i in range(iterations):
            self.step()


#def evolve_visual(msnake, levelset = None, num_iters = 20, background = None, result_path = None, mask = 0, H = 400, W = 300):
def evolve_visual(msnake, levelset = None, num_iters = 20, background = None):
    """
    Visual evolution of a morphological snake.
    
    Parameters
    ----------
    msnake : MorphGAC or MorphACWE instance
        The morphological snake solver.
    levelset : array-like, optional
        If given, the levelset of the solver is initialized to this. If not
        given, the evolution will use the levelset already set in msnake.
    num_iters : int, optional
        The number of iterations.
    background : array-like, optional
        If given, background will be shown behind the contours instead of
        msnake.data.
    """
       
    #load the image and perform pyramid mean shift filtering to aid the thresholding step
    im_color = cv2.imread('/home/suxingliu/test/active_snake/0348.jpg')
    
    if levelset is not None:
        msnake.levelset = levelset
        
    # Iterate.
    for i in range(num_iters):
        # Evolve.
        msnake.step()
        
    '''
        #visualize the iteration steps
        # Prepare the visual environment.
        fig = plt.gcf()
        fig.clf()
        
        ax = plt.Axes(fig, [0., 0., 1.0, 1.0])
        ax.set_axis_off()
        
        ax.set_frame_on(False)
        plt.axis('off')
        
        
        fig.add_axes(ax)
            
        if background is None:
            ax.imshow(msnake.data, cmap=plt.cm.gray)
        else:
            ax.imshow(background, cmap=plt.cm.gray)
            
        ax.contour(msnake.levelset, [0.5], colors='r')
        
        plt.pause(0.001)
        
           
        # Update figure.
        del ax.collections[0]
        ax.contour(msnake.levelset, [0.5], colors='r')
        fig.canvas.draw()
        
        result_img_path = '/home/suxingliu/active_snake_steps/active_component/' + str('{:03}'.format(i)) + '.png'
        
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        
        fig.savefig(result_img_path, bbox_inches=extent)
        
    '''
    #plt.show()
    
    
    
    '''
    if mask==1:
        #print("mask == 1")
        ######################################################
        # Prepare the visual environment.
        #fig = plt.gcf()
        #fig.set_size_inches(10, 6)
        #fig.clf()
        
        fig = plt.figure(frameon=False)
        
        DPI = fig.get_dpi()
        
        fig.set_size_inches(W/float(DPI),H/float(DPI))

        #fig.set_size_inches(w,h)
    
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off() 
        ax.set_frame_on(False)
        
        #ax.imshow(data, cmap=plt.cm.gray)
        
        fig.add_axes(ax)



       
        #if background is None:
            #ax1.imshow(msnake.data, cmap=plt.cm.gray)
        #else:
            #ax1.imshow(background, cmap=plt.cm.gray)
        #ax1.contour(msnake.levelset, [0.5], colors='r')
        
        
        #ax = fig.add_subplot(1,2,2)
        
        #ax_u = ax.imshow(msnake.levelset, cmap = plt.cm.gray)
        
        ax_u = ax.imshow(msnake.levelset)

        
        plt.pause(0.001)
        
        # Iterate.
        for i in range(num_iters):
            # Evolve.
            msnake.step()
            
            # Update figure.
            #del ax1.collections[0]
            #ax1.contour(msnake.levelset, [0.5], colors='r')
            ax_u.set_data(msnake.levelset)
            
            fig.canvas.draw()
            #plt.pause(0.001)
        
        #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #fig.savefig('ax1_figure.png', bbox_inches=extent)
        
    
        if result_path is None:
            fig.savefig('active_snake.png', bbox_inches = 0, pad_inches = 0)
        else:
            fig.savefig(result_path, bbox_inches = 0, pad_inches = 0)
            
        plt.close(fig)
        
        ##########################################################
    
    elif mask == 0: 
        
        ############################################################
        # Prepare the visual environment.
        fig = plt.gcf()
        fig.clf()
        
        #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
        
        #print("mask == 0")
        
        ax = plt.Axes(fig, [0., 0., 1.0, 1.0])
        ax.set_axis_off()
        
        ax.set_frame_on(False)
        plt.axis('off')
        
        
        fig.add_axes(ax)
            
        if background is None:
            ax.imshow(msnake.data, cmap=plt.cm.gray)
        else:
            ax.imshow(background, cmap=plt.cm.gray)
            
        ax.contour(msnake.levelset, [0.5], colors='r')
        
        plt.pause(0.001)
        
        # Iterate.
        for i in range(num_iters):
            # Evolve.
            msnake.step()
            
            # Update figure.
            del ax.collections[0]
            ax.contour(msnake.levelset, [0.5], colors='r')
            fig.canvas.draw()
        
        #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        
        if result_path is None:
            fig.savefig('active_snake.png', bbox_inches = 0, pad_inches = 0)
        else:
            fig.savefig(result_path, bbox_inches = 0, pad_inches = 0)
            
        plt.close(fig)
        #########################################################################3
        
        # Iterate.
        for i in range(num_iters):
            # Evolve.
            msnake.step()
        
    else:
        print('Wrong parameter!')
        
    '''

    
    # Return the last levelset.
    return msnake.levelset

def evolve_visual3d(msnake, levelset=None, num_iters=20):
    """
    Visual evolution of a three-dimensional morphological snake.
    
    Parameters
    ----------
    msnake : MorphGAC or MorphACWE instance
        The morphological snake solver.
    levelset : array-like, optional
        If given, the levelset of the solver is initialized to this. If not
        given, the evolution will use the levelset already set in msnake.
    num_iters : int, optional
        The number of iterations.
    """
    from mayavi import mlab
    import matplotlib.pyplot as plt
    
    if levelset is not None:
        msnake.levelset = levelset
    
    fig = mlab.gcf()
    mlab.clf()
    src = mlab.pipeline.scalar_field(msnake.data)
    mlab.pipeline.image_plane_widget(src, plane_orientation='x_axes', colormap='gray')
    cnt = mlab.contour3d(msnake.levelset, contours=[0.5])
    
    @mlab.animate(ui=True)
    def anim():
        for i in range(num_iters):
            msnake.step()
            cnt.mlab_source.scalars = msnake.levelset
            print("Iteration %s/%s..." % (i + 1, num_iters))
            yield
    
    anim()
    mlab.show()
    
    # Return the last levelset.
    return msnake.levelset
