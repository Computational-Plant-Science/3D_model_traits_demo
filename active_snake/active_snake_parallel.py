"""
Version: 1.0
Summary: active contour and levelset
Author: suxing liu
Author-email: suxingliu@gmail.com

USAGE

python3 active_snake_parallel.py -p ~/ply_data/cross_section_scan/ 


"""
#!/usr/bin/env python

# import the necessary packages
import morphsnakes
import numpy as np

#import imageio

import argparse
import math

from skimage import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.util import invert
from skimage.measure import regionprops, label
from skimage.morphology import convex_hull_image

import glob,os
import cv2

import multiprocessing
from multiprocessing import Pool
from contextlib import closing

def mkdir(path):
    # import module
    import os
 
    # remove space at the beginning
    path=path.strip()
    # remove slash at the end
    path=path.rstrip("\\")
 
    # path exist?   # True  # False
    isExists=os.path.exists(path)
 
    # process
    if not isExists:
        # construct the path and folder
        print (path + ' folder constructed!')
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        #print path+' path exists!'
        return False

# distance function
def dist(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def rgb2gray(img):
    
    # Convert a RGB image to gray scale
    return 0.2989*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    
    # Build a binary function with a circle as the 0.5-levelset
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u


def active_snake(image_file):
    
    
    #Parse image path  and create result image path
    path, filename = os.path.split(image_file)
    
    print("processing image : {0} \n".format(str(filename)))
    
    #load the image and perform pyramid mean shift filtering to aid the thresholding step
    imgcolor = cv2.imread(image_file)
    img_gray = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)
    
    # Image binarization by applying otsu threshold
    thresh = threshold_otsu(img_gray)
    binary = img_gray > thresh
 
    # Extract convex hull of the binary image   
    convexhull = convex_hull_image(invert(binary))
    
    # label image regions
    label_image_convexhull = label(convexhull)
    
    # Measure properties of labeled image regions.
    regions = regionprops(label_image_convexhull)
    
    # center location of region
    y0, x0 = regions[0].centroid
    #print(y0,x0)
    print("Coordinates of centroid: {0} , {0} \n".format(y0,x0))
    
    # axis length of region
    d_major = regions[0].major_axis_length
    d_minor = regions[0].minor_axis_length
    
    diameter = regions[0].equivalent_diameter
    
    minr, minc, maxr, maxc = regions[0].bbox
    d_bbox = max(maxr - minr, maxc - minc)
    radius = int(max(d_major, d_minor, d_bbox)/2) + 20
    
    print("Radius of convex hull region is: {0} \n".format(radius))
    
    gI = morphsnakes.gborders(img_gray, alpha = 5, sigma = 1)
        
    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing = 1, threshold = 0.24, balloon = -1)
    
    mgac.levelset = circle_levelset(img_gray.shape, (y0, x0), radius, scalerow = 0.75)
  
    # Visual evolution.
    morphsnakes.evolve_visual(mgac, num_iters = num_iters, background = imgcolor)
    
    #define result path for simplified segmentation result
    result_img_path = save_path_ac + str(filename[0:-4]) + '.png'
    
    # suppose that img's dtype is 'float64'
    img_uint8 = img_as_ubyte(mgac.levelset)
    
    cv2.imwrite(result_img_path,img_uint8)
    


if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help="path to image file")
    ap.add_argument("-ft", "--filetype", required = False, default = 'jpg',   help = "Image filetype")
    #ap.add_argument("-m", "--mask", required = False,  type = int, default = 0, help = "1 for contour or 0 for component")
    args = vars(ap.parse_args())
    
    global save_path_ac, num_iters
    # setting path to cross section image files
    file_path = args["path"]
    ext = args['filetype']
    
    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype

    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))

    # make the folder to store the results
    parent_path = os.path.abspath(os.path.join(file_path, os.pardir))
    mkpath = parent_path + '/' + str('active_component')
    mkdir(mkpath)
    save_path_ac = mkpath + '/'
    
    num_iters = 230

    
    '''
    for idx, image_file in enumerate(imgList):
        
        num_iters_update = num_iters + idx*0.5
        
        
        if idx < 100:
            num_iters_update = 80
        elif idx < 300:
            num_iters_update = 120
        else:
            num_iters_update = 150
        
        print(num_iters_update)
        
        
        active_snake(image_file, int(num_iters))
    '''
    
    # Run this with a pool of avaliable agents having a chunksize of 3 until finished
    agents = multiprocessing.cpu_count() - 2
    chunksize = 3
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(active_snake, imgList, chunksize)
        pool.terminate()
    
    
     

