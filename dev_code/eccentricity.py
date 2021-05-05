"""
Version: 1.0
Summary: Mutiple object eccentricity computation
Author: suxing liu
Author-email: suxingliu@gmail.com

USAGE

python eccentricity.py -p /home/suxingliu/MOF_track/brace3/ -ext png 


"""
#!/usr/bin/env python

# import the necessary packages
import cv2

#import morphsnakes
import numpy as np

from scipy.misc import imread
from matplotlib import pyplot as plt
#import matplotlib.patches as mpatches

#import pylab as pl

from scipy.misc import imsave


import argparse
import math
import os
import fnmatch

from skimage.filters import threshold_otsu
from skimage.util import invert
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.morphology import convex_hull_image

from skimage.filters import sobel
from skimage.draw import ellipse


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
        print path + ' folder constructed!'
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        #print path+' path exists!'
        return False
        
def rgb2gray(img):
    
    # Convert a RGB image to gray scale
    return 0.2989*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]


if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to image file")
    ap.add_argument("-ext", "--extension", required = False, default = 'png', help = "extension name. default is 'png'.")  
    #ap.add_argument("-f", "--filename", required = True, help="video file name.")
    #ap.add_argument("-i", "--patternID", type = int, required = True, help="pattern ID")
    args = vars(ap.parse_args())
      
    # Arguments
    file_path = args["path"]
    ext = args['extension']
    
    mkpath = file_path + 'eccentricity'
    mkdir(mkpath)
    save_path = mkpath + '/'
    
    
    
    #accquire image file list
    filetype = '*.' + ext
    images = sorted(fnmatch.filter(os.listdir(file_path), filetype))
    
    
    plt.figure(1)
    
    fig = plt.gcf()
    #fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
   
    for image in images:
        
        plt.clf()
            
        # Capture frame-by-frame
        image_path = os.path.join(file_path, image)
        
        #Load the image
        imgcolor = imread(image_path)
        img = rgb2gray(imgcolor)
        
      
        # Image binarization by apltying otsu threshold
        thresh = threshold_otsu(img)
        binary = img > thresh
     
        binary = invert(binary)
        
        # Extract convex hull of the binary image   
        convexhull = convex_hull_image(binary)
      
        img_mask_convexhull = np.where(convexhull == False, img, 0)
        
        # label image regions
        label_image_convexhull = label(convexhull)
        
        #label_image_convexhull = label(binary)

        
        #label_image_binary = label(binary) 
        #plt.imshow(label_image_convexhull, cmap = 'gray', interpolation='nearest')
        #plt.imshow(label_image_convexhull)
        
        # overlay display
        image_label_overlay = label2rgb(label_image_convexhull, image = imgcolor)
       
       
        image_label_overlay = rgb2gray(image_label_overlay)
        
        #show image overlay with convexhull
        plt.imshow(image_label_overlay, cmap = 'gray', interpolation='nearest')

        # Measure properties of labeled image regions.
        regions = regionprops(label_image_convexhull)

        #unique values in label results
        #print("Length of region is: {0} \n".format(len(np.unique(label_image_binary))))

        # center location of region
        y0, x0 = regions[0].centroid
        #print("Coordinates of centroid: {0} , {0} \n".format(int(y0),int(x0)))
        
        eccentricity = regions[0].eccentricity

        # axis length of region
        d_major = regions[0].major_axis_length
        d_minor = regions[0].minor_axis_length

        diameter = regions[0].equivalent_diameter

        minr, minc, maxr, maxc = regions[0].bbox
        d_bbox = max(maxr - minr, maxc - minc)
        radius = int(max(d_major, d_minor, d_bbox)/2) + 20

        print("Eccentricity of image {0} is: {1} \n".format(str(image), eccentricity))
        
        #Get the ellipse of convexhull 
        rr, cc = ellipse(y0, x0, 0.5 * d_major, 0.5 * d_minor)
        
        #mask_ellipse = np.zeros([img.shape[0], img.shape[1]], dtype = np.uint8)
      
        #mask_ellipse[rr, cc] = 1
       
        #edge = sobel(mask_ellipse)
        
        '''
        indices = np.where(edge == [0])
        
        #print indices
        
        coordinates = zip(indices[0], indices[1])
                
        coordinates = np.asarray(coordinates)
        
        #print(coordinates.shape)
        '''

        for props in regions:
            y0, x0 = props.centroid
            orientation = props.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
            x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length
            
            plt.plot((x0, x1), (y0, y1), '-r', linewidth = 2.5)
            plt.plot((x0, x2), (y0, y2), '-r', linewidth = 2.5)
            plt.plot(x0, y0, '.g', markersize = 15)
            
            #plt.plot(rr,cc, 'r' ,'LineWidth', 1)
            
            #minr, minc, maxr, maxc = props.bbox
            #bx = (minc, maxc, maxc, minc, minc)
            #by = (minr, minr, maxr, maxr, minr)
            #plt.plot(bx, by, '-b', linewidth = 2.5)

            plt.pause(0.1)
            #plt.close(fig)
        
        plt.title(str(image))
        
        result_file_name_path = save_path + str(image) + '_eccentricity.png'
        
        fig.savefig(result_file_name_path, bbox_inches = 0, pad_inches = 0)
        
        plt.pause(0.05)
        
    
    plt.close()
    
    
    
############opencv method
'''
frame = cv2.imread(image_path)

height, width, channels = frame.shape

#Filter image using Meanshif 
#shifted = cv2.pyrMeanShiftFiltering(frame, 3, 3)

#Convert the mean shift image to grayscale, then aplty Otsu's thresholding
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#Obtain the threshold image using OTSU adaptive filter
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


# Finding contours for the thresholded image
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# create hull array for convex hull points
hull = []
 
# calculate points for each contour
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull.append(cv2.convexHull(contours[i], False))
    
# create an empty black image
drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
 
# draw contours and hull points
for i in range(len(contours)):
    color_contours = (0, 255, 0) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw ith contour
    cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    cv2.drawContours(drawing, hull, i, color, 1, 8)


cv2.imshow('Original', img_mask_convexhull)

# Slower the FPS
cv2.waitKey(500)

'''
    
    
    
     

