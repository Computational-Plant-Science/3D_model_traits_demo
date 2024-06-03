"""
Version: 1.5

Summary: ConvelHull based bounding box segmenation for image sequence

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 bbox_seg_mask.py -p ~/example/test/ -ft jpg 


argument:
("-p", "--path", required = True,    help="path to image file")
("-ft", "--filetype", required = True,    help="Image filetype")

"""

#!/usr/bin/python
# Standard Libraries

import os,fnmatch
import argparse
import shutil
import cv2

import numpy as np
#import matplotlib.pyplot as plt

import glob
from math import atan2, cos, sin, sqrt, pi, radians

import multiprocessing
from multiprocessing import Pool
from contextlib import closing

import cv2
#import psutil

import resource
import os

from pathlib import Path 

from rembg import remove



# create result folder
def mkdir(path):
    # import module
    #import os
 
    # remove space at the beginning
    path=path.strip()
    # remove slash at the end
    path=path.rstrip("\\")
 
    # path exist?   # True  # False
    isExists=os.path.exists(path)
 
    # process
    if not isExists:
        # construct the path and folder
        #print path + ' folder constructed!'
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        #print path+' path exists!'
        return False


def get_orientation(pts):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    '''
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    draw_axis(img, cntr, p1, (0, 150, 0), 1)
    draw_axis(img, cntr, p2, (200, 150, 0), 5)
    '''
    
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    print("orientation in radians is {}".format(angle*180/pi))
    
    return angle


def slope(x1, y1, x2, y2):
    if x1 == x2:
        return 0
    slope = (y2-y1)/(x2-x1)
    theta = np.rad2deg(np.arctan(slope))
    return theta


def createMask(rows, cols, hull, value):
    
    # black image
    mask = np.zeros((rows, cols), dtype=np.uint8)
    
    # assign contours in white color
    cv2.drawContours(mask, [hull], 0, 255, -1)
    
    return mask
    

def foreground_substractor(image_file):
    
  
    #parse the file name 
    path, filename = os.path.split(image_file)
    
    # construct the result file path
    #result_img_path = save_path + str(filename[0:-4]) + '_seg.png'
    
    print("Extracting foreground for image : {0} \n".format(str(filename)))
    
    # Load the image
    image = cv2.imread(image_file)
    
    if image is None:
        print(f"Could not load image {image_file}, skipping")
        return
    
    
    gamma = 1.0
    
    # apply gamma correction and show the images
    gamma = gamma if gamma > 0 else 0.1
    
    adjusted = adjust_gamma(image, gamma=gamma)
    
    image = image_enhance(adjusted)
    
    
    
    ori = image.copy()
    
    #get size of image
    img_height, img_width = image.shape[:2]
    
    #scale_factor = 1
    
    #image = cv2.resize(image, (0,0), fx = scale_factor, fy = scale_factor) 
    
    # Convert BGR to GRAY
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.blur(gray, (3, 3)) # blur the image
    #blur = cv2.GaussianBlur(gray, (25, 25), 0)
    
    #Obtain the threshold image using OTSU adaptive filter
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    #thresh = cv2.erode(thresh, None, iterations=2)
    
    #thresh = cv2.dilate(thresh, None, iterations=2)
 

            
    # extract the contour of subject
    #im, cnts, hier = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts, hier = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #find the max contour 
    c = max(cnts, key = cv2.contourArea)
    

    mask_contour = createMask(img_height, img_width, c, 0)
    
    masked_fg_contour = cv2.bitwise_and(image, image, mask = mask_contour)
    
    
    linewidth = 10
    
    hull = cv2.convexHull(c)
    
    mask_hull = createMask(img_height, img_width, hull, 0)
    
    kernel = np.ones((10,10), np.uint8)
    
    mask_hull_dilation = cv2.dilate(mask_hull, kernel, iterations=1)
    
    # apply individual object mask
    masked_fg = cv2.bitwise_and(image, image, mask = mask_hull_dilation)
    
    masked_bk = cv2.bitwise_and(image, image, mask = ~ mask_hull_dilation)
    
    masked_bk_gray = cv2.cvtColor(masked_bk, cv2.COLOR_BGR2GRAY)
    
    avg_color_per_row = np.average(masked_bk_gray, axis=0)
    
    avg_color = int(np.average(avg_color_per_row, axis=0))
    

    
    #print(avg_color)
    
    # black image
    #mask_hull_avg = np.full((img_height, img_width), avg_color, dtype=np.uint8)
    
    # assign contours in white color
    #cv2.drawContours(mask_hull_avg, [hull], 0, 255, -1)
    
    #masked_bk_avg = cv2.bitwise_and(mask_hull_avg, mask_hull_avg, mask = ~ mask_hull_dilation)
    
    
    # replace background color with average color value
    # load background (could be an image too)
    bk = np.full(image.shape, avg_color, dtype=np.uint8)
    
    # get masked foreground
    
    fg_masked = cv2.bitwise_and(image, image, mask=mask_hull_dilation)
    
    # get masked background, mask must be inverted
    
    bk_masked = cv2.bitwise_and(bk, bk, mask=cv2.bitwise_not(mask_hull_dilation))
    
    # combine masked foreground and masked background
    combined_fg_bk = cv2.bitwise_or(fg_masked, bk_masked)
    
 
    ##############################################################
    
    # find the bouding box of the max contour
    (x, y, w, h) = cv2.boundingRect(c)

    # draw the max bounding box
    #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # construct the result file path
    #result_img_path = save_path + str(filename[0:-4]) + '_seg.' + ext
    
    # save result as images for reference
    #cv2.imwrite(result_img_path,image)
    
    #print(int(x/scale_factor),int(y/scale_factor),int(w/scale_factor),int(h/scale_factor))
    
    #return int(x/scale_factor),int(y/scale_factor),int(w/scale_factor),int(h/scale_factor), int(img_width), int(img_height)
    

    
    
    margin = 150
    
    # define crop region
    start_y = int((y - margin) if (y - margin )> 0 else 0)
    
    start_x = int((x - margin) if (x - margin )> 0 else 0)
    
    crop_width = int((x + margin + w) if (x + margin + w) < img_width else (img_width))
    
    crop_height = int((y + margin + h) if (y + margin + h) < img_height else (img_height))
    
    #print img_width , img_height 
    
    # construct the result file path
    result_img_path = save_path + str(filename[0:-4]) + '_seg.' + ext
    
    #crop_img = combined_fg_bk[start_y:crop_height, start_x:crop_width]
    
    #crop_img = masked_fg_contour[start_y:crop_height, start_x:crop_width]
    
    if args['bounding_box'] == 1:
        crop_img = ori[start_y:crop_height, start_x:crop_width]
    else:
        crop_img = combined_fg_bk[start_y:crop_height, start_x:crop_width]
        #crop_img = masked_fg_contour[start_y:crop_height, start_x:crop_width]
    
    
    # PhotoRoom Remove Background API
    ai_crop = remove(crop_img).copy()

    #orig = roi_image.copy()
    if ai_crop.shape[2] > 3:

        ai_crop = cv2.cvtColor(ai_crop, cv2.COLOR_RGBA2RGB)
    
    
    cv2.imwrite(result_img_path, ai_crop)
    
    
    #return int(x),int(y),int(w),int(h), int(img_width), int(img_height)


#adjust the gamma value to increase the brightness of image
def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)



#apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to perfrom image enhancement
def image_enhance(img):

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    # convert from BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  
    
    # split on 3 different channels
    l, a, b = cv2.split(lab)  

    # apply CLAHE to the L-channel
    l2 = clahe.apply(l)  

    # merge channels
    lab = cv2.merge((l2,a,b))  
    
    # convert from LAB to BGR
    img_enhance = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  
    
    return img_enhance





if __name__ == '__main__':

    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", required = False,  default = 'jpg' ,    help = "image filetype")
    ap.add_argument("-b", "--bounding_box", required = False,  type = int, default = 0, help = "using bounding box")
    args = vars(ap.parse_args())

    # setting path to model file
    file_path = args["path"]
    ext = args['filetype']

    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype

    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))

    #print((imgList))


    # make the folder to store the results
    parent_path = os.path.abspath(os.path.join(file_path, os.pardir))
    mkpath = file_path + '/' + str('segmented')
    mkdir(mkpath)
    save_path = mkpath + '/'

    print ("results_folder: " + save_path)

    
    # get cpu number for parallel processing
    #agents = psutil.cpu_count()   
    agents = multiprocessing.cpu_count()-1
    
    
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        pool.map(foreground_substractor, imgList)
        pool.terminate()
    
    
    
    '''
    for image_file in imgList:
        foreground_substractor(image_file)
        
    
    
    # parse the result 
    x = min(list(zip(*result)[0]))
    
    y = min(list(zip(*result)[1]))
    
    width = max(list(zip(*result)[2]))
    
    height = max(list(zip(*result)[3]))
    
    img_width = min(list(zip(*result)[4]))
    
    img_height = min(list(zip(*result)[5]))
    
    print("Coordinates of max bounding box: {0} , {1} \n".format(int(x),int(y)))
    
    print("Dimension of max bounding box: {0} , {1} \n".format(int(width),int(height)))
    
    print("Image size: {0} , {1} \n".format(int(img_width),int(img_height)))
    
    
    
    # perfrom crop action based on bouding box results in parallel way
    with closing(Pool(processes = agents)) as pool:
        #pool.map(crop_pil, imgList)
        pool.map(crop_image, imgList)
        pool.terminate()
    

    # monitor memory usage 
    rusage_denom = 1024.0
    
    print("Memory usage: {0} MB\n".format(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom)))
    '''
    
    
    '''
    #rename all the images in soting order
    ######################################################
     # setting path to result file
    ori_path = current_path
    dst_path = save_path

     #accquire image file list
    imgList_ori = sorted(fnmatch.filter(os.listdir(ori_path), filetype))
    imgList_dst = sorted(fnmatch.filter(os.listdir(dst_path), filetype))

    imgList = sorted(imgList_ori + imgList_dst)

    # make the folder to store the results
    mkpath = current_path + str('interpolation_result')
    mkdir(mkpath)
    save_path = mkpath + '/'

    #Combine the interpolation result with original images and rename all the results
    file_sort(imgList)
    
    #delete the interpolation result folder
    try:
        shutil.rmtree(dst_path, ignore_errors=True)
        
        print "Phase based motion frame prediction and interpolation was finished!\n"
        
        print "results_folder: " + save_path  
        
    except OSError:
        pass

    '''



 

    

    

    
  

   
    
    



