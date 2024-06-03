'''
Name: trait_extract_parallel.py

Version: 1.0

Summary: Extract plant shoot traits (larea, solidity, max_width, max_height, avg_curv, color_cluster) by paralell processing 
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2018-09-29

USAGE:

    python3 template_match.py -p ~/example/plant_test/mi_test/ -ft png  

'''

# import the necessary packages
import os
import glob


import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.feature import match_template
from skimage.feature import peak_local_max


import imutils


import argparse
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import math
import openpyxl
import csv
    

from pathlib import Path 



# generate foloder to store the output results
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
        #print path + ' folder constructed!'
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        #print path+' path exists!'
        return False
        


def circle_detection(image):

    """Detecting Circles in Images using OpenCV and Hough Circles
    
    Inputs: 
    
        image: image loaded 

    Returns:
    
        circles: detcted circles
        
        circle_detection_img: circle overlayed with image
        
        diameter_circle: diameter of detected circle
        
    """
    
    # create background image for drawing the detected circle
    output = image.copy()
    
    # obtain image dimension
    img_height, img_width, n_channels = image.shape
    
    #backup input image
    circle_detection_img = image.copy()
    
    # change image from RGB to Gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # apply blur filter 
    blurred = cv2.medianBlur(gray, 25)
    
    # setup parameters for circle detection
    
    # This parameter is the inverse ratio of the accumulator resolution to the image resolution 
    #(see Yuen et al. for more details). Essentially, the larger the dp gets, the smaller the accumulator array gets.
    dp = 1.5
    
    #Minimum distance between the center (x, y) coordinates of detected circles. 
    #If the minDist is too small, multiple circles in the same neighborhood as the original may be (falsely) detected. 
    #If the minDist is too large, then some circles may not be detected at all.
    minDist = 100
    
    #Gradient value used to handle edge detection in the Yuen et al. method.
    #param1 = 30
    
    #accumulator threshold value for the cv2.HOUGH_GRADIENT method. 
    #The smaller the threshold is, the more circles will be detected (including false circles). 
    #The larger the threshold is, the more circles will potentially be returned. 
    #param2 = 30  
    
    #Minimum/Maximum size of the radius (in pixels).
    #minRadius = 80
    #maxRadius = 120 
    
    # detect circles in the image
    #circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    
    # detect circles in the image
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, minDist)
    
    # initialize diameter of detected circle
    diameter_circle = 0
    
    
    circle_center_coord = []
    circle_center_radius = []
    idx_closest = 0
    
    
    # At leaset one circle is found
    if circles is not None:
        
        # Get the (x, y, r) as integers, convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
       
        if len(circles) < 2:
           
            print("Only one circle was found!\n")
           
        else:
            
            print("More than one circles were found!\n")
        
            idx_closest = 0
        
            #cv2.circle(output, (x, y), r, (0, 255, 0), 2)
          
        # loop over the circles and the (x, y) coordinates to get radius of the circles
        for (x, y, r) in circles:
            
            coord = (x, y)
            
            circle_center_coord.append(coord)
            circle_center_radius.append(r)

        if idx_closest == 0:

            print("Circle marker with radius = {} was detected!\n".format(circle_center_radius[idx_closest]))
        
        '''
        # draw the circle in the output image, then draw a center
        circle_detection_img = cv2.circle(output, circle_center_coord[idx_closest], circle_center_radius[idx_closest], (0, 255, 0), 4)
        circle_detection_img = cv2.circle(output, circle_center_coord[idx_closest], 5, (0, 128, 255), -1)

        # compute the diameter of coin
        diameter_circle = circle_center_radius[idx_closest]*2


        tmp_mask = np.zeros([img_width, img_height], dtype=np.uint8)

        tmp_mask = cv2.circle(tmp_mask, circle_center_coord[idx_closest], circle_center_radius[idx_closest] + 5, (255, 255, 255), -1)

        tmp_mask_binary = cv2.threshold(tmp_mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        masked_tmp = cv2.bitwise_and(image.copy(), image.copy(), mask = ~tmp_mask_binary)
        '''

        (startX, startY) = circle_center_coord[idx_closest]

        endX = startX + int(r*1.2) + 1050
        endY = startY + int(r*1.2) + 1050

        #sticker_crop_img = output[startY:endY, startX:endX]
        
        sticker_crop_img = output
    
    else:
        
        print("No circle was found!\n")
        
        sticker_crop_img = output
        
        diameter_circle = 0
    
    return circles, sticker_crop_img, diameter_circle



# Detect stickers in the image
def template_detect_CV(image_file, template):
    
    
    # load the image, clone it for output, and then convert it to grayscale
    img_rgb = cv2.imread(image_file)
    
    # Convert it to grayscale 
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 
      
    # Store width and height of template in w and h 
    w, h = template.shape[::-1] 
      
    # Perform match operations. 
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    
    #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(res)
    
    
    # Specify a threshold 
    #threshold = 0.4
    
    if np.amax(res) > threshold:
        
        print("Found matched template...")
        
        # Store the coordinates of matched area in a numpy array 
        loc = np.where( res >= threshold)  
    
        if len(loc):
        
            (y,x) = np.unravel_index(res.argmax(), res.shape)
        
            (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(res)
        
            #print(y,x)
            
            #print(min_val, max_val, min_loc, max_loc)
            
            
            (startX, startY) = max_loc
            endX = startX + template.shape[0] 
            endY = startY + template.shape[1] 
            

            
            # Draw a rectangle around the matched region. 
            for pt in zip(*loc[::-1]): 
                
                template_overlay = cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 1)
            
            
            template_crop_img = img_rgb[startY:endY, startX:endX]


        return  template_crop_img, template_overlay, template.shape[0], template.shape[1] 
    else:

        print("Template not found, adjust threshold for template matching...")
        
        return  0, 0, 0, 0 
    
    
    






def template_detect_ski(image_file, template):
    

    
    ##########################################################
    img_rgb = imread(image_file)
    
    img_gray = rgb2gray(img_rgb)
    
    
    result = match_template(img_gray, template)
    
    (x, y) = np.unravel_index(np.argmax(result), result.shape)
    
    print((x, y))

    
    
    
    template_width, template_height = template.shape
    
    rect = plt.Rectangle((y, x), template_height, template_width, color='y', fc='none')
    
    plt.gca().add_patch(rect)
    
    imshow(img_gray)
    
    plt.show()
    
    
    




if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-tp", "--temp_path", required = False,  help = "template image path")
    ap.add_argument("-ft", "--filetype", required=True,    help = "Image filetype")
    ap.add_argument('-th', '--threshold', type = float, required = False, default = 0.4,  help = 'threshold for template matching')
    
    args = vars(ap.parse_args())
    
    
    # setting path to model file
    file_path = args["path"]
    ext = args['filetype']
    threshold = args['threshold']


    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype
    
    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    
    
    print(imgList)
    
    #global  template
    template_path = args['temp_path']
    
    # load template image 
    template_file = sorted(glob.glob(template_path + filetype))
    
    print((template_file[0]))
    
    
    #template_file = 
    # Read the template 
    #template = cv2.imread(template_path, 0) 
    
    if template_file is None:
        print("template image is empty!\n")
    else:
        
        template_rgb = imread(template_file[0])
    
        template_gray = rgb2gray(template_rgb)
        
        template = cv2.imread(template_file[0], 0) 
        
        
        print("template image loaded!\n")
        
        
    # save folder construction
    mkpath = os.path.dirname(file_path) +'/marker_detection'
    mkdir(mkpath)
    marker_save_path = mkpath + '/'
    
    


    
    n_images = len(imgList)

    
    #loop execute
    for image_id, image_file in enumerate(imgList):
        
        ###########################################################
        abs_path = os.path.abspath(image_file)

        filename, file_extension = os.path.splitext(abs_path)

        base_name = os.path.splitext(os.path.basename(filename))[0]

        image_file_name = Path(image_file).name

        print("Analyzing image : {0}\n".format(str(image_file_name)))
        
        #template_detect_ski(image, template_gray)
        
        
        
        (template_crop_img, template_overlay, template_h, template_w) = template_detect_CV(image_file, template)
        
        
        print("template_w = {0}, template_h = {1}\n".format(template_w, template_h))
        
        
        if template_w > 0 and template_h > 0:
            # save segmentation result
            result_file = (marker_save_path + base_name + '.' + ext)
            #print(result_file)
            cv2.imwrite(result_file, template_overlay)

    print("results_file path: {0}\n".format(marker_save_path))

        
    
    
        
    
    


    
