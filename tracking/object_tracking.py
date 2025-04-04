"""
Version: 1.5

Summary: Multi Object Tracker Using Kalman Filter and Hungarian Algorithm

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

python3 object_tracking.py -p /home/suxingliu/ply_data/sequence/ -d 10 -mfs 15 -mtl 15 -rmin 1 -rmax 100


argument:
("-p", "--path", required = True, help = "path to image file")
("-ext", "--extension", required = False, default = 'png', help = "extension name. default is 'png'.")  

"""


# Import python libraries
import argparse
import os
import fnmatch
import re

import cv2
import copy

from detectors import Detectors
#from root_detector import Root_Detectors
from tracker import Tracker

import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib._png import read_png
from matplotlib.cbook import get_sample_data
import matplotlib.image as mpimg

#from mpl_toolkits.mplot3d import Axes3D

import itertools

from scipy.optimize import curve_fit

from sklearn.preprocessing import normalize, MinMaxScaler

import csv
import math

import multiprocessing
from multiprocessing import Pool
from contextlib import closing

def mkdir(path):
    """Create result folder"""
 
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



def Distance(x1, y1, x2, y2):
    """compute distance between two points"""
    
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist

'''
def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    """generate data for cylinder"""
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid
'''

def center_radius(x, y, radius_track, coord_radius):
    """compute average radius"""
    
    idx = []
    index = 0
    
    for i in range(0, len(x)):
        result = np.argmax((coord_radius[:,0] == x[i]) & (coord_radius[:,1] == y[i]))
        
        if result:
            idx.append(result)
        else:
            index = i
    
    avg_radius = 1
    
    if len(idx)>0:
        avg_radius = np.average(radius_track[idx])

    return avg_radius


def func(x, a, b, c):
    """compute curve fit function"""
    
    return a * np.exp(-b * x) + c



def Trace_tracking(images):
    """tracking mutiple traces"""
    
    #extract tfirst file name 
    first_filename = os.path.splitext(images[0])[0]
    
    #extract first file index
    #first_number = int(filter(str.isdigit, first_filename))
    
    first_number = int(re.search(r'\d+', first_filename).group(0))
    
    #define trace result path
    outfile = save_path_track +  str('{:04}'.format(first_number)) + '.txt'
        
    print(outfile)

    image_path = os.path.join(dir_path, images[0])
    
    #load image from image path
    frame = cv2.imread(image_path)
    
    #Determine the width and height from the first image
    height, width, channels = frame.shape
    length = len(images)
    
    print("Image sequence size: {0} {1} {2}\n".format(width, height, length))
   
    #Create Object Detector
    detector = Detectors()
    
    #detector = Root_Detectors(pattern_id)    

    # Create Object Tracker, arguments:
    # dist_thresh, max_frames_to_skip, max_trace_length, trackIdCount
    #distance threshold. When exceeds the threshold, track will be deleted and new track is created
    tracker = Tracker(dist_thresh, max_frames_to_skip, max_trace_length, trackIdCount)

    # Variables initialization
    skip_frame_count = 0
    
    #frame ID
    ID = 0
    
    #stem_track = np.zeros(3)
    
    #initilize parameters for record radius and center locations
    radius_track = []
    
    centers_track = []
    
     #Begin of process each image for tracking
    ###################################################################################
    # loop to process video frames
    for frame_ID, image in enumerate(images):

        # Capture frame-by-frame
        image_path = os.path.join(dir_path, image)
        
        #load image frame
        frame = cv2.imread(image_path)
        
        # exit the loop if reach the end frame
        if ID == len(images):
            print("End of frame sequence!")
            break
        
        # Make copy of original frame
        orig_frame = copy.copy(frame)
        
        print("Processing frame {}...".format(frame_ID))
        
        # Detect and return centeroids of the objects in the frame
        (centers,radius_rec) = detector.Detect(frame, ID, radius_min, radius_max)

        # record radius and center locations
        radius_track.append(radius_rec)
        centers_track.append(centers)
        
        #centers, stem_center = detector.Detect_root(frame, ID, pattern_id, stem_track)
        
        #centers = detector.Detect_root_blob(frame, ID, pattern_id, stem_track)
        
        
        # If centroids are detected then track them
        if (len(centers) > 0):

            # Track object using Kalman Filter
            tracker.Update(centers)

            print("Tracker size: {}...".format(len(tracker.tracks)))
            
    #End of process each image for tracking
    ###################################################################################
    
    
    radius_track = np.hstack(radius_track)
    
    coord_radius = []
    
    # combine x, y coordinates
    for i in range(0, len(centers_track)):
        for j in range(0, len(centers_track[i])):
            coord_radius.append(np.array(centers_track[i][j]))
            
    coord_radius = np.array(coord_radius)

    #start index value along Z axis 
    offset = first_number
    
    # write output as txt file
    with open(outfile, 'w') as f:
        
        #loop all tracked objects
        for i in range(len(tracker.tracks)):
            
            if (len(tracker.tracks[i].trace) > 2):
                
                #accquire dimension of current tracker
                dim = len(tracker.tracks[i].trace)
                
                #extract point data from current tracker
                point = np.asarray(tracker.tracks[i].trace)
                
                #print(type(tracker.tracks[i].trace))
                
                # accquire shape of points
                nsamples, nx, ny = point.shape
                
                #reshape points 
                point = point.reshape((nsamples,nx*ny))
                
                #extract x,y,z coordinates 
                x = np.asarray(point[:,0]).flatten()
                y = np.asarray(point[:,1]).flatten()
                z = np.asarray(range(offset , dim + offset)).flatten() 
                
                #curve fitting of xy trace in 2D space
                #popt, pcov = curve_fit(func, x, y)
                #y = func(x, *popt)
                
                #compute average radius 
                avg_radius = center_radius(x,y,radius_track,coord_radius)
                
                #reshape radius array 
                r = np.asarray(avg_radius * np.ones((len(x),1))).flatten() 

                #print("Average radius: {0} \n".format(avg_radius))
                
                # write out tracing trace result
                #if ( (len(x) == len(y) == len(z)) and (np.count_nonzero(x) == dim) and (np.count_nonzero(y) == dim) and sum(x) !=0 ):
                if ( (len(x) == len(y) == len(z)) and sum(x) !=0):

                    # save trace points as txt file
                    f.write("#Trace {0} \n".format(i))
                    np.savetxt(f, np.c_[x,y,z,r], fmt = '%-7.2f')
                    
                #else:
                    #print("Inconsistant length pf 3D array!")
                    #ax.scatter(x, y, z, c = 'b', marker = 'o')
                    #ax.plot(x, y, z, label='Tracking root trace')
                    #f.write("#End\n")
                
        # write end mark and close file 
        f.write("#End\n")
        f.close()

    
    

if __name__ == "__main__":
    
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to image file")
    ap.add_argument("-ext", "--extension", required = False, default = 'png', help = "extension name. default is 'png'.")
    ap.add_argument('-d', '--dist_thresh', required = False, type = int, default = 10 , help = 'dist_thresh.')
    ap.add_argument('-mfs', '--max_frames_to_skip', required = False, type = int, default = 15 , help = 'max_frames_to_skip.')
    ap.add_argument('-mtl', '--max_trace_length', required = False, type = int, default = 15 , help = 'max_trace_length.')
    ap.add_argument('-rmin', '--radius_min', required = False, type = int, default = 1 , help = 'radius_min.')
    ap.add_argument('-rmax', '--radius_max', required = False, type = int, default = 100 , help = 'radius_max.')
    args = vars(ap.parse_args())
    

    # Arguments
    global pattern_id, dir_path, ext, save_path_track
    global dist_thresh, max_frames_to_skip, max_trace_length, trackIdCount
    global radius_min, radius_max
    
    #pattern_id = args["patternID"]
    dir_path = args["path"]
    ext = args['extension']
    
    #accquire image file list
    filetype = '*.' + ext
    images = sorted(fnmatch.filter(os.listdir(dir_path), filetype))
    
    # make the folder to store the results
    parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
    mkpath = parent_path + '/' + str('trace_track')
    mkdir(mkpath)
    save_path_track = mkpath + '/'

    # parameters for tracking
    dist_thresh = args["dist_thresh"]
    max_frames_to_skip = args["max_frames_to_skip"]
    max_trace_length = args["max_trace_length"]
    trackIdCount = 1 
    
    radius_min = args["radius_min"]
    radius_max = args["radius_max"]
    
    #accquire length of image list
    imagelist_length = len(images)
    
    #compute intervals between files based on max_trace_length value
    if imagelist_length % max_trace_length == 0: #even 
        interval = max_trace_length
    else: #odd
        interval = max_trace_length + 1
    
    #number of chunks of file list
    num_sublist = int(imagelist_length/interval)

    # Divide image list into chunks 
    sub_list = np.array_split(images, num_sublist)
    print("Number of sublist is: {0} \n".format(num_sublist))

    # parallel tracking for divided image chunks
    agents = multiprocessing.cpu_count() -2
    chunksize = 3
    with closing(Pool(processes = agents)) as pool:
        
        result = pool.map(Trace_tracking, sub_list, chunksize)
        
        pool.terminate()

    print ("results_folder: " + save_path_track)
