
"""
Version: 1.0

Summary: Phase based motipn frame prediction and interpolation

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

python3 frame_interpolation_phase.py -p /home/suxingliu/ply_data/cross_section_scan/ -n_frames 10


argument:
("-p", "--path", required = True,    help = "path to image file")
("-ft", "--filetype", required = False, default = 'jpg', help = "Image filetype")
('-n_frames', '-n', required = True, type = int, default = 1 , help = 'Number of new frames.')
('-dev', '-d', required = False, type = str, default = 'cpu', help = 'Choose a device to run on.')
('-gpu', type = int, required = False, default = 0, help = 'Choose which GPU to use.')

"""

#!/usr/bin/python
# Standard Libraries

import numpy as np
import imageio

import argparse
import time
from matplotlib import pyplot as plt

import shutil

from frame_interp import interpolate_frame
from skimage import img_as_ubyte

import sys
import os,fnmatch,os.path
import glob
from itertools import tee
import itertools
zip = getattr(itertools, 'izip', zip)

import multiprocessing
from multiprocessing import Pool
from contextlib import closing

import warnings
warnings.filterwarnings("ignore")

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



def pairwise(iterable):
    """generate image file pair list with adjacent two files combination"""
    
    #return combination of file list like "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    
    return zip(a, b)
    
    
def frame_interpolation(imgList_pair):
    """compute interpolation fram e based on phase """

    for v, w in imgList_pair:
        
        # accquire filename without extension
        filename, file_extension = os.path.splitext(v)
        
        print("proceeesing image pair: " + v + " & " + w + "\n")
        
        
        #load image pairs
        img1 = imageio.imread(current_path + v)
        img2 = imageio.imread(current_path + w)
        
        #generate interpolation images
        new_frames = interpolate_frame(img1, img2, n_frames = args["n_frames"], scale = .5**.25, xp = xp)
        

        
        #save interpolated images 
        for j in range(args["n_frames"]):
            
            save_name = save_path + filename + '_'+ str(j) +'.' + args["filetype"]
        
            new_img = img_as_ubyte(new_frames[j])
                    
            imageio.imsave(save_name, new_img)
        

def file_sort(imgList):
    """rename, sorting and move image files"""
    
    for i in range(0,(len(imgList))):
        
        new_file = save_path + str('{:05}'.format(i)) + '.' + args["filetype"]
        
        filepath = ori_path + imgList[i]
        
        if os.path.isfile(filepath):
            shutil.copy(ori_path + imgList[i], new_file)
        else:
            shutil.copy(dst_path + imgList[i], new_file)



if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", required = False, default = 'jpg', help = "Image filetype")
    ap.add_argument('-n_frames', '-n', required = True, type = int, default = 1 , help = 'Number of new frames.')
    ap.add_argument('-dev', '-d', required = False, type = str, default = 'cpu', help = 'Choose a device to run on.')
    ap.add_argument('-gpu', type = int, required = False, default = 0, help = 'Choose which GPU to use.')
    args = vars(ap.parse_args())

    if args["dev"] == 'cpu':
        print('Using CPU.')
        xp = np
    elif args["dev"] == 'gpu':
        try:
            import cupy as cp
            xp = cp
            print('Using GPU.')
            xp.cuda.Device(args.gpu).use()
        except ImportError:
            xp = np
            print('No CUPY available. Using NUMPY instead.')
    else:
        raise NotImplementedError('Unknown choice of device.')

    # setting path to result file
    current_path = args["path"]
    
    #accquire image file list
    filetype = '*.' + args["filetype"]
    imgList = sorted(fnmatch.filter(os.listdir(current_path), filetype))
        
    # make the folder to store the results
    parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
    mkpath = parent_path + '/' + str('interpolation')
    mkdir(mkpath)
    save_path = mkpath + '/'
    
    #generate image file pair list with adjacent two files combination
    imgList_pair = pairwise(imgList)
    
    #start = time.time()
    
    #frame_interpolation(imgList_pair)
    
    
    # get cpu number for parallel processing
    agents = multiprocessing.cpu_count() - 2
    
    print("Using {0} cores for parallel processing... \n".format(int(agents)))
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # compute the interpolation frame  for each image pairs in file list
    with closing(Pool(processes = agents)) as pool:
        pool.map(frame_interpolation, zip(imgList_pair))
        pool.terminate()
    
    
    
    '''
    #loop for frame interpolation
    for v, w in pairwise(imgList):
        
        print("proceeesing image pairs: " + v + " & " + w)
    
    
    for i in range(0,(len(imgList)-1)):
        
        print("proceeesing image pairs: " + imgList[i] + " & " + imgList[i+1])
        
         # accquire filename without extension
        filename, file_extension = os.path.splitext(imgList[i])
        #filename = filename.replace(current_path,"")
        
        img1 = imageio.imread(current_path + imgList[i])
        
        img2 = imageio.imread(current_path + imgList[i+1])
        
        new_frames = interpolate_frame(img1, img2, n_frames = args["n_frames"], scale = .5**.25, xp = xp)
        
        for j in range(args["n_frames"]):
            
            save_name = save_path + filename + '_'+ str(j) +'.' + args["filetype"]
            
            new_img = img_as_ubyte(new_frames[j])
        
            imageio.imsave(save_name, new_img)
        
    '''

    # setting path to result file
    ori_path = current_path
    dst_path = save_path

     #accquire image file list
    imgList_ori = sorted(fnmatch.filter(os.listdir(ori_path), filetype))
    imgList_dst = sorted(fnmatch.filter(os.listdir(dst_path), filetype))

    imgList = sorted(imgList_ori + imgList_dst)

    # make the folder to store the results
    mkpath = parent_path + '/' + str('interpolation_result')
    mkdir(mkpath)
    save_path = mkpath + '/'

    #Combine the interpolation result with original images and rename all the results
    file_sort(imgList)
    
    #delete the interpolation result folder
    try:
        shutil.rmtree(dst_path, ignore_errors=True)
        
        print ("Phase based motion frame prediction and interpolation was finished!\n")
        
        print ("results_folder: " + save_path)
        
    except OSError:
        pass
    
    
    


    
    
    
