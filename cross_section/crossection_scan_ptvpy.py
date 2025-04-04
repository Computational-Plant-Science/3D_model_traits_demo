"""
Version: 1.5

Summary: compute the segmentaiton and label of cross section image sequence 

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

python3 crossection_scan_ptvpy.py -p ~/ply_data/test/


argument:
("-p", "--path", required = True,    help = "path to image file")
("-ft", "--filetype", required = False, default = 'png',   help = "Image filetype")

"""

import subprocess, os
import sys

import argparse
import glob
import fnmatch
import os, os.path

'''
def execute_script(cmd_line):
    """execute script inside program"""
    try:
        print(cmd_line)
        #os.system(cmd_line)
        
        process = subprocess.Popen(cmd_line, shell=True, stdout=subprocess.PIPE)
        
        stdout = process.communicate()[0]
        
        print ('STDOUT:{}'.format(stdout))
        
        process.wait()
        
        #print process.returncode
        
    except OSError:
        
        print("Failed ...!\n")
'''

def execute_script(command):
    
    try:
        subprocess.run(command, shell = True)
        
    except OSError:
        
        print("Failed ...!\n")


def remove_file(filePath):
    
    #print(filePath)
    
    if os.path.exists(filePath):
        os.remove(filePath)
        print("File {} was updated\n".format(filePath))
    else:
        print("Create new {}\n".format(filePath))


def sequence_scan_pipeline(file_path, result_file):
    
    #delete ptvpy.toml if exist
    
    print("Current working directory path {}\n".format(os.getcwd()))

    filePath_ptvpy = os.getcwd() + "/ptvpy.toml"
    remove_file(filePath_ptvpy)
    
    filePath_ptvpy = os.getcwd() + "/ptvpy.h5"
    remove_file(filePath_ptvpy)
    
    
    #create a new profile file
    print("Level set scan computating...\n")
    profile_file = "ptvpy profile create --data-files '" + file_path + "*" + ext + "'"
    print(profile_file)
    
    execute_script(profile_file)
    
    
    #compute and tracking traces from image slices 
    print("Tracking individual root particles...\n")
    track_trace = "ptvpy process"
    execute_script(track_trace)
    
    #print tracking results 
    #view_result = "ptvpy view summary --all"
    #execute_script(view_result)
    
    #delete result file if exist
    remove_file(result_file)
    
    export_result = "ptvpy export --type csv " + "'" + result_file + "'" 
    execute_script(export_result)
    

if __name__ == '__main__':

    # construct and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", required = False, default = 'png',   help = "Image filetype")
    args = vars(ap.parse_args())

   
    # setting path to cross section image files
    file_path = args["path"]
    ext = args['filetype']
     
    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype

    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    
    #print(imgList)

    #Create result file path
    result_file = (file_path + 'trace_result.csv')
    
    sequence_scan_pipeline(file_path, result_file)
    
