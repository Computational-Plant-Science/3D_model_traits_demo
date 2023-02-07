"""
Version: 1.5

Summary: Process all skeleton graph data in each individual folders

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 skeleton_graph_pipeline.py -p ~/example/

"""

import subprocess, os
import sys
import argparse
import numpy as np 
import pathlib


import psutil
import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from contextlib import closing


# execute script inside program
def execute_script(cmd_line):
    
    try:
        #print(cmd_line)
        #os.system(cmd_line)

        process = subprocess.getoutput(cmd_line)
        
        print(process)
        
        #process = subprocess.Popen(cmd_line, shell = True, stdout = subprocess.PIPE)
        #process.wait()
        #print (process.communicate())
        
    except OSError:
        
        print("Failed ...!\n")

# execute pipeline scripts in order
def skeleton_analysis_pipeline(file_path):
    
    folder_name = os.path.basename(file_path)

    model_skeleton_name = folder_name + '_skeleton.ply'
    
    print("Processing folder '{}'...\n".format(model_skeleton_name))
    
    file_path_full = file_path + '/'

    # python3 skeleton_graph.py -p ~/example/pt_cloud/tiny/ -m1 tiny_skeleton.ply
    #skeleton_analysis = "python3 skeleton_graph.py -p " + file_path_full + " -m1 " + model_skeleton_name
    
    filename = folder_name + '_his.png'
    
    #cp ~/example/B73_test/01/01_his.png ~/example/B73_resulls/histogram/
    skeleton_analysis = "cp " + file_path_full + filename + " /home/suxing/example/B73_resulls/histogram/" 
    
    #print(skeleton_analysis)
    
    execute_script(skeleton_analysis)


def fast_scandir(dirname):
    
    subfolders= sorted([f.path for f in os.scandir(dirname) if f.is_dir()])
    
    return subfolders
    
    
if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to individual folders")
    args = vars(ap.parse_args())
    
    
    #parameter sets
    # path to individual folders
    current_path = args["path"]
    
    #subfolders = sorted([ f.path for f in os.scandir(current_path) if f.is_dir() ])
    
    subfolders = fast_scandir(current_path)
    
    #print("Processing folder in path '{}' ...\n".format(subfolders))
    
    '''
    #loop execute
    for subfolder_id, subfolder_path in enumerate(subfolders):
        
        #folder_name = os.path.basename(subfolder_path)
        
        #model_skeleton_name = folder_name + '_skeleton.ply'
        
        #model_skeleton_name = folder_name + '_his.png'
        
        
        
        #print("Processing folder '{}'...\n".format(subfolder_path))
        
        skeleton_analysis_pipeline(subfolder_path)
        

    '''
    ###########################################################
    #parallel processing module
    
    # get cpu number for parallel processing
    agents = psutil.cpu_count() - 2 
    #agents = multiprocessing.cpu_count() 
    #agents = 8
    
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(skeleton_analysis_pipeline, subfolders)
        pool.terminate()
    
