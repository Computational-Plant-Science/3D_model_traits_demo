"""
Version: 1.5

Summary: Analyze model data in each individual folders

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

      python3 traits_batch_compute.py -p ~/example/ 

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
def model_analysis_pipeline(file_path):
    
    
    print("Processing file_path '{}'...".format(file_path))

    basename = os.path.basename(file_path)
    
    print("folder_name is {}".format(basename))
    
    filename = pathlib.PurePath(file_path).name + ".ply"
        
    print("3D model file name is {}\n".format(filename))

    
    file_path_full = file_path + '/'
    
    
    '''
    # step 1  python3 model_alignment.py -p ~/example/ -m test.ply
    print("Transform point cloud model to its rotation center and align its upright orientation with Z direction...\n")

    format_convert = "python3 model_alignment.py -p " + file_path_full + " -m " + filename + " -t " + str(args["test"])
    
    #print(format_convert)
    
    #execute_script(format_convert)
    
    
    # step 2 ./AdTree/Release/bin/AdTree ~/example/pt_cloud/test.xyz ~/example/pt_cloud/ -s
    print("Compute structure and skeleton from point cloud model ...\n")
    
    skeleton_graph = "./AdTree/Release/bin/AdTree " + file_path_full + basename + ".xyz " + file_path_full + " -s"
    
    #print(skeleton_graph)
    
    execute_script(skeleton_graph)
    
    
    # step 3  python3 extract_slice.py -p ~/example/pt_cloud/ -f test_branches.obj -n 100 
    print("Generate cross section sequence ...\n")
   
    cross_section_scan = "python3 extract_slice.py -p " + file_path_full + " -f " + basename + "_branches.obj " + "-n " + str(n_slices)
    
    #print(cross_section_scan)
    
    execute_script(cross_section_scan)
    '''
    
    # step 4 python3 skeleton_analyze.py -p ~/example/pt_cloud/ -m1 test_skeleton.ply -m2 test_aligned.ply -m3 ~/example/pt_cloud/slices/ -v 1
    print("Analyze skeleton / structure and compute traits...\n")

    traits_computation = "python3 skeleton_analyze.py -p " + file_path_full + " -m1 " + basename + "_skeleton.ply " + " -m2 " + basename + "_aligned.ply " + " -m3 " + file_path_full + "slices/ " + "-v " + str(args["visualize_model"])
    
    #print(traits_computation)
    
    execute_script(traits_computation)
    
        
    
    
def fast_scandir(dirname):
    
    subfolders= sorted([f.path for f in os.scandir(dirname) if f.is_dir()])
    
    return subfolders
    
    

if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-t", "--test", required = False, type = int, default = 0, help = "if using test setup")
    ap.add_argument("-n", "--n_slices", required = False, type = int, default = 500 , help = 'Number of slices for 3d model.')
    ap.add_argument("-v", "--visualize_model", required = False, type = int, default = 0, help = "Display model or not, deafult as no due to headless display in cluster")
    
    args = vars(ap.parse_args())
    
    
    #parameter sets
    # path to model file 
    current_path = args["path"]
    
    # number of slices for cross section 
    n_slices = args["n_slices"]
    
    subfolders = fast_scandir(current_path)
    
    #print("Processing folder in path '{}' ...\n".format(subfolders))
    
    '''
    ###########################################################
    #loop execute
    for subfolder_id, subfolder_path in enumerate(subfolders):
        
        #folder_name = os.path.basename(subfolder_path)
        
        #model_skeleton_name = folder_name + '_skeleton.ply'
        
        #model_skeleton_name = folder_name + '_his.png'
        
        
        
        #print("Processing folder '{}'...\n".format(subfolder_path))
        
        model_analysis_pipeline(subfolder_path)
   

    '''
    ###########################################################
    #parallel processing module
    
    # get cpu number for parallel processing
    agents = psutil.cpu_count() - 1 
    #agents = multiprocessing.cpu_count() 
    #agents = 8
    
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(model_analysis_pipeline, subfolders)
        pool.terminate()
    


    
