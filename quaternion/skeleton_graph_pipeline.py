"""
Version: 1.5

Summary: Process all skeleton graph data in each individual folders

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 skeleton_graph_pipeline.py -p ~/example/B73_test/

"""

import subprocess, os
import sys
import argparse
import numpy as np 
import pathlib
import os
import glob

import psutil
import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from contextlib import closing


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
        #shutil.rmtree(path)
        #os.makedirs(path)
        return False
        



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
    
    #execute_script(skeleton_analysis)
    ####################################################################

    filename = folder_name + '_quaternion.xlsx'
    batch_cmd = "cp " + file_path_full + filename + " /home/suxing/example/quaternion/B73_result/values/" 
    execute_script(batch_cmd)
    
    filename = folder_name + '_his.png' 
    batch_cmd = "cp " + file_path_full + filename + " /home/suxing/example/quaternion/B73_result/histogram/" 
    execute_script(batch_cmd)
    
    filename = folder_name + '_quaternion_4D.html'
    batch_cmd = "cp " + file_path_full + filename + " /home/suxing/example/quaternion/B73_result/scatterplot/" 
    execute_script(batch_cmd)
    
    ####################################################################
    #print(skeleton_analysis)
    
    
    


def fast_scandir(dirname):
    
    subfolders= sorted([f.path for f in os.scandir(dirname) if f.is_dir()])
    
    return subfolders
    
    
if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to individual folders")
    args = vars(ap.parse_args())
    
    
    
    '''
    ###################################################################
    current_path = args["path"]
    
    file_path = current_path + '*.ply'

    # get the absolute path of all Excel files 
    Files_list = glob.glob(file_path)
    
    for image_file in Files_list:
    
        abs_path = os.path.abspath(image_file)
    
        filename, file_extension = os.path.splitext(abs_path)
        
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        base_name = base_name.replace("_skeleton", "")

        mkpath = os.path.dirname(abs_path) + '/' + base_name + '/'
        mkdir(mkpath)
        save_path = mkpath + '/'
        
        cp_model = "cp " + image_file + ' ' + save_path
        
        execute_script(cp_model)
    ####################################################################
    '''
    
    
    #parameter sets
    # path to individual folders
    current_path = args["path"]

    subfolders = fast_scandir(current_path)
    
    #print("Processing folder in path '{}' ...\n".format(subfolders))
    
    
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
    '''
