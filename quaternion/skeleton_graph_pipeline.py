"""
Version: 1.5

Summary: Process all skeleton graph data in each individual folders

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 skeleton_graph_pipeline.py -p ~/example/B73_test/ -n 4 -r 20 -tq 0

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
    
    #model_skeleton_name = 'trait.xlsx' 
    
    #file_path_full = file_path + '/' + folder_name + '_trait.xlsx' 
    
    file_path_full = file_path + '/'
    
    print("Processing folder {} in folder {}...\n".format(file_path, folder_name))
    

    
    ################################################################################

    
    # python3 skeleton_graph.py -p ~/example/pt_cloud/tiny/ -m1 tiny_skeleton.ply
    if args["n_cluster"] > 0:
        
        skeleton_analysis = "python3 skeleton_graph.py -p " + file_path_full + " -m1 " + model_skeleton_name + ' -n ' + str(args["n_cluster"]) + ' -r ' + str(len_ratio) + ' -tq  ' + str(type_quaternion)
    else:
        skeleton_analysis = "python3 skeleton_graph.py -p " + file_path_full + " -m1 " + model_skeleton_name + ' -r ' + str(len_ratio) + ' -tq  ' + str(type_quaternion)

        
    print(skeleton_analysis)
    
    execute_script(skeleton_analysis)

    
    
    


def fast_scandir(dirname):
    
    subfolders= sorted([f.path for f in os.scandir(dirname) if f.is_dir()])
    
    return subfolders
    
    
if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to individual folders")
    ap.add_argument("-r", "--len_ratio", required = False, type = int, default = 50, help = "length threshold to filter the roots, number of nodes in the shortest length path")
    ap.add_argument("-n", "--n_cluster", required = False, type = int, default = 0, help = "Number of clusters to filter the small length paths")
    ap.add_argument("-tq", "--type_quaternion", required = False, type = int, default = 0, help = "analyze quaternion type, average_quaternion=0, composition_quaternion=1, diff_quaternion=2, distance_quaternion=3")

    args = vars(ap.parse_args())
    
    
    
   
    #parameter sets
    # path to individual folders
    current_path = args["path"]
    type_quaternion = args["type_quaternion"]
    len_ratio = args["len_ratio"]
    
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
    #agents = psutil.cpu_count() - 2 
    #agents = multiprocessing.cpu_count() 
    agents = 10
    
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result = pool.map(skeleton_analysis_pipeline, subfolders)
        pool.terminate()
    

    

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
    
