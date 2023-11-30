"""
Version: 1.5

Summary: Process all skeleton graph data in each individual folders

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 batch_file_move.py -p ~/example/molly_HLN_models_skeleton/Models/HighN/ -r ~/example/molly_HLN_models_skeleton/Average/ -tq 0

"""

import subprocess, os
import sys
import argparse
import numpy as np 
import pathlib
import os
import glob
import shutil

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
def file_move(source_file_path, target_file_path):
    
    #file_path = folder_name + '_quaternion.xlsx'
    
    #file_path = folder_name + '_seg.png'
    
    batch_cmd = "cp " + source_file_path + " " + target_file_path
    
    print(batch_cmd)
    
    execute_script(batch_cmd)




# execute pipeline scripts in order
def folder_delete(file_path):
    
    isExists = os.path.exists(file_path)
    
    if isExists:
        
        shutil.rmtree(file_path)
    else:
        print("Path {} not exist!\n".format(file_path))
        

def create_folders(genotype_list):
    
    for item in genotype_list:
        
        folder_path = current_path + item + '/'
        
        mkdir(folder_path)


def fast_scandir(dirname):
    
    subfolders= sorted([f.path for f in os.scandir(dirname) if f.is_dir()])
    
    return subfolders
    
    
if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to individual folders")
    ap.add_argument("-r", "--target_path", required = False, help = "path to target folders")
    ap.add_argument("-tq", "--type_quaternion", required = True, type = int, default = 0, help = "analyze quaternion type, average_quaternion=0, composition_quaternion=1, diff_quaternion=2, distance_quaternion=3")
    args = vars(ap.parse_args())
    
    
   
    
    #parameter sets
    # path to individual folders
    current_path = args["path"]
    type_quaternion = args["type_quaternion"]
    
    target_path = args["target_path"]
    
    subfolders = fast_scandir(current_path)
    
    #print("Processing folder in path '{}' ...\n".format(subfolders))
    
    if type_quaternion == 0:
        tq_folder = 'average'
        tq_file = 'average_quaternion.xlsx'
    elif type_quaternion == 1:
        tq_folder = 'composition'
        tq_file = 'composition_quaternion.xlsx'
    elif type_quaternion == 2:
        tq_folder = 'diff'
        tq_file = 'diff_quaternion.xlsx'
    elif type_quaternion == 3:
        tq_folder = 'distance'
        tq_file = 'distance_quaternion.xlsx'
        
        
    #B101	B112	DKIB014	LH123HT	Pa762	PHZ51	DKPB80_3IIH6	H96_3IIH6	LH59_PHG29	Pa762_3IIH6	PHG50_PHG47	PHZ51_LH59
    
    genotype_list = ['B101', 'B112', 'DKIB014', 'LH123HT', 'Pa762', 'PHZ51', \
                'DKPB80_3IIH6', 'H96_3IIH6', 'LH59_PHG29', 'Pa762_3IIH6', 'PHG50_PHG47', 'PHZ51_LH59']

        
    #create_folders(genotype_list)
    
    
    #loop execute
    for subfolder_id, subfolder_path in enumerate(subfolders):
        
        '''
        folder_name = os.path.basename(subfolder_path)

        source_file = subfolder_path + '/' + tq_folder + '/' + tq_file

        target_file = target_path + folder_name + '_' + tq_file 
        
        print("Moving file '{}' to {}\n".format(source_file, target_file))
        
        file_move(source_file, target_file)
        '''
        
        
        
        folder_name = os.path.basename(subfolder_path)
        
        print(folder_name)
        
        source_file = subfolder_path + '/' + folder_name + '.xlsx' 

        target_file = target_path 
        
        print("Moving file '{}' to {}\n".format(source_file, target_file))
        
        file_move(source_file, target_file)
        
        
        '''
        ##################################################
        #Delete folder
        
        target_folder = subfolder_path + '/' + tq_folder + '/'
        
        print("Deleting folder '{}'\n".format(target_folder))

        #folder_delete(target_folder)
        '''
        
        '''
        #############################################
        label = 'pie_color.png'
        
        source_file = subfolder_path + '/leaf1/' + label
        
        target_file = target_path + '/' + folder_name + '_' +label
        
        file_move(source_file, target_file)
        '''
    
    

