"""
Version: 1.5

Summary: Analyze and visualzie tracked traces

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 pipeline.py -p ~/example/ -m test.ply

"""

import subprocess, os
import sys
import argparse
import numpy as np 
import pathlib

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
def model_analysis_pipeline(file_path, filename, basename):

    '''
    # step 1  python3 model_alignment.py -p ~/example/ -m test.ply
    print("Transform point cloud model to its rotation center and align its upright orientation with Z direction...\n")

    format_convert = "python3 model_alignment.py -p " + file_path + " -m " + filename + " -t " + str(args["test"])
    
    #print(format_convert)
    
    execute_script(format_convert)
    '''
    
    # step 2 ./AdTree_compiled/Release/bin/AdTree ~/example/pt_cloud/test.xyz ~/example/pt_cloud/ 
    print("Compute structure and skeleton from point cloud model ...\n")
    
    skeleton_graph = "./AdTree_compiled/Release/bin/AdTree " + file_path + basename + ".xyz " + file_path 
    
    execute_script(skeleton_graph)
    
    
    # step 3  python3 extract_slice.py -p ~/example/pt_cloud/ -f test_branches.obj -n 500 
    print("Generate cross section sequence ...\n")
   
    cross_section_scan = "python3 extract_slice.py -p " + file_path + " -f " + basename + "_branches.obj " + "-n " + str(n_slices)

    execute_script(cross_section_scan)
    
    
    # step 4 python3 skeleton_analyze.py -p ~/example/pt_cloud/ -m1 test_skeleton.ply -m2 test_aligned.ply -m3 ~/example/pt_cloud/slices/ -v 0
    print("Compute all the traits...\n")

    traits_computation = "python3 skeleton_analyze.py -p " + file_path + " -m1 " + basename + "_skeleton.ply " + " -m2 " + basename + "_aligned.ply " + " -m3 " + file_path + "slices/ " + "-v " + str(args["visualize_model"])
    
    #print(traits_computation)
    
    execute_script(traits_computation)
    
    
   
    
    

if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m", "--model", required = False, help = "model file name")
    ap.add_argument("-t", "--test", required = False, type = int, default = 0, help = "if using test setup")
    ap.add_argument("-n", "--n_slices", required = False, type = int, default = 500 , help = 'Number of slices for 3d model.')
    ap.add_argument("-v", "--visualize_model", required = False, type = int, default = 0, help = "Display model or not, deafult as no due to headless display in cluster")
    
    args = vars(ap.parse_args())
    
    
    #parameter sets
    # path to model file 
    file_path = args["path"]
    #filename = args["model"]
    
    if args["model"] is None:
        
        filename = pathlib.PurePath(file_path).name + ".ply"
        
        print("3D model file name is {}\n".format(filename))
    
    else:
        
        filename = args["model"]
    
    #ratio = args["ratio"]
    #angle = args["angle"]
    file_full_path = file_path + filename
    
    #print(file_full_path)
    
    file_base_name = os.path.basename(file_full_path).split('.')[0]

    print("Processing 3d model point cloud file '{}' ...\n".format(file_full_path))
    
    # number of slices for cross section 
    n_slices = args["n_slices"]
    
    
    if args["visualize_model"] == True:
        
        print("Visualize skeleton and sturcture in 3D graph... \n")
    else:
        print("Skip Visualization steps... \n")
    
    model_analysis_pipeline(file_path, filename, file_base_name)
    
    
