"""
Version: 1.5

Summary: Compute traits from a 3D model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:


    python3 /opt/code/pipeline.py -i /srv/test/test.ply -o /srv/test/result/ -md 35 -n 1000  -np 10
    

Arguments:

("-md", "--min_dis", dest = "min_dis", required = False, type = int, default = 35,   help = "min distance for watershed segmentation")
("-n", "--n_slices", dest = "n_slices", required = False, type = int, default = 1000 , help = 'Number of slices for 3d model.')

"""

import subprocess, os, glob
import sys
import argparse

import pathlib


'''
import numpy as np 


import psutil
import concurrent.futures
import multiprocessing
from multiprocessing import Pool
from contextlib import closing
'''

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
def model_analysis_pipeline(file_path, filename, basename, result_path):

    
    # step 1  python3 /opt/code/model_alignment.py -p ~/example/ -m test.ply  -o ~/example/result/
    print("Transform point cloud model to its rotation center and align its upright orientation with Z direction...\n")

    format_convert = "python3 /opt/code/model_alignment.py -p " + file_path + " -m " + filename + " -o " + result_path
    
    print(format_convert)
    
    execute_script(format_convert)
    
    
    # step 2 ./opt/code/compiled/Release/bin/AdTree ~/example/result/test.xyz ~/example/result/ -s  
    print("Compute structure and skeleton from point cloud model ...\n")
    
    #skeleton_graph = "./compiled/Release/bin/AdTree " + result_path + basename + ".xyz " + result_path + " -s"
    
    skeleton_graph = "/opt/code/compiled/Release/bin/AdTree " + result_path + basename + ".xyz " + result_path + " -s"
    
    print(skeleton_graph)
    
    execute_script(skeleton_graph)
    
    
    # step 3  python3 /opt/code/extract_slice.py -p ~/example/result/ -f test_branches.obj -o ~/example/result/ -n 1000 
    print("Generate cross section sequence ...\n")
   
    cross_section_scan = "python3 /opt/code/extract_slice.py -p " + result_path + " -f " + basename + "_branches.obj " + " -o " + result_path + " -n " + str(n_slices)
    
    print(cross_section_scan)
    
    execute_script(cross_section_scan)
    
    
    # step 4 python3 /opt/code/skeleton_analyze.py -p ~/example/result/ -m1 test_skeleton.ply -m2 test_aligned.ply -o ~/example/result/ -md 12 -np 10 -v 0
    print("Compute all the traits...\n")

    traits_computation = "python3 /opt/code/skeleton_analyze.py -p " + result_path + " -m1 " + basename + "_skeleton.ply " + " -m2 " + basename + "_aligned.ply " + " -o " + result_path + " -md " + str(min_dis) + " -np " + str(n_plane) + " -v " + str(visualize_model)
    
    print(traits_computation)
    
    execute_script(traits_computation)
    
    
    # step 5 grants read and write access to all result folders
    print("Compute all the traits...")

    access_grant = "chmod 777 -R " + result_path 
    
    print(access_grant + '\n')
    
    execute_script(access_grant)
    
    


# parelle processing of folders for local test only
def parallel_folders(subfolder_path):

    folder_name = os.path.basename(subfolder_path) 

    subfolder_path = os.path.join(subfolder_path, '')

    m_file = subfolder_path + folder_name + '.' + ext

    print("Processing 3d model point cloud file '{}'...\n".format(m_file))

    (filename, basename) = get_fname(m_file)

    #print("Processing 3d model point cloud file '{}'...\n".format(filename))

    #print("Processing 3d model point cloud file basename '{}'...\n".format(basename))

    model_analysis_pipeline(subfolder_path, filename, basename, subfolder_path)




# get file information from the file path uisng os for python 2.7
def get_fname(file_full_path):
    
    abs_path = os.path.abspath(file_full_path)

    filename= os.path.basename(abs_path)

    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    return filename, base_name




# get sub folders from a inout path for local test only
def fast_scandir(dirname):
    
    subfolders= sorted([f.path for f in os.scandir(dirname) if f.is_dir()])
    
    return subfolders




# get file information from the file path using pathon 3
def get_file_info(file_full_path):

    p = pathlib.Path(args["input"])
    
    filename = p.name
    
    basename = p.stem


    file_path = p.parent.absolute()
    
    file_path = os.path.join(file_path, '')
    
    return file_path, filename, basename







if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", dest = "input", required = True, type = str, help = "full path to 3D model file")
    #ap.add_argument("-p", "--path", dest = "path", required = True, type = str, help = "path to 3D model file")
    #ap.add_argument("-ft", "--filetype", dest = "filetype", type = str, required = False, default = 'ply', help = "3D model file filetype, default *.ply")
    ap.add_argument("-o", "--output_path", dest = "output_path", required = False, type = str, help = "result path")
    ap.add_argument("-md", "--min_dis", dest = "min_dis", required = False, type = int, default = 35,   help = "min distance for watershed segmentation")
    ap.add_argument("-np","--n_plane", type = int, required = False, default = 10,  help = "Number of planes to segment the 3d model along Z direction")
    ap.add_argument("-n", "--n_slices", dest = "n_slices", required = False, type = int, default = 1000 , help = 'Number of slices for 3d model.')
    ap.add_argument("-v", "--visualize_model", dest = "visualize_model", required = False, type = int, default = 0, help = "Display model or not, deafult as no due to headless display in cluster")
    
    args = vars(ap.parse_args())
    
    '''
    # path to model file 
    file_path = args["path"]
    
    
    
    # setting path to model file
    file_path = args["path"]

    ext = args['filetype'].split(',') if 'filetype' in args else []
    
    patterns = [os.path.join(file_path, f"*.{p}") for p in ext]
    
    model_List = [f for fs in [glob.glob(pattern) for pattern in patterns] for f in fs]
    
    
    
    # load input model files
    
    if len(model_List) > 0:
    
        print("Model files in input folder: '{}'\n".format(model_List))
        
        
    
    else:
        print("3D model file does not exist")
        sys.exit()
        

    '''
    # get input file information
    
    if os.path.isfile(args["input"]):
    
        (file_path, filename, basename) = get_file_info(args["input"])
        
        print("Processing 3d model point cloud file '{} {} {}'...\n".format(file_path, filename, basename))
        
        
        # result path
        result_path = args["output_path"] if args["output_path"] is not None else file_path

        result_path = os.path.join(result_path, '')

        # print out result path
        print ("results_folder: {}\n".format(result_path))

        # number of slices for cross section 
        n_slices = args["n_slices"]
        
        min_dis = args["min_dis"]
        
        n_plane = args["n_plane"]
        
        visualize_model = args["visualize_model"]


        # start pipeline
        ########################################################################################3

        model_analysis_pipeline(file_path, filename, basename, result_path)

    
    else:
        
        print("The input file is missing or not readable!\n")

        print("Exiting the program...")

        sys.exit(0)
    
    
    
    

    
    
    '''
    #loop execute
    for model_id, model_file in enumerate(model_List):
        
        print("Processing 3d model point cloud file '{}'...\n".format(model_file))

        (filename, basename) = get_fname(model_file)

        #print("Processing 3d model point cloud file {} {}\n".format(filename, basename))

        model_analysis_pipeline(file_path, filename, basename, result_path)
    
    '''
    
    '''
    ######################################################################################
    # docker version

    folder_name = os.path.basename(file_path[:-1])
    
    file_path = os.path.join(file_path, '')
    
    m_file = file_path + folder_name + '.' + ext
    
    print("Processing 3d model point cloud file '{}'...\n".format(m_file))
    
    (filename, basename) = get_fname(m_file)

    print("Processing 3d model point cloud file {} {}\n".format(filename, basename))
    
    model_analysis_pipeline(file_path, filename, basename, result_path)
    '''
    
    '''
    ####################################################################################
    # local test loop version
    subfolders = fast_scandir(file_path)
    
    for subfolder_id, subfolder_path in enumerate(subfolders):
    
        
        folder_name = os.path.basename(subfolder_path) 
        
        subfolder_path = os.path.join(subfolder_path, '')
        
        m_file = subfolder_path + folder_name + '.' + ext
        
        print("Processing 3d model point cloud file '{}'...\n".format(m_file))
        
        (filename, basename) = get_fname(m_file)

        #print("Processing 3d model point cloud file '{}'...\n".format(filename))
        
        #print("Processing 3d model point cloud file basename '{}'...\n".format(basename))

        model_analysis_pipeline(subfolder_path, filename, basename, subfolder_path)
    
    '''

    
    ########################################################################################
    # local test parellel version
    '''
    subfolders = fast_scandir(file_path)
    
    print(len(subfolders))
    
    # get cpu number for parallel processing
    agents = psutil.cpu_count() - 4  

    
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result_list = pool.map(parellel_folders, subfolders)
        pool.terminate()
    '''
