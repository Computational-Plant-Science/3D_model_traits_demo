"""
Version: 1.5

Summary: Analyze and visualzie tracked traces

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 /opt/code/pipeline.py -p ~/example/ -o ~/example/result/

"""

import subprocess, os, glob
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
def model_analysis_pipeline(file_path, filename, basename, result_path):

    
    # step 1  python3 model_alignment.py -p ~/example/ -m test.ply  -o ~/example/result/
    print("Transform point cloud model to its rotation center and align its upright orientation with Z direction...\n")

    format_convert = "python3 /opt/code/model_alignment.py -p " + file_path + " -m " + filename + " -o " + result_path
    
    print(format_convert)
    
    #execute_script(format_convert)
    
    
    # step 2 ./compiled/Release/bin/AdTree ~/example/result/test.xyz ~/example/result/ -s
    print("Compute structure and skeleton from point cloud model ...\n")
    
    skeleton_graph = "/opt/code/compiled/Release/bin/AdTree " + result_path + basename + ".xyz " + result_path + " -s"
    
    print(skeleton_graph)
    
    #execute_script(skeleton_graph)
    
    
    # step 3  python3 extract_slice.py -p ~/example/result/ -f test_branches.obj -o ~/example/result/ -n 500 
    print("Generate cross section sequence ...\n")
   
    cross_section_scan = "python3 /opt/code/extract_slice.py -p " + result_path + " -f " + basename + "_branches.obj " + " -o " + result_path + " -n " + str(n_slices)
    
    print(cross_section_scan)
    
    #execute_script(cross_section_scan)
    
    
    # step 4 python3 skeleton_analyze.py -p ~/example/result/ -m1 test_skeleton.ply -m2 test_aligned.ply -m3 -o ~/example/result/ -v 0
    print("Compute all the traits...\n")

    traits_computation = "python3 /opt/code/skeleton_analyze.py -p " + result_path + " -m1 " + basename + "_skeleton.ply " + " -m2 " + basename + "_aligned.ply " + " -o " + result_path + " -v " + str(args["visualize_model"])
    
    print(traits_computation)
    
    #execute_script(traits_computation)





def get_fname(file_full_path):
    
    abs_path = os.path.abspath(file_full_path)

    filename= os.path.basename(abs_path)

    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    return filename, base_name





if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", dest = "path", required = True, type = str, help = "path to 3D model file")
    ap.add_argument("-ft", "--filetype", dest = "filetype", type = str, required = False, default = 'ply', help = "3D model file filetype, default *.ply")
    #ap.add_argument("-m", "--model", dest = "model", required = True, type = str, help = "model file name")
    ap.add_argument("-o", "--output_path", dest = "output_path", required = False, type = str, help = "result path")
    ap.add_argument("-t", "--test", dest = "test", required = False, type = int, default = 0, help = "if using test setup")
    ap.add_argument("-n", "--n_slices", dest = "n_slices", required = False, type = int, default = 500 , help = 'Number of slices for 3d model.')
    ap.add_argument("-v", "--visualize_model", dest = "visualize_model", required = False, type = int, default = 0, help = "Display model or not, deafult as no due to headless display in cluster")
    
    args = vars(ap.parse_args())
    
    
    
    # path to model file 
    file_path = args["path"]
    
    ext = args['filetype'].split(',') if 'filetype' in args else []
    
    patterns = [os.path.join(file_path, f"*.{p}") for p in ext]
    
    files = [f for fs in [glob.glob(pattern) for pattern in patterns] for f in fs]
    
    
    # load input model files
    model_files = sorted(files)
    
    if len(model_files) > 0:
    
        print("Input folder: '{}'\n".format(file_path))
    
    else:
        print("3D model file does not exist")
        sys.exit()
    
    '''
    # input model file path
    #file_full_path = file_path + filename
    
    file_full_path = args["path"]
    
    if os.path.isfile(path):
        
        print("Processing 3D model file {}\n".format(filename))
    
    else:
        print("3D model file does not exist")
        sys.exit()
        
    '''
    
    # output path
    result_path = args["output_path"] if args["output_path"] is not None else os.getcwd()
    
    result_path = os.path.join(result_path, '')
    
    # result path
    print ("results_folder: {}\n".format(result_path))
    
    # number of slices for cross section 
    n_slices = args["n_slices"]
    
    

    #loop execute
    for mfile_id, im_file in enumerate(model_files):
        
        (filename, basename) = get_fname(im_file)

        print("Processing 3d model point cloud file '{}'...\n".format(filename))
        
        print("Processing 3d model point cloud file basename '{}'...\n".format(basename))
        
        if args["visualize_model"] == True:
            
            print("Visualize skeleton and sturcture in 3D graph... \n")
        else:
            print("Skip Visualization steps... \n")
        
        model_analysis_pipeline(file_path, filename, basename, result_path)
    
    
    

