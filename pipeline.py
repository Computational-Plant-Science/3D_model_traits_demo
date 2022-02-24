"""
Version: 1.5

Summary: Analyze and visualzie tracked traces

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

test
python3 pipeline.py -p ~/example/ -m test.ply -t 1

normal
python3 pipeline.py -p ~/example/ -m test.ply

"""

import subprocess, os
import sys
import argparse
import numpy as np 

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

    
    # step 1  python3 model_alignment.py -p ~/example/ -m test.ply
    print("Transform point cloud to its rotation center and align its upright orientation with Z direction...\n")

    format_convert = "python3 model_alignment.py -p " + file_path + " -m " + filename + " -t " + str(args["test"])
    
    #print(format_convert)
    
    execute_script(format_convert)

    
    # step 2 ./AdTree/Release/bin/AdTree ~/example/pt_cloud/test.xyz ~/example/pt_cloud/
    print("Compute structure and skeleton from point cloud model ...\n")
    
    skeleton_graph = "./AdTree/Release/bin/AdTree " + file_path + basename + ".xyz " + file_path
    
    execute_script(skeleton_graph)
    

    # step 3  python3 extract_slice.py -p ~/example/pt_cloud/ -f test_branches.obj -n 100
    print("Generate cross section sequence ...\n")
   
    cross_section_scan = "python3 extract_slice.py -p " + file_path + " -f " + basename + "_branches.obj " + "-n " + str(n_slices)

    execute_script(cross_section_scan)
    
    
    # step 4 python3 skeleton_analyze.py -p ~/example/pt_cloud/ -m1 test_skeleton.ply -m2 test_aligned.ply -m3 ~/example/pt_cloud/slices/ 
    print("Analyze skeleton / structure and compute traits...\n")

    traits_computation = "python3 skeleton_analyze.py -p " + file_path + " -m1 " + basename + "_skeleton.ply " + " -m2 " + basename + "_aligned.ply " + " -m3 " + file_path + "slices/ "  
    
    execute_script(traits_computation)
    
    
   
    
    

if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m", "--model", required = False, help = "model file name")
    #ap.add_argument("-a", "--angle", required = False, type = int, default = 1, help = "rotation_angle")
    #ap.add_argument("-r", "--ratio", required = False, type = float, default = 0.01, help = "outlier remove ratio")
    ap.add_argument("-t", "--test", required = False, type = int, default = 0, help = "if using test setup")
    ap.add_argument("-n", "--n_slices", required = False, type = int, default = 500 , help = 'Number of slices for 3d model.')
    
    
    '''
    ap.add_argument("-i", "--interval", required = False, default = '1',  type = int, help= "intervals along sweeping plane")
    ap.add_argument("-de", "--direction", required = False, default = 'X', help = "direction of sweeping plane, X, Y, Z")
    ap.add_argument('-frames', '-n_frames', required = False, type = int, default = 2 , help = 'Number of new frames.')
    ap.add_argument("-th", "--threshold", required = False, default = '2.35', type = float, help = "threshold to remove outliers")
    ap.add_argument('-d', '--dist_thresh', required = False, type = int, default = 10 , help = 'dist_thresh.')
    ap.add_argument('-mfs', '--max_frames_to_skip', required = False, type = int, default = 15 , help = 'max_frames_to_skip.')
    ap.add_argument('-mtl', '--max_trace_length', required = False, type = int, default = 15 , help = 'max_trace_length.')
    ap.add_argument('-rmin', '--radius_min', required = False, type = int, default = 1 , help = 'radius_min.')
    ap.add_argument('-rmax', '--radius_max', required = False, type = int, default = 100 , help = 'radius_max.')
    ap.add_argument("-dt", "--dis_tracking", required = False, type = float, default = 50.5, help = "dis_dis_tracking")
    ap.add_argument("-ma", "--min_angle", required = False, type = float, default = 0.1, help = "min_angle")
    ap.add_argument("-dr", "--dist_ratio", required = False, type = float, default = 4.8, help = "dist_ratio")
    '''
    args = vars(ap.parse_args())
    
    
    #parameter sets
    # path to model file 
    file_path = args["path"]
    filename = args["model"]
    #ratio = args["ratio"]
    #angle = args["angle"]
    file_full_path = file_path + filename
    
    #print(file_full_path)
    
    file_base_name = os.path.basename(file_full_path).split('.')[0]

    print("Processing 3d model point cloud file '{}' ...\n".format(file_full_path))
    
    # number of slices for cross section 
    n_slices = args["n_slices"]
    
    
    
    '''
    interval = args["interval"]
    direction = args["direction"]
    
    #frame interpolation 
    n_frames = args["n_frames"]
    
    #cross section scan 
    thresh_value = args["threshold"]
    
    # parameters for tracking
    dist_thresh = args["dist_thresh"]
    max_frames_to_skip = args["max_frames_to_skip"]
    max_trace_length = args["max_trace_length"]
    radius_min = args["radius_min"]
    radius_max = args["radius_max"]
    
    #define min distance tracking threshold
    dis_tracking = args["dis_tracking"]
    min_angle = args["min_angle"]
    dist_ratio = args["dist_ratio"]
    '''

    
    model_analysis_pipeline(file_path, filename, file_base_name)
    
    
