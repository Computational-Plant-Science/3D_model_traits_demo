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
from os.path import join
from pathlib import Path

import numpy as np


# execute script inside program
def execute_script(cmd_line):
    try:
        # print(cmd_line)
        # os.system(cmd_line)

        process = subprocess.getoutput(cmd_line)

        print(process)

        # process = subprocess.Popen(cmd_line, shell = True, stdout = subprocess.PIPE)
        # process.wait()
        # print (process.communicate())

    except OSError:

        print("Failed ...!\n")


def model_analysis_pipeline(model_path: str, output_directory: str, n_slices: int, test: bool = False):

    # step 1  python3 model_alignment.py -p ~/example/test.ply
    print("Transform point cloud to its rotation center and align its upright orientation with Z direction...\n")
    subprocess.getoutput(f"python3 model_alignment.py -p {model_path} -o {output_directory} -t {str(test)}")

    # step 2 ./AdTree/Release/bin/AdTree ~/example/pt_cloud/test.xyz ~/example/pt_cloud/
    print("Compute structure and skeleton from point cloud model ...\n")
    model_stem = Path(model_path).stem
    aligned_model_path = join(output_directory, model_stem + ".xyz")
    subprocess.getoutput(f"./AdTree/Release/bin/AdTree {aligned_model_path} {output_directory}")

    # step 3  python3 extract_slice.py -p ~/example/pt_cloud/test_branches.obj -n 100
    print("Generate cross section sequence ...\n")
    obj_model_path = join(output_directory, model_stem + "_branches.obj")
    slice_directory_path = join(output_directory, 'slices')
    subprocess.getoutput(f"python3 extract_slice.py -p {obj_model_path} -o {slice_directory_path} -n {str(n_slices)}")

    # step 4 python3 skeleton_analyze.py -m1 ~/example/pt_cloud/test_skeleton.ply -m2 ~/example/pt_cloud/test_aligned.ply -m3 ~/example/pt_cloud/slices/
    print("Analyze skeleton&structure and compute traits...\n")
    skeleton_model_path = join(output_directory, model_stem + "_skeleton.ply")
    traits_computation = f"python3 skeleton_analyze.py -m1 {skeleton_model_path} -m2 {aligned_model_path} -m3 {slice_directory_path} -o {output_directory}"
    execute_script(traits_computation)


if __name__ == '__main__':
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="path to *.ply model file")
    ap.add_argument("-o", "--output", required=True, help="path to directory to save results")
    ap.add_argument("-t", "--test", required=False, type=int, default=0, help="if using test setup")
    ap.add_argument("-n", "--n_slices", required=False, type=int, default=500, help='Number of slices for 3d model.')
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
    file_path = args["path"]            # path to model file
    output_directory = args['output']   # directory for results
    n_slices = args["n_slices"]         # number of slices for cross section
    test = args['test']                 # if using test setup

    print("Processing 3d model point cloud file '{}' ...\n".format(file_path))

    # if no output directory provided, use current working directory
    if output_directory is None: output_directory = os.getcwd()

    # make sure output directory exists
    if not Path(output_directory).is_dir():
        print(f"Output directory does not exist: {output_directory}")
        sys.exit(1)

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

    model_analysis_pipeline(file_path, output_directory, n_slices, test)
