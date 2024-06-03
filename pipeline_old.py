"""
Version: 1.5

Summary: Analyze and visualzie tracked traces

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

python3 pipeline.py -p /home/suxingliu/ply_data/ -m surface.ply

parameter list:

ap.add_argument('-n_frames', '-n', required = True, type = int, default = 1 , help = 'Number of new frames.')


"""

import subprocess, os
import sys
import argparse


def execute_script(cmd_line):
    """execute script inside program"""
    try:
        print(cmd_line)
        #os.system(cmd_line)
        
        process = subprocess.Popen(cmd_line, shell=True, stdout=subprocess.PIPE)
        
        process.wait()
        
        #print process.returncode
        
    except OSError:
        
        print("Failed ...!\n")


def model_analysis_pipeline(current_path, filename):
    """execute pipeline scripts in order"""
    
    
    # step 1
    format_convert = "python3 /opt/code/format_converter.py -p " + current_path + " -m " + filename
    
    execute_script(format_convert)

    model_point_scan = "python3 /opt/code/pt_scan_engine.py -p " + current_path + " -m converted.ply " + " -i " + str(interval) +  " -de " + str(direction) 
    
    print("Computing cross section image sequence from 3D model file...\n")
    
    execute_script(model_point_scan)
   
  
    # step 3
    #cross_section_scan = "python3 /opt/code/crossection_scan.py -p " + current_path + "interpolation_result/" + " -th " + str(thresh_value)
    cross_section_scan = "python3 /opt/code/crossection_scan.py -p " + current_path + "cross_section_scan/" + " -th " + str(thresh_value)

    print("Analyzing cross section image sequence to generate labeled segmentation results...\n")
    
    execute_script(cross_section_scan)
    
    
    # step 4
    object_tracking = "python3 /opt/code/object_tracking.py -p " + current_path + "active_component/" + " -d " + str(dist_thresh) + " -mfs " + str(max_frames_to_skip) + " -mtl " + str(max_trace_length) + " -rmin " + str(radius_min) + " -rmax " + str(radius_max)   
    
    print("Analyzing root system traits by tracking individual root trace from each crosssection image...\n")
    
    execute_script(object_tracking)
    
    
    # step 5
    trace_analysis = "python3 /opt/code/trace_load_connect.py -p " + current_path + "trace_track/" + " -dt " + str(dis_tracking) + " -ma " + str(min_angle) + " -dr " + str(dist_ratio)

    print("Analyzing tracked root system traits...\n")
    
    execute_script(trace_analysis)
    
    

if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m", "--model", required = False, help = "model file name")
    ap.add_argument("-i", "--interval", required = False, default = '1',  type = int, help= "intervals along sweeping plane")
    ap.add_argument("-de", "--direction", required = False, default = 'X', help = "direction of sweeping plane, X, Y, Z")
    #ap.add_argument("-r", "--reverse", required = False, default = '1', type = int, help = "Reverse model top_down, 1 for Ture, 0 for False")
    ap.add_argument('-n_frames', '-n', required = False, type = int, default = 2 , help = 'Number of new frames.')
    ap.add_argument("-th", "--threshold", required = False, default = '2.35', type = float, help = "threshold to remove outliers")
    ap.add_argument('-d', '--dist_thresh', required = False, type = int, default = 10 , help = 'dist_thresh.')
    ap.add_argument('-mfs', '--max_frames_to_skip', required = False, type = int, default = 15 , help = 'max_frames_to_skip.')
    ap.add_argument('-mtl', '--max_trace_length', required = False, type = int, default = 15 , help = 'max_trace_length.')
    ap.add_argument('-rmin', '--radius_min', required = False, type = int, default = 1 , help = 'radius_min.')
    ap.add_argument('-rmax', '--radius_max', required = False, type = int, default = 100 , help = 'radius_max.')
    ap.add_argument("-dt", "--dis_tracking", required = False, type = float, default = 50.5, help = "dis_dis_tracking")
    ap.add_argument("-ma", "--min_angle", required = False, type = float, default = 0.1, help = "min_angle")
    ap.add_argument("-dr", "--dist_ratio", required = False, type = float, default = 4.8, help = "dist_ratio")
    args = vars(ap.parse_args())
    
    
    #parameter sets
    # point_cloud_scan
    # setting path to model file 
    current_path = args["path"]
    filename = args["model"]
    file_path = current_path + filename
    interval = args["interval"]
    direction = args["direction"]
    #flag_reverse = args["reverse"]
    
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
    

    
    model_analysis_pipeline(current_path, filename)
    
    
