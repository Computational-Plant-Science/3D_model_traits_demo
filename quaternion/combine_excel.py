"""
Version: 1.5

Summary: CDF visualization

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 combine_excel.py -p ~/cluster_test/syngenta_data/results/


argument:
("-p", "--path", required = True,    help = "path to image file")

"""

#!/usr/bin/python
# Standard Libraries



import glob
import os,fnmatch,os.path
import argparse
import shutil
import pandas as pd
from pathlib import Path  


def mkdir(path):
    """Create result folder"""
 
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
        return False



if __name__ == '__main__':

    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to excel file")
    #ap.add_argument("-f", "--filename", required = True, help = "data file name")
    args = vars(ap.parse_args())

    ###################################################################
    
    current_path = args["path"]
    
    file_path = current_path + "*.xlsx"

    # get the absolute path of all Excel files 
    ExcelFiles_list = glob.glob(file_path)


    ####################################################################
    # add filename to the first column of all excel files
    # loop over the list of excel files
    for f in ExcelFiles_list:
        
        filename = Path(f).name
        filename = filename.replace("_trait.xlsx", "")
        
        print("Processing file '{}'...\n".format(filename))
        
        # read the csv file
        data = pd.read_excel(f)
        
        if 'file_name' in data.columns:
            
            print("file_name already exists!")
        else:
            data.insert(loc = 0, column = 'file_name', value = filename)
        
        data.to_excel(f)
    
    #print(df)
    
    
    ####################################################################
    # merge all excel files
    # read all Excel files at once
    df = pd.concat(pd.read_excel(excelFile) for excelFile in ExcelFiles_list)

    # save folder construction
    mkpath = os.path.dirname(current_path) +'/combined'
    mkdir(mkpath)
    save_path = mkpath + '/'
        
    output_file = save_path + "combined.xlsx"
    
    # create excel writer object
    #combined_excel = pd.ExcelWriter(output_file)
    
    # write dataframe to excel
    df.to_excel(output_file)
    
    # save the excel
    #combined_excel.save()
    
    print('DataFrame is written successfully to Excel File.')
    

    ####################################################################
