"""
Version: 1.5

Summary: compbine mutiple excels files 

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 combine_excel.py -p ~/example/quaternion/species_comp/excels/ 


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


def fast_scandir(dirname):
    
    subfolders= sorted([f.path for f in os.scandir(dirname) if f.is_dir()])
    
    return subfolders



def add_file_name(ExcelFiles_list):
    
    ####################################################################
    # add filename to the first column of all excel files
    # loop over the list of excel files
    for f in ExcelFiles_list:
        
        filename = Path(f).name
        #filename = filename.replace("_trait.xlsx", "")
        
        print("Processing file '{}'...\n".format(filename))
        
        # read the csv file
        data = pd.read_excel(f)
        
        if 'file_name' in data.columns:
            
            print("file_name already exists!")
        else:
            data.insert(loc = 0, column = 'file_name', value = filename)
        
        data.to_excel(f)
    


def merge_files(ExcelFiles_list, folder_name, subfolder_path):
    
    # merge all excel files
    # read all Excel files at once
    df = pd.concat(pd.read_excel(excelFile) for excelFile in ExcelFiles_list)

    '''
    if type_quaternion == 0:
        combined_file_name = folder_name + '_average.xlsx'
    elif type_quaternion == 1:
        combined_file_name = folder_name + '_composition.xlsx'
    elif type_quaternion == 2:
        combined_file_name = folder_name + '_diff.xlsx'
    elif type_quaternion == 3:
        combined_file_name = folder_name + '_distance.xlsx'
    '''
    
    combined_file_name = folder_name + '.xlsx'
    
    # save folder construction
    mkpath = os.path.dirname(subfolder_path) +'/combined'
    mkdir(mkpath)
    save_path = mkpath + '/'

    output_file = save_path + combined_file_name
    
    print("Result file path '{}'\n".format(output_file))
    
    # create excel writer object
    #combined_excel = pd.ExcelWriter(output_file)
    
    # write dataframe to excel
    df.to_excel(output_file)
    
    # save the excel
    #combined_excel.save()
    
    print('DataFrame is written successfully to Excel File.')

        
        
        


if __name__ == '__main__':

    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to excel file")
    #ap.add_argument("-tq", "--type_quaternion", required = False, type = int, default = 0, help = "analyze quaternion type, average_quaternion=0, composition_quaternion=1, diff_quaternion=2, distance_quaternion=3")

    args = vars(ap.parse_args())

    ###################################################################
    
    current_path = args["path"]
    
    file_path = current_path + "*.xlsx"
    
    #type_quaternion = args["type_quaternion"]
    
    #folder_name = os.path.basename(current_path)
    
    
    subfolders = fast_scandir(current_path)
    
    if any(subfolders):
        
        for subfolder_id, subfolder_path in enumerate(subfolders):
        
            folder_name = os.path.basename(subfolder_path)
     
            print("Combine files in folder {}\n".format(folder_name))
            
            excel_file_path = subfolder_path + "/*.xlsx"
            
            #print(excel_file_path)
            
            
            # get the absolute path of all Excel files 
            ExcelFiles_list = sorted(glob.glob(excel_file_path))
            
            
            #add_file_name(ExcelFiles_list)
            
            merge_files(ExcelFiles_list, folder_name, subfolder_path)
            
        
    
    else:
        print("No sub folders found, combine files in current folder...\n")
        
        ExcelFiles_list = sorted(glob.glob(file_path))
        
        
        #print(ExcelFiles_list)
        #print()
        #print(sorted(ExcelFiles_list))
        
        folder_name = 'Mahalanobis'
        
        merge_files(ExcelFiles_list, folder_name, current_path)

    
