"""
Version: 1.5

Summary: Use Mahalanobis Function to calculate the Mahalanobis distance

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 Ma_distance.py -p ~/example/quaternion/species_comp/bean/average/ -tq 0 

    python3 Ma_distance.py -p ~/example/quaternion/species_comp/bean/average/ -gl 1 -gn bean -tq 0 

argument:
("-p", "--path", required = True,    help = "path to image file")

"""


# Importing libraries

import numpy as np
import pandas as pd
import scipy as stats
from scipy.stats import chi2

import glob
import os,fnmatch,os.path
import argparse
import shutil
from pathlib import Path 

import plotly.express as px
import plotly

import itertools

from pyquaternion import Quaternion


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


# Mahalanobis Function to calculate the Mahalanobis distance
def calculateMahalanobis(y=None, data=None, cov=None):

    y_mu = y - np.mean(data, axis=0)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()



if __name__ == '__main__':

    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to excel file")
    ap.add_argument("-gl", "--genotype_label", required = True, type = int, default = -1, help = "genotype_label, represented as integer")
    ap.add_argument("-gn", "--genotype_name", required = True, type = str, default = 'empty', help = "genotype_name, represented as string")
    ap.add_argument("-tq", "--type_quaternion", required = False, type = int, default = 0, help = "analyze quaternion type, average_quaternion=0, composition_quaternion=1, diff_quaternion=2, distance_quaternion=3")

    args = vars(ap.parse_args())

    ###################################################################
    
    current_path = args["path"]
    
    if args["genotype_label"] == -1:
        print("No genotype_label was assigned!\n")
    else:
        genotype_label = args["genotype_label"]
    
    if args["genotype_name"] == 'empty':
        print("No genotype_name was assigned!\n")
    else:
        genotype_name = args["genotype_name"]
    
    
    type_quaternion = args["type_quaternion"]
    
    file_path = current_path + "*.xlsx"

    # get the absolute path of all Excel files 
    ExcelFiles_list = sorted(glob.glob(file_path))
    '''
    if type_quaternion == 0:
        str_replace = '_average.xlsx'
    elif type_quaternion == 1:
        str_replace = '_composition.xlsx'
    elif type_quaternion == 2:
        str_replace = '_diff.xlsx'
    elif type_quaternion == 3:
        str_replace = '_distance.xlsx'
    '''
    str_replace = '.xlsx'
    
    ####################################################################
    # add filename to the first column of all excel files
    # loop over the list of excel files
    for f_id, f in enumerate(ExcelFiles_list):
        
        filename = Path(f).name
        
        base_name = filename.replace(str_replace, "")
        
        print("Processing file '{}'...\n".format(filename))
        
        mkpath = current_path + base_name + '_Mahalanobis'
        mkdir(mkpath)
        save_path = mkpath + '/'
        
        
        if args["genotype_label"] == -1:
            print("No genotype_label was assigned!\n")
            genotype_label = f_id + 0
        else:
            genotype_label = args["genotype_label"]
        
        if args["genotype_name"] == 'empty':
            print("No genotype_name was assigned!\n")
            genotype_name = base_name
        else:
            genotype_name = args["genotype_name"]
        
        
        
        print("genotype_name = {} genotype_label = {}\n".format(genotype_name, genotype_label))
        
        #output_file = save_path + base_name + '_Mahalanobis.xlsx'
        
        #print("output_file '{}'...\n".format(output_file))
        
       
        
        
        # read the csv file
        xls = pd.ExcelFile(f)
        
        #sheet_name_list = ['sheet_quaternion_1', 'sheet_quaternion_2', 'sheet_quaternion_3']
        
        sheet_name_list = xls.sheet_names
        

        
        df_list = []
        
        for sheet_name in sheet_name_list:
            
            df = pd.read_excel(xls, sheet_name)

            ############################################################
            # select specific columns 
            if type_quaternion == 0:
                cols_q = ['quaternion_a','quaternion_b','quaternion_c', 'quaternion_d']
            elif type_quaternion == 1:
                cols_q = ['composition_quaternion_a','composition_quaternion_b','composition_quaternion_c', 'composition_quaternion_d']
            elif type_quaternion == 2:
                cols_q = ['diff_quaternion_a','diff_quaternion_b','diff_quaternion_c', 'diff_quaternion_d']
            elif type_quaternion == 3:
                cols_q = ['distance_absolute','distance_intrinsic', 'distance_symmetrized']
            
            
            print("Extracting data '{}'...\n".format(cols_q))
            
            data_q = df[cols_q]
            
            #data_q_list = data_q.values.tolist()
            

            
            # Mahalanobis distance
            #################################################################################################                        
            # Creating a new column in the dataframe that holds the Mahalanobis distance for each row
            df['quaternion_Mahalanobis'] = calculateMahalanobis(y = data_q, data = df[cols_q])
            
            # compute the p-value for every Mahalanobis distance of each observation of the dataset. 
            # Creating a new column in the dataframe that calculates p-value for each mahalanobis distance
            df['quaternion_p'] = 1 - chi2.cdf(df['quaternion_Mahalanobis'], 3)
            
            # draw Mahalanobis and p values as 2d scatter plot
            cols_ma = ['quaternion_Mahalanobis','quaternion_p']
            data_ma = df[cols_ma]
            
            fig = px.scatter(data_ma, x = "quaternion_Mahalanobis", y = "quaternion_p", color = 'quaternion_Mahalanobis')
            
            Mahalanobis_p_file = (save_path + base_name + sheet_name.replace("Sheet1", "") +'_quaternion_Mahalanobis_p.html')
    
            plotly.offline.plot(fig, auto_open = False, filename = Mahalanobis_p_file)
            
            
            df_q = df[['quaternion_Mahalanobis']]
            
            fig = px.histogram(df_q, x = "quaternion_Mahalanobis", nbins=20)
            
            Mahalanobis_p_file = (save_path + base_name + sheet_name.replace("Sheet1", "") +'_quaternion_Mahalanobis_his.html')
    
            plotly.offline.plot(fig, auto_open = False, filename = Mahalanobis_p_file)
            
            
            ############################################################
            # compute Mahalanobis distance of rotation vectors 
            if type_quaternion == 0:
                cols_vec = ['rotVec_avg_0','rotVec_avg_1','rotVec_avg_2']
            elif type_quaternion == 1:
                cols_vec = ['rotVec_composition_0','rotVec_composition_1','rotVec_composition_2']
            elif type_quaternion == 2:
                cols_vec = ['rotVec_diff_0','rotVec_diff_1','rotVec_diff_2']
            elif type_quaternion == 3:
                cols_vec = ['rotVec_avg_0','rotVec_avg_1','rotVec_avg_2']
            
            data_vec = df[cols_vec]
            
            print(data_vec)
                                    
            # Creating a new column in the dataframe that holds the Mahalanobis distance for each row
            df['rotVec_Mahalanobis'] = calculateMahalanobis(y = data_vec, data = df[cols_vec])

            # compute the p-value for every Mahalanobis distance of each observation of the dataset. 
            # Creating a new column in the dataframe that calculates p-value for each mahalanobis distance
            df['rotVec_p'] = 1 - chi2.cdf(df['rotVec_Mahalanobis'], 3)
            
            # note:
            #the observation having a p-value less than 0.001 is assumed to be an outlier. 
            
            
            # draw Mahalanobis and p values as 2d scatter plot
            cols_ma = ['rotVec_Mahalanobis','rotVec_p']
            data_ma = df[cols_ma]
            
            fig = px.scatter(data_ma, x = "rotVec_Mahalanobis", y = "rotVec_p", color = 'rotVec_Mahalanobis')
            
            Mahalanobis_p_file = (save_path + base_name + sheet_name.replace("Sheet1", "") +'_rotVec_Mahalanobis_p.html')
    
            plotly.offline.plot(fig, auto_open = False, filename = Mahalanobis_p_file)


            df_v = df[['rotVec_Mahalanobis']]
            
            fig = px.histogram(df_v, x = "rotVec_Mahalanobis", nbins = 20)
            
            Mahalanobis_p_file = (save_path + base_name + sheet_name.replace("Sheet1", "") +'_rotVec_Mahalanobis_his.html')
    
            plotly.offline.plot(fig, auto_open = False, filename = Mahalanobis_p_file)
            
            #print(df)
            
            #############################################################################################
            # add genotype_label and genotype
            df['genotype_label'] = np.repeat(genotype_label, len(data_q))
            df['genotype'] = np.repeat(genotype_name, len(data_q))
            
            
            
            
            ##########################################################################################
            # filter data with p vlaue less than 0.001
            p_thresh = 0.001
        
            df_sel = df.loc[(df["quaternion_p"] >= p_thresh) & (df["rotVec_p"] >= p_thresh)]
            
            df_list.append(df_sel)
            
            print("Original path number = {}, filtered path number = {}\n".format(df.shape[0], df_sel.shape[0]))
         
            
        #####################################################
        output_file = save_path + base_name + '_Mahalanobis.xlsx'
        
        #print(output_file)
        
        with pd.ExcelWriter(output_file, engine = "openpyxl") as writer:
        
            for sheet_name_cur, df_cur in zip(sheet_name_list, df_list):
                
                #print(df_cur.shape[0])
                
                df_cur.to_excel(writer, sheet_name = sheet_name_cur)
            
       





