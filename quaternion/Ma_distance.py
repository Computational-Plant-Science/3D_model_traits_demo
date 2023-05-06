"""
Version: 1.5

Summary: Use Mahalanobis Function to calculate the Mahalanobis distance

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 Ma_distance.py -p ~/example/quaternion/tiny/ 


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
        
        base_name = filename.replace("_quaternion.xlsx", "")
        
        print("Processing file '{}'...\n".format(base_name))
        
        mkpath = current_path +  base_name + '_Mahalanobis_p'
        mkdir(mkpath)
        save_path = mkpath + '/'
        
        
        # read the csv file
        xls = pd.ExcelFile(f)
        
        sheet_name_list = ['sheet_quaternion_1', 'sheet_quaternion_2', 'sheet_quaternion_3']
        
        
        
        
        df_list = []
        
        for sheet_name in sheet_name_list:
            
            df = pd.read_excel(xls, sheet_name)

            ############################################################
            # select specific columns 
            cols_q = ['quaternion_a','quaternion_b','quaternion_c', 'quaternion_d']
            data_q = df[cols_q]
            
            #print(data_vec)
                                    
            # Creating a new column in the dataframe that holds the Mahalanobis distance for each row
            df['quaternion_Mahalanobis'] = calculateMahalanobis(y = data_q, data = df[cols_q])
            
            # compute the p-value for every Mahalanobis distance of each observation of the dataset. 
            # Creating a new column in the dataframe that calculates p-value for each mahalanobis distance
            df['quaternion_p'] = 1 - chi2.cdf(df['quaternion_Mahalanobis'], 3)
            
            # draw Mahalanobis and p values as 2d scatter plot
            cols_ma = ['quaternion_Mahalanobis','quaternion_p']
            data_ma = df[cols_ma]
            
            fig = px.scatter(data_ma, x = "quaternion_Mahalanobis", y = "quaternion_p", color = 'quaternion_Mahalanobis')
            
            Mahalanobis_p_file = (save_path + base_name + sheet_name.replace("sheet_quaternion", "") +'_quaternion__Mahalanobis_p.html')
    
            plotly.offline.plot(fig, auto_open = False, filename = Mahalanobis_p_file)
            
            
            df_q = df[['quaternion_Mahalanobis']]
            
            fig = px.histogram(df_q, x = "quaternion_Mahalanobis", nbins=20)
            
            Mahalanobis_p_file = (save_path + base_name + sheet_name.replace("sheet_quaternion", "") +'_quaternion_Mahalanobis_his.html')
    
            plotly.offline.plot(fig, auto_open = False, filename = Mahalanobis_p_file)
            
            ############################################################
            # compute Mahalanobis distance of rotation vectors
            # select specific columns 
            cols_vec = ['rotVec_rec_0','rotVec_rec_1','rotVec_rec_2']
            data_vec = df[cols_vec]
            
            #print(data_vec)
                                    
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
            
            Mahalanobis_p_file = (save_path + base_name + sheet_name.replace("sheet_quaternion", "") +'_rotVec_Mahalanobis_p.html')
    
            plotly.offline.plot(fig, auto_open = False, filename = Mahalanobis_p_file)


            df_v = df[['rotVec_Mahalanobis']]
            
            fig = px.histogram(df_v, x = "rotVec_Mahalanobis", nbins = 20)
            
            Mahalanobis_p_file = (save_path + base_name + sheet_name.replace("sheet_quaternion", "") +'_rotVec_Mahalanobis_his.html')
    
            plotly.offline.plot(fig, auto_open = False, filename = Mahalanobis_p_file)
            
            #print(df)
            
            
            ##########################################################################################
            # filter data with p vlaue less than 0.001
            p_thresh = 0.001
        
            df_sel = df.loc[(df["quaternion_p"] >= p_thresh) & (df["rotVec_p"] >= p_thresh)]
            
            df_list.append(df_sel)
            
            print("Original path number = {}, filtered path number = {}\n".format(df.shape[0], df_sel.shape[0]))
         
            
        #####################################################
        output_file = save_path + base_name + '_sel.xlsx'
        
        #print(output_file)
        
        with pd.ExcelWriter(output_file, engine = "openpyxl") as writer:
        
            for sheet_name_cur, df_cur in zip(sheet_name_list, df_list):
                
                #print(df_cur.shape[0])
                
                df_cur.to_excel(writer, sheet_name = sheet_name_cur)
            
            





