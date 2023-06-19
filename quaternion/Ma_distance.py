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




# compute mutltipy two quaternions in a pair:
def quaternion_multiply(Q0, Q1):
    """
    Multiplies two quaternions.
 
    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31) 
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32) 
 
    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33) 
 
    """
    
    
    # Extract the values from Q0
    w0 = Q0[0]
    x0 = Q0[1]
    y0 = Q0[2]
    z0 = Q0[3]
     
    # Extract the values from Q1
    w1 = Q1[0]
    x1 = Q1[1]
    y1 = Q1[2]
    z1 = Q1[3]
     
    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
     
    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])
     
    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32) 
    return final_quaternion



# get consecutive elements pairing in list:
def pairwise(q_list):
 
    # use itertools.tee to create two iterators from the list
    a, b = itertools.tee(q_list)
     
    # advance the iterator by one element
    next(b, None)
     
    # use zip to pair the elements from the two iterators
    res = list(zip(a, b))  
    
    return res



# compute mutltipy of a list of quaternions:
def quaternion_list_multiply(q_list):

    # get the pair of the elements from the list
    res = pairwise(q_list)
    
    # Create a 4 element array containing the final quaternion mutltipy results
    q_mutiply = np.array([0, 0, 0, 0])
     
    #compute mutltipy of adjacent pair of quaternions and then loop all elements 
    for idx, q_pair_value in enumerate(res):
        
        if idx < 1: 
            q_mutiply = quaternion_multiply(res[0][0], res[0][1])
        else:
            q_mutiply = quaternion_multiply(q_mutiply, res[idx][1])

    
    return q_mutiply


# compute the distance between two quaternions accounting for the sign ambiguity.
def quaternion_list_distance(q_list):

    # get the pair of the elements from the list
    res = pairwise(q_list)
    
    # Create a 4 element array containing the final quaternion mutltipy results
    q_distance = []
    
    # make the column of the same length for easy operate
    q_distance.append([0,0,0])
     
    #compute mutltipy of adjacent pair of quaternions and then loop all elements 
    for idx, q_pair_value in enumerate(res):
        
        #Quaternion(numpy.array([a, b, c, d]))
        
        Q_Current = Quaternion(np.array(q_pair_value[0]))
        Q_Next = Quaternion(np.array(q_pair_value[1]))
        
        #This function does not measure the distance on the hypersphere, 
        #but it takes into account the fact that q and -q encode the same rotation. 
        #It is thus a good indicator for rotation similarities.
        # Quaternion absolute distance.
        Q_D_absolute = Quaternion.absolute_distance(Q_Current, Q_Next)
        
        # Quaternion intrinsic distance.
        #Although q0^(-1)*q1 != q1^(-1)*q0, the length of the path joining them is given by the logarithm of those product quaternions, the norm of which is the same.
        Q_D_intrinsic = Quaternion.distance(Q_Current, Q_Next)
        
        # Quaternion symmetrized distance.
        #Find the intrinsic symmetrized geodesic distance between q0 and q1.
        Q_D_symmetrized = Quaternion.sym_distance(Q_Current, Q_Next)
        
        q_distance.append([Q_D_absolute, Q_D_intrinsic, Q_D_symmetrized])


    #print(q_distance)
    
    return q_distance







if __name__ == '__main__':

    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to excel file")
    ap.add_argument("-gl", "--genotype_label", required = True, type = int, help = "genotype_label, represented as integer")
    ap.add_argument("-gn", "--genotype_name", required = True, type = str, help = "genotype_name, represented as string")
    ap.add_argument("-tq", "--type_quaternion", required = False, type = int, default = 0, help = "analyze quaternion type, 0 = average_quaternion, 1 = composition_quaternion")
    args = vars(ap.parse_args())

    ###################################################################
    
    current_path = args["path"]
    genotype_label = args["genotype_label"]
    genotype_name = args["genotype_name"]
    
    type_quaternion = args["type_quaternion"]
    
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
        
        #sheet_name_list = ['Sheet1']
        

        
        df_list = []
        
        for sheet_name in sheet_name_list:
            
            df = pd.read_excel(xls, sheet_name)

            ############################################################
            # select specific columns 
            if type_quaternion == 0:
                cols_q = ['quaternion_a','quaternion_b','quaternion_c', 'quaternion_d']
            elif type_quaternion == 1:
                cols_q = ['composition_quaternion_a','composition_quaternion_b','composition_quaternion_c', 'composition_quaternion_d']
            
            data_q = df[cols_q]
            
            data_q_list = data_q.values.tolist()
            

            # quaternion distance 
            ###########################################################################################
            '''
            q_composition = quaternion_list_multiply(data_q_list)
            
            print(q_composition)
            
            q_distance = quaternion_list_distance(data_q_list)
            
            q_distance_arr = np.vstack(q_distance)
            
            print(q_distance_arr.shape)
            
            #df['quaternion__composition'] = np.repeat(q_composition, len(q_distance_arr))
            
            # Creating a new column in the dataframe that holds the quaternion distance for each row
            df['absolute_distance'] = q_distance_arr[:, 0]
            df['intrinsic_distance'] = q_distance_arr[:, 1]
            df['symmetrized_distance'] = q_distance_arr[:, 2]
            '''
            
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
            
            Mahalanobis_p_file = (save_path + base_name + sheet_name.replace("sheet_quaternion", "") +'_quaternion__Mahalanobis_p.html')
    
            plotly.offline.plot(fig, auto_open = False, filename = Mahalanobis_p_file)
            
            
            df_q = df[['quaternion_Mahalanobis']]
            
            fig = px.histogram(df_q, x = "quaternion_Mahalanobis", nbins=20)
            
            Mahalanobis_p_file = (save_path + base_name + sheet_name.replace("sheet_quaternion", "") +'_quaternion_Mahalanobis_his.html')
    
            plotly.offline.plot(fig, auto_open = False, filename = Mahalanobis_p_file)
            
            ############################################################
            # compute Mahalanobis distance of rotation vectors select specific columns 
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
            
            #############################################################################################
            # add genotype_label and genotype
            df['genotype_label'] = np.repeat(genotype_label, len(data_q_list))
            df['genotype'] = np.repeat(genotype_name, len(data_q_list))
            
            
            
            
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
            
            





