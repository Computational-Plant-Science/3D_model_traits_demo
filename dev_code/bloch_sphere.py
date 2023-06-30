"""
Version: 1.5

Summary: Use Mahalanobis Function to calculate the Mahalanobis distance

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 bloch_sphere.py -p ~/example/quaternion/species_comp/ -v 1 -tq 0


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

from qutip import *
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize


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
    ap.add_argument("-v", "--visualize", required = False, type= int, default = 0, help = "Visualize rotation vector or not")
    ap.add_argument("-tq", "--type_quaternion", required = False, type = int, default = 0, help = "analyze quaternion type, average_quaternion=0, composition_quaternion=1, diff_quaternion=2, distance_quaternion=3")

    args = vars(ap.parse_args())

    ###################################################################
    
    current_path = args["path"]
    type_quaternion = args["type_quaternion"]
    
    file_path = current_path + "*.xlsx"
    
    # get the absolute path of all Excel files 
    ExcelFiles_list = glob.glob(file_path)
    
    '''
    b = qutip.Bloch()
    
    b.make_sphere()
    
    pnt = [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
    b.add_points(pnt)

    vec = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    b.add_vectors(vec)

    b.show()
    
    plt.show()
    '''




    ####################################################################
    # loop over the list of excel files
    data_q = []
    
    data_v = []
    
    genotype_label = []
    
    #sample_rate = 100
    
    for f in ExcelFiles_list:
        
        filename = Path(f).name
        
        base_name = filename.replace(".xlsx", "")
       
        print("Processing file '{}'...\n".format(filename))
        
        # read the csv file
        df = pd.read_excel(f)
        ###############################################################
        #get downsampled rotation vectors
        #rotVec = np.vstack((data['rotVec_rec_0'],data['rotVec_rec_1'],data['rotVec_rec_2'])).T
        
        if type_quaternion == 0:
            cols_vec = ['rotVec_avg_0','rotVec_avg_1','rotVec_avg_2']
        elif type_quaternion == 1:
            cols_vec = ['rotVec_composition_0','rotVec_composition_1','rotVec_composition_2']
        elif type_quaternion == 2:
            cols_vec = ['rotVec_diff_0','rotVec_diff_1','rotVec_diff_2']
        elif type_quaternion == 3:
            cols_vec = ['rotVec_avg_0','rotVec_avg_1','rotVec_avg_2']
        

        data_v = df[cols_vec].values.tolist()
        

        # downsample along coloum direction, every 10th
        #data_v = np.asarray(data_v)[::sample_rate,:]
        
        data_v_arr = np.asarray(data_v)
        
        
        ################################################################
        #get downsampled genotype values
        genotype_label = df['genotype_label'].values.tolist()

        # downsample along coloum direction, every 10th
        #genotype_label = genotype_v[::sample_rate,:]
        #genotype_label = np.asarray(genotype_label)[::sample_rate]
        
        genotype_label_arr = np.asarray(genotype_label)
        
        
        
        ################################################################
        genotype_type = df['genotype'].values.tolist()
        
        genotype_type_arr = np.asarray(genotype_label)
        
        genotype_unique = list(set(genotype_type))
        
        #genotype_unique_arr = np.array(genotype_unique)
        
        print("Genotypes are {} \n".format(genotype_unique))
        
        
        
        

    ######################################################################
    #normalize into unit vectors
    normalized_data_v = []
    
    for vector in data_v:
        
        n_vector = normalize([vector])
        
        n_vector_1D = [item for sub_list in n_vector for item in sub_list]
       
        #print(type(n_vector_1D))
        
        normalized_data_v.append(n_vector_1D)
        
    #print(type(normalized_data_v))
    
    #print(data_v)
    #print(normalized_data_v)
    
    #######################################################################
    #visualize vectors in sphere 
    
    #vector_sphere = qutip.Bloch3d()
    vector_sphere = qutip.Bloch()
    
    vector_sphere.make_sphere()
    
    #pnt = [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
    #vector_sphere.add_points(data_v_arr)

    #vec = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    vector_sphere.add_vectors(normalized_data_v)
    
    vector_sphere.vector_color = ['r']

    vector_sphere.show()
    
    plt.show() 
    
    

    
    ########################################################################
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from qutip import *
    
    H = sigmaz() + 0.3 * sigmay()
    e_ops = [sigmax(), sigmay(), sigmaz()]
    times = np.linspace(0, 10, 100)
    psi0 = (basis(2, 0) + basis(2, 1)).unit()
    result = mesolve(H, psi0, times, [], e_ops)
    
    
    b = Bloch()
    b.add_vectors(expect(H.unit(), e_ops))
    b.add_points(result.expect, meth='l')
    b.make_sphere()
    b.show()
    
    plt.show()
    '''
    ########################################################################
    '''
    import matplotlib as mpl

    from matplotlib import cm

    psi = (basis(10, 0) + basis(10, 3) + basis(10, 9)).unit()

    xvec = np.linspace(-5, 5, 500)

    W = wigner(psi, xvec, xvec)

    wmap = wigner_cmap(W)  # Generate Wigner colormap

    nrm = mpl.colors.Normalize(-W.max(), W.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    plt1 = axes[0].contourf(xvec, xvec, W, 100, cmap=cm.RdBu, norm=nrm)

    axes[0].set_title("Standard Colormap");

    cb1 = fig.colorbar(plt1, ax=axes[0])

    plt2 = axes[1].contourf(xvec, xvec, W, 100, cmap=wmap)  # Apply Wigner colormap

    axes[1].set_title("Wigner Colormap");

    cb2 = fig.colorbar(plt2, ax=axes[1])

    fig.tight_layout()

    plt.show()

    '''
    
    





