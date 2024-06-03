"""
Version: 1.5

Summary: compute the segmentaiton and label of cross section image sequence 

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

python3 cross_watershed.py -p ~/example/test/slices/  -ft png


argument:
("-p", "--path", required = True,    help = "path to image file")
("-ft", "--filetype", required = False, default = 'jpg',   help = "Image filetype")

"""


# import the necessary packages
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import convex_hull_image
from skimage.measure import regionprops

from sklearn.cluster import KMeans

from scipy.signal import find_peaks
from scipy import ndimage

import numpy as np
import argparse
import imutils
import cv2
 
import glob
import os,fnmatch

import math

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pandas as pd


import itertools

from findpeaks import findpeaks



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
        
        

# compute diameter from area
def area_radius(area_of_circle):
    radius = ((area_of_circle/ math.pi)** 0.5)
    
    #note: return diameter instead of radius
    return 2*radius 



# segmentation of overlapping components 
def watershed_seg(orig, thresh, min_distance_value):
    
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    
    localMax = peak_local_max(D, indices = False, min_distance = min_distance_value,  labels = thresh)
    
    #localMax = peak_local_max(D, min_distance = min_distance_value,  labels = thresh)
    
    #localMax = peak_local_max(D,  min_distance = min_distance_value,  labels = thresh)
     
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure = np.ones((3, 3)))[0]
    
    #print("markers")
    #print(type(markers))
    
    labels = watershed(-D, markers, mask = thresh)
    
    #print("[INFO] {} unique segments found\n".format(len(np.unique(labels)) - 1))
    
    return labels
    


# analyze cross sections 
def crosssection_analysis(image_file):
    

    # load the image 
    imgcolor = cv2.imread(image_file)
    
    # if cross scan images are white background and black foreground
    #imgcolor = ~imgcolor
    
    # accquire image dimensions 
    height, width, channels = imgcolor.shape
    #shifted = cv2.pyrMeanShiftFiltering(image, 5, 5)

    #Image binarization by apltying otsu threshold
    img = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)
    
    # Convert BGR to GRAY
    img_lab = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2LAB)
    
    gray = cv2.cvtColor(img_lab, cv2.COLOR_BGR2GRAY)
    
    #Obtain the threshold image using OTSU adaptive filter
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    #Obtain the threshold image using OTSU adaptive filter
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    
    
    #find contours and fill contours 
    ####################################################################
    #container version
    contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #local version
    #_, contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_img = []
    
    #define image morphology operation kernel
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    
    #draw all the filled contours
    for c in contours:
        
        #fill the connected contours
        contours_img = cv2.drawContours(binary, [c], -1, (255, 255, 255), cv2.FILLED)
        
        #contours_img = cv2.erode(contours_img, kernel, iterations = 5)

    #Obtain the threshold image using OTSU adaptive filter
    thresh_filled = cv2.threshold(contours_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    #Obtain the threshold image using OTSU adaptive filter
    ret, binary_filled = cv2.threshold(contours_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    
    
    # process filled contour images to extract connected Components
    ####################################################################
    
    contours, hier = cv2.findContours(binary_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #local version
    #_, contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #draw all the filled contours
    for c in contours:
        
        #fill the connected contours
        contours_img = cv2.drawContours(binary, [c], -1, (255, 255, 255), cv2.FILLED)

    # define kernel
    connectivity = 8
    
    #find connected components 
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(contours_img, connectivity , cv2.CV_32S)
    
    #find the component with largest area 
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) 
    
    #collection of all component area
    areas = [s[4] for s in stats]
    
    # average of component area
    area_avg = sum(areas)/len(np.unique(labels))
    
    #area_avg = sum(areas)
    
    ###################################################################
    # segment overlapping components
    #make backup image
    orig = imgcolor.copy()

    #watershed based segmentaiton 
    labels = watershed_seg(contours_img, thresh_filled, min_distance_value)
    
    
    N_seg = len(np.unique(labels))-1
    
    
    
    #Map component labels to hue val
    label_hue = np.uint8(128*labels/np.max(labels))
    #label_hue[labels == largest_label] = np.uint8(15)
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set background label to black
    labeled_img[label_hue==0] = 0
    
    # save label results
    #result_file = (save_path_label + base_name + '_label.png')
    
    #print(result_file)

    #cv2.imwrite(result_file, labeled_img)
    

    
    return N_seg, labeled_img
    


#cluster 1D list using Kmeans
def cluster_list(list_array, n_clusters):
    
    data = np.array(list_array)
    
    if data.ndim == 1:
        
        data = data.reshape(-1,1)
    
    #kmeans = KMeans(n_clusters).fit(data.reshape(-1,1))
        
    kmeans = KMeans(n_clusters, init='k-means++', random_state=0).fit(data)
    
    #kmeans = KMeans(n_clusters).fit(data)
    
    labels = kmeans.labels_
    
    centers = kmeans.cluster_centers_
    
    center_labels = kmeans.predict(centers)
    
    #print(kmeans.cluster_centers_)
    
    '''
    #visualzie data clustering 
    ######################################################
    y_kmeans = kmeans.predict(data)
    
    plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=5, cmap='viridis')

    centers = kmeans.cluster_centers_
    
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    
    #plt.legend()

    plt.show()
    '''
    
    
    
    '''
    from sklearn.metrics import silhouette_score
    
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    silhouette_avg = []
    for num_clusters in range_n_clusters:
     
         # initialize kmeans
         kmeans = KMeans(n_clusters=num_clusters)
         kmeans.fit(data)
         cluster_labels = kmeans.labels_
         
         # silhouette score
         silhouette_avg.append(silhouette_score(data, cluster_labels))
    
    plt.plot(range_n_clusters,silhouette_avg,'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette analysis For Optimal k')
    plt.show()
    
    
    Sum_of_squared_distances = []
    K = range(2,8)
    for num_clusters in K :
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data)
        Sum_of_squared_distances.append(kmeans.inertia_)
        
    plt.plot(K,Sum_of_squared_distances,'bx-')
    plt.xlabel('Values of K') 
    plt.ylabel('Sum of squared distances/Inertia') 
    plt.title('Elbow Method For Optimal k')
    plt.show()
    '''
    ######################################################
    
    
    return labels, centers, center_labels
    

# smooth 1d list
def smooth_data_convolve_average(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")

    # The "my_average" part: shrinks the averaging window on the side that 
    # reaches beyond the data, keeps the other side the same size as given 
    # by "span"
    re[0] = np.average(arr[:span])
    for i in range(1, span + 1):
        re[i] = np.average(arr[:i + span])
        re[-i] = np.average(arr[-i - span:])
    return re    



# Find closest number to k in given list
def closest(lst, K):
     
    v_closet = lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
    
    idx_list = [i for i, value in enumerate(lst) if value == v_closet]
    
    return idx_list, v_closet
     


if __name__ == '__main__':

    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", required = False, default = 'jpg',   help = "Image filetype")
    ap.add_argument("-md", "--min_dis", required = False, type = int, default = 25,   help = "min distance for watershed segmentation")
    ap.add_argument("-sp", "--span", required = False, type = int, default = 3,   help = "paramter to smooth the 1d curve")
    args = vars(ap.parse_args())


    
    # setting path to model file
    file_path = args["path"]
    ext = args['filetype']
     
    min_distance_value = args['min_dis']
    span = args['span']
    
    print("Min dis = {}\n".format(min_distance_value))
    
    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype

    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))

    # make the folder to store the results
    mkpath = file_path + str('lable')
    mkdir(mkpath)
    save_path_label = mkpath + '/'

    List_N_seg = []
    
    ####################################################################
    for image_file in imgList:
        
        path, filename = os.path.split(image_file)
    
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        (n_seg, labeled_img) = crosssection_analysis(image_file)
        
        print("{} N_labels = {}".format(base_name, n_seg))
        
        result_file = (save_path_label + base_name + '_label.png')
        
        cv2.imwrite(result_file, labeled_img)
        
        List_N_seg.append(n_seg)
        

    
    #########################################################################
    # find the first 1 in list 
    idx_f = List_N_seg.index(1)
    
    print("first index {}\n".format(idx_f))
    
    # adjust the list values 
    if idx_f > 0:
        
        List_N_seg[0:idx_f] = [1 for x in List_N_seg[0:idx_f]]
    
    
    # find the index of peak max value
    if np.argmax(List_N_seg) > 0:
        
        List_N_seg = List_N_seg[0: np.argmax(List_N_seg)]
    
    ###################################################################
    #span = 3
  
    List_N_seg_smooth = smooth_data_convolve_average(np.array(List_N_seg), span)
    
    
    #List_N_seg_smooth = List_N_seg

    ####################################################################
    # peak detection
    # Data
    X = List_N_seg_smooth
    
    # Initialize
    fp = findpeaks(method='peakdetect', lookahead = 1)
    
    # return dictionary object
    results = fp.fit(X)

    # Plot
    fp.plot()
    
    result_file = file_path + 'N_seg.png'
    
    plt.savefig(result_file)
    
    plt.close()
    #####################################################################

    # parse the dictionary object and ge the pd.DataFrame
    df = results.get("df")
 
    # print df names
    print(list(df.columns))      #x    y  labx  valley   peak

    # Use boolean indexing to extract peak locations to a new pd frame
    df_peak = df.loc[df['peak'] == True]

    print(df_peak)
    
    # convert x, y values into list
    peak_y = df_peak['y'].tolist()
    peak_x = df_peak['x'].tolist()
    
    
    
    ###############################################################################
    N_cluster = 4
    
    # cluster list to find the center values
    (labels, centers, center_labels) = cluster_list(peak_y, n_clusters = N_cluster)
    
    sorted_idx = np.argsort(centers[:,0])
    
    centers = centers[sorted_idx]
    
    print("centers = {}\n".format(centers))
    
    #print("sorted_idx = {}\n".format(sorted_idx))
    
    
    #print("centers = {}\n".format(centers[sorted_idx]))
    

    
    '''
    plt.scatter([i for i in range(len(peak_y))], peak_y, c=labels)
    
    result_file = file_path + 'scatter.png'
    
    plt.savefig(result_file)
    
    plt.close()
    '''
    
    N_count = []
    idx_N_count = []
    
    # get idx of the cluster center in "y" series 
    for idx, (center_value) in enumerate(centers):
        
        (idx_list, v_closet) = closest(peak_y, center_value)
        
        
        if len(idx_list) > 0:
        
            print("idx_list = {}, v_closet = {}\n".format(peak_x[idx_list[0]], v_closet))
            
            N_count.append(v_closet)
            idx_N_count.append(peak_x[idx_list[0]])
    
    print(N_count)
    print(idx_N_count)
    
    
    
    '''
    ####################################################################
    # find the peak location index
    x = np.array(List_N_seg_smooth)
    
    (peaks, properties) = find_peaks(x, prominence = 1, width = 2)
    
    print(peaks)
    
    
    if len(peaks) > 4:
        
        # initialize N
        N = 4

        # Indices of N largest elements in list
        sorted_list = sorted(List_N_seg, reverse=True)
        
        res = [i for i,x in enumerate(List_N_seg) if x in itertools.islice(sorted_list, N)]

        # printing result
        print("Indices list of max N elements is : " + str(res))
    
    '''


    
    


    

    
    

    

    
