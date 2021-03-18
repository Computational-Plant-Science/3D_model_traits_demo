#required libraries


#python libraries
import argparse
import time
import shutil
import sys
import os,fnmatch
import glob
from itertools import tee, izip
from itertools import compress
import multiprocessing
from multiprocessing import Pool
from contextlib import closing
import warnings
from operator import itemgetter
import math
import copy
import csv


import numpy as np
from numpy import NaN, Inf, arange, isscalar, asarray, array
from numpy import arctan2, sqrt


import numexpr as ne


from rdp import rdp


from scipy import misc
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.spatial import distance
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, gaussian_gradient_magnitude



from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import normalize, MinMaxScaler



from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib 
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib._png import read_png
from matplotlib.cbook import get_sample_data
import matplotlib.image as mpimg



from mayavi import mlab
from mayavi.core.ui.mayavi_scene import MayaviScene


import cv2



from skimage import color
from skimage import transform
from skimage.measure import regionprops, label
from skimage.morphology import watershed, convex_hull_image
from skimage.color import label2rgb
from skimage.util import invert
from skimage.feature import peak_local_max
from skimage.morphology import watershed



from openpyxl import load_workbook
from openpyxl import Workbook



from plyfile import PlyData, PlyElement

'''
from frame_interp import interpolate_frame
import morphsnakes
from detectors import Detectors
from tracker import Tracker
from kalman_filter import KalmanFilter
'''

'''
sudo pip install 
rdp
scikit-learn
mayavi
opencv-python
opencv-contrib-pythonb
scikit-image
openpyxl
plyfile
'''

'''
sudo apt-get update
sudo apt-get install 
python-numpy 
python-scipy 
python-matplotlib 
ipython 
ipython-notebook 
python-pandas 
python-sympy 
python-nose
python-numexpr
build-essential 
cmake 
git 
libgtk2.0-dev 
pkg-config 
libavcodec-dev 
libavformat-dev 
libswscale-dev 
python-dev 
python-numpy 
libtbb2 
libtbb-dev 
libjpeg-dev 
libpng-dev 
libtiff-dev 
libjasper-dev 
libdc1394-22-dev
'''

'''
pkg-config --modversion opencv 
opencv3.4.0


sudo apt-get install 
build-essential 
cmake 
git 
libgtk2.0-dev 
pkg-config 
libavcodec-dev 
libavformat-dev 
libswscale-dev 
python-dev 
python-numpy 
libtbb2 
libtbb-dev 
libjpeg-dev 
libpng-dev 
libtiff-dev 
libjasper-dev 
libdc1394-22-dev

sudo pip install opencv-python
sudo pip install opencv-contrib-python

'''
