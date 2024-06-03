"""
Version: 1.5

Summary: Class to detect objects in video frame

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

from detectors import Detectors
 

"""


# Import python libraries
import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.measure import regionprops



class Detectors(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """
    def __init__(self):
        """Initialize variables used by Detectors class
        Args:
            None
        Return:
            None
        """
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def auto_canny(self, image_gray, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image_gray)
     
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image_gray, lower, upper)
     
        # return the edged image
        return edged
    
    
    def Detect(self, frame, ID, radius_min, radius_max):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """
        
        #perform pyramid mean shift filtering to aid the thresholding step
        #shifted = cv2.pyrMeanShiftFiltering(frame, 5, 5)

        # Convert BGR to GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Perform Background Subtraction
        fgmask = self.fgbg.apply(gray)

        # Detect edges
        #edges = cv2.Canny(fgmask, 50, 190, 3)
        
        edges = self.auto_canny(frame, sigma = 0.33)
        
        # Retain only edges within the threshold
        #ret, thresh = cv2.threshold(edges, 127, 255, 0)
        ret, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
       
        centers = []  # vector of object centroids in a frame
        
        # Find contours
        #_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers = []  # vector of object centroids in a frame

        #blob_radius_thresh = 10        
        radius_min = 1
        radius_max = 100
        
        radius_rec = []
        
        # Find centroid for each valid contours
        for cnt in contours:
            try:
                # Calculate and draw circle
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                centeroid = (int(x), int(y))
                radius = int(radius)
                
                if ((radius > radius_min) and (radius < radius_max) ):
                    cv2.circle(frame, centeroid, radius, (0, 255, 0), 2)
                    b = np.array([[x], [y]])
                    centers.append(np.round(b))
                    radius_rec.append(np.round(radius))
            except ZeroDivisionError:
                pass

        
        # show contours of tracking objects
        #cv2.imshow('Track Bugs', frame)
        
        return centers, radius_rec


