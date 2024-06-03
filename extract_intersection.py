'''
Name: extract_intersection.py

Version: 1.0

Summary:  extract the intersection data at given z value
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2018-04-29


'''

#!/usr/bin/env python3



class Point:

    def __init__(self, coord_x, coord_y, normal = None):
        self.c_x = coord_x
        self.c_y = coord_y
        self.nl = normal

    def __str__(self): #Useful only for testing
        return "({},{},{})".format(str(self.coord), str(self.ord), str(self.nl))


#Select slice segments
def filter_data(data, h):

    data_filter = []
    
    for i in range(2, len(data), 9): 
        
        if (data[i]-h)*(data[i+3]-h) < 0:  
            data_filter += [(Point(data[i-2], data[i-1], data[i]), Point(data[i+1], data[i+2], data[i+3]))]
            
        if (data[i]-h)*(data[i+6]-h) < 0:
            data_filter += [(Point(data[i-2], data[i-1], data[i]), Point(data[i+4], data[i+5], data[i+6]))]
            
        if (data[i+3]-h)*(data[i+6]-h) < 0:
            data_filter += [(Point(data[i+1], data[i+2], data[i+3]), Point(data[i+4], data[i+5], data[i+6]))]
            
    return data_filter



#returns the coordinates of the points of intersection of the selected segments of the plane z = h
def intersection(segment, h):

    data_final = []
    
    for elt in segment:#tuple
        
        coord_x = (elt[1].c_x - elt[0].c_x) * (h - elt[0].nl) / (elt[1].nl - elt[0].nl) + elt[0].c_x
        
        coord_y = (elt[1].c_y - elt[0].c_y) * (h - elt[0].nl) / (elt[1].nl - elt[0].nl) + elt[0].c_y
        
        data_final.append(Point(coord_x, coord_y))
        
    return data_final



#find the bounds of each coordinate
def find_boundary(data):

    xmin = min(data[i] for i in range(0, len(data), 3))
    xmax = max(data[i] for i in range(0, len(data), 3))
    
    ymin = min(data[i] for i in range(1, len(data), 3))
    ymax = max(data[i] for i in range(1, len(data), 3))
    
    zmin = min(data[i] for i in range(2, len(data), 3))
    zmax = max(data[i] for i in range(2, len(data), 3))
    
    return xmin, xmax, ymin, ymax, zmin, zmax
