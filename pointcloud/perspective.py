"""
Version: 1.0

Summary: calculte slicing direction of point cloud 

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

from pointcloud.perspective import Perspective

"""

class Perspective:
    
    ALL_PERSPECTIVES = ["X", "Y", "Z"]

    def __init__(self, perspective):
        
        perspective.strip()
        
        self.perspective = perspective
        
        self.perspectiveIndex = self.ALL_PERSPECTIVES.index(self.perspective)
        
        self.complementValues = self.ALL_PERSPECTIVES.copy()
        
        self.complementValues.remove(self.perspective)

    def getComplementValues(self):
        
        return self.complementValues

    def getComplementsFromTuple(self, tupleVal):
        
        complements = {}
        
        complements[self.complementValues[0]] = float(tupleVal[self.ALL_PERSPECTIVES.index(self.complementValues[0])])
        
        complements[self.complementValues[1]] = float(tupleVal[self.ALL_PERSPECTIVES.index(self.complementValues[1])])

        return complements

    def getIndex(self):
        
        return self.perspectiveIndex
    
