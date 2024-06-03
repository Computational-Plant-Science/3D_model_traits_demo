"""
Version: 1.0

Summary: extract point cloud data with specified perspective

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

from pointcloud.pointmap import PointMap

"""

import math
from pointcloud.perspective import Perspective

class PointMap:
    
    def __init__(self, perspective):
        
        self.perspective = Perspective(perspective)
        
        self.map = {}
        
        self.minComps = {}
        
        self.maxComps = {}
    
    def addPoint(self, pointTuple):
        
        perspectiveKey = int(self.truncate(float(pointTuple[self.perspective.getIndex()]), 0));
        
        complements = self.perspective.getComplementsFromTuple(pointTuple)

        if perspectiveKey not in self.map:
            
            self.map[perspectiveKey] = []

        self.map[perspectiveKey].append(complements)
        
        self.__checkMinMaxComps(complements)

    def __checkMinMaxComps(self, comps):
        
        compKeys = self.perspective.getComplementValues()

        if (compKeys[0] not in self.minComps or self.minComps[compKeys[0]] > comps[compKeys[0]]):
            self.minComps[compKeys[0]] = comps[compKeys[0]]
            
        if (compKeys[0] not in self.maxComps or self.maxComps[compKeys[0]] < comps[compKeys[0]]):
            self.maxComps[compKeys[0]] = comps[compKeys[0]]
            
        if (compKeys[1] not in self.minComps or self.minComps[compKeys[1]] > comps[compKeys[1]]):
            self.minComps[compKeys[1]] = comps[compKeys[1]]
            
        if (compKeys[1] not in self.maxComps or self.maxComps[compKeys[1]] < comps[compKeys[1]]):
            self.maxComps[compKeys[1]] = comps[compKeys[1]]
            

    def computeMinMaxKeys(self):
        
        self.minKey = min(self.map, key=float)
        
        self.maxKey = max(self.map, key=float)

    def containsKey(self, key):
        
        return key in self.map
        

    def getOrEmpty(self, key):
        
        if key in self.map:
            
            return self.map[key]
            
        return []

    def getMinKey(self):
        
        return self.minKey
    
    def getMaxKey(self):
        
        return self.maxKey
    
    def truncate(self, number, digits) -> float:
        
        stepper = pow(10.0, digits)
        
        return math.trunc(stepper * number) / stepper
