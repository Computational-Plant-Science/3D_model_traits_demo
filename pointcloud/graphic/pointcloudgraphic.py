"""
Version: 1.0

Summary: render current cross section of the point cloud.

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

from pointcloud.graphic.pointcloudgraphic import PointCloudGraphic

"""


from pointcloud.pointmap import PointMap
from graphic.circle import Circle
from tkinter import *
from image.imagedrawing import ImageDrawing

class PointCloudGraphic:
    
    PORTRAIT = "portrait"
    
    LANDSCAPE = "landscape"
    
    POINT_RADIUS = 1
    
    FILE_NAME_PREFIX = "cross-section-"
    
    SCALE_FACTOR = 1

    def __init__(self, pointMap, csThickness, destinationFile, imageWidth, imageHeight, xOffset = None, yOffset = None):
        
        if xOffset is None:
            
            self.xOffset = 0
        
        if yOffset is None:
            
            self.yOffset = 0

        self.pointMap = pointMap
        
        self.csThickness = csThickness
        
        self.csIteratorIndex = 0
        
        self.destinationFile = destinationFile
        
        self.imageWidth = imageWidth
        
        self.imageHeight = imageHeight

        self.__computeGraphicProperties()
        
        self.__precomputeCrossSections()

    def __computeGraphicProperties(self):
        
        self.__computeAxisLabels()
        self.__computeWidth()
        self.__computeHeight()
        self.__computeScale()
        self.__computeScaledBounds()
        self.__computeScaledCenter()
        self.__computeScaledHeight()
        self.__computeScaledWidth()
        self.__computeOrientation()

    def __computeWidth(self):
        
        self.width = self.pointMap.maxComps[self.axisLabels[0]] - self.pointMap.minComps[self.axisLabels[0]]
    
    def __computeHeight(self):
        
        self.height = self.pointMap.maxComps[self.axisLabels[1]] - self.pointMap.minComps[self.axisLabels[1]]

    def __computeScale(self):
        
        self.scale = self.imageWidth * self.SCALE_FACTOR / self.width
        
        if (self.height * self.scale) > self.imageHeight:
            
            self.scale = self.imageHeight * self.SCALE_FACTOR / self.height

    def __computeOrientation(self):
        self.orientation = self.LANDSCAPE if self.width > self.height else self.PORTRAIT

    def __computeAxisLabels(self):
        self.axisLabels = self.pointMap.perspective.getComplementValues()

    def __computeScaledWidth(self):
        self.width = self.rightBound - self.leftBound 

    def __computeScaledHeight(self):
        self.height = self.bottomBound - self.topBound

    def __computeScaledBounds(self):
        
        self.__computeRightBound()
        
        self.__computeBottomBound()
        
        self.__computeTopBound()
        
        self.__computeLeftBound()

    def __computeRightBound(self):
        
        self.rightBound = self.pointMap.maxComps[self.axisLabels[0]] * self.scale

    def __computeBottomBound(self):
        
        self.bottomBound = self.pointMap.maxComps[self.axisLabels[1]] * self.scale

    def __computeLeftBound(self):
        
        self.leftBound = self.pointMap.minComps[self.axisLabels[0]] * self.scale

    def __computeTopBound(self):
        
        self.topBound = self.pointMap.minComps[self.axisLabels[1]] * self.scale

    def __computeScaledCenter(self):
        
        self.centerX = (self.rightBound + self.leftBound)/2
        
        self.centerY = (self.bottomBound + self.topBound)/2

    def __precomputeCrossSections(self):
        
        minKey = self.pointMap.getMinKey()
        
        maxKey = self.pointMap.getMaxKey()
        
        self.crossSections = []
        
        self.imageDrawingList = []
        
        #print("minkey".format(minKey))
        #print("maxKey".format(maxKey))

        while minKey <= maxKey:
            
            startLayer = minKey
            
            endLayer = minKey + self.csThickness
            
            endLayer = maxKey if endLayer > maxKey else (minKey + self.csThickness)

            self.crossSections.append(self.__aggregateLayers(startLayer, endLayer))
            
            self.imageDrawingList.append(ImageDrawing(self.destinationFile, self.imageWidth, self.imageHeight))
            
            minKey = endLayer + 1

        
        self.numCrossSections = len(self.crossSections)
        
        #print(self.numCrossSections)
        
        print("Generating {0} cross section of 3D model...\n".format(self.numCrossSections))

    def __aggregateLayers(self, startLayer, endLayer):
        
        aggregatedLayers = []
        
        non_empty_index = 0
        
        for i in range(startLayer, endLayer + 1):
            
            if not self.pointMap.getOrEmpty(i):
                
                j = i
                while j > 0:              
                   j = j - 1
                   if self.pointMap.getOrEmpty(j):
                      break
                #print("non_empty_index value {}:".format(j))
                
                aggregatedLayers.extend(self.pointMap.getOrEmpty(j))
                
            else:
                
                aggregatedLayers.extend(self.pointMap.getOrEmpty(i))
                
                #print("current i = {}\n".format(i))
        
        return aggregatedLayers

    # Draws current cross section of the point cloud.  Automatically advances into
    # the next one after printing the current one.
    def drawCrossSection(self, xTranslate = 0, yTranslate = 0, targetCanvas = None):
        
        csCurrent = self.crossSections[self.csIteratorIndex]
        
        imageDrawingCurrent = self.imageDrawingList[self.csIteratorIndex]
        
        for i in range(0, len(csCurrent)):
            #self.__printPoint(targetCanvas, csCurrent[i][self.axisLabels[0]], csCurrent[i][self.axisLabels[1]], xTranslate, yTranslate)
            self.__savePointToImage(imageDrawingCurrent, csCurrent[i][self.axisLabels[0]], csCurrent[i][self.axisLabels[1]], xTranslate, yTranslate)

        imageDrawingCurrent.saveImage(self.__generateCrossSectionFileName())
        
        self.csIteratorIndex = self.csIteratorIndex + 1

    def __printPoint(self, targetCanvas, x, y, xTranslate = 0, yTranslate = 0):
        
        xPos = x * self.scale + xTranslate + self.xOffset
        
        yPos = y * self.scale + yTranslate + self.yOffset
        
        (Circle(xPos, yPos, self.POINT_RADIUS)).draw(targetCanvas)
    
    def __savePointToImage(self, imageDrawing, x, y, xTranslate = 0, yTranslate = 0):
        
        xPos = x * self.scale + xTranslate + self.xOffset
        
        yPos = y * self.scale + yTranslate + self.yOffset
        
        imageDrawing.addPoint((Circle(xPos, yPos, self.POINT_RADIUS)))

    def drawAllCrossSections(self, targetCanvas, xTranslate = 0, yTranslate = 0):
        
        for i in range(0, len(self.crossSections)):
            
            csCurrent = self.crossSections[i]
            
            for j in range(0, len(csCurrent)):
                
                self.__printPoint(targetCanvas, csCurrent[j][self.axisLabels[0]], csCurrent[j][self.axisLabels[1]], xTranslate, yTranslate)


    def __generateCrossSectionFileName(self):
        
        #return self.FILE_NAME_PREFIX + str(self.csIteratorIndex)
        
        return str('{:04}'.format(self.csIteratorIndex)) + ".png"
        


