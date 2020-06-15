"""
Version: 1.0

Summary: render engine for cross section of point cloud data

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

from pointcloud.render.pointcloudrenderengine import PointCloudRenderEngine

"""

from tkinter import *
from pointcloud.filedataextractor import FileDataExtractor 
from pointcloud.pointmap import PointMap
from render.renderengine import RenderEngine
from pointcloud.graphic.pointcloudgraphic import PointCloudGraphic
from graphic.circle import Circle

class PointCloudRenderEngine(RenderEngine):
    
    DEFAULT_PADDING = 0
    POINT_RADIUS = 1.0
    CROSS_SECTION_THICKNESS = 1
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"

    def __init__(self, modelFileName, destinationFile, perspective, csThickness, width = None, height = None):
        
        super().__init__(width, height)
        
        self.destinationFile = destinationFile
        
        self.perspective = perspective
        
        self.csThickness = csThickness

        pointMap = (FileDataExtractor(modelFileName, self.perspective)).extractPointCloud()
        
        self.pointCloudGraphic = PointCloudGraphic(pointMap, self.csThickness, self.destinationFile, self.width, self.height)
        
        self.xCenterShift = self.canvasCenterX - self.pointCloudGraphic.centerX
        
        self.yCenterShift = self.canvasCenterY - self.pointCloudGraphic.centerY
        
    
    def render(self):
        
        self.drawCrossSections()
        
        #super().render()

    def drawCrossSections(self):
        
        for i in range(0, self.pointCloudGraphic.numCrossSections):
            
            self.pointCloudGraphic.drawCrossSection(self.xCenterShift, self.yCenterShift)

        #self.pointCloudGraphic.drawAllCrossSections(self.__generateCanvas(), self.xCenterShift, self.yCenterShift)


    def __generateCanvas(self):
        
        tk = Tk()
        
        canvas = Canvas(tk, width=self.width, height=self.height)
        
        canvas.pack()
        
        return canvas

