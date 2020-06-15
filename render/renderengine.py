"""
Version: 1.0

Summary: engine for render image on canvas 

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

from render.renderengine import RenderEngine

"""

from tkinter import *
from pointcloud.filedataextractor import FileDataExtractor 
from pointcloud.pointmap import PointMap

class RenderEngine:
    
    DEFAULT_CANVAS_WIDTH = 600
    
    DEFAULT_CANVAS_HEIGHT = 600

    def __init__(self, width = None, height = None):
        
        if width is None:
            
            self.width = self.DEFAULT_CANVAS_WIDTH
            
        if height is None:
            
            self.height = self.DEFAULT_CANVAS_HEIGHT
            
        self.canvasCenterX = self.width/2
        
        self.canvasCenterY = self.height/2

    def render(self):
        
        mainloop()
