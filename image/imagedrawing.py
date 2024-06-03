"""
Version: 1.0

Summary: draw image using pillow library and save image as jpg file

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

from image.imagedrawing import ImageDrawing

"""

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
from graphic.circle import Circle


class ImageDrawing:
    
    FILE_EXTENSION = "PNG"
    
    DEFAULT_FILL_COLOR_BLACK = (0, 0, 0)
    
    DEFAULT_BACKGROUND_COLOR_WHITE = (255, 255, 255)

    def __init__(self, destination, width, height):
        
        #self.image = Image.new("RGB", (width, height), self.DEFAULT_BACKGROUND_COLOR_WHITE)
        
        self.image = Image.new("RGB", (width, height), self.DEFAULT_FILL_COLOR_BLACK)
        
        self.draw = ImageDraw.Draw(self.image)
        
        self.destination = destination

    def saveImage(self, fileName):
        
        fullFileName = str(self.destination) + str(fileName)
        
        self.image = self.image.convert('L')
        
        self.image.save(fullFileName, self.FILE_EXTENSION)

    def addPoint(self, circle: Circle):
        
        #self.draw.ellipse((circle.x0, circle.y0, circle.x1, circle.y1), self.DEFAULT_FILL_COLOR_BLACK)
        
        self.draw.ellipse((circle.x0, circle.y0, circle.x1, circle.y1), self.DEFAULT_BACKGROUND_COLOR_WHITE)
