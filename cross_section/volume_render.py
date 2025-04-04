"""
Version: 1.5

Summary: render the cross section skeleton

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 volume_render.py -p ~/example/B73_test/01/slices/


arguments:
("-p", "--path", required=True,    help="path to cross section files")

"""


import vtk
import glob
import os
import argparse


if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to cross section files")
    args = vars(ap.parse_args())
    
    # setting path to cross section image files
    file_path = args["path"]
    ext = "png"
     
    #accquire image file list (stack of images)
    filetype = '*.' + ext
    image_file_path = file_path + filetype

    #accquire image file list
    files = sorted(glob.glob(image_file_path))
    
    # Setting the file path
    filePath = vtk.vtkStringArray()

    # Sorting file to arrange in ascending order to get slices correctly
    files.sort(key = lambda x: int(''.join(filter(str.isdigit, x))))

    filePath.SetNumberOfValues(len(files))


    for i in range(0,len(files),1):
        
        filePath.SetValue(i,files[i])
        #print(files[i])


    #Source -Reader
    #reader=vtk.vtkJPEGReader()
    reader = vtk.vtkPNGReader()
    
    reader.SetFileNames(filePath)
    
    reader.SetDataSpacing(1,1,1)
    
    reader.Update()
    
    print(reader)

    #Filter --&gt; Setting the color mapper, Opacity for VolumeProperty
    colorFunc = vtk.vtkColorTransferFunction()
    
    colorFunc.AddRGBPoint(1, 1, 0.0, 0.0) # Red

    # To set different colored pores
    colorFunc.AddRGBPoint(2, 0.0, 1, 0.0) # Green
    #colorFunc.AddRGBPoint(3, 0.0, 0, 1.0) # Black
    #colorFunc.AddRGBPoint(4, 0.0, 0.0, 1) # Blue

    opacity = vtk.vtkPiecewiseFunction()
    # opacity.AddPoint(1, 1, 0.0, 0.0)
    # opacity.AddPoint(2, 0.0, 0.0, 0.0)

    # The previous two classes stored properties and we want to apply
    # these properties to the volume we want to render,
    # we have to store them in a class that stores volume properties.
    volumeProperty = vtk.vtkVolumeProperty()
    # set the color for volumes
    volumeProperty.SetColor(colorFunc)
    # To add black as background of Volume
    volumeProperty.SetScalarOpacity(opacity)
    
    volumeProperty.SetInterpolationTypeToLinear()
    
    volumeProperty.SetIndependentComponents(2)

    #Ray cast function know how to render the data
    volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
    #volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
    #volumeMapper = vtk.vtkUnstructuredGridVolumeRayCastMapper()

    volumeMapper.SetInputConnection(reader.GetOutputPort())
    
    volumeMapper.SetBlendModeToMaximumIntensity()

    # Different modes are available in vtk for Blend mode functions
    #volumeMapper.SetBlendModeToAverageIntensity()
    #volumeMapper.SetBlendModeToMinimumIntensity()
    #volumeMapper.SetBlendModeToComposite()
    #volumeMapper.SetBlendModeToAdditive()

    volume = vtk.vtkVolume()
    
    volume.SetMapper(volumeMapper)
    
    volume.SetProperty(volumeProperty)

    ren = vtk.vtkRenderer()
    
    ren.AddVolume(volume)
    
    #No need to set by default it is black
    ren.SetBackground(0, 0, 0)

    renWin = vtk.vtkRenderWindow()
    
    renWin.AddRenderer(ren)
    
    renWin.SetSize(900, 900)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renWin)

    interactor.Initialize()
    
    renWin.Render()
    
    interactor.Start()



