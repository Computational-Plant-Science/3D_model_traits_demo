#!/usr/bin/env python

# This simple example shows how to do basic rendering and pipeline
# creation.
import numpy as np 
import argparse
import glob
import os
import open3d as o3d

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)


def read_model(filename_pcloud):
    '''
    colors = vtkNamedColors()
    # Set the background color.
    bkg = map(lambda x: x / 255.0, [26, 51, 102, 255])
    colors.SetColor("BkgColor", *bkg)

    # This creates a polygonal cylinder model with eight circumferential
    # facets.
    cylinder = vtkCylinderSource()
    cylinder.SetResolution(8)

    # The mapper is responsible for pushing the geometry into the graphics
    # library. It may also do color mapping, if scalars or other
    # attributes are defined.
    cylinderMapper = vtkPolyDataMapper()
    cylinderMapper.SetInputConnection(cylinder.GetOutputPort())

    # The actor is a grouping mechanism: besides the geometry (mapper), it
    # also has a property, transformation matrix, and/or texture map.
    # Here we set its color and rotate it -22.5 degrees.
    cylinderActor = vtkActor()
    cylinderActor.SetMapper(cylinderMapper)
    cylinderActor.GetProperty().SetColor(colors.GetColor3d("Tomato"))
    cylinderActor.RotateX(30.0)
    cylinderActor.RotateY(-45.0)

    # Create the graphics structure. The renderer renders into the render
    # window. The render window interactor captures mouse events and will
    # perform appropriate camera or actor manipulation depending on the
    # nature of the events.
    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Add the actors to the renderer, set the background and size
    ren.AddActor(cylinderActor)
    ren.SetBackground(colors.GetColor3d("BkgColor"))
    renWin.SetSize(300, 300)
    renWin.SetWindowName('CylinderExample')

    # This allows the interactor to initalize itself. It has to be
    # called before an event loop.
    iren.Initialize()

    # We'll zoom in a little by accessing the camera and invoking a "Zoom"
    # method on it.
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1.5)
    renWin.Render()

    # Start the event loop.
    iren.Start()
    
    
    
    from numpy import pi, sin, cos, mgrid
    dphi, dtheta = pi/250.0, pi/250.0
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
    r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
    x = r*sin(phi)*cos(theta)
    y = r*cos(phi)
    z = r*sin(phi)*sin(theta)

    # View it.
    from mayavi import mlab
    s = mlab.mesh(x, y, z)
    mlab.show()
    '''
    
    

    print("Loading 3D point cloud {}...\n".format(filename_pcloud))

    model_pcloud_name_base = os.path.splitext(model_pcloud)[0]


    pcd = o3d.io.read_point_cloud(model_pcloud)

    Data_array_pcloud = np.asarray(pcd.points)

    print(Data_array_pcloud.shape)
    
    
    
    if pcd.has_colors():

        print("Render colored point cloud\n")

        pcd_color = np.asarray(pcd.colors)

        if len(pcd_color) > 0: 
        
            pcd_color = np.rint(pcd_color * 255.0)

    #pcd_color = tuple(map(tuple, pcd_color))
    else:

        print("Generate random color\n")

        pcd_color = np.random.randint(256, size = (len(Data_array_pcloud),3))
    
    
    #o3d.visualization.draw_geometries([pcd])
    
    from numpy import pi, sin, cos, mgrid
    dphi, dtheta = pi/250.0, pi/250.0
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
    r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
    x = r*sin(phi)*cos(theta)
    y = r*cos(phi)
    z = r*sin(phi)*sin(theta)

    # View it.
    from mayavi import mlab
    from tvtk.api import tvtk
    
    mlab.figure("Structure_graph", size = (800, 800), bgcolor = (0, 0, 0))
        
    mlab.clf()
    
    s = mlab.mesh(x, y, z)
    
    mlab.show()
    



if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m2", "--model_pcloud", required = False, default = None, help = "point cloud model file name, same path with ply model")
    args = vars(ap.parse_args())
    
    # setting path to model file 
    current_path = args["path"]


    if args["model_pcloud"] is None:
        filename_pcloud = None
    else:
        filename_pcloud = args["model_pcloud"]
    
    model_pcloud = current_path + filename_pcloud
    
    read_model(model_pcloud)
    
    
