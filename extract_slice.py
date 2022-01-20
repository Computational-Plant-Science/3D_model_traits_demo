'''
Name: extract_slice.py

Version: 1.0

Summary:  extract the intersection plane/cross section from 3D model in obj format
    
Author: suxing liu

Author-email: suxingliu@gmail.com

Created: 2018-04-29

USAGE:

python3 extract_slice.py -p ~/path/model.obj -n 10


'''
# !/usr/bin/env python3

import argparse
import io
import os
import shutil
import struct
import sys
from os.path import join
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from cairosvg import svg2png
from sklearn import preprocessing

import extract_intersection


# generate foloder to store the output results
def mkdir(path):
    # import module
    import os

    # remove space at the beginning
    path = path.strip()
    # remove slash at the end
    path = path.rstrip("\\")

    # path exist?   # True  # False
    isExists = os.path.exists(path)

    # process
    if not isExists:
        # construct the path and folder
        # print path + ' folder constructed!'
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        # print path+' path exists!'
        shutil.rmtree(path)
        os.makedirs(path)
        return False


def OBJ2STL(model_path, output_directory):
    stem = Path(model_path).stem
    name = Path(model_path).name
    print("Converting file format for 3D point cloud model: {}".format(name))

    mesh = o3d.io.read_triangle_mesh(model_path)
    print("3D model file infomation: {}".format(mesh))

    stl_mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)
    stl_model_file = join(output_directory, stem + '.stl')
    if o3d.io.write_triangle_mesh(stl_model_file, stl_mesh):
        print("Converted 3d model was saved at: {0}".format(stl_model_file))
        return stl_mesh
    else:
        print("Model file converter failed!")
        sys.exit(1)


# get data from STL file as point coordinates
def get_data(stl_model_path):
    data_points = []

    with open(stl_model_path, "rb") as f:

        stl_bytes = f.read()

    n_triangle = (len(stl_bytes) - 84) // 50

    fmt = '<'

    for k in range(n_triangle):
        fmt += 'ffffffffffffh'

    # All of a sudden otherwise '<' does everything buggy
    data_temp = struct.unpack(fmt, stl_bytes[84:])

    for i in range(n_triangle):  # sort the data to have only the points

        data_points += data_temp[i * 13 + 3:i * 13 + 12]

    return data_points


# normalize data 
def normalize_data(data_points, data_range):
    # Normalize data
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, data_range))

    # min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1*data_range, data_range))

    point_normalized = min_max_scaler.fit_transform(np.asarray(data_points).reshape(-1, 1))

    return list(point_normalized.flatten())


# Generate svg file from slice
def save_file(data_list, width, height, slice_name, xmin, ymin, output_directory: str):
    result_file = join(output_directory, slice_name + '.png')

    print("Generating slice image: '{}'".format(slice_name))

    # svg_data = bytearray()

    header_line = '<svg width="{}" height="{}" viewBox="0 0 {} {}" >\n'.format(width + 300, height + 300, width + 2, height + 2)

    svg_data = bytearray(header_line, 'utf-8')

    for indice in range(0, len(data_list), 2):
        data = '<line x1="{}" x2="{}" y1="{}" y2="{}" stroke = "black" stroke-width="{}"/>\n'.format(data_list[indice].c_x - xmin,
                                                                                                     data_list[indice + 1].c_x - xmin,
                                                                                                     data_list[indice].c_y - ymin,
                                                                                                     data_list[indice + 1].c_y - ymin, height / 200)

        svg_data.extend(bytearray(data, 'utf-8'))

    svg_data.extend(b'</svg>')

    '''
    # save svg format
    file_svg = open(slice_name, "w")
    
    file_svg.write('<svg width="{}" height="{}" viewBox="0 0 {} {}" >\n'.format(width + 300, height + 300, width + 2, height + 2))

    for indice in range(0, len(data_list), 2):
        
        data = '<line x1="{}" x2="{}" y1="{}" y2="{}" stroke = "black" stroke-width="{}"/>\n'.format(data_list[indice].c_x-xmin, data_list[indice+1].c_x-xmin, data_list[indice].c_y-ymin, data_list[indice+1].c_y-ymin, height/200)
        
        svg_data.extend(bytearray(data, 'utf-8'))
        
        file_svg.write(data)
    
    file_svg.write('</svg>')
    '''

    # image_bytes = svg2png( url = image_file, write_to = result_file, scale = 1.0)

    png_data = svg2png(bytestring=svg_data, scale=1.0)

    pil_image = Image.open(io.BytesIO(png_data))

    # pil_image_gray = pil_image.convert('LA')

    # pil_image_gray.save(result_file)

    # print(pil_image)

    opencvImage = np.array(pil_image)

    # print(opencvImage.shape)

    # make mask of where the transparent bits are
    trans_mask = opencvImage[:, :, 3] == 0

    # replace areas of transparency with white and not transparent
    opencvImage[trans_mask] = [255, 255, 255, 255]

    # new image without alpha channel...
    opencvImage_BRG = cv2.cvtColor(opencvImage, cv2.COLOR_BGRA2BGR)

    opencvImage_BRG = ~opencvImage_BRG

    opencvImage_gray = cv2.cvtColor(opencvImage_BRG, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(result_file, opencvImage_gray)

    return opencvImage


# get data point from model files
def get_slice_data(plane_h, output_directory: str):
    slice_planes = extract_intersection.filter_data(data_points, plane_h)
    data_slice_planes = extract_intersection.intersection(slice_planes, plane_h)
    max_width = max(xmax - xmin, ymax - ymin)
    save_file(data_slice_planes, max_width, max_width,
              "slice_" + str(int(plane_h)).zfill(3) + ".svg", xmin, ymin, output_directory)

    return data_slice_planes


# compute slice data at a set of depth values
def slice_model(model_path: str, output_directory: str, n_slices: int):
    global data_points, xmin, xmax, ymin, ymax, zmin, zmax

    data_points_model = get_data(model_path)
    data_points = normalize_data(data_points_model, n_slices)

    (xmin, xmax, ymin, ymax, zmin, zmax) = extract_intersection.find_boundary(data_points)
    print(xmin, xmax, ymin, ymax, zmin, zmax)

    planes = np.linspace(zmin + 1, zmax - 1, n_slices - 2)
    for index, plane_h in enumerate(planes):
        # slice_planes = extract_intersection.filter_data(data_points, plane_h)

        # data_slice_planes = extract_intersection.intersection(slice_planes, plane_h)

        # save_file(data_slice_planes, max(xmax-xmin, ymax-ymin), max(xmax-xmin, ymax-ymin), save_path + "slice_" + str(index).zfill(3) + ".svg", xmin, ymin)

        data_slice_planes = get_slice_data(plane_h, output_directory)

    '''
    # get cpu number for parallel processing
    agents = psutil.cpu_count()   
    #agents = multiprocessing.cpu_count() 
    #agents = 8
    
    print("Using {0} cores to perfrom parallel processing... \n".format(int(agents)))
    
    # Create a pool of processes. By default, one is created for each CPU in the machine.
    # extract the bouding box for each image in file list
    with closing(Pool(processes = agents)) as pool:
        result_list = pool.map(get_slice_data, planes)
        pool.terminate()
    '''


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help=".obj model file path")
    ap.add_argument("-o", "--output", required=False, help="path to output directory")
    ap.add_argument('-n', '--num_slices', required=False, type=int, default=500, help='Number of slices')

    args = vars(ap.parse_args())

    # path to model file
    model_path = args["path"]
    output_directory = args['output']
    slices = int(args['num_slices'])

    # make sure model file and output directory exist
    if not Path(model_path).exists():
        print(f"Model file does not exist: {model_path}")
        sys.exit(1)

    # make sure model file and output directory exist
    if not Path(output_directory).exists():
        print(f"Output directory does not exist: {output_directory}")
        sys.exit(1)

    # if no output directory provided, use current working directory
    if output_directory is None: output_directory = os.getcwd()

    # convert .obj to .stl
    print("Converting .obj file to .stl")
    stl_mesh = OBJ2STL(model_path, output_directory)

    # extract slices from .stl model
    print("Computing slices")
    slice_model(join(output_directory, Path(model_path).stem + '.stl'), output_directory, slices)
