import csv
import multiprocessing
import os
import time
from contextlib import closing
from glob import glob
from math import ceil
from os.path import join
from pathlib import Path

import click
from openpyxl import load_workbook

from crossection_measure import root_area_label
from crossection_scan import CDF_visualization, parallel_root_system_trait
from options import DIRT3DOptions
from pointcloud.render.pointcloudrenderengine import PointCloudRenderEngine


def cross_section_scan(options: DIRT3DOptions):
    images = sorted(glob(join(options.output_directory, '*.jpg')))
    Path(join(options.output_directory, 'active_component')).mkdir(exist_ok=True)
    Path(join(options.output_directory, 'label')).mkdir(exist_ok=True)

    # Run this with a pool of avaliable agents having a chunksize of 3 until finished
    # run image labeling fucntion to accquire segmentation for each cross section image
    chunk_size = 3
    with closing(multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2)) as pool:
        result = pool.map(root_area_label, images, chunk_size)
        pool.terminate()

    list_thresh = sorted(CDF_visualization(result))

    # compute plateau in curve
    dis_array = [j - i for i, j in zip(list_thresh[:-1], list_thresh[1:])]

    # get index of plateau location
    index = [i for i in range(len(dis_array)) if dis_array[i] <= 1.3]
    dis_index = [j - i for i, j in zip(index[:-1], index[1:])]

    for idx, value in enumerate(dis_index):
        if idx < len(index) - 2:
            if value == dis_index[idx + 1]:
                index.remove(index[idx + 1])

    reverse_index = sorted(index, reverse=True)
    count = len(index)

    # compute whorl location
    whorl_dis = []
    whorl_loc = []

    for idx, value in enumerate(reverse_index, start=1):
        loc_value = int(len(images) - list_thresh[value + 1])
        whorl_loc.append(loc_value)

    # compute whorl distance
    whorl_dis_array = [j - i for i, j in zip(whorl_loc[:-1], whorl_loc[1:])]
    whorl_loc.extend([0, len(images)])
    whorl_loc = list(dict.fromkeys(whorl_loc))
    whorl_loc_ex = sorted(whorl_loc)

    print("list_thresh : {0} \n".format(str(list_thresh)))
    print("dis_array : {0} \n".format(str(dis_array)))
    print("index : {0} \n".format(str(index)))
    print("dis_index : {0} \n".format(str(dis_index)))
    print("reverse_index : {0} \n".format(str(reverse_index)))
    print("whorl_loc : {0} \n".format(str(whorl_loc)))
    print("whorl_dis_array : {0} \n".format(str(whorl_dis_array)))
    print("whorl_loc_ex : {0} \n".format(str(whorl_loc_ex)))

    # divide the image list into n chunks
    list_part = []
    for idx, val in enumerate(whorl_loc_ex):
        print(idx, val)
        if idx < len(whorl_loc_ex) - 1:
            sublist = images[val:whorl_loc_ex[idx + 1]]
            list_part.append(sublist)

    print(len(list_part))

    for i in range(0, len(list_part)):
        if i == 0:
            pattern_id = 1
        elif i == 1:
            pattern_id = 2
        else:
            pattern_id = 3

        print(pattern_id)
        parallel_root_system_trait(list_part[i])

    with open(join(options.output_directory, 'system_traits.csv'), 'w', newline="") as f:
        csv_writer = csv.writer(f)
        for r in load_workbook(join(options.output_directory, 'system_traits.xlsx')).active.rows:
            csv_writer.writerow([cell.value for cell in r])


@click.command()
@click.argument('input_file')
@click.option('-o', '--output_directory', required=False, type=str, default='')
@click.option("-i", "--interval", required=False, default='1', type=int, help="Intervals along sweeping plane")
@click.option("-de", "--direction", required=False, type=click.Choice(['X', 'Y', 'Z'], case_sensitive=False), default='X', help="Direction of sweeping plane (X, Y, or Z)")
@click.option("-r", "--reverse", required = False, default=False, type =bool, help = "Reverse model (top down)")
@click.option('-f', '--frames', required=False, type=int, default=2, help='Number of new frames')
@click.option("-th", "--threshold", required=False, default='2.35', type=float, help="Threshold to remove outliers")
@click.option('-d', '--distance_threshold', required=False, type=int, default=10)
@click.option("-dt", "--distance_tracking", required=False, type=float, default=50.5)
@click.option("-dr", "--distance_ratio", required=False, type=float, default=4.8)
@click.option('-mfs', '--max_skipped_frames', required=False, type=int, default=15)
@click.option('-mtl', '--max_trace_length', required=False, type=int, default=15)
@click.option('-rmin', '--min_radius', required=False, type=int, default=1)
@click.option('-rmax', '--max_radius', required=False, type=int, default=100)
@click.option("-ma", "--min_angle", required=False, type=float, default=0.1)
def cli(
        input_file,
        output_directory,
        interval,
        direction,
        reverse,
        frames,
        threshold,
        distance_threshold,
        distance_tracking,
        distance_ratio,
        max_skipped_frames,
        max_trace_length,
        min_radius,
        max_radius,
        min_angle):
    start = time.time()
    options = DIRT3DOptions(
        input_file=input_file,
        output_directory=output_directory,
        interval=interval,
        direction=direction,
        reverse=reverse,
        frames=frames,
        threshold=threshold,
        distance_threshold=distance_threshold,
        distance_tracking=distance_tracking,
        distance_ratio=distance_ratio,
        max_skipped_frames=max_skipped_frames,
        max_trace_length=max_trace_length,
        min_radius=min_radius,
        max_radius=max_radius,
        min_angle=min_angle)

    print("Scanning 3D point cloud")
    # model_path = join(options.output_directory, 'converted.ply')
    rendering = PointCloudRenderEngine(options.input_file, options.output_directory, options.direction, options.interval)
    rendering.render()

    print("Scanning cross sections")
    cross_section_scan(options)

    duration = ceil((time.time() - start))
    print(f"Finished in {duration} seconds.")


if __name__ == '__main__':
    cli()
