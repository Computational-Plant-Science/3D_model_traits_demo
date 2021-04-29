# 3D_model_traits_measurement

Function: Extract gemetrical traits of 3D model 

Author            : Suxing Liu

Date created      : 04/04/2018

Date last modified: 04/25/2020

Python Version    : 3.7


Example of structure vs. 3D root model
![Optional Text](../master/media/image3.png) ![Optional Text](../master/media/image2.gif)

Example of connecting disconnected root path
![Optional Text](../master/media/image7.png) ![Optional Text](../master/media/image8.png)


usage: 

python pipeline.py -p /$path_to_your_3D_model/ -m 3D_model_name.ply

Singularity test:

sudo singularity build --writable model-scan.img Singularity

singularity exec model-scan.img python /opt/code/pipeline.py -p /$path_to_your_3D_model/ -m surface.ply

singularity exec shub://lsx1980/3D_model_traits_measurement python /opt/code/pipeline.py -p /$path_to_your_3D_model/ -m surface.ply

- Pre-requisite:  
    - Python3.7  
    - Numpy  
    - SciPy  
    - Opencv 3.0 for Python - [Installation](http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/)
    

Visualization requirement:

  pip3 install numba \
                imagesize \
                progressbar2 \
                mayavi \
                PyQt5 \
                networkx
  
  python3 graph_compute.py -p /&path/active_component/
  
