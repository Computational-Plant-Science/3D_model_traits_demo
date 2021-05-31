#!/bin/bash
# pipeline of colmap and visualSFM for 3D model reconstruction from images
# /images/ are input data files inside docker container


#feature extraction
python3 /opt/code/crossection_scan_ptvpy.py -p /srv/images/cross_section_scan/
    
#feature mathcing
python3 /opt/code/track_load_ori.py -p /srv/images/cross_section_scan/  -f trace_result.csv -v True 


#chmod 777 -R /srv/images/
