FROM ubuntu:22.04

LABEL maintainer='Suxing Liu, Wes Bonelli'

COPY ./ /opt/code
WORKDIR /opt/code

RUN apt-get update

RUN apt-get -y dist-upgrade

RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install -y \
    build-essential \
    aptitude \
    libglu1-mesa-dev \
    mesa-common-dev \
    libboost-all-dev \
    python3-setuptools \
    python3-pip \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    cmake-gui \
    freeglut3-dev \
    freeglut3 \
    libopengl0 -y \
    mesa-utils \
    software-properties-common \
    libcairo2 \
    python3-cairo \
    nano \
    xorg-dev \
    gnupg2 \
    ca-certificates 

RUN cd /opt/code/compiled/ && rm -rf Release && mkdir Release && cd Release && cmake -DCMAKE_BUILD_TYPE=Release ..   && make

RUN pip3 install numpy==1.22.4 \
    contourpy==1.2.0 \
    matplotlib==3.8.0 \
    Pillow \
    scipy==1.8.1 \
    scikit-image \
    scikit-learn \
    scikit-build \
    plyfile \
    open3d \
    openpyxl \
    imutils \
    findpeaks 
    


RUN echo -n 'deb [arch=amd64 ] https://downloads.skewed.de/apt jammy main' >> /etc/apt/sources.list

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key 612DEFB798507F25 

RUN apt-get -y update 

RUN apt-get install python3-graph-tool -y


