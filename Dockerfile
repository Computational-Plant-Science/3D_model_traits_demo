FROM ubuntu:20.04

LABEL maintainer='Suxing Liu, Wes Bonelli'

COPY ./ /opt/3D_model_traits_demo

RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt install -y \
    build-essential \
    python3-setuptools \
    python3-pip \
    python3 \
    python3-tk \
    python3-numexpr \
    python3-pil.imagetk \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    cmake-gui \
    libglu1-mesa-dev \
    freeglut3-dev \
    freeglut3 \
    libopengl0 -y \
    mesa-common-dev \
    mesa-utils \
    software-properties-common \
    libcairo2 \
    python-cairo \
    nano
    

ENV PYTHONPATH=$PYTHONPATH:/opt/3D_model_traits_demo/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/3D_model_traits_demo/

# ENV LC_ALL=C.UTF-8
# ENV LANG=C.UTF-8

RUN pip3 install --upgrade pip && \
    pip3 install numpy \
    Pillow \
    rdp \
    scipy \
    scikit-image \
    scikit-learn \
    scikit-build \
    matplotlib \
    mahotas \
    plyfile \
    psutil \
    cairosvg \
    certifi \
    pandas \
    pytest \
    coverage \
    coveralls \
    open3d \
    opencv-python-headless \
    openpyxl \
    click \
    PyYAML \
    imutils 
    
RUN pip3 install --upgrade numpy

RUN apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25

RUN add-apt-repository 'deb [ arch=amd64 ] https://downloads.skewed.de/apt focal main'

RUN apt update 

RUN apt install python3-graph-tool -y
