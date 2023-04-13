FROM ubuntu:20.04

LABEL maintainer='Suxing Liu, Wes Bonelli'

COPY ./ /opt/3D_model_traits_demo
WORKDIR /opt/3D_model_traits_demo

RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install -y \
    build-essential \
    python3-setuptools \
    cmake-gui \
    xorg-dev \
    libglu1-mesa-dev \ 
    mesa-utils \
    libboost-all-dev \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libcairo2 \
    python3-pip \
    python3 \
    python3-tk \
    python3-pil.imagetk \
    python3-cairo \
    freeglut3-dev \
    freeglut3 \
    libopengl0 -y \
    mesa-common-dev \
    software-properties-common \
    nano


RUN pip3 install --upgrade pip && \
    pip3 install numpy \
    Pillow \
    rdp \
    scipy \
    scikit-image==0.19.3 \
    scikit-learn \
    scikit-build \
    matplotlib \
    mahotas \
    numexpr \
    plyfile \
    psutil \
    cairosvg \
    certifi \
    pandas \
    pytest \
    coverage \
    coveralls \
    open3d \
    opencv-python \
    openpyxl \
    click \
    PyYAML \
    imutils 
    

RUN cd /opt/3D_model_traits_demo/AdTree/ && mkdir /opt/3D_model_traits_demo/AdTree/Release && cd /opt/3D_model_traits_demo/AdTree/Release  && cmake -DCMAKE_BUILD_TYPE=Release .. && make


RUN pip3 install --upgrade numpy

RUN apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25

RUN add-apt-repository 'deb [ arch=amd64 ] https://downloads.skewed.de/apt focal main'

RUN apt update 

RUN apt install python3-graph-tool -y


RUN chmod +x /opt/3D_model_traits_demo/shim.sh 


ENV PYTHONPATH=$PYTHONPATH:/opt/3D_model_traits_demo/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/3D_model_traits_demo/
