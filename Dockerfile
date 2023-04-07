FROM ubuntu:22.04

LABEL maintainer='Suxing Liu, Wes Bonelli'

COPY ./ /opt/3D_model_traits_demo

RUN apt-get update && apt-get upgrade -y



RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install -y \
    build-essential \
    python3-dev \
    python3-setuptools \
    python3-pip \
    python3 \
    python3-tk \
    python3-numexpr \
    python3-pil.imagetk \
    pkg-config \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libcairo2-dev \
    cmake-gui \
    libglu1-mesa-dev \
    freeglut3-dev \
    freeglut3 \
    libopengl0 -y \
    mesa-common-dev \
    mesa-utils \
    software-properties-common \
    nano 
    

ENV PYTHONPATH=$PYTHONPATH:/opt/3D_model_traits_demo/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/3D_model_traits_demo/



RUN python3 -m pip install --upgrade pip

RUN pip3 install numpy \
    Pillow \
    rdp \
    scipy \
    scikit-image==0.19.3 \
    scikit-learn \
    scikit-build \
    matplotlib \
    mahotas \
    plyfile \
    psutil \
    pycairo \
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



RUN echo "deb https://deb.debian.org/debian experimental main" | tee -a /etc/apt/sources.list 



#RUN echo "deb-src http://downloads.skewed.de/apt/stretch stretch main" | tee -a /etc/apt/sources.list

RUN apt-key adv --keyserver pgp.skewed.de --recv-key 648ACFD622F3D138 

RUN apt-key adv --keyserver pgp.skewed.de --recv-key 0E98404D386FA1D9


#RUN add-apt-repository 'deb [ arch=amd64 ] https://downloads.skewed.de/apt focal main'

RUN apt-get update 

RUN apt -t experimental install python3-graph-tool



RUN chmod +x /opt/3D_model_traits_demo/shim.sh 

