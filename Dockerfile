FROM ubuntu:20.04

LABEL maintainer='Suxing Liu, Wes Bonelli'

# copy project source
COPY ./ /opt/code

# update OS and install dependencies
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

# install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install -r /opt/code/requirements.txt

# install Python graph-tool (not available via pip)
RUN apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25 && \
    add-apt-repository 'deb [ arch=amd64 ] https://downloads.skewed.de/apt focal main' && \
    apt update && \
    apt install python3-graph-tool -y

# make sure Python modules/scripts are available
ENV PYTHONPATH=$PYTHONPATH:/opt/code/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/code/

# is this necessary?
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
