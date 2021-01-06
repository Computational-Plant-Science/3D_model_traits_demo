FROM python:3.7.9-buster

LABEL maintainer='Suxing Liu'

COPY ./ /opt/3D_model_traits_demo

RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt install -y \
    build-essential \
    python3-tk \
    python3-numexpr \
    python3-pil.imagetk \
    libgl1-mesa-glx \
    libsm6 \
    libxext6

RUN pip install --upgrade pip && \
    pip install numpy \
    Pillow \
    rdp \
    scipy \
    scikit-image \
    scikit-learn \
    scikit-build \
    matplotlib \
    networkx \
    plyfile \
    open3d \
    opencv-python-headless \
    openpyxl