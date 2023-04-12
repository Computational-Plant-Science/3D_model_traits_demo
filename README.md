# 3D_model_traits_measurement

Function: Extract gemetrical traits of 3D root model 

Author            : Suxing Liu

Date created      : 04/04/2018

Date last modified: 04/25/2020

Python Version    : 3.7


Example of structure vs. 3D root model

![Optional Text](../master/media/image3.png) ![Optional Text](../master/media/image2.gif)

![Optional Text](../master/media/image2_1.gif)

Example of connecting disconnected root path

![Optional Text](../master/media/image7.png) ![Optional Text](../master/media/image8.png)


usage: 

python3 pipeline.py -p /$path_to_your_3D_model/ -m 3D_model_name.ply



## Requirements

[Docker](https://www.docker.com/) is required to run this project in a Linux environment.

Install Docker Engine (https://docs.docker.com/engine/install/)


## Usage


1. Build docker image on your PC under linux environment
```shell
docker build -t 3d-model-traits -f Dockerfile .
```
2. Download prebuild docker image from Docker hub
```shell
docker pull computationalplantscience/3d-model-traits
```
3. Run the pipeline inside the docker container 

link your test image path to the /images/ path inside the docker container
 ```shell
docker run -v /path_to_your_3D_model:/srv/test -it 3d-model-traits

or 

docker run -v /path_to_your_3D_model:/srv/test -it computationalplantscience/3d-model-traits

```
(For example: docker run -v /your local directory to cloned "Syngenta_PhenoTOOLs"/Syngenta_PhenoTOOLs/sample_test/Ear_test:/images -it syngenta_phenotools)

4. Run the pipeline inside the container
```shell
python3 pipeline.py -p /$path_to_your_3D_model/ -m 3D_model_name.ply


  

Reference:

Shenglan Du, Roderik Lindenbergh, Hugo Ledoux, Jantien Stoter, and Liangliang Nan.
AdTree: Accurate, Detailed, and Automatic Modelling of Laser-Scanned Trees.
Remote Sensing. 2019, 11(18), 2074.

@article{du2019adtree,
  title={AdTree: Accurate, detailed, and automatic modelling of laser-scanned trees},
  author={Du, Shenglan and Lindenbergh, Roderik and Ledoux, Hugo and Stoter, Jantien and Nan, Liangliang},
  journal={Remote Sensing},
  volume={11},
  number={18},
  pages={2074},
  year={2019}
}
