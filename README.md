# -------------Work in Progress--------------------------
# ICPR2020 Attention2Angio

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attention2angiogan-synthesizing-fluorescein/fundus-to-angiography-generation-on-fundus)](https://paperswithcode.com/sota/fundus-to-angiography-generation-on-fundus?p=attention2angiogan-synthesizing-fluorescein)

This code is part of the supplementary materials for the ICPR 2020 for our paper Attention2AngioGAN: Synthesizing FluoresceinAngiography from Retinal Fundus Images usingGenerative Adversarial Networks . The paper has since been accpeted to ICPR 2020 and will be preseneted in January 2021.

![](img1.png)

### Arxiv Pre-print
```
https://arxiv.org/abs/2007.09191
```
# Citation 
```
@inproceedings{kamran2020attention2angiogan,
  title={Attention2AngioGAN: Synthesizing Fluorescein Angiography from Retinal Fundus Images using Generative Adversarial Networks},
  author={Kamran, Sharif Amit and Hossain, Khondker Fariha and Tavakkoli, Alireza and Zuckerbrod, Stewart Lee},
  booktitle={International Conference on Pattern Recognition},
  year={2020}
}
```

## Pre-requisite
- Ubuntu 18.04 / Windows 7 or later
- NVIDIA Graphics card

## Installation Instruction for Ubuntu
- Download and Install [Nvidia Drivers](https://www.nvidia.com/Download/driverResults.aspx/142567/en-us)
- Download and Install via Runfile [Nvidia Cuda Toolkit 10.0](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)
- Download and Install [Nvidia CuDNN 7.6.5 or later](https://developer.nvidia.com/rdp/cudnn-archive)
- Install Pip3 and Python3 enviornment
```
sudo apt-get install pip3 python3-dev
```
- Install Tensorflow-Gpu version-2.0.0 and Keras version-2.3.1
```
sudo pip3 install tensorflow-gpu==2.0.3
sudo pip3 install keras==2.3.1
```
- Install packages from requirements.txt
```
sudo pip3 -r requirements.txt
```

### Dataset download link for Hajeb et al.
```
https://sites.google.com/site/hosseinrabbanikhorasgani/datasets-1/fundus-fluorescein-angiogram-photographs--colour-fundus-images-of-diabetic-patients
```
