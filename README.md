# GeoLifeCLEF-2022 :palm_tree:
Authored by: Zafir Momin & Farris Atif  
NYU Center for Data Science

# Introduction:

This repository contains the code for the GeoLifeClef 2022 competition hosted by the LifeCLEF 2022 lab of the CLEF 2022 conference, and of the FGVC9 workshop organized in conjunction with CVPR 2022 conference.
https://www.kaggle.com/c/geolifeclef-2022-lifeclef-2022-fgvc9/overview

# Background

The rise of deep learning in image/object recognition tasks has opened the gate for uncovering elusive relationships in nature previously unbeknownst to humans. This particular project aims to leverage this, by utilizng CNN's for plant and species recognition in a variety of areas scattered throughout the United States and France. More specifically, for an arbitrary patch of land we aim to predict 17K+ species (9K plant species and 8K animal species). In doing so , we hope to improve species identification tools for biodiversity management and conservation.

# Data

Put simply, this dataset is huge. In total there are 6.72M files and 61 columns of data. More specifically, there are 1.6M geo-localized observations from France and the US, each of which has 4 corresponding images, summarized as follows:
1. Remote sensing imagery: 256mx256m RGB-IR patches centered at each observation. 
* Format: 256x256 JPEG images, a color JPEG file for RGB data and a grayscale one for near-infrared. 
* Resolution: 1 meter per pixel. 
* Source: NAIP for US and IGN for France. 
2. Land cover data: 256mx256m patches centered at each observation. 
* Format: 256x256 TIFF files with Deflate compression. 
* Resolution: 1 meter per pixel. 
* Source: NLCD for US and Cesbio for France. 
3. Altitude data: 256mx256m patches centered at each observation. 
* Format: 256x256 TIFF files with Deflate compression. 
* Resolution: 1 meter per pixel. 
* Source: SRTMGL1 for US and FranceAdditionally environmental rasters (metadata) are available for each observation which could be fed into a model for added information.  

As a result, the data was stored on the [NYU Greene Cluster](https://sites.google.com/nyu.edu/nyu-hpc/home?authuser=0). Due to quota limitations however, we only were able to obtain image scans from 'patches_fr' (France geo observations), and were not able to store the rasters. In summary, we were able to store 221588/671246 of the image scans from France, and the corresponding target species. This was divided for the training+val+test splits. Below is a snapshot of one of the RGB images used for training:


<img src="https://github.com/farris/GeoLifeCLEF-2022/blob/master/images/rgb_show.png" width="400" height="400">

# File Structure & Guide

(placeholder)

# Implementation

(placeholder)

# Evaluation

(placeholder)


