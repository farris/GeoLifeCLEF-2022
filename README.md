# GeoLifeCLEF-2022
Authored by: Zafir Momin & Farris Atif
NYU Center for Data Science

# Introduction:

This repository contains the code for the GeoLifeClef 2022 competition hosted by the LifeCLEF 2022 lab of the CLEF 2022 conference, and of the FGVC9 workshop organized in conjunction with CVPR 2022 conference.
https://www.kaggle.com/c/geolifeclef-2022-lifeclef-2022-fgvc9/overview

# Background

The rise of deep learning in image/object recognition tasks has opened the gate for uncovering elusive relationships in nature previously unbeknownst to humans. This particular project aims to leverage this, by utilizng CNN's for plant and species recognition in a variety of areas scattered throughout the United States and France. More specifically, for an arbitrary patch of land we aim to predict 17K+ species (9K plant species and 8K animal species). In doing so , we hope to improve species identification tools for biodiversity management and conservation.

# Data

Put simply, this dataset is huge. There are 1.6M geo-localized observations from France and the US, each of which has 4 corresponding images (rgb aerial image, tiff aerial image, altitude scan, and landcover). Additionally environmental rasters (metadata) are available for each observation which could be fed into a model for added information. In total there are 6.72M files and 61 columns of data.

As a result, the data was stored on the NYU Greene Cluster. Due to quota limitations however, we only were able to obtain image scans from 'patches_fr' (france geo observations), and were not able to store the rasters. In summary, we were able to store 2215888/671246 of the image scans from France, and the corresponding target species. This was devided for the training+val+test splits. Below is a snapshot of one of the rgb images used for training:
![alt text](https://github.com/farris/GeoLifeCLEF-2022/blob/master/test.png)

#Implementation

(placeholder)

