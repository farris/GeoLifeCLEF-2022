# :evergreen_tree: :palm_tree: GeoLifeCLEF-2022 :palm_tree: :evergreen_tree:
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

As a result, the data was stored on the [NYU Greene Cluster](https://sites.google.com/nyu.edu/nyu-hpc/home?authuser=0). Due to quota limitations however, we only were able to obtain image scans from 'patches_fr' (France geo observations), and were not able to store the rasters. In summary, we were able to store 221588/671246 of the image scans from France, and the corresponding target species. This was divided for the training+val+test splits. Below is a snapshot of one of the infrared images used for training:


<img src="https://github.com/farris/GeoLifeCLEF-2022/blob/master/images/near_ir_show.png" width="400" height="400">

# File Structure & Guide

```
GeoLifeCLEF-2022/
├─ GLC (competition source code)/
├─ wandb/
├─ data (on cluster)/
│  ├─ metadata
│  ├─ observations
│  ├─ patches-fr
├─ Images/
│  ├─ gimme_picture.py
├─ models/
│  ├─ main_v1.py (baseline implementation)
│  ├─ main_v2.py (ResNet50 trained on RGB scans)
│  ├─ main_v3.py (ResNet18 trained on all images (1) )
│  ├─ main_v4.py (ResNet18 trained on all images (2) )
├─ .gitignore
├─ requirements.txt
├─ README.md
├─ main.sbatch
├─ filterobs.py
```
# Implementation

Pull dataset w/ Kaggle API

```
kaggle competitions download -c geolifeclef-2022-lifeclef-2022-fgvc9
```

Clone competition repo containing dataloaders and metric calculation scripts

``` 
git clone git@github.com:maximiliense/GLC.git
``` 

After unzipping the data, unless you have unlimited file storage on your cluster/cloud instance, you'll probably realize that you can't store all the image data (patches-fr), and the dataloader given doesn't function as is unless you have the full dataset. At that point you can run 
```
python filterobs.py
```
to return a new observations csv containing instances only existing in your data/patches-fr or data/patches-us directory(ies)

If on Linux cluster, you can use main.sbatch to run the different model variations we have under models/

# Evaluation

Below is a table summarizing configurations run, their WandB code names, and associated file under model(s)

|<p> </p><p>**Architecture**</p><p></p>|<p>**WandB** </p><p>**Code Name**</p>|**File**|**Batch Size**|<p>**Epochs**</p><p></p>|<p>**Optimizer/Lr**</p><p><br></p>|<p>**LR Scheduler**</p><p>**(Step/Gamma)**</p>|<p>**Data Types Included**</p><p></p>|**Gpu Type**|
| :- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|<p>ResNet50</p><p></p>|<p>Chocolate-glitter-9</p><p>helpful-sponge-2</p>|main\_v2|<p>128</p><p></p>|50|<p>Adam/5e-3</p><p></p>|*NA*|RGB|Tesla V100|
|ResNet50|lively-music-11|main\_v2|164|50|Adam/5e-3|20/0.2|RGB|RTX 8000|
|ResNet18|<p>apricot-cherry-16</p><p>serene-totem-15</p>|<p>main\_v4</p><p>main\_v3</p>|<p>512</p><p>*(128x4 image types)*</p>|50|Adam/3e-4|10/.1|All|<p>Tesla V100</p><p></p>|


Since we are trying to predict 17k+ classes in total, error is difficult to minimize/quantify as a metric. As a result, the chosen metric of the competition is top_30 error (For each of the top 30 softmax scores for each input x, +1 if the correct target y label is in that vector). The graph below shows the top_30 error on the holdout set across training epochs

<img width="600" img height = '500' alt="Screen Shot 2022-05-15 at 2 10 15 PM" src="https://user-images.githubusercontent.com/70980118/168487605-daa91534-a522-487d-8b5e-f517763b46ec.png" >
