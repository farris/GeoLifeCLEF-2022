import skimage.io
from skimage.io import imread
import tifffile 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm.notebook import tqdm
import cv2
import shutil, json
import glob, os
import seaborn as sns
import gc, pandas as pd, numpy as np
import warnings
from warnings import WarningMessage, filterwarnings
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

DATA_PATH = Path("/scratch/fda239/Kaggle/data")

df_obs_fr_train = pd.read_csv(DATA_PATH / "observations" / "observations_fr_train.csv", sep=";", index_col="observation_id")



def load_patch(
    observation_id,
    patches_path,
    data='all',

     ):
   
    observation_id = str(observation_id)

    region_id = observation_id[0]
    if region_id == "1":
        region = "patches-fr"
    elif region_id == "2":
        region = "patches-us"
    else:
        raise ValueError(
            "Incorrect 'observation_id' {}, can not extract region id from it".format(
                observation_id
            )
        )

    subfolder1 = observation_id[-2:]
    subfolder2 = observation_id[-4:-2]

    filename = Path(patches_path) / region / subfolder1 / subfolder2 / observation_id

    obs_id = []

    if data == "all":
        data = ["rgb", "near_ir", "landcover", "altitude"]

    if "rgb" in data:
       
        rgb_filename = filename.with_name(filename.stem + "_rgb.jpg")
        if os.path.exists(rgb_filename):
            obs_id.append(observation_id)
            
    if "near_ir" in data:
        near_ir_filename = filename.with_name(filename.stem + "_near_ir.jpg")
        if os.path.exists(near_ir_filename):
            obs_id.append(observation_id)
            
    if "altitude" in data:
        altitude_filename = filename.with_name(filename.stem + "_altitude.tif")
        if os.path.exists(altitude_filename):
            obs_id.append(observation_id)
         

    if "landcover" in data:
        landcover_filename = filename.with_name(filename.stem + "_landcover.tif")
        if os.path.exists(landcover_filename):
            obs_id.append(observation_id)
    
    
    if not list(set(obs_id)):
        pass
    else:
        if list(set(obs_id))[0] == None:
            pass
        else:
            return int(list(set(obs_id))[0])


DATA_PATH = Path("/scratch/fda239/Kaggle/data")
out = []
idx = df_obs_fr_train.index.tolist()
for i in tqdm(range(len(idx))):
    out.append(load_patch(idx[i],DATA_PATH))
    
out = list(filter(None, out))
df_new_obs_train = df_obs_fr_train[df_obs_fr_train.index.isin(out)]
print('-------------------')
print(len(df_obs_fr_train))
print(len(df_new_obs_train))
print('-------------------')
df_new_obs_train.to_csv('observations_fr_train_filter.csv')
 



