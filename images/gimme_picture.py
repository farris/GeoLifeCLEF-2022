import skimage.io
from skimage.io import imread
import tifffile 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm.notebook import tqdm
import cv2
import shutil, json
import tensorflow as tf
import glob, os
import seaborn as sns
import gc, pandas as pd, numpy as np
import warnings
from warnings import WarningMessage, filterwarnings
import os
from pathlib import Path
import pandas as pd
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from GLC.data_loading.environmental_raster import PatchExtractor
from PIL import Image
import copy 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from tqdm import tqdm
torch.backends.cudnn.enabled = False

def get_train_transforms():
    return Compose([
            #RandomResizedCrop(256, 256),
            #Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            #ShiftScaleRotate(p=0.5),
            #HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            #RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            #CoarseDropout(p=0.5),
            #Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms():
    return Compose([
            CenterCrop(256,256, p=1.),
            #Resize(CFG['img_size'], CFG['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def load_patch(
    observation_id,
    patches_path,
    *,
    data="all",
    landcover_mapping=None,
    return_arrays=True
                            ):
    """Loads the patch data associated to an observation id
    Parameters
    ----------
    observation_id : integer
        Identifier of the observation.
    patches_path : string / pathlib.Path
        Path to the folder containing all the patches.
    data : string or list of string
        Specifies what data to load, possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'.
    landcover_mapping : 1d array-like
        Facultative mapping of landcover codes, useful to align France and US codes.
    return_arrays : boolean
        If True, returns all the patches as Numpy arrays (no PIL.Image returned).
    Returns
    -------
    patches : tuple of size 4 containing 2d array-like objects
        Returns a tuple containing all the patches in the following order: RGB, Near-IR, altitude and landcover.
    """
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

    patches = []

    if data == "all":
        data = ["rgb", "near_ir", "landcover", "altitude"]

    if "rgb" in data:
       
        rgb_filename = filename.with_name(filename.stem + "_rgb.jpg")
        if os.path.exists(rgb_filename):
            rgb_patch = Image.open(rgb_filename)
            if return_arrays:
                rgb_patch = np.asarray(rgb_patch)
            patches.append(rgb_patch)
        
            
    if "near_ir" in data:
        near_ir_filename = filename.with_name(filename.stem + "_near_ir.jpg")
        near_ir_patch = Image.open(near_ir_filename)
        if return_arrays:
            near_ir_patch = np.asarray(near_ir_patch)
        patches.append(near_ir_patch)

    if "altitude" in data:

        altitude_filename = filename.with_name(filename.stem + "_altitude.tif")
       
        if os.path.exists(altitude_filename):
            altitude_patch = tifffile.imread(altitude_filename)
            patches.append(altitude_patch)

    if "landcover" in data:
  
        landcover_filename = filename.with_name(filename.stem + "_landcover.tif")

        if os.path.exists(landcover_filename):
            landcover_patch = tifffile.imread(landcover_filename)
            if landcover_mapping is not None:
                landcover_patch = landcover_mapping[landcover_patch]
            patches.append(landcover_patch)
    
    return patches
    
class GeoLifeCLEF2022Dataset(Dataset):
    """Pytorch dataset handler for GeoLifeCLEF 2022 dataset.
    Parameters
    ----------
    root : string or pathlib.Path
        Root directory of dataset.
    subset : string, seither "train", "val", "train+val" or "test"
        Use the given subset ("train+val" is the complete training data).
    region : string, either "both", "fr" or "us"
        Load the observations of both France and US or only a single region.
    patch_data : string or list of string
        Specifies what type of patch data to load, possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'.
    use_rasters : boolean (optional)
        If True, extracts patches from environmental rasters.
    patch_extractor : PatchExtractor object (optional)
        Patch extractor to use if rasters are used.
    transform : callable (optional)
        A function/transform that takes a list of arrays and returns a transformed version.
    target_transform : callable (optional)
        A function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        root,
        subset,
        *,
        region="both",
        patch_data="all",
        use_rasters=True,
        patch_extractor=None,
        transform=None,
        target_transform=None
    ):
        self.root = Path(root)
        self.subset = subset
        self.region = region
        self.patch_data = patch_data
        self.transform = transform
        self.target_transform = target_transform

        possible_subsets = ["train", "val", "train+val", "test"]
        if subset not in possible_subsets:
            raise ValueError(
                "Possible values for 'subset' are: {} (given {})".format(
                    possible_subsets, subset
                )
            )

        possible_regions = ["both", "fr", "us"]
        if region not in possible_regions:
            raise ValueError(
                "Possible values for 'region' are: {} (given {})".format(
                    possible_regions, region
                )
            )

        if subset == "test":
            subset_file_suffix = "test"
            self.training_data = False
        else:
            subset_file_suffix = "train"
            self.training_data = True

        df_fr = pd.read_csv(
            self.root
            / "observations"
            / "observations_fr_{}_filter.csv".format(subset_file_suffix),
            sep=",",
            index_col="observation_id",nrows = 50000
        )
        
        n_unique = len(set(df_fr.species_id))
        species_unique = sorted(df_fr['species_id'].unique())
        species_map = {s : i for s,i in zip(species_unique,range(1,n_unique+1)) }
        df_fr['target_map'] = df_fr.species_id.map(species_map)
   
        df = copy.deepcopy(df_fr)
        
        if self.training_data and subset != "train+val": #if val
            ind = df.index[df["subset"] == subset]
            df = df.loc[ind]

        self.observation_ids = df.index
        self.coordinates = df[["latitude", "longitude"]].values
        
        if self.training_data:
            self.targets =df.target_map.values #df.species_id ---> df.target_map
        else:
            self.targets = None

        if use_rasters:
            if patch_extractor is None:
        
                patch_extractor = PatchExtractor(self.root / "rasters", size=256)
                patch_extractor.add_all_rasters()

            self.patch_extractor = patch_extractor
        else:
            self.patch_extractor = None

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, index):

        latitude = self.coordinates[index][0]
        longitude = self.coordinates[index][1]
        observation_id = self.observation_ids[index]

        observation_id = str(observation_id)

        subfolder1 = observation_id[-2:]
        subfolder2 = observation_id[-4:-2]

        filename = Path(self.root) / "patches-fr" / subfolder1 / subfolder2 / observation_id
        rgb_filename = filename.with_name(filename.stem + "_rgb.jpg")
        patches = load_patch(
            observation_id, self.root, data=self.patch_data
        )
        patches = torch.tensor(np.array(patches))
        
        if patches is not None:
            
            if len(patches) == 1:
                patches = patches[0]

            if self.transform:
                patches = self.transform(patches)
              

            if self.training_data:
                target = self.targets[index]

                if self.target_transform:
                    target = self.target_transform(target)

                return patches, target
            else:
                return patches
        else:
            return patches

DATA_PATH = Path("/scratch/fda239/Kaggle/data")
# possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'


for angle in ['rgb', 'near_ir', 'landcover' ,'altitude']:
    dataset = GeoLifeCLEF2022Dataset(DATA_PATH,subset = "train", 
                                region = 'fr', 
                                patch_data = angle, \
                                use_rasters = None,\
                                #transform = get_train_transforms(),\
                                transform = None,\
                                patch_extractor = None )
    print("Dataset created....")


    #Set of target species in filtered csv
    N_CLASSES = 4426
    N_EPOCHS = 100
    BATCH_SIZE = 1

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,num_workers = 0,shuffle = True,drop_last=True)

    # Print image and target size and show image
    rgb_batch, target = iter(train_loader).next()
    plt.figure(figsize=(10, 12))
    plt.imshow(rgb_batch[0])
    plt.savefig(angle+'_show.png')
    del dataset
    del train_loader


