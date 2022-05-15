import skimage.io
from skimage.io import imread
import tifffile 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import cv2
import tensorflow as tf
import shutil, json
import glob, os
import gc, pandas as pd, numpy as np
import warnings
from warnings import WarningMessage, filterwarnings
import sys
import os
from pathlib import Path
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
import torchvision.models as models
import torch.backends.cudnn as cudnn
from tqdm import tqdm
torch.backends.cudnn.enabled = False
import wandb
from torch.optim.lr_scheduler import StepLR


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
        rgb_patches,near_ir_patches,altitude_patches,landcover_patches = [],[],[],[]
        data = ["rgb", "near_ir", "landcover", "altitude"]

        rgb_filename = filename.with_name(filename.stem + "_rgb.jpg")
        if os.path.exists(rgb_filename):
            rgb_patch = Image.open(rgb_filename)
            rgb_patch = np.asarray(rgb_patch)
            rgb_patches.append(rgb_patch)

        near_ir_filename = filename.with_name(filename.stem + "_near_ir.jpg")
        if os.path.exists(near_ir_filename):
            near_ir_patch = Image.open(near_ir_filename)
            near_ir_patch = np.asarray(near_ir_patch)
            near_ir_patches.append(near_ir_patch)

        altitude_filename = filename.with_name(filename.stem + "_altitude.tif")
        if os.path.exists(altitude_filename):
            altitude_patch = tifffile.imread(altitude_filename)
            altitude_patches.append(altitude_patch)

        landcover_filename = filename.with_name(filename.stem + "_landcover.tif")
        if os.path.exists(landcover_filename):
            landcover_patch = tifffile.imread(landcover_filename)
            if landcover_mapping is not None:
                landcover_patch = landcover_mapping[landcover_patch]
            landcover_patches.append(landcover_patch)

        return rgb_patches,near_ir_patches,altitude_patches,landcover_patches

    elif "rgb" in data:
       
        rgb_filename = filename.with_name(filename.stem + "_rgb.jpg")
        if os.path.exists(rgb_filename):
            rgb_patch = Image.open(rgb_filename)
            if return_arrays:
                rgb_patch = np.asarray(rgb_patch)
            patches.append(rgb_patch)
        
            
    elif "near_ir" in data:
        near_ir_filename = filename.with_name(filename.stem + "_near_ir.jpg")
        near_ir_patch = Image.open(near_ir_filename)
        if return_arrays:
            near_ir_patch = np.asarray(near_ir_patch)
        patches.append(near_ir_patch)

    elif "altitude" in data:

        altitude_filename = filename.with_name(filename.stem + "_altitude.tif")
       
        if os.path.exists(altitude_filename):
            altitude_patch = tifffile.imread(altitude_filename)
        patches.append(altitude_patch)

    elif "landcover" in data:
  
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
        species_map = {s : i for s,i in zip(species_unique,range(0,n_unique)) }
        df_fr['target_map'] = df_fr.species_id.map(species_map)
   
        df = copy.deepcopy(df_fr)
        
        if self.training_data and subset != "train+val": #if val
            ind = df.index[df["subset"] == subset]
            df = df.loc[ind]

        self.observation_ids = df.index
        self.coordinates = df[["latitude", "longitude"]].values
        
        if self.training_data:
            self.targets = df.target_map.values #df.species_id ---> df.target_map
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
        rgb_patches,near_ir_patches,altitude_patches,landcover_patches = load_patch(
            observation_id, self.root, data=self.patch_data
        )
        rgb_patches = torch.tensor(np.array(rgb_patches))
        near_ir_patches = torch.tensor(np.array(near_ir_patches)).unsqueeze_(-1).repeat(1,1,1,3)
        altitude_patches = torch.tensor(np.array(altitude_patches)).unsqueeze_(-1).repeat(1,1,1,3)
        landcover_patches = torch.tensor(np.array(landcover_patches)).unsqueeze_(-1).repeat(1,1,1,3)
        patches = torch.cat([rgb_patches,near_ir_patches,altitude_patches,landcover_patches])
        patches = patches.view(-1, 256,256, 3)
       
        if patches is not None:
            
            if len(patches) == 1:
                patches = patches[0]

            if self.transform:
                patches = self.transform(patches)
              

            if self.training_data:
                target = self.targets[index]
                target = np.asarray([target for i in range(4)])
                if self.target_transform:
                    target = self.target_transform(target)

                return patches, target
            else:
                return patches
        else:
            return patches

def train_model(model, criterion, optimizer,scheduler ,num_epochs=3, top_k=30):
    wandb.watch(model, criterion, log="all", log_freq=1)

    for epoch in pbar:
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_topk_error = 0

            for batch in tqdm(dataloaders[phase]):
                inputs, labels = batch

                inputs = inputs.to(device)
                inputs = inputs.float()
                inputs = inputs.view([len(labels)*4, 3, 256, 256])

                labels = labels.to(device)
                labels = labels.view([-1])
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                
                #top k-------------------------#
                top_30_values,top_30_indices = torch.topk(F.log_softmax(outputs, dim=1), top_k)
                batch_top30_acc = torch.sum(top_30_indices == labels.view(-1,1)).float()

                batch_top30_err = 1 - (batch_top30_acc / len(labels))
                running_topk_error += batch_top30_err.item()

                del _
                del preds
                del top_30_values
                del top_30_indices
                del inputs
                # gc.collect()
                # torch.cuda.empty_cache()
                #bottom of batch train loop --------------------------------------
            
            if phase == "train": image_len = train_size
            else: image_len = val_size

            epoch_loss = running_loss / image_len
            epoch_acc = running_corrects.double() / image_len
            epoch_topk = running_topk_error / (np.floor(image_len) / len(labels))
            # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
            #                                             epoch_loss,
            #                                             epoch_acc))
            pbar.set_postfix({'Phase': phase, 'Loss': epoch_loss, 'Acc': epoch_acc.item(), 'Topk_Err': epoch_topk/4})
            if phase == "train":
                train_loss = epoch_loss
                train_acc = epoch_acc.item()
                train_topk = epoch_topk
        
        wandb.log({
            "Epoch": epoch, "Train_Loss": train_loss/4, "Train_Acc": train_acc/4, "Train_TopK": train_topk/4,
            "Val_Loss": epoch_loss/4, "Val_Acc": epoch_acc/4, "Val_TopK": epoch_topk/4
            })
        # scheduler.step()
    return model

DATA_PATH = Path("/scratch/fda239/Kaggle/data")
# possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'
dataset = GeoLifeCLEF2022Dataset(DATA_PATH,subset = "train", 
                                 region = 'fr', 
                                 patch_data = 'all', \
                                 use_rasters = None,\
                                 #transform = get_train_transforms(),\
                                 transform = None,\
                                 patch_extractor = None )
print("Dataset created....")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Set of target species in filtered csv
N_CLASSES = 4426
N_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
TOP_K = 30

config_dict = {
  "learning_rate": LEARNING_RATE,
  "epochs": N_EPOCHS,
  "batch_size": BATCH_SIZE,
  "TopK":TOP_K
}

print("WandB Project Initialized")
wandb.init(config=config_dict, project="DLS", entity="dls-glc")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,num_workers =int(os.environ["SLURM_CPUS_PER_TASK"]),shuffle = False,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=int(BATCH_SIZE/4), num_workers=0,shuffle = False,drop_last=True)
dataloaders = {"train": train_loader, "eval":val_loader}
rgb_batch, target = iter(train_loader).next()


model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
               nn.Linear(512, 1024),
               nn.ReLU(inplace=True),
               nn.Linear(1024, N_CLASSES))

model.to(device)
model = model.float()

criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)


pbar = tqdm(range(N_EPOCHS))
model_trained = train_model(model, criterion, optimizer,None, num_epochs=N_EPOCHS, top_k=TOP_K)
