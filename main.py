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
        # if os.path.exists(rgb_filename):
        patches = load_patch(
            observation_id, self.root, data=self.patch_data
        )
        patches = torch.tensor(patches)
        
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
        # else:
        #     pass


# def collate_fn(batch):

#     batch = list(filter(lambda x: x is not None, batch))
#     return torch.utils.data.dataloader.default_collate(batch)

DATA_PATH = Path("/scratch/fda239/Kaggle/data")
# possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'
dataset = GeoLifeCLEF2022Dataset(DATA_PATH,subset = "train", 
                                 region = 'fr', 
                                 patch_data = 'rgb', \
                                 use_rasters = None,\
                                 #transform = get_train_transforms(),\
                                 transform = None,\
                                 patch_extractor = None )
print("Dataset created....")


# train_loader = DataLoader(dataset, batch_size=2,num_workers = 0,shuffle = False,drop_last=True,collate_fn=lambda x: x )
# train_loader = DataLoader(dataset, batch_size=2,num_workers = 0,shuffle = False,drop_last=True)



# print("Len train_loader:", len(train_loader))
# count = 0
# for batch in train_loader:
#     # print("Batch size:", len(batch))
#     print('-----------------------------------')
#     count += 1
#     # print("count:", count)
#     print(batch)
#     print('-----------------------------------')
#     if count == 20:
#         break

# raise SystemExit(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------COPY AND PASTE FROM DL HOMEWORK 3 CNN ---------------------------------------------
class ResidualBlock(torch.nn.Module):
  def __init__(self, in_dim, out_dim, ksize):
      super(ResidualBlock, self).__init__()
      self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=ksize, padding="same")
      self.conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=ksize, padding="same")
      self.relu = nn.ReLU()

  def forward(self, x):
    # forward pass
    identity = x 

    out = self.relu(self.conv1(x))
    out = self.conv2(out)
    # combine forward pass and identity through addition

    out += identity

    return self.relu(out) 

class CNN(torch.nn.Module):

  def __init__(self, nfeatures, nclasses):
    super().__init__()
    self.nfeatures = nfeatures
    self.nclasses = nclasses
    self.kernel = 5
    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = self.nfeatures, kernel_size=self.kernel, padding="same")
    self.bnorm1 = nn.BatchNorm2d(self.nfeatures)
    self.pool = nn.MaxPool2d(self.kernel)
    self.relu = nn.ReLU()

    self.residual = ResidualBlock(self.nfeatures, self.nfeatures, self.kernel)
    self.bnorm3 = nn.BatchNorm2d(self.nfeatures)

    self.conv2 = nn.Conv2d(self.nfeatures, 50, kernel_size=5, padding="same")
    self.bnorm2 = nn.BatchNorm2d(50)
    self.pool = nn.MaxPool2d(self.kernel)
    self.relu = nn.ReLU()

    self.reshape = nn.Flatten()

    self.fc1 = nn.Linear(5000, 7500)
    self.relu = nn.ReLU()
    self.bnorm4 = nn.BatchNorm1d(7500)

    self.fc2 = nn.Linear(7500, 5000)
    self.relu = nn.ReLU()
    self.bnorm5 = nn.BatchNorm1d(5000)

    self.fc3 = nn.Linear(5000, self.nclasses)

    self.LSM = nn.LogSoftmax(dim=1)

  def forward(self, x):
    # print("Input shape:", x.shape)
    x = self.conv1(x)
    # print("Conv1:", x.shape)
    x = self.bnorm1(x)
    # print("BNorm1:", x.shape)
    x = self.pool(x)
    # print("Pool1:", x.shape)
    x = self.relu(x)
    # print("Relu1:", x.shape)

    x = self.residual(x)
    # print("Residual:", x.shape)
    x = self.bnorm3(x)
    # print("BNorm2:", x.shape)

    x = self.conv2(x)
    # print("Conv2:", x.shape)
    x = self.bnorm2(x)
    # print("BNorm2:", x.shape)
    x = self.pool(x)
    # print("Pool1:", x.shape)
    x = self.relu(x)
    # print("Relu2:", x.shape)

    x = self.reshape(x)
    # print("Reshape:", x.shape)
    x = self.relu(self.fc1(x))
    x = self.bnorm4(x)
    # print("FC1:",x.shape)

    x = self.relu(self.fc2(x))
    x = self.bnorm5(x)
    # print("FC2:",x.shape)

    x = self.fc3(x)

    x = self.LSM(x)
    return x

def get_loss_and_correct(model, batch, criterion, device):
  # Implement forward pass and loss calculation for one batch.
  # Remember to move the batch to device.
  # 
  # Return a tuple:
  # - loss for the batch (Tensor)
  # - number of correctly classified examples in the batch (Tensor)

  data, labels = batch
  data, labels = data.to(device), labels.to(device)
  data = data.view([len(labels), 3, 256, 256])
  data = data.float()

  output = model(data)
  loss = criterion(output, labels)

  pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
  correct = pred.eq(labels.data.view_as(pred)).cpu().sum()

  return loss, correct

def step(loss, optimizer):
  # Implement backward pass and update.
  
  optimizer.zero_grad()

  loss.backward()

  optimizer.step()


#Set of target species in filtered csv
N_CLASSES = 4426
N_EPOCHS = 1
BATCH_SIZE = 256

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,num_workers = 0,shuffle = True,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers = 0,shuffle = False,drop_last=True)

# Print image and target size and show image
# rgb_batch, target = iter(train_loader).next()
# plt.figure(figsize=(10, 12))
# print("RGB image size: ", rgb_batch.shape)
# print ("Target size: ", target.shape)
# plt.imshow(rgb_batch[0])

model = CNN(25, N_CLASSES)
model.to(device)
model = model.float()
criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5, weight_decay=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, amsgrad=True)

train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies = []

pbar = tqdm(range(N_EPOCHS))

for i in pbar:
  total_train_loss = 0.0
  total_train_correct = 0.0
  total_validation_loss = 0.0
  total_validation_correct = 0.0

  model.train()

  for batch in tqdm(train_loader, leave=False):
    loss, correct = get_loss_and_correct(model, batch, criterion, device)
    step(loss, optimizer)
    total_train_loss += loss.item()
    total_train_correct += correct.item()

  with torch.no_grad():
    for batch in val_loader:
      loss, correct = get_loss_and_correct(model, batch, criterion, device)
      total_validation_loss += loss.item()
      total_validation_correct += correct.item()

  mean_train_loss = total_train_loss / len(train_dataset)
  train_accuracy = total_train_correct / len(train_dataset)

  mean_validation_loss = total_validation_loss / len(val_dataset)
  validation_accuracy = total_validation_correct / len(val_dataset)

  train_losses.append(mean_train_loss)
  validation_losses.append(mean_validation_loss)

  train_accuracies.append(train_accuracy)
  validation_accuracies.append(validation_accuracy)

  pbar.set_postfix({'train_loss': mean_train_loss, \
  'validation_loss': mean_validation_loss, 'train_accuracy': train_accuracy, 'validation_accuracy': validation_accuracy})