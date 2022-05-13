import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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
import sys
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
import torchvision.models as models
import torch.backends.cudnn as cudnn
from tqdm import tqdm
torch.backends.cudnn.enabled = False


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)

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
        species_map = {s : i for s,i in zip(species_unique,range(0,n_unique)) }
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

def train_model(model, device,criterion, optimizer, num_epochs=3):
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
                labels = labels.to(device)
                inputs = inputs.float()
                inputs = inputs.view([len(labels), 3, 256, 256])

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
                top_30_values,top_30_indices = torch.topk(F.log_softmax(outputs, dim=1), 30)
                labels.unsqueeze_(1)
                labels = labels.repeat(1,30)
                batch_top30_acc = torch.sum(torch.sum(top_30_indices == labels)).item()

                batch_top30_err = 1 - batch_top30_acc/len(labels)
                running_topk_error += batch_top30_err
            
            if phase == "train": image_len = train_size
            else: image_len = val_size

            epoch_loss = running_loss / image_len
            epoch_acc = running_corrects.double() / image_len
            epoch_topk = running_topk_error / (np.floor(image_len) / len(labels))
            # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
            #                                             epoch_loss,
            #                                             epoch_acc))
            pbar.set_postfix({'Phase': phase, 'Loss': epoch_loss, 'Acc': epoch_acc.item(), 'Topk_Err': epoch_topk})
    return model

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Lets Go')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
 

    args = parser.parse_args()
                               
    DATA_PATH = Path("/scratch/fda239/Kaggle/data")
    dataset = GeoLifeCLEF2022Dataset(DATA_PATH,subset = "train", 
                                    region = 'fr', 
                                    patch_data = 'rgb', \
                                    use_rasters = None,\
                                    #transform = get_train_transforms(),\
                                    transform = None,\
                                    patch_extractor = None )
    print("Dataset created....")

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)
    
    setup(rank, world_size)
    
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset1, num_replicas=world_size, rank=rank)
    # train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, sampler=train_sampler, \
    #                                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True,shuffle = True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True,shuffle = False,drop_last=True)
    
    global dataloaders
    dataloaders = {"train": train_loader, "eval":val_loader}

    model = ResNet18()
    model.fc = nn.Sequential(
                nn.Linear(2048, 5000),
                nn.ReLU(inplace=True),
                nn.Linear(5000, N_CLASSES))

    model = model.to(local_rank)
    model = model.float()

    ddp_model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adadelta(ddp_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(range(N_EPOCHS))
        model_trained = train_model(ddp_model, local_rank, criterion, optimizer, num_epochs=args.epochs)
        if rank == 0: test(ddp_model, local_rank, test_loader,criterion)
        scheduler.step()

    dist.destroy_process_group()

