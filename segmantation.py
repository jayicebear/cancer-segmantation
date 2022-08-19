import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import animation, rc
import imageio
import time
import cv2
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
import tensorflow as tf
from skimage.morphology import label
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from skimage.segmentation import watershed
from skimage.measure import label
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import torch
import cv2
import re
import albumentations as A
import random
import pickle
import os
import collections
from sklearn.model_selection import train_test_split
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pathlib import Path
import random
from PIL import Image as Img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color
SEED = 42


home_dir = Path('/content/')
data_dir = home_dir / 'train'
train_data = pd.read_csv(home_dir / 'train.csv')
train_data.sample(3)

list_images = map(str, data_dir.rglob('*/*/*/*'))
image_properties = pd.DataFrame([(c, c.split('/')[-3], c.split('/')[-1]) for c in list_images], columns = ['whole_path', 'case_day', 'file'])
image_properties['slice'] = image_properties['file'].apply(lambda x: f"slice_{x.split('_')[1]}")
image_properties['height'] = image_properties['file'].apply(lambda x: int(x.split('_')[2]))
image_properties['width']  = image_properties['file'].apply(lambda x: int(x.split('_')[3]))
image_properties['id']     = image_properties['case_day'] + '_' + image_properties['slice']

train_data = pd.merge(train_data, image_properties, on='id', how='left')
train_data.sample(3)

# Helper functions
def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height, width, channels) of array to return 
    color: color for the mask
    Returns numpy array (mask)

    '''
    s = mask_rle.split()
    
    starts = list(map(lambda x: int(x) - 1, s[0::2]))
    lengths = list(map(int, s[1::2]))
    ends = [x + y for x, y in zip(starts, lengths)]
    
    if len(shape)==3:
        img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    else:
        img = np.zeros(shape[0] * shape[1], dtype=np.float32)
        
    for start, end in zip(starts, ends):
        img[start : end] = color
    return img.reshape(shape)
    
# Plot image    
def plot_images_and_masks(sample_id):
    case_num = sample_id.split('_')[0]
    day_num = sample_id.split('_')[1]
    slice_num = [x for x in os.listdir(f'/content/train/{case_num}/{case_num + "_" + day_num}/scans/') if sample_id.split('_')[3] in x.split('_')[1]][0]
    
    path = f'/content/train/{case_num}/{case_num + "_" + day_num}/scans/{slice_num}'
    image_data = plt.imread(path)
    print('\nID:', sample_id,'\n')
    print('\nTrain data records:')
    display(train_data[(train_data['id']==sample_id) & (train_data['segmentation'].notna())].reset_index(drop = True))
    print('\nImage data shape:', image_data.shape)
    print('Min and max pixels:', image_data.min(),',', image_data.max(),'\n')
    plt.subplot(1, 2, 1)
    plt.imshow(image_data, cmap = 'gray')
    plt.title('Input image')
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(image_data, cmap = 'gray')
    for i in range(0, train_data[(train_data['id']==sample_id) & (train_data['segmentation'].notna())].shape[0]):
        plt.imshow(rle_decode(train_data[(train_data['id']==sample_id) & (train_data['segmentation'].notna())]['segmentation'].tolist()[i], shape = image_data.shape), alpha = 0.4, cmap ='gray')
    plt.title('Input image with mask')
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
sample_id = 'case144_day0_slice_0068'
plot_images_and_masks(sample_id)
