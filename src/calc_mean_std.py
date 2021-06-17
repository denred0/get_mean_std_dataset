import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pathlib import Path
import cv2

from tqdm import tqdm

####### PARAMS

device = torch.device('cpu')
num_workers = 4
image_size = 1024
batch_size = 8
data_path = Path('data/images')
images_ext = ['*.jpg']

augs = A.Compose([A.Resize(height=image_size, width=image_size),
                  A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                  ToTensorV2()])


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


images = get_all_files_in_folder(data_path, images_ext)


class LeafData(Dataset):

    def __init__(self,
                 files,
                 transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # import
        # path = os.path.join(self.directory, self.data.iloc[idx]['image_id'])
        image = cv2.imread(str(self.files[idx]), cv2.COLOR_BGR2RGB)

        # augmentations
        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image


# dataset
image_dataset = LeafData(files=images,
                         transform=augs)

# data loader
image_loader = DataLoader(image_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True)

####### COMPUTE MEAN / STD

# placeholders
psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# loop through images
for inputs in tqdm(image_loader):
    psum += inputs.sum(axis=[0, 2, 3])
    psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])

####### FINAL CALCULATIONS

# pixel count
count = len(images) * image_size * image_size

# mean and std
total_mean = psum / count
total_var = (psum_sq / count) - (total_mean ** 2)
total_std = torch.sqrt(total_var)

total_mean_255 = total_mean * 255
total_std_255 = total_std * 255

# output
print('mean: ' + str(total_mean))
print('std:  ' + str(total_std))
print('mean_255: ' + str(total_mean_255))
print('std_255:  ' + str(total_std_255))


