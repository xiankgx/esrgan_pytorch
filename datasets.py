import glob
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TANH_MEAN = [0.5, 0.5, 0.5]
TANH_STD = [0.5, 0.5, 0.5]


def normalize(x, mean, std):
    assert x.ndim == 4
    return (x - torch.tensor(mean).view(1, 3, 1, 1).to(x))/torch.tensor(std).view(1, 3, 1, 1).to(x)


def inv_normalize(x, mean, std):
    assert x.ndim == 4
    return x * torch.tensor(std).view(1, 3, 1, 1).to(x) + torch.tensor(mean).view(1, 3, 1, 1).to(x)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape, extra_files=[]):
        h, w = hr_shape
        lr_size = (h//4, w//4)

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=(h, w),
                                         scale=(0.08, 1.0),
                                         ratio=(3./4., 4./3.),
                                         interpolation=transforms.InterpolationMode.BICUBIC)
        ])
        self.transform_lr = transforms.Compose([
            transforms.RandomChoice([
                transforms.Resize(size=lr_size,
                                  interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Resize(size=lr_size,
                                  interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Resize(size=lr_size,
                                  interpolation=transforms.InterpolationMode.NEAREST),
                transforms.Resize(size=lr_size,
                                  interpolation=transforms.InterpolationMode.LANCZOS),
            ])
        ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TANH_MEAN, TANH_STD)
        ])

        self.files = sorted(glob.glob(root + "/**/*.*", recursive=True))
        print(f"Found {len(self)} files in dir: {root}")
        if extra_files:
            self.files += extra_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img_hr = self.transform(img)
        img_lr = self.transform_lr(img_hr)
        return {"lr": self.to_tensor(img_lr), "hr": self.to_tensor(img_hr)}
