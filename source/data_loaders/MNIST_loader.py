import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

def MNIST_loader(batch_size=60000, subset=None, split='train', flatten=False, norm_transform=None, augment_transform=None, num_workers=8):

    transform_list = []

    if split == 'train':
        if augment_transform is not None and augment_transform != False:
            for augmentation in augment_transform:
                if isinstance(augmentation, torchvision.transforms):
                    transform_list.append(augmentation)

    transform_list.append(ToTensor())

    if isinstance(norm_transform, Normalize):
        transform_list.append(norm_transform)
    elif norm_transform == 'default':
        transform_list.append(Normalize((0.1307,), (0.3081,)))

    if flatten:
        transform_list.append(Lambda(lambda x: torch.flatten(x)))
        
    transform = Compose(transform_list)

    if split == 'train' or split == 'val':
        set = MNIST(
            root="datasets", train=True, 
            download=True, transform=transform)
        train_indices, val_indices, _, _ = train_test_split(
            range(len(set)),
            set.targets,
            stratify=set.targets,
            test_size=0.1,
        )
        if split == 'train':
            train_split = Subset(set, train_indices)
            loader = DataLoader(
                train_split, batch_size=batch_size, shuffle=True, 
                num_workers=num_workers)
        if split == 'val':
            val_split = Subset(set, val_indices)
            loader = DataLoader(
                val_split, batch_size=batch_size, shuffle=True, 
                num_workers=num_workers)
    elif split == 'test':
        set = MNIST(
            root="datasets", train=False, 
            download=True, transform=transform)
        loader = DataLoader(
            set, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers)
    
    if norm_transform == 'auto':
        x, y = next(iter(loader))
        mean, std = np.round(x.mean(),2), np.round(x.std(),2) 

        loader = MNIST_loader(batch_size, subset, split, flatten, norm_transform=Normalize((mean,), (std,)), 
                                                  augment_transform=augment_transform, num_workers=num_workers)
    
    loader.num_classes = 10

    return loader