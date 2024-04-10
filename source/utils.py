import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def augment_positive_samples(x, x_test, dims):
    rotation_transform = transforms.RandomRotation(degrees=10)
    x_shape = x.shape
    x_test_shape = x_test.shape

    if x.dim() == 2:
        x = x.view(-1, dims[0], dims[1], dims[2])
        x_test = x_test.view(-1, dims[0], dims[1], dims[2])
    
    rotated_images = torch.stack([rotation_transform(img) for img in x])
    rotated_images_test = torch.stack([rotation_transform(img) for img in x_test])
    
    if len(x_shape) == 2:
        rotated_images = rotated_images.view(x_shape)
        rotated_images_test = rotated_images_test.view(x_test_shape)

    return rotated_images, rotated_images_test

def log_loss(g_pos: torch.Tensor, g_neg: torch.Tensor, threshold: torch.Tensor):
    return torch.log(1 + torch.exp(torch.cat([
                -g_pos + threshold,
                g_neg - threshold]))).mean()
