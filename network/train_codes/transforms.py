import sys, os
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import random

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).reshape(1,-1,1,1)
        self.std = torch.FloatTensor(std).reshape(1,-1,1,1)

    def __call__(self, sample):
        for key in sample:
            if key[0] in ['y']:
                sample[key] = (sample[key] - self.mean) / self.std
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for key in sample:
            if key[0] in ['y']:
                sample[key]= np.expand_dims(sample[key], 0) # Must have shape [1, 224,224,3]
                sample[key] = torch.from_numpy(sample[key])
                sample[key] = sample[key].permute(0,3,2,1)
                sample[key] = sample[key].float()
            elif key == 'label':
                sample[key] = torch.tensor(sample[key])
        return sample
