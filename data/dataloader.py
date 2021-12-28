'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from skimage import io
from data.solar_dataset import SolarDataset
import json
import numpy as np
from PIL import Image

def load_dataset(opts, kwargs=None):

    if opts.dataset.lower() == 'solar':
        normalize = transforms.Normalize(
            mean=[660.5929, 812.9481, 1080.6552, 1398.3968, 1662.5913, 1899.4804, 2061.932, 2100.2792, 2214.9325, 2230.5973, 2443.3014, 1968.1885],
            std=[137.4943, 195.3494, 241.2698, 378.7495, 383.0338, 449.3187, 511.3159, 547.6335, 563.8937, 501.023, 624.041, 478.9655]
        )
        all_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        trn = SolarDataset("train", opts,  transform=all_transforms)
        val = SolarDataset("val", opts, transform=all_transforms)

        trn_loader = torch.utils.data.DataLoader(trn, batch_size=opts.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val, batch_size=opts.batch_size, shuffle=False)
    else:
        raise ValueError("Dataset %s is not recognized" % (opts.dataset))

    dataloaders = {'train': trn_loader, 'val': val_loader}
    
    return dataloaders