'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import rasterio as rio
import glob, os
from skimage import io
import numpy as np
import tifffile as tiff
from skimage.transform import resize

mean=[660.5929,812.9481,1080.6552,1398.3968,1662.5913,1899.4804,2061.932,2100.2792,2214.9325,2230.5973,2443.3014,1968.1885],
std=[137.4943,195.3494,241.2698,378.7495,383.0338,449.3187,511.3159,547.6335,563.8937,501.023,624.041,478.9655]


class SolarDataset(Dataset):
    def __init__(self, split_name, opts, transform=None):
        super(SolarDataset, self).__init__()
        assert split_name in ["train", "val"]
        assert opts.dataset.lower() == "solar"
        self.img_files = []
        self.folder_path = None

        if split_name == "train":
            self.folder_path = os.path.join(opts.data_dir, opts.dataset + "/train/")  
        elif split_name == "val":
            self.folder_path = os.path.join(opts.data_dir, opts.dataset + "/val/")  

        assert os.path.exists(self.folder_path)
        self.img_files.extend(glob.glob(os.path.join(self.folder_path + '/images', '*.tif'))) 
        
        self.transform = transform
        self.mask_files = []
        for image_path in self.img_files:
            self.mask_files.append(os.path.join(os.path.dirname(image_path), '../labels/', str(os.path.basename(image_path)).replace('.tif', '.png')))
                
    def __getitem__(self, index):
        img = tiff.imread(self.img_files[index])
        img = np.moveaxis(img, 0, 2).astype(float)
        img = resize(img, (128, 128, 12), anti_aliasing=True)
        label = io.imread(self.mask_files[index])
        label = resize(label, (128, 128), anti_aliasing=True)
        label[label != 0] = 1
        if self.transform is not None:
            img = self.transform(img)
        return (img, label)
         

    def __len__(self):
        return len(self.img_files)
