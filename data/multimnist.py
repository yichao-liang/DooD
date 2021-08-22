import sys
import random
from pathlib import Path
from itertools import product

from PIL import Image
import torch
from tqdm import tqdm
import glob
import numpy as np

import util

class MultiMnistDataset(torch.utils.data.Dataset):
    """MultiMnist dataset
    """
    def __init__(self, root, device, keep_classes, transform):


        self.root_path = Path(root)
        self.keep_classes = keep_classes
        self.device = device
        self.transform = transform

        self.load_dataset()
    
    def __getitem__(self, index: int):
        image = torch.tensor(self.data[index], dtype=torch.float).unsqueeze(0)
        image = util.normalize_pixel_values(img=image.unsqueeze(0), 
                                            method='maxnorm').squeeze(0)
        image = self.transform(image)
        return image, 1

    def __len__(self):
        return self.num_data

    
    def load_dataset(self):
        self.data = []
        self.num_data = 0
        for length in [2, 3]:
            for number in self.get_all_numbers(num_digits=length):
                if number in self.keep_classes:
                    target = number
                    sub_path = self.root_path / number / "*"
                    for img_path in glob.glob(str(sub_path)):
                        self.data.append(np.array(pil_loader(img_path)))
                        self.num_data += 1


    def get_all_numbers(self, num_digits):
        '''Get all numbers of digits `num_digits` in their num_digits-digits 
        form. E.g. num_digits=2 results in 00, 01, ... 99.
        '''
        l = range(10)
        for i in product(*[l]*num_digits):
            out = ''
            for j in i:
                out += str(j)
            yield out

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning 
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
        
def get_dataloader(keep_classes = None, 
                    root='./data/MultiDigitMNIST/datasets/multimnist/train/', 
                    batch_size=None,
                    device=None,
                    transform=None,
                    shuffle=True):
    '''Get a dataloader for the MultiMnist dataset
    Args:
        keep_classes [str]: a list of classes that should be kept
            in the dataset. If none then return all classes.
    Return:
        a dataloader
    '''
    
    dataset = MultiMnistDataset(root=root, device=device, 
                                keep_classes=keep_classes,
                                transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle)

    return dataloader