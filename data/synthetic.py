import sys
import random
from pathlib import Path

import torch
from tqdm import tqdm

import models.base

class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset based on the generative model
    """
    def __init__(self, generative_model, train=True, return_latent:bool=True):
        self.generative_model = generative_model
        self.train = train
        self.return_latent = return_latent
        self.train_set_size = 60800
        self.test_set_size = 1024

        save_path = Path(f"./data/synthetic_dataset/").mkdir(
                                                            parents=True, 
                                                            exist_ok=True)
        self.dataset_path = ("./data/synthetic_dataset/train.pt" if train else
                        "./data/synthetic_dataset/test.pt")
        self.load_dataset()
    
    def __getitem__(self, index: int):
        imgs, control_points = self.dataset

        if self.return_latent:
            return imgs[index].squeeze(0), control_points[index].squeeze(0)
        else:
            return imgs[index].squeeze(0)

    def __len__(self):
        if self.train:
            return 60800
        else:
            return 1024
    
    def load_dataset(self):
        try:
            dataset = torch.load(self.dataset_path)
            self.dataset = dataset['images'], dataset['control_points']
        except FileNotFoundError:
            self.make_dataset()
    
    def make_dataset(self,):
        # make and save the dataset
        imgs_lst, control_points_lst = [], []
        size = self.train_set_size if self.train else self.test_set_size
        for i in tqdm(range(size)):
            img, points = self.generative_model.sample()
            imgs_lst.append(img)
            control_points_lst.append(points)
        dataset = {"images": imgs_lst,
                    "control_points": control_points_lst}
        torch.save(dataset, self.dataset_path)
        self.dataset = imgs_lst, control_points_lst

        
def get_data_loader(generative_model=None, control_points_dim=2, 
                                            batch_size=None,
                                            device=None):
    if generative_model is None:
        gen = models.base.GenerativeModel(control_points_dim=control_points_dim, 
                                            prior_dist='Normal',device=device)
    else:
        gen = generative_model
    
    trn_dataset = SyntheticDataset(gen, train=True)
    tst_dataset = SyntheticDataset(gen, train=False)

    trn_dataloader = torch.utils.data.DataLoader(trn_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,)
    tst_dataloader = torch.utils.data.DataLoader(tst_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,)

    return trn_dataloader, tst_dataloader   