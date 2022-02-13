'''MLP modules for the sequential model
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util

class PresMLP(nn.Module):
    """
    Infer presence from RNN hidden state
    """
    def __init__(self, in_dim):
        nn.Module.__init__(self)
        self.seq = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
    
    def forward(self, h):
        # todo make capacible with other z_where_types
        z = self.seq(h)
        z_pres_p = torch.sigmoid(z[:, :1])
        return z_pres_p

class WhereMLP(nn.Module):
    """
    Infer presence and location from RNN hidden state
    """
    def __init__(self, in_dim, z_where_type, z_where_dim):
        nn.Module.__init__(self)
        self.z_where_dim = z_where_dim
        self.type = z_where_type
        self.seq = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, z_where_dim * 2),
            )
    
    def forward(self, h):
        # todo make capacible with other z_where_types
        z = self.seq(h)
        # z_pres_p = util.constrain_parameter(z[:, :1], min=0, max=1.) + 1e-6
        if self.type == '3':
            z_where_loc = z[:, 0:3]
            z_where_scale = F.softplus(z[:, 3:])
        else: 
            raise NotImplementedError
        return z_where_loc, z_where_scale
class PresWhereMLP(nn.Module):
    """
    Infer presence and location from RNN hidden state
    """
    def __init__(self, in_dim, z_where_type, z_where_dim):
        nn.Module.__init__(self)
        self.z_where_dim = z_where_dim
        self.type = z_where_type
        self.seq = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1 + z_where_dim * 2),
            )
    
    def forward(self, h):
        # todo make capacible with other z_where_types
        z = self.seq(h)
        z_pres_p = torch.sigmoid(z[:, :1])
        # z_pres_p = util.constrain_parameter(z[:, :1], min=0, max=1.) + 1e-6
        if self.type == '3':
            z_where_loc = z[:, 1:4]
            z_where_scale = F.softplus(z[:, 4:])
        else: 
            raise NotImplementedError
        return z_pres_p, z_where_loc, z_where_scale
    

class WhatMLP(nn.Module):
    def __init__(self, in_dim=256, z_what_dim=50, hid_dim=512, num_layers=2):
        super().__init__()
        self.z_what_dim = z_what_dim
        self.out_dim = z_what_dim * 2
        self.mlp = util.init_mlp(in_dim=in_dim, 
                                out_dim=self.out_dim,
                                hidden_dim=hid_dim,
                                num_layers=num_layers)
    def forward(self, x):
        out = self.mlp(x)
        z_what_loc = F.tanh(out[:, 0:self.z_what_dim])
        z_what_scale = F.softplus(out[:, self.z_what_dim:]) + 1e-6
        return z_what_loc, z_what_scale

class Decoder(nn.Module):
    def __init__(self, z_what_dim=50, 
                       img_dim=[1, 50, 50], 
                       hidden_dim=256, 
                       num_layers=2,
                       bias=-2.0):
        super().__init__()
        self.img_dim = img_dim
        self.z_what_dim = z_what_dim
        self.out_dim = np.prod(img_dim)
        self.net = util.init_mlp(in_dim=z_what_dim,
                                 out_dim=self.out_dim, 
                                 hidden_dim=hidden_dim, 
                                 num_layers=num_layers)
        self.bias = bias
    
    def forward(self, z_what):
        out = self.net(z_what)
        # the one that works:
        # out_loc = torch.sigmoid(out + self.bias
        #                                 ).view(*z_what.shape[:2], *self.img_dim)

        # exp: unnormalized
        out_loc = F.softplus(out + self.bias
                                        ).view(*z_what.shape[:2], *self.img_dim)
        return out_loc