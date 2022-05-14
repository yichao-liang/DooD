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
        self.seq.linear_modules[-1].weight.data.zero_()
        # [pres,  loc:scale,shift,rot,  std:scale,shift,rot]
        init_bias = 6
        if dataset in ['KMNIST']:
            init_bias = 4
        if dataset in [
                'Omniglot',
                'Quickdraw'
            ]:
            # init_bias = 15
            # init_bias = 8 # this works well with [.02]+4 comp+rend 1
            # if bzRnn:
            #     init_bias = 7 # works well with [.01]+4 comp+rend1
            #                 # [.01]+20 comp+rend1+bzRnn
            # else:
            # if trans_what:
                init_bias = 7 # ok for [.01]+20comp+[bzrnn,mlp]
            # else:
            #     init_bias = 6.5 # ok for [.01]+20comp+[bzrnn,mlp]
                
        self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
            [init_bias], dtype=torch.float)) # works for stable models
 
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
    def __init__(self, in_dim, z_where_type, z_where_dim, dataset):
        nn.Module.__init__(self)
        self.z_where_dim = z_where_dim
        self.type = z_where_type
        self.seq = util.init_mlp(in_dim=in_dim,
                                 out_dim=1 + z_where_dim * 2,
                                 hidden_dim=256,
                                 num_layers=2)
        # init_bias = 4
        # if dataset == "Omniglot":
        #     init_bias = 20
        # print("pres_bias=", init_bias)
        # self.seq.linear_modules[-1].weight.data.zero_()
        # # [pres,  loc:scale,shift,rot,  std:scale,shift,rot]
        # self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
        #     [init_bias,0,0,1.,0, 0,0,0,0], dtype=torch.float)) # works for stable models

    def forward(self, h):
        # todo make capacible with other z_where_types
        z = self.seq(h)
        z_pres_p = torch.sigmoid(z[:, :1])
        # z_pres_p = util.constrain_parameter(z[:, :1], min=0, max=1.) + 1e-6
        if self.type == '3':
            z_where_shift_loc = z[:, 1:3]
            z_where_scale_loc = F.softplus(z[:, 3:4]) + 1e-12
            z_where_loc = torch.cat([z_where_shift_loc, z_where_scale_loc], 
                                    dim=-1)
            
        elif self.type == '4_rotate':
            z_where_shift_loc = util.constrain_parameter(
                                                    z[:, 1:3], min=-.8, max=.8)
            z_where_scale_loc = F.softplus(z[:, 3:4]) + 1e-6
            z_where_rot_loc = z[:, 4:5]
            z_where_loc = torch.cat([
                                        z_where_shift_loc, 
                                        z_where_scale_loc,
                                        z_where_rot_loc
                                    ], dim=-1)
        else: 
            raise NotImplementedError
        # to avoid 0 in scale
        z_where_scale = F.softplus(z[:, self.z_where_dim+1:])
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