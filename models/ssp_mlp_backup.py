'''MLP modules for the sequential model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import util

class RendererParamMLP(nn.Module):
    """Predict the render parameters
    """
    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()
        self.seq = util.init_mlp(in_dim=in_dim, 
                                 out_dim=3,
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers)        

        self.seq.linear_modules[-1].weight.data.zero_()
        self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
            [6,2,2], dtype=torch.float)) # with maxnorm
            # [-2,10,0], dtype=torch.float)) # without maxnorm

    def forward(self, h):
        z = self.seq(h)
        sigma = util.constrain_parameter(z[:, 0:1], min=.02, max=.04)
        strk_slope = util.constrain_parameter(z[:, 1:2], min=.1, max=.9)
        # strk_slope = F.softplus(z[:, 1:2]) + .1
        add_slope = util.constrain_parameter(z[:, 2:3], min=.1, max=.9)
        return sigma, strk_slope, add_slope

class PresWhereMLP(nn.Module):
    """Infer presence and location from RNN hidden state
    """
    def __init__(self, in_dim, z_where_type, z_where_dim, hidden_dim, num_layers):
        nn.Module.__init__(self)
        self.z_where_dim = z_where_dim
        self.type = z_where_type
        self.seq = util.init_mlp(in_dim=in_dim, 
                                 out_dim=1 + z_where_dim * 2,
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers)        

        if z_where_type == '4_rotate':
            # Initialize the weight/bias with identity transformation
            self.seq.linear_modules[-1].weight.data.zero_()
            # [pres,  scale_loc,shift_loc,rot_loc,  scale_std,shift_std,rot_std]
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [4, 4,0,0,0, -4,-4,-4,-4], dtype=torch.float)) 
        # elif z_where_type == '3':
        #     # Initialize the weight/bias with identity transformation
        #     self.seq.linear_modules[-1].weight.data.zero_()
        #     # [pres, scale_loc, shift_loc, scale_std, shift_std
        #     self.seq.linear_modules[-1].bias = torch.nn.Parameter(
        #         torch.tensor([4,4,0,0,-4,-4,-4], dtype=torch.float))
        else:
            raise NotImplementedError
    
    def forward(self, h):
        # todo make capacible with other z_where_types
        z = self.seq(h)
        # z_pres_p = torch.sigmoid(z[:, :1])
        z_pres_p = util.constrain_parameter(z[:, :1], min=0, max=1.)

        if self.type == '4_rotate':
            z_where_scale_loc = util.constrain_parameter(z[:, 1:2], min=0, 
                                                                        max=1)
            z_where_shift_loc = util.constrain_parameter(z[:, 2:4], min=-1, 
                                                                        max=1)
            z_where_ang_loc = util.constrain_parameter(z[:, 4:5], min=-45, 
                                                                        max=45)
            z_where_loc = torch.cat(
                [z_where_scale_loc, z_where_shift_loc, z_where_ang_loc], dim=1)
            # z_where_scale = F.softplus(z[:, (1+self.z_where_dim):]) + 1e-6
            z_where_scale = util.constrain_parameter(z[:, 5:9], min=1e-6, max=2)

            return z_pres_p, z_where_loc, z_where_scale
        # elif self.type == '3':
        #     z_where_loc_scale = util.constrain_parameter(z[:, 1:1+1], min=0, max=1)
        #     z_where_loc_shift = util.constrain_parameter(z[:, 1+1:1+self.z_where_dim], 
        #                                                          min=-1., max=1.)
        #     z_where_loc = torch.cat([z_where_loc_scale, z_where_loc_shift], 
        #                                                          dim=1)
        #     # z_where_scale = F.softplus(z[:, (1+self.z_where_dim):]) + 1e-6
        #     z_where_scale = torch.sigmoid(z[:, (1+self.z_where_dim):]) + 1e-6
        else: 
            raise NotImplementedError
        return z_pres_p, z_where_loc, z_where_scale
    
class PresWherePriorMLP(nn.Module):
    """
    Infer presence and location from RNN hidden state
    """
    def __init__(self, in_dim, z_where_type, z_where_dim, hidden_dim, num_layers):
        nn.Module.__init__(self)
        self.z_where_dim = z_where_dim
        self.type = z_where_type
        self.seq = util.init_mlp(in_dim=in_dim, 
                                 out_dim=1 + z_where_dim * 2,
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers)                

        if z_where_type == '4_rotate':
            # Initialize the weight/bias with identity transformation
            self.seq.linear_modules[-1].weight.data.zero_()
            # [pres, scale_loc, shift_loc, rot_loc, scale_std, shift_std, rot_std
            #  10  , 4 for normal digits
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [4, 4,0,0,0, 0,0,0,0], dtype=torch.float)) 
        # elif z_where_type == '3':
        #     # Initialize the weight/bias with identity transformation
        #     self.seq.linear_modules[-1].weight.data.zero_()
        #     # [pres, scale_loc, shift_loc, scale_std, shift_std
        #     #  10  , 4 for normal digits
        #     self.seq.linear_modules[-1].bias = torch.nn.Parameter(
        #         torch.tensor([4,4,0,0,-4,-4,-4], dtype=torch.float))
        else:
            raise NotImplementedError

    def forward(self, h, gen=False):
        '''
        Args:
            gen: whether it's in Generation model or just evaluating priors.
        '''
        # todo make capacible with other z_where_types
        z = self.seq(h)
        # z_pres_p = torch.sigmoid(z[:, :1])
        z_pres_p = util.constrain_parameter(z[:, :1], min=0, max=1.)

        if self.type == '4_rotate':
            z_where_scale_loc = util.constrain_parameter(
                                                    z[:, 1:2], min=0, max=1)
            z_where_shift_loc = util.constrain_parameter(
                                                    z[:, 2:4], min=-1, max=1)
            z_where_ang_loc = util.constrain_parameter(
                                                    z[:, 4:5], min=-45, max=45)
            z_where_loc = torch.cat(
                [z_where_scale_loc, z_where_shift_loc, z_where_ang_loc], dim=1)
            z_where_scale = util.constrain_parameter(z[:, 5:9], min=1e-6, max=2)
            # z_where_scale = F.softplus(z[:, 5:9]) + 1e-6
            return z_pres_p, z_where_loc, z_where_scale
        # elif self.type == '3':
        #     z_where_loc_scale = util.constrain_parameter(z[:, 1:2], min=0, max=1)
        #     z_where_loc_shift = util.constrain_parameter(z[:, 2:4], 
        #                                                          min=-1, max=1)
        #     z_where_loc = torch.cat([z_where_loc_scale, z_where_loc_shift], 
        #                                                          dim=1)
        #     # z_where_scale = constrain_parameter(z[:, (1+self.z_where_dim):], 
        #     #                                                      min=.1, max=.2)
        #     z_where_scale = F.softplus(z[:, 4:]) + 1e-6
        else: 
            raise NotImplementedError
        return z_pres_p, z_where_loc, z_where_scale

class WhatMLP(nn.Module):
    def __init__(self, in_dim=256, pts_per_strk=5, hid_dim=256, num_layers=1):
        super().__init__()
        self.out_dim = pts_per_strk * 2 * 2
        self.mlp = util.init_mlp(in_dim=in_dim, 
                                out_dim=self.out_dim,
                                hidden_dim=hid_dim,
                                num_layers=num_layers)
    def forward(self, x):
        # out = constrain_parameter(self.mlp(x), min=.3, max=.7)
        out = self.mlp(x)
        # z_what_loc = torch.sigmoid(out[:, 0:(int(self.out_dim/2))])
        # z_what_scale = F.softplus(out[:, (int(self.out_dim/2)):]) + 1e-6
        out = torch.sigmoid(out)
        z_what_loc = out[:, 0:(int(self.out_dim/2))]
        z_what_scale = out[:, (int(self.out_dim/2)):] + 1e-6
        return z_what_loc, z_what_scale

class WhatPriorMLP(nn.Module):
    def __init__(self, in_dim=256, pts_per_strk=5, hid_dim=256, num_layers=1):
        super().__init__()
        self.out_dim = pts_per_strk * 2 * 2
        self.mlp = util.init_mlp(in_dim=in_dim, 
                            out_dim=self.out_dim,
                            hidden_dim=hid_dim,
                            num_layers=num_layers)
        self.mlp.linear_modules[-1].weight.data.zero_()
        self.mlp.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                    [0] * pts_per_strk * 2 +
                    [-1.386] * pts_per_strk * 2, dtype=torch.float)) # sigmoid
                    # [-1.50777] * pts_per_strk * 2, dtype=torch.float)) # softplus

    def forward(self, x):
        # out = constrain_parameter(self.mlp(x), min=.3, max=.7)
        out = self.mlp(x)
        # z_what_loc = torch.sigmoid(out[:, 0:(int(self.out_dim/2))])
        # z_what_scale = F.softplus(out[:, (int(self.out_dim/2)):]) + 1e-6
        out = torch.sigmoid(out)
        z_what_loc = out[:, 0:(int(self.out_dim/2))]
        z_what_scale = out[:, (int(self.out_dim/2)):] + 1e-6
        return z_what_loc, z_what_scale