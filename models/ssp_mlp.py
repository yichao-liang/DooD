'''MLP modules for the sequential model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import util

def constrain_z_where(z_where_type, z_where_loc, clamp=False):
    '''Constrain z_where mean or sample
    Args:
        clamp: if True, use `torch.clamp`;
               if False, use `util.constrain_parameter.
    '''
    # z_where mean/sample
    #   constrain
    assert z_where_type in ['3', '4_rotate'], "NotImplementedError"
    shape_len = len(z_where_loc.shape)
    assert shape_len == 2 or shape_len == 3

    if clamp: 
        constrain_f = torch.clamp
    else:
        constrain_f = util.constrain_parameter
    # z_where_scale_loc = constrain_f(z_where_loc[:, 0:1], min=.3, max=1)
    z_where_scale_loc = constrain_f(z_where_loc[:, 0:1], min=0.3, max=1)
    z_where_shift_loc = constrain_f(z_where_loc[:, 1:3], min=-.7, max=.7)
    if z_where_type == '4_rotate':
        z_where_ang_loc = constrain_f(z_where_loc[:, 3:4], min=-45, max=45)

    #   concat
    z_where_loc = [z_where_scale_loc, z_where_shift_loc]
    if z_where_type =='4_rotate':
        z_where_loc.append(z_where_ang_loc)
    z_where_loc = torch.cat(z_where_loc, dim=1)

    return z_where_loc

def constrain_z_what(z_what_loc, clamp=False):
    '''Constrain z_what mean or sample
    '''
    if clamp:
        z_what_loc = torch.clamp(z_what_loc, min=0., max=1.)
    else:
        z_what_loc = torch.sigmoid(z_what_loc)
    return z_what_loc

class RendererParamMLP(nn.Module):
    """Predict the render parameters
    """
    def __init__(self, in_dim, hidden_dim, num_layers, maxnorm, sgl_strk_tanh,
                 spline_decoder=True):
        super().__init__()
        self.maxnorm = maxnorm
        self.sgl_strk_tanh = sgl_strk_tanh
        self.spline_decoder=spline_decoder
        self.seq = util.init_mlp(in_dim=in_dim, 
                                 out_dim=3,
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers)        
        self.seq.linear_modules[-1].weight.data.zero_()
        if self.maxnorm and self.sgl_strk_tanh:
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(
                        # works well with no canvas
                        torch.tensor([-2,2,2], dtype=torch.float))
                        # torch.tensor([2,2,2], dtype=torch.float))

        elif not self.sgl_strk_tanh and not self.maxnorm:
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(
                    torch.tensor([0,0,2], dtype=torch.float)) 

        elif not self.maxnorm and self.sgl_strk_tanh:
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [0,1,0], dtype=torch.float)) # without maxnorm
        else:
            raise NotImplementedError

    def forward(self, h):
        z = self.seq(h)
        # renderer sigma
        sigma = util.constrain_parameter(z[:, 0:1], min=.02, max=.04)

        # stroke slope
        if self.maxnorm:
            sgl_strk_slope = util.constrain_parameter(z[:, 1:2], 
                                                      min=.1, max=.9) # maxnorm
        else:
            sgl_strk_slope = F.softplus(z[:, 1:2]) + 1e-3 # tanh

        # add slope
        if self.sgl_strk_tanh:
            # works well with no canvas
            add_slope = util.constrain_parameter(z[:, 2:3], min=.1, max=1.5)
        else:
            add_slope = F.softplus(z[:, 2:3]) + 1e-3

        return sigma, sgl_strk_slope, add_slope

class PresWhereMLP(nn.Module):
    """Infer presence and location from RNN hidden state
    """
    def __init__(self, in_dim, z_where_type, z_where_dim, hidden_dim, num_layers,
        constrain_param=False, spline_decoder=True):
        nn.Module.__init__(self)
        self.z_where_dim = z_where_dim
        self.type = z_where_type
        self.seq = util.init_mlp(in_dim=in_dim, 
                                 out_dim=1 + z_where_dim * 2,
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers)        
        self.constrain_param = constrain_param

        # if spline_decoder:
        if z_where_type == '4_rotate':
            # has minimal constrain
            self.seq.linear_modules[-1].weight.data.zero_()
            # [pres,  loc:scale,shift,rot,  std:scale,shift,rot]
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                # this works normally -> init to all steps; and 
                # works to constrain #steps with β4 -> init to 1 step
                [4, 4,0,0,0, -4,-4,-4,-4], dtype=torch.float)) 
                # works to constrain #steps with β3 -> init to 1 step
                # [2, 4,0,0,0, -4,-4,-4,-4], dtype=torch.float)) 
                # works with β2  -> init to 1 step
                # (init to 2 didn't work)
                # [1, 4,0,0,0, -4,-4,-4,-4], dtype=torch.float)) 

            # AIR constrain
            # self.seq.linear_modules[-1].weight.data.zero_()
            # # [pres,  loc:scale,shift,rot,  std:scale,shift,rot]
            # self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
            #     [4, 1,0,0,0, 1,1,1,1], dtype=torch.float)) 
        elif z_where_type == '3':
            self.seq.linear_modules[-1].weight.data.zero_()
            # [pres, loc:scale,shift, std:scale,shift
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(
                torch.tensor([4, 4,0,0, -4,-4,-4], dtype=torch.float))
        else:
            raise NotImplementedError
    
    def forward(self, h):
        # todo make capacible with other z_where_types
        z = self.seq(h)
        z_pres_p = z[:, :1]
        z_where_std = z[:, self.z_where_dim+1:] # '4' 5:9; '3': 4:7
        z_where_loc = z[:, 1:self.z_where_dim+1] # '4' 1:5 ; '3' 1:4
        
        z_pres_p = util.constrain_parameter(z_pres_p, min=1e-12, max=1-(1e-12))

        # has minimal constrain
        z_where_loc = constrain_z_where(z_where_type=self.type,
                                        z_where_loc=z_where_loc)
        z_where_std = util.constrain_parameter(z_where_std, min=1e-9, max=1)

        return z_pres_p, z_where_loc, z_where_std

class WhereMLP(nn.Module):
    """Infer z_where variable from rnn hidden state
    """
    def __init__(self, in_dim, z_where_type, z_where_dim, hidden_dim, 
                 num_layers):
        super().__init__()
        self.z_where_dim = z_where_dim
        self.type = z_where_type
        self.seq = util.init_mlp(in_dim=in_dim, 
                                 out_dim=z_where_dim * 2,
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers)        
        if z_where_type == '4_rotate':
            # has minimal constrain
            self.seq.linear_modules[-1].weight.data.zero_()
            # [pres,  loc:scale,shift,rot,  std:scale,shift,rot]
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [4,0,0,0, -4,-4,-4,-4], dtype=torch.float)) 
        elif z_where_type == '3':
            self.seq.linear_modules[-1].weight.data.zero_()
            # [pres, loc:scale,shift, std:scale,shift
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(
                torch.tensor([4,0,0, -4,-4,-4], dtype=torch.float))
        else:
            raise NotImplementedError
    def forward(self, h):
        # todo make capacible with other z_where_types
        z = self.seq(h)
        z_where_std = z[:, self.z_where_dim:] 
        z_where_loc = z[:, :self.z_where_dim] 
        # has minimal constrain
        z_where_loc = constrain_z_where(z_where_type=self.type,
                                        z_where_loc=z_where_loc)
        z_where_std = util.constrain_parameter(z_where_std, min=1e-9, max=1)
        return z_where_loc, z_where_std

class PresMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()
        self.seq = util.init_mlp(in_dim=in_dim,
                                 out_dim=1,
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers)
        self.seq.linear_modules[-1].weight.data.zero_()
        # [pres,  loc:scale,shift,rot,  std:scale,shift,rot]
        self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
            # [1], dtype=torch.float)) 
            [4], dtype=torch.float))
    def forward(self, h):
        z = self.seq(h)
        z_pres_p = util.constrain_parameter(z, min=1e-12, max=1-(1e-12))
        return z_pres_p

class WhatMLP(nn.Module):
    def __init__(self, in_dim=256, pts_per_strk=5, hid_dim=256, num_layers=1,
        constrain_param=False):
        super().__init__()
        self.out_dim = pts_per_strk * 2 * 2
        self.constrain_param = constrain_param
        self.mlp = util.init_mlp(in_dim=in_dim, 
                                out_dim=self.out_dim,
                                hidden_dim=hid_dim,
                                num_layers=num_layers)
    def forward(self, x):
        # out = constrain_parameter(self.mlp(x), min=.3, max=.7)
        out = self.mlp(x)
        z_what_loc = out[:, 0:(int(self.out_dim/2))]
        z_what_std = out[:, (int(self.out_dim/2)):]

        z_what_loc = constrain_z_what(z_what_loc)

        # has minimal constrain (no constrain previously)
        z_what_std = torch.sigmoid(z_what_std) + 1e-9

        return z_what_loc, z_what_std


class PresWherePriorMLP(PresWhereMLP):
    def __init__(self, in_dim, z_where_type, z_where_dim, hidden_dim, 
                                            num_layers, constrain_param=False):
        super().__init__(
                            in_dim, 
                            z_where_type, 
                            z_where_dim, 
                            hidden_dim, 
                            num_layers,
                            constrain_param=False
                        )
        if z_where_type == '4_rotate':
            self.seq.linear_modules[-1].weight.data.zero_()
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [4, 4,0,0,0, 0,0,0,0], dtype=torch.float)) 
        elif z_where_type == '3':
            self.seq.linear_modules[-1].weight.data.zero_()
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(
                torch.tensor([4, 4,0,0, 0,0,0], dtype=torch.float))
        else: raise NotImplementedError

class WhatPriorMLP(WhatMLP):
    def __init__(self, in_dim=256, pts_per_strk=5, hid_dim=256, num_layers=1,
    constrain_param=False):

        super().__init__(
                            in_dim=256, 
                            pts_per_strk=5, 
                            hid_dim=256, 
                            num_layers=1,
                            constrain_param=False
                        )
        self.mlp.linear_modules[-1].weight.data.zero_()
        self.mlp.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                    [0] * pts_per_strk * 2 +
                    [-1.386] * pts_per_strk * 2, dtype=torch.float)) # sigmoid
                    # [-1.50777] * pts_per_strk * 2, dtype=torch.float)) # softplus