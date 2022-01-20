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
    def __init__(self, in_dim, hidden_dim, num_layers, maxnorm, strk_tanh):
        super().__init__()
        self.maxnorm = maxnorm
        self.strk_tanh = strk_tanh
        self.seq = util.init_mlp(in_dim=in_dim, 
                                 out_dim=3,
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers)        
        self.seq.linear_modules[-1].weight.data.zero_()
        if self.maxnorm and self.strk_tanh:
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [6,2,2], dtype=torch.float)) # with maxnorm
        elif not self.strk_tanh and not self.maxnorm:
            # without both
            # used when no execution_guided
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [0,20,30], dtype=torch.float)) 
        elif not self.maxnorm and self.strk_tanh:
            # used when execution_guided
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [0,5,0], dtype=torch.float)) # without maxnorm
        else:
            raise NotImplementedError

    def forward(self, h):
        z = self.seq(h)
        # renderer sigma
        sigma = util.constrain_parameter(z[:, 0:1], min=.02, max=.04)

        # stroke slope
        if self.maxnorm:
            strk_slope = util.constrain_parameter(z[:, 1:2], min=.1, max=.9) # maxnorm
        else:
            strk_slope = F.softplus(z[:, 1:2]) + 1e-3 # tanh

        # add slope
        if self.strk_tanh:
            add_slope = util.constrain_parameter(z[:, 2:3], min=.1, max=1.5)
            # add_slope = F.softplus(z[:, 2:3]) + 1e-3
        else:
            add_slope = F.softplus(z[:, 2:3]) + 1e-3

        return sigma, strk_slope, add_slope

class PresWhereMLP(nn.Module):
    """Infer presence and location from RNN hidden state
    """
    def __init__(self, in_dim, z_where_type, z_where_dim, hidden_dim, num_layers,
        constrain_param=False):
        nn.Module.__init__(self)
        self.z_where_dim = z_where_dim
        self.type = z_where_type
        self.seq = util.init_mlp(in_dim=in_dim, 
                                 out_dim=1 + z_where_dim * 2,
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers)        
        self.constrain_param = constrain_param

        if z_where_type == '4_rotate':
            self.seq.linear_modules[-1].weight.data.zero_()
            # [pres,  loc:scale,shift,rot,  std:scale,shift,rot]
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [4, 4,0,0,0, -4,-4,-4,-4], dtype=torch.float)) 
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
        z_pres_p = util.constrain_parameter(z_pres_p, min=1e-12, max=1-(1e-12))

        z_where_std = z[:, self.z_where_dim+1:] # '4' 5:9; '3': 4:7
        z_where_loc = z[:, 1:self.z_where_dim+1] # '4' 1:5 ; '3' 1:4
        if self.constrain_param:
            z_where_loc = constrain_z_where(z_where_type=self.type,
                                            z_where_loc=z_where_loc)
        else:
            z_where_loc = constrain_z_where(z_where_type=self.type,
                                            z_where_loc=z_where_loc)
        z_where_std = util.constrain_parameter(z_where_std, min=1e-6, max=1)
        # z_where_std = util.constrain_parameter(z_where_std, min=0.01, max=1)

        return z_pres_p, z_where_loc, z_where_std

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
        # z_what_loc = torch.sigmoid(out[:, 0:(int(self.out_dim/2))])
        # z_what_scale = F.softplus(out[:, (int(self.out_dim/2)):]) + 1e-6

        z_what_loc = out[:, 0:(int(self.out_dim/2))]
        if self.constrain_param:
            z_what_loc = constrain_z_what(z_what_loc)
        else:
            z_what_loc = constrain_z_what(z_what_loc)

        z_what_std = out[:, (int(self.out_dim/2)):]
        z_what_std = torch.sigmoid(z_what_std) + 1e-6
        # z_what_std = util.constrain_parameter(z_what_std, min=0.01, max=1) 
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

# class WhatPriorMLP(nn.Module):
#     def __init__(self, in_dim=256, pts_per_strk=5, hid_dim=256, num_layers=1,
#     constrain_param=False):
#         super().__init__()
#         self.out_dim = pts_per_strk * 2 * 2
#         self.constrain_param = constrain_param
#         self.mlp = util.init_mlp(in_dim=in_dim, 
#                             out_dim=self.out_dim,
#                             hidden_dim=hid_dim,
#                             num_layers=num_layers)
#         self.mlp.linear_modules[-1].weight.data.zero_()
#         self.mlp.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
#                     [0] * pts_per_strk * 2 +
#                     [-1.386] * pts_per_strk * 2, dtype=torch.float)) # sigmoid
#                     # [-1.50777] * pts_per_strk * 2, dtype=torch.float)) # softplus

#     def forward(self, x):
#         # out = constrain_parameter(self.mlp(x), min=.3, max=.7)
#         out = self.mlp(x)
#         # z_what_loc = torch.sigmoid(out[:, 0:(int(self.out_dim/2))])
#         # z_what_scale = F.softplus(out[:, (int(self.out_dim/2)):]) + 1e-6
#         if self.constrain_param:
#             out = torch.sigmoid(out)
#         z_what_loc = out[:, 0:(int(self.out_dim/2))]
#         z_what_scale = out[:, (int(self.out_dim/2)):] + 1e-6
#         return z_what_loc, z_what_scale
# class PresWherePriorMLP(nn.Module):
#     """
#     Infer presence and location from RNN hidden state
#     """
#     def __init__(self, in_dim, z_where_type, z_where_dim, hidden_dim, num_layers,
#         constrain_param=False):
#         nn.Module.__init__(self)
#         self.z_where_dim = z_where_dim
#         self.type = z_where_type
#         self.seq = util.init_mlp(in_dim=in_dim, 
#                                  out_dim=1 + z_where_dim * 2,
#                                  hidden_dim=hidden_dim,
#                                  num_layers=num_layers)                
#         self.constrain_param = constrain_param

#         if z_where_type == '4_rotate':
#             # Initialize the weight/bias with identity transformation
#             self.seq.linear_modules[-1].weight.data.zero_()
#             # [pres, scale_loc, shift_loc, rot_loc, scale_std, shift_std, rot_std
#             #  10  , 4 for normal digits
#             self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
#                 [4, 4,0,0,0, 0,0,0,0], dtype=torch.float)) 
#         elif z_where_type == '3':
#             # Initialize the weight/bias with identity transformation
#             self.seq.linear_modules[-1].weight.data.zero_()
#             # [pres, scale_loc, shift_loc, scale_std, shift_std
#             #  10  , 4 for normal digits
#             self.seq.linear_modules[-1].bias = torch.nn.Parameter(
#                 torch.tensor([4, 4,0,0, 0,0,0], dtype=torch.float))
#         else:
#             raise NotImplementedError

#     def forward(self, h, gen=False):
#         '''
#         Args:
#             gen: whether it's in Generation model or just evaluating priors.
#         '''
#         # todo make capacible with other z_where_types
#         z = self.seq(h)

#         z_pres_p = z[:, :1]
#         z_pres_p = util.constrain_parameter(z_pres_p, min=1e-12, max=1-(1e-12))

#         if self.type == '4_rotate':
#             z_where_scale_loc = z[:, 1:2]
#             z_where_shift_loc = z[:, 2:4]
#             z_where_ang_loc = z[:, 4:5]
#             z_where_std = z[:, 5:9]

#             if self.constrain_param:
#                 z_where_scale_loc = util.constrain_parameter(
#                                         z_where_scale_loc, min=.3, max=1)
#                 z_where_shift_loc = util.constrain_parameter(
#                                         z_where_shift_loc, min=-.7, max=.7)
#                 z_where_ang_loc = util.constrain_parameter(
#                                         z_where_ang_loc, min=-45, max=45)
#                 z_where_std = util.constrain_parameter(
#                                         z_where_std, min=1e-6, max=1)

#             z_where_loc = torch.cat([z_where_scale_loc, 
#                                      z_where_shift_loc, 
#                                      z_where_ang_loc], dim=1)
#         elif self.type == '3':
#             z_where_scale_loc = z[:, 1:2]
#             z_where_shift_loc = z[:, 2:4]
#             z_where_std = z[:, 4:7]

#             if self.constrain_param:
#                 z_where_scale_loc = util.constrain_parameter(
#                                         z_where_scale_loc, min=.3, max=1)
#                 z_where_shift_loc = util.constrain_parameter(
#                                         z_where_shift_loc, min=-.7, max=.7)
#                 z_where_std = util.constrain_parameter(
#                                         z_where_std, min=1e-6, max=1)
#             z_where_loc = torch.cat([z_where_scale_loc, 
#                                             z_where_shift_loc], dim=1)
#         else: 
#             raise NotImplementedError

#         return z_pres_p, z_where_loc, z_where_std