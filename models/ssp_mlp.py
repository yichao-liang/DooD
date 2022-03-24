'''MLP modules for the sequential model
'''
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Independent, Normal
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal

import util

def constrain_z_where(z_where_type, z_where_loc, clamp=False, more_range=False):
    '''Constrain z_where mean or sample
    Args:
        clamp: if True, use `torch.clamp`;
               if False, use `util.constrain_parameter.
    '''
    # z_where mean/sample
    #   constrain
    assert z_where_type in ['3', '4_rotate', '5'], "NotImplementedError"
    shape_len = len(z_where_loc.shape)
    assert shape_len == 2 or shape_len == 3

    if clamp:
        constrain_f = torch.clamp
    else:
        constrain_f = util.constrain_parameter
    if z_where_type == '5':
        z_where_shift_loc = constrain_f(z_where_loc[:, 0:2], min=-.8, max=.8)
        z_where_scale_loc = constrain_f(z_where_loc[:, 2:4], min=0.25, max=1)
        z_where_ang_loc = constrain_f(z_where_loc[:, 4:5], 
                                                    min=-np.pi/4, max=np.pi/4)
        z_where_loc = [z_where_shift_loc, z_where_scale_loc, z_where_ang_loc]
        z_where_loc = torch.cat(z_where_loc, dim=1)
    else:
        z_where_shift_loc = constrain_f(z_where_loc[:, 0:2], min=-.8, max=.8)
        if more_range:
            z_where_scale_loc = constrain_f(z_where_loc[:, 2:3], min=.25, max=1)
        else:
            z_where_scale_loc = constrain_f(z_where_loc[:, 2:3], min=.25, max=1)

        # concat
        z_where_loc_list = [z_where_shift_loc, z_where_scale_loc]
        if z_where_type =='4_rotate':
            z_where_ang_loc = constrain_f(z_where_loc[:, 3:4], min=-np.pi/4, 
                                                               max=np.pi/4)
            z_where_loc_list.append(z_where_ang_loc)

        z_where_loc = torch.cat(z_where_loc_list, dim=1)

    return z_where_loc

def constrain_z_what(z_what_loc, clamp=False, more_range=False):
    '''Constrain z_what mean or sample
    '''
    if clamp:
        z_what_loc = torch.clamp(z_what_loc, min=-.2, max=1.2)
    else:
        # safe
        if more_range:
            z_what_loc = util.constrain_parameter(z_what_loc, min=-.5, max=1.5)
        else:
            # z_what_loc = util.constrain_parameter(z_what_loc, min=-.2, max=1.2)
            z_what_loc = util.constrain_parameter(z_what_loc, min=-.5, max=1.5)
            # z_what_loc = util.constrain_parameter(z_what_loc, min=0., max=1.)
    return z_what_loc

class RendererParamMLP(nn.Module):
    """Predict the render parameters
    """
    def __init__(self, in_dim, hidden_dim, num_layers, maxnorm, sgl_strk_tanh,
                 spline_decoder=True, dataset=None):
        super().__init__()
        self.maxnorm = maxnorm
        self.sgl_strk_tanh = sgl_strk_tanh
        self.spline_decoder=spline_decoder
        self.seq = util.init_mlp(in_dim=in_dim, 
                                 out_dim=3,
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers)        
        self.seq.linear_modules[-1].weight.data.zero_()

        init_b1, init_b2 = -6, 6
        if dataset in [
                    'Omniglot', 
                    'Quickdraw'
                    ]:
            init_b1, init_b2 = -15, 15
        if self.maxnorm and self.sgl_strk_tanh:
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(
                        # works well with no canvas
                        torch.tensor([init_b1,init_b2,init_b2], 
                        dtype=torch.float))
                        # torch.tensor([2,2,2], dtype=torch.float))

        elif not self.sgl_strk_tanh and not self.maxnorm:
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(
                    torch.tensor([0,0,2], dtype=torch.float)) 

        # elif not self.maxnorm and self.sgl_strk_tanh:
        #     self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
        #         [0,1,0], dtype=torch.float)) # without maxnorm
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


class PresMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dataset=None):
        super().__init__()
        self.seq = util.init_mlp(in_dim=in_dim,
                                 out_dim=1,
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers)
        self.seq.linear_modules[-1].weight.data.zero_()
        # [pres,  loc:scale,shift,rot,  std:scale,shift,rot]
        init_bias = 6
        # if dataset in ['Quickdraw']:
        #     init_bias = 10
        if dataset in ['Omniglot']:
            init_bias = 15
        self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
            [init_bias], dtype=torch.float)) # works for stable models
    
    def forward(self, h):
        z = self.seq(h)
        z_pres_p = util.constrain_parameter(z, min=1e-9, max=1-(1e-9))
        return z_pres_p

class WhereMLP(nn.Module):
    """Infer z_where variable from rnn hidden state
    """
    def __init__(self, in_dim, z_where_type, z_where_dim, hidden_dim, 
                 num_layers, constrain_param=True, dataset=None):
        super().__init__()
        self.z_where_dim = z_where_dim
        self.type = z_where_type
        self.seq = util.init_mlp(in_dim=in_dim, 
                                 out_dim=z_where_dim * 2,
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers)
        self.constrain_param = constrain_param
        self.more_range = False        
        # init_b = 0
        init_b = 6
        if dataset == [
            'Omniglot', 
            # 'Quickdraw'
            ]:
            init_b = 15
        # has minimal constrain
        self.seq.linear_modules[-1].weight.data.zero_()
        if z_where_type == '4_rotate':
            # [loc:shift,scale,rot,  std:scale,shift,rot]
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                # init at 0, 0, 1, 0 after the sigmoid
                [0,0,init_b,0, 
                 -4,-4,-4,-4], dtype=torch.float)) 
        elif z_where_type == '5':
            # [loc:shift,scale,rot,  std:scale,shift,rot]
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [0,0,init_b,init_b,0, 
                 -4,-4,-4,-4,-4], dtype=torch.float)) 
        elif z_where_type == '3':
            # [loc:shift,scale, std:scale,shift]
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [0,0,init_b, -4,-4,-4], dtype=torch.float))
        else:
            raise NotImplementedError
        
    def forward(self, h):
        # todo make capacible with other z_where_types
        z = self.seq(h)
        z_where_std = z[:, self.z_where_dim:] 
        z_where_loc = z[:, :self.z_where_dim] 
        # has minimal constrain
        if self.constrain_param:
            z_where_loc = constrain_z_where(z_where_type=self.type,
                                        z_where_loc=z_where_loc, 
                                        more_range=self.more_range)
        z_where_std = util.constrain_parameter(z_where_std, min=1e-9, max=1)
        return z_where_loc, z_where_std

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
        elif z_where_type == '5':
            # has minimal constrain
            self.seq.linear_modules[-1].weight.data.zero_()
            # [pres,  loc:scale,shift,rot,  std:scale,shift,rot]
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [4, 4,4,0,0,0, -4,-4,-4,-4,-4], dtype=torch.float)) 
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


class WhatMLP(nn.Module):
    def __init__(self, in_dim=256, pts_per_strk=5, hid_dim=256, num_layers=1,
        constrain_param=False, dataset=None):
        super().__init__()
        self.out_dim = pts_per_strk * 2 * 2
        self.constrain_param = constrain_param
        self.mlp = util.init_mlp(in_dim=in_dim, 
                                out_dim=self.out_dim,
                                hidden_dim=hid_dim,
                                num_layers=num_layers)
        self.more_range = False

    def forward(self, x,):
        # out = constrain_parameter(self.mlp(x), min=.3, max=.7)
        out = self.mlp(x)
        z_what_loc = out[:, 0:(int(self.out_dim/2))]
        z_what_std = out[:, (int(self.out_dim/2)):]

        if self.constrain_param:
            z_what_loc = constrain_z_what(z_what_loc, 
                                          more_range=self.more_range)

        # has minimal constrain (no constrain previously)
        z_what_std = torch.sigmoid(z_what_std) + 1e-9

        return z_what_loc, z_what_std

class PresPriorMLP(PresMLP):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__(in_dim, hidden_dim, num_layers)
        self.seq.linear_modules[-1].weight.data.zero_()
        self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
            [4], dtype=torch.float))

class WherePriorMLP(nn.Module):
    def __init__(self, 
                in_dim, 
                z_where_type, 
                z_where_dim, 
                hidden_dim, 
                num_layers, 
                n_comp):
        super().__init__()
        self.z_where_dim = z_where_dim
        self.type = z_where_type
        self.n_comp = n_comp
        self.more_range = False        

        self.seq = util.init_mlp(in_dim=in_dim,
                                 out_dim=n_comp * (1 + z_where_dim*2),
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers,)

        if z_where_type == '4_rotate':
            self.seq.linear_modules[-1].weight.data.zero_()
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [.1]*n_comp + 
                ([4] + (torch.rand(3)*.6-.3).tolist())*n_comp +
                [0]*4*n_comp, 
                dtype=torch.float)) 
        elif z_where_type == '5':
            self.seq.linear_modules[-1].weight.data.zero_()
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [.1]*n_comp + 
                ([4,4] + (torch.rand(3)*.6-.3).tolist())*n_comp +
                [0]*5*n_comp, 
                dtype=torch.float)) 
        elif z_where_type == '3':
            self.seq.linear_modules[-1].weight.data.zero_()
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [.1]*n_comp + 
                ([4] + (torch.rand(2)*.6-.3).tolist())*n_comp +
                [0]*3*n_comp, 
                dtype=torch.float))
        else: raise NotImplementedError

    def forward(self, h):
        '''
        Return:
            all_loc, all_std [bs, self.n_comp, z_where_dim]
        '''
        z = self.seq(h)
        bs = z.shape[0]

        logits = z[:, :self.n_comp]
        all_loc = z[:, self.n_comp: 
                       self.n_comp + (self.z_where_dim * self.n_comp)]
        all_loc = constrain_z_where(
                            z_where_type=self.type,
                            z_where_loc=all_loc.view(bs * self.n_comp, -1),
                            more_range=self.more_range
                        ).view(bs, self.n_comp, -1)

        all_std = z[:, self.n_comp + (self.z_where_dim * self.n_comp):]
        all_std = util.constrain_parameter(
                            all_std.view(bs * self.n_comp, -1),
                            min=1e-9, max=1
                        ).view(bs, self.n_comp, -1)
        return logits, all_loc, all_std
class PresWherePriorMLP(nn.Module):
    def __init__(self, in_dim, z_where_type, z_where_dim, hidden_dim, 
                                            num_layers, n_comp):
        super().__init__()
        self.z_where_dim = z_where_dim
        self.type = z_where_type
        self.n_comp = n_comp
        self.more_range = False        

        self.seq = util.init_mlp(in_dim=in_dim,
                    # 1 pres + n_comp * (mix_weight + 2 * z_where) + corr
                                 out_dim=1 + n_comp * (1 + z_where_dim*2 + 1),
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers,)

        if z_where_type == '4_rotate':
            self.seq.linear_modules[-1].weight.data.zero_()
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                # pres: 1 after sigmoid + mix_weight
                [4] + [.1]*n_comp + 
                # all means: at 0, 0, 1, 0 after sigmoid
                (torch.tensor([0,0,4,0]*n_comp)+torch.randn(4*n_comp)*.01
                 ).tolist()+
                # std: at .018 after sigmoid + corr: at 0
                [-4] * 4*n_comp + [0] * n_comp, 
                dtype=torch.float)) 
        elif z_where_type == '5':
            raise NotImplementedError
            self.seq.linear_modules[-1].weight.data.zero_()
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [4] + [.1]*n_comp + 
                ([4] + (torch.rand(5*n_comp - 1)*.6-.3).tolist()) +
                # ([4,4] + (torch.rand(3)*.6-.3).tolist())*n_comp +
                [0]*5*n_comp, 
                dtype=torch.float)) 
        elif z_where_type == '3':
            raise NotImplementedError
            self.seq.linear_modules[-1].weight.data.zero_()
            self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                [4] + [.1]*n_comp + 
                # (torch.rand(4*n_comp)*.6-.3).tolist() +
                ([4] + (torch.rand(3*n_comp - 1)*.6-.3).tolist()) +
                [0]*3*n_comp, 
                dtype=torch.float))
        else: raise NotImplementedError
        
    def forward(self, h):
        '''
        logits [bs, n_comp]
        all_loc [bs, n_comp, z_where_dim]
        all_cor [bs, n_comp]
        '''
        z = self.seq(h)
        bs = z.shape[0]

        # z_pres
        z_pres_p = z[:, :1]
        z_pres_p = util.constrain_parameter(z_pres_p, min=1e-12, max=1-(1e-12))
        
        # z_where
        logits = z[:, 1: 1 + self.n_comp]
        all_loc = z[:, 1 + self.n_comp: 
                       1 + self.n_comp + (self.z_where_dim * self.n_comp)]
        all_loc = constrain_z_where(
                            z_where_type=self.type,
                            z_where_loc=all_loc.reshape(bs * self.n_comp, -1),
                            more_range=self.more_range
                        ).view(bs, self.n_comp, -1)

        all_std = z[:, 1 + self.n_comp + (self.z_where_dim * self.n_comp): 
                       -self.n_comp]
        all_std = util.constrain_parameter(
                            all_std.reshape(bs * self.n_comp, -1),
                            min=1e-9, max=1
                        ).view(bs, self.n_comp, -1)
        
        # [bs, n_comp]
        all_cor = util.constrain_parameter(
                            z[:, -self.n_comp:],
                            min=-1, max=1)
        return z_pres_p, logits, all_loc, all_std, all_cor

class WhatPriorMLP(nn.Module):
    def __init__(self, in_dim=256, pts_per_strk=5, hid_dim=256, num_layers=1,
                        constrain_param=False, 
                        n_comp=4):

        super().__init__()
        self.n_comp = n_comp
        self.pts_per_strk = pts_per_strk
        self.more_range = True
        self.seq = util.init_mlp(in_dim=in_dim, 
                                out_dim=
                        # mix_weight + pts loc, std + pts cor
                n_comp + n_comp * pts_per_strk * 2 * 2 + n_comp * pts_per_strk,
                                hidden_dim=hid_dim,
                                num_layers=num_layers)
        self.seq.linear_modules[-1].weight.data.zero_()
        self.seq.linear_modules[-1].bias = torch.nn.Parameter(torch.tensor(
                    # mix_weight
                    [.1] * n_comp +
                    # loc: init at .5 + randn noise after the sigmoid
                    (torch.randn(n_comp*pts_per_strk*2) * .01).tolist() +
                    # std: init at .2 after the sigmoid
                    [-1.386] * n_comp* pts_per_strk * 2 +
                    # cor
                    [0] * n_comp * pts_per_strk, dtype=torch.float)) # sigmoid
                    # [-1.50777] * pts_per_strk * 2, dtype=torch.float)) # softplus

    def forward(self, h):
        '''
        logits = [bs, n_comp]
        all_loc, all_std [bs, n_comp, pts_per_strk * 2]
        all_cor [bs, n_comp, pts_per_strk]
        '''
        z = self.seq(h)
        bs = z.shape[0]

        logits = z[:, :self.n_comp]
        all_loc = constrain_z_what(
                        z[:, self.n_comp: 
                             self.n_comp + self.n_comp * (self.pts_per_strk) * 2
                                    ].reshape(bs * self.n_comp, -1),
                        more_range=True
                    ).view(bs, self.n_comp, -1)
        all_std = torch.sigmoid(z[:, 
                        self.n_comp + self.n_comp * (self.pts_per_strk) * 2:
                        -self.n_comp * self.pts_per_strk
                                    ].reshape(bs * self.n_comp, -1)
                    ).view(bs, self.n_comp, -1) + 1e-9

        all_cor = util.constrain_parameter(
                        z[:, -self.n_comp * self.pts_per_strk:],
                        min=-1, max=1
                    ).view(bs, self.n_comp, -1)
        
        return logits, all_loc, all_std, all_cor

class ImageMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, num_layers):
        super().__init__()
        self.mlp = util.WhatPriorMLP(in_dim=in_dim, 
                                out_dim=out_dim,
                                hidden_dim=hid_dim,
                                num_layers=num_layers)
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        out = self.mlp(x)

        return out

PtRnnState = namedtuple('PtRnnState',
                        'z_what h')
class ControlPointPriorRNN(nn.Module):
    def __init__(self, in_dim, pts_per_strk, hid_dim, n_comp, 
                 correlated_latent=True):
        super().__init__()
        self.hid_dim = hid_dim
        self.rnn = torch.nn.GRUCell(in_dim, hid_dim)
        self.pts_per_strk = pts_per_strk
        self.n_comp = n_comp
        self.correlated_latent = correlated_latent
        self.mlp = WhatPriorMLP(in_dim=hid_dim,
                                 pts_per_strk=1,
                                 hid_dim=hid_dim,
                                 num_layers=1,
                                 n_comp=n_comp)
    
    def forward(self, x):
        '''
        in: [bs, hid_dim] output of the z_what_rnn
        dist
        Return:
            logits = [bs, pts_per_strk, n_comp]
            all_loc, all_std [bs, pts_per_strk, n_comp, 2]
            all_cor [bs, pts_per_strk, n_comp]
        '''
        # init state
        bs = x.shape[0]
        device = x.device
        state = PtRnnState(
                    z_what = torch.zeros(bs, 2).to(device),
                    h = torch.zeros(bs, self.hid_dim).to(device)
                )
        pis, locs, stds, cors = [], [], [], []
        samples = []

        for _ in range(self.pts_per_strk):
            # [bs, in_dim]
            rnn_in = torch.cat([state.z_what, x], dim=-1)
            h = self.rnn(rnn_in, state.h)
            
            # control point dist param
            # [bs, n_comp], [bs, n_comp, 2], _, [bs, n_comp, 1]
            pi, loc, std, cor = self.mlp(h)

            if self.correlated_latent:
                # [bs, n_comp, 2, 2]
                tril = torch.diag_embed(std)
                tril[:, :, 1, 0:1] = cor
                comp = MultivariateNormal(loc, scale_tril=tril)
            else:
                comp = Independent(Normal(loc, std), 
                                        reinterpreted_batch_ndims=1)
            mix = Categorical(logits=pi)
            dist = MixtureSameFamily(mix, comp)

            z_what = dist.sample()
            
            pis.append(pi), locs.append(loc)
            stds.append(std), cors.append(cor)

            state = PtRnnState(
                z_what=z_what,
                h=h,
            )
        
        pi = torch.stack(pis, dim=-1).view(bs, self.n_comp, self.pts_per_strk
                                            ).transpose(1, 2)

        loc = torch.stack(locs, dim=-1).view(bs, self.n_comp, 
                                                self.pts_per_strk, 2
                                            ).transpose(1,2)
        std = torch.stack(stds, dim=-1).view(bs, self.n_comp, 
                                                self.pts_per_strk, 2
                                            ).transpose(1,2)
        cor = torch.stack(cors, dim=-1).view(bs, self.n_comp, self.pts_per_strk
                                            )
        
        return pi, loc, std, cor