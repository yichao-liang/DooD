'''
Attend, Infer, Repeat-style model
'''
import pdb
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, Laplace, Bernoulli
from einops import rearrange
from kornia.morphology import dilation, erosion

import util

# latent variable tuple
GuideReturn = namedtuple('GuideReturn', ['z_smpl', 'z_lprb', 'z_pms'])

class GenerativeModel(nn.Module):
    def __init__(self, hidden_dim=512, z_dim=8, res=50):
        super().__init__()
        self.z_dim = z_dim
        self.res = res

        # z dist
        self.register_buffer("z_loc", torch.zeros(z_dim))
        self.register_buffer("z_std", torch.ones(z_dim))

        # Decoder
        self.decoder = Decoder(z_dim=z_dim, 
                               img_dim=[1, res, res],
                               hidden_dim=hidden_dim,
                               num_layers=2)

    def z_dist(self, bs=[1, 3]):
        '''(z_what Prior) Batched control points distribution
        Return: dist of
            bs: [bs, max_strks]
            h_c [bs, h_dim]: hidden-states for computing sequential prior dist
            event_shape: [zwhat_dim, 2]
        '''
        loc, std = self.z_loc, self.z_std

        dist =  Independent(
                    Normal(loc, std), reinterpreted_batch_ndims=1).expand(bs)

        assert (dist.event_shape == torch.Size([self.z_dim]) and 
                dist.batch_shape == torch.Size([*bs]))
        return dist
        
    def img_dist(self, latents=None):
        '''Batched `Likelihood distribution` of `image` conditioned on `latent
        parameters`.
        Args:
            latents: [bs, z_dim]
        Return:
            Dist over images: [bs, 1 (channel), H, W]
        '''
        imgs_dist_loc, imgs_dist_std = self.decoder(latents)
        bs = imgs_dist_loc.shape[0]
        dist = Independent(Laplace(imgs_dist_loc, imgs_dist_std), 
                            reinterpreted_batch_ndims=3)
        assert (dist.event_shape == torch.Size([1, self.res, self.res]) and 
                dist.batch_shape == torch.Size([bs]))
        return dist

    def log_prob(self, latents, imgs):
        '''
        Args:
            latents: [bs, z_dim]
            imgs: [bs, 1, res, res]
        Return:
            Joint log probability
        '''
        bs = imgs.shape[:-3]
        # Prior and  Likelihood
        log_prior = self.z_dist(bs=bs).log_prob(latents)
        log_likelihood = self.img_dist(latents=latents).log_prob(imgs)
        return log_prior, log_likelihood

    def sample(self, bs=[1]):
        latents = self.z_dist(bs).rsample()
        imgs = self.img_dist(latents).rsample()
        return imgs, latents

class Guide(nn.Module):
    def __init__(self, img_dim=[1,28,28], z_dim=8, hidden_dim=512):

        super().__init__()
        # Parameters
        self.img_dim = img_dim
        self.img_numel = np.prod(img_dim)
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        # Encoder
        self.what_mlp_in_dim = np.prod(img_dim)
        self.encoder = Encoder(in_dim=self.what_mlp_in_dim,
                                  zwhat_dim=self.z_dim,
                                  hid_dim=self.hidden_dim,
                                  num_layers=2)

    def forward(self, imgs):
        '''
        Args: 
            img: [bs, 1, H, W]
        Returns:
            data: GuideReturn
        '''
        bs = imgs.size(0)

        z_loc, z_std = self.encoder(imgs.view(bs, -1))

        z_loc = z_loc.view([bs, self.z_dim])
        z_std = z_std.view([bs, self.z_dim])
            
        z_post = Independent(Normal(z_loc, z_std), 
                                        reinterpreted_batch_ndims=1)
        assert (z_post.event_shape == torch.Size([self.z_dim]) and
                z_post.batch_shape == torch.Size([bs]))

        # [bs, z_dim, 2] 
        z = z_post.rsample()

        # log_prob(z): [bs, 1]
        # z_pres: [bs, 1]
        z_lprb = z_post.log_prob(z)

        data = GuideReturn(z_smpl=z,
                           z_pms=(z_loc, z_std),
                           z_lprb=z_lprb)    
        return data
        
class Encoder(nn.Module):
    def __init__(self, in_dim=256, zwhat_dim=50, hid_dim=256, num_layers=1):
        super().__init__()
        self.out_dim = zwhat_dim * 2
        self.mlp = util.init_mlp(in_dim=in_dim, 
                                out_dim=self.out_dim,
                                hidden_dim=hid_dim,
                                num_layers=num_layers)
    def forward(self, x):
        # out = constrain_parameter(self.mlp(x), min=.3, max=.7)
        out = self.mlp(x)
        z_what_loc = F.tanh(out[:, 0:(int(self.out_dim/2))])
        z_what_scale = F.softplus(out[:, (int(self.out_dim/2)):]) + 1e-6
        # out = torch.cat([z_what_loc, z_what_scale])
        # out = torch.sigmoid(out)
        # z_what_loc = out[:, 0:(int(self.out_dim/2))]
        # z_what_scale = out[:, (int(self.out_dim/2)):] + 1e-6
        return z_what_loc, z_what_scale

class Decoder(nn.Module):
    def __init__(self, z_dim=8, 
                       img_dim=[1, 50, 50], 
                       hidden_dim=256, 
                       num_layers=3):
        super().__init__()
        self.img_dim = img_dim
        self.z_dim = z_dim
        self.out_dim = np.prod(img_dim)
        self.net = util.init_mlp(in_dim=z_dim,
                                 out_dim=self.out_dim * 2, 
                                 hidden_dim=hidden_dim, 
                                 num_layers=num_layers)
    
    def forward(self, zwhat):
        out = self.net(zwhat)
        out_loc = torch.sigmoid(out[:, :self.out_dim]).view(zwhat.shape[0], *self.img_dim)
        out_std = F.softplus(out[:, self.out_dim:]).view(zwhat.shape[0], *self.img_dim) + 1e-6
        return out_loc, out_std