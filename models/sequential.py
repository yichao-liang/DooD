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
from splinesketch.code.bezier import Bezier

def schedule_model_parameters(gen, guide, iteration, loss, device):
    # if loss == 'l1':
    #     if iteration == 0:
    #         # this loss doesn't use a prior
    #         # .02 works well for 1 stroke
    #         # might need higher for more strokes
    #         gen.sigma = torch.log(torch.tensor(.04))
    #     if iteration == 3e3:
    #         gen.sigma = torch.log(torch.tensor(.03))
    #     if iteration == 5e3:
    #         gen.sigma = torch.log(torch.tensor(.02))
    if loss == 'elbo':
        if iteration == 0:
            # sigma = .04 worked for "1, 7"
            # scale = 1/5 worked well for 1 stroke settings, but smaller than 1/5
            # second update at 100 works for "1, 7" but not for all
            # comment out when sigma is learnable
            # gen.sigma = torch.log(torch.tensor(.04)) 
            # doesn't work well
            gen.control_points_scale = (torch.ones(gen.strks_per_img, 
                                                gen.ctrl_pts_per_strk, 2
                                            )/5).to(device)
        # if iteration == 2e3: 
        #     gen.sigma = torch.log(torch.tensor(.03))
    elif loss == 'nll':
        # working with σ in renderer set to >=.02, σ for image Gaussian <=.2
        if iteration == 0:
            gen.sigma = torch.log(torch.tensor(.02))
            gen.control_points_scale = (torch.ones(
                                    gen.strks_per_img, 
                                    gen.ctrl_pts_per_strk, 2
                                )/5).to(device)

# latent variable tuple
Z = namedtuple("Z", "z_pres z_what z_where")

class GenerativeModel(nn.Module):
    def __init__(self, max_strks=2, pts_per_strk=5):
        self.max_strks = max_strks
        self.pts_per_strk = pts_per_strk


        # Prior parameters
        # z_what
        self.register_buffer("pts_loc", torch.zeros(self.max_strks,
                                                    self.pts_per_strk, 2)+.5)
        self.register_buffer("pts_std", torch.ones(self.max_strks, 
                                                    self.pts_per_strk, 2)/5)
        # z_pres
        self.register_buffer("z_pres_prob", torch.zeros(self.max_strks)+.5)
        # z_where: default '3'
        # '3': (scale, shift x, y)
        # '4_rotate': (scale, shift x, y, rotate) or 
        # '4_no_rotate': (scale x, y, shift x, y)
        # '5': (scale x, y, shift x, y, rotate)
        self.z_where_type = '3'
        z_where_loc, z_where_std, self.z_where_dim = util.init_z_where(
                                                            self.z_where_type)
        self.register_buffer("z_where_loc", z_where_loc.expand(self.max_strks, 
                                                            self.z_where_dim))
        self.register_buffer("z_where_std", z_where_std.expand(self.max_strks,
                                                            self.z_where_dim))
        
        # Image renderer, and its parameters
        self.res = 28
        self.bezier = Bezier(res=self.res, steps=500, method='bounded')
        self.sigma = torch.nn.Parameter(torch.tensor(6.), requires_grad=True)
        self.norm_pixel_method = 'tanh' # maxnorm or tanh
        self.register_buffer("dilation_kernel", torch.ones(2,2))
        self.tanh_norm_slope = torch.nn.Parameter(torch.tensor(6.), 
                                                          requires_grad=True)

    def control_points_dist(self, batch_shape=[1]):
        '''(z_what Prior) Batched control points distribution
        Return: dist of
            batch_shape: [batch_shape]
            event_shape: [max_strks, pts_per_strk, 2]
        '''
        dist =  Independent(
            Normal(self.pts_loc, self.pts_std), reinterpreted_batch_ndims=3,
        ).expand(batch_shape)
        assert (dist.event_shape == torch.Size([self.max_strks, 
                                                self.pts_per_strk, 2]) and 
                dist.batch_shape == torch.Size(batch_shape))
        return dist
        
    def presence_dist(self, batch_shape=[1]):
        '''(z_pres Prior) Batched presence distribution 
        Return: dist of
            batch_shape [batch_shape]
            event_shape [max_strks]
        '''
        dist = Independent(
            Bernoulli(self.z_pres_prob), reinterpreted_batch_ndims=1,
        ).expand(batch_shape)

        assert (dist.event_shape == torch.Size([self.max_strks]) and 
                dist.batch_shape == torch.Size(batch_shape))
        return dist

    def transformation_dist(self, batch_shape=[1]):
        '''(z_where Prior) Batched transformation distribution
        Return: dist of
            batch_shape [batch_shape]
            event_shape [max_strks, z_where_dim (3-5)]
        '''
        dist = Independent(
            Normal(self.z_where_loc, self.z_where_std), 
            reinterpreted_batch_ndims=2
        ).expand(batch_shape)
        assert (dist.event_shape == torch.Size([self.max_strks, 
                                                self.z_where_dim]) and 
                dist.batch_shape == torch.Size(batch_shape))
        return dist

    def img_dist(self, latents):
        '''Batched `Likelihood distribution` of `image` conditioned on `latent
        parameters`.
        Args:
            latents: 
                z_pres: [batch_shape, max_strks] 
                z_what: [batch_shape, max_strks, pts_per_strk, 2 (x, y)]
                z_where:[batch_shape, max_strks, z_where_dim]
        Return:
            Dist over images: [batch_shape, 1 (channel), H, W]
        '''
        z_pres, z_what, z_where = latents
        b_strk_shape = z_pres.shape
        # size: [batch_size*n_strk, 2, 3]
        z_where = util.get_affine_matrix_from_param(
                        z_where.view(np.prod(b_strk_shape)), self.z_where_type)

        # todo: make it stops rendering after the first 0 in each z_pres.
        # [batch_size, n_strk, n_channel (1), H, W]
        imgs_dist_loc = self.bezier(z_what, sigma=util.constrain_parameter(
                                                self.sigma, min=.01, max=.05))  
        imgs_dist_loc = imgs_dist_loc * z_pres.unsqueeze(-1).unsqueeze(-1)

        # max normalized so each image has pixel values [0, 1]
        # size: [batch_size*n_strk, n_channel (1), H, W]
        imgs_dist_loc = util.normalize_pixel_values(imgs_dist_loc.view(
                                np.prod(b_strk_shape), 1, self.res, self.res), 
                                method="maxnorm")

        # Transform back. z_where goes from a standardized img to the observed.
        imgs_dist_loc = util.inverse_stn_transformation(imgs_dist_loc, z_where)

        # Change back to [batch_size, n_strk, n_channel (1), H, W]
        imgs_dist_loc = imgs_dist_loc.view(*b_strk_shape, 1, self.res, self.res)
        # Change to [batch_size, n_channel (1), H, W] through `sum`
        imgs_dist_loc = imgs_dist_loc.sum(1) 

        # tanh normalize
        imgs_dist_loc = util.normalize_pixel_values(imgs_dist_loc, 
                            method=self.norm_pixel_method,
                            slope=util.constrain_parameter(self.tanh_norm_slope,
                                                            min=.1,max=.7))
        
        assert not imgs_dist_loc.isnan().any()
        assert imgs_dist_loc.max() <= 1.

        imgs_dist_std = torch.ones_like(imgs_dist_loc) 
        dist = Independent(Laplace(imgs_dist_loc, imgs_dist_std), 
                            reinterpreted_batch_ndims=3
                        )
        assert (dist.event_shape == torch.Size([1, self.res, self.res]) and 
                dist.batch_shape == torch.Size(b_strk_shape[0]))
        return dist

    def log_prob(self, latents, imgs):
        '''
        Args:
            latents: 
                z_pres: [batch_shape, max_strks] 
                z_what: [batch_shape, max_strks, pts_per_strk, 2 (x, y)]
                z_where:[batch_shape, max_strks, z_where_dim]
            imgs: [batch_shape, 1, res, res]
        Return:
            Joint log probability
        '''
        z_pres, z_what, z_where = latents
        shape = imgs.shape[:-3]

        log_prior = (self.control_points_dist(shape).log_prob(z_what) +
                     self.presence_dist(shape).log_prob(z_pres) +
                     self.transformation_dist(shape).log_prob(z_where)
                    )
        log_likelihood = self.img_dist(latents).log_prob(imgs)
        return log_prior, log_likelihood

    def sample(self, batch_shape=[1]):
        z_pres = self.control_points_dist(batch_shape).sample()
        z_what = self.presence_dist(batch_shape).sample()
        z_where = self.transformation_dist(batch_shape).sample()
        latents = Z(z_pres, z_what, z_where)
        imgs = self.img_dist(latents).sample()

        return imgs, latents


class Guide(nn.Module):
    def __init__(self):
        pass

    def get_control_points_dist(self, imgs):
        pass

    def log_prob(self, imgs, latent):
        pass

    def rsample(self, imgs, samples_shape=[], stn_out=False):
        pass