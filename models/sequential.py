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

# latent variable tuple
ZSample = namedtuple("ZSample", "z_pres z_what z_where")
ZLogProb = namedtuple("ZLogProb", "z_pres z_what z_where")
DecoderParam = namedtuple("DecoderParam", "sigma slope")
GuideState = namedtuple('GuideState', 'h z_what_h bl_h z_pres z_where z_what')
GuideReturn = namedtuple('GuideReturn', ['z_smpl', 
                                         'z_lprb', 
                                         'mask_prev',
                                         # 'z_pres_dist', 'z_what_dist','z_where_dist', 
                                         'baseline_value', 
                                         'z_pres_pms',
                                         'decoder_param',
                                         'canvas',
                                         'residual'])

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
            pass
            # gen.control_points_scale = (torch.ones(gen.strks_per_img, 
            #                                     gen.ctrl_pts_per_strk, 2
            #                                 )/5).to(device)
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



class GenerativeModel(nn.Module):
    def __init__(self, max_strks=2, pts_per_strk=5, res=28, z_where_type='3',
                                                    execution_guided=False, 
                                                    transform_z_what=True,
                                                    input_dependent_param=True,
                                                    prior_dist='Independent'):
        super().__init__()
        self.max_strks = max_strks
        self.pts_per_strk = pts_per_strk
        self.execution_guided=execution_guided

        # Prior parameters
        self.prior_dist = prior_dist
        # z_what
        self.register_buffer("pts_loc", torch.zeros(self.pts_per_strk, 2)+.5)
        self.register_buffer("pts_std", torch.ones(self.pts_per_strk, 2)/5)
        # z_pres
        self.register_buffer("z_pres_prob", torch.zeros(self.max_strks)+.5)
        # z_where: default '3'
        # '3': (scale, shift x, y)
        # '4_rotate': (scale, shift x, y, rotate) or 
        # '4_no_rotate': (scale x, y, shift x, y)
        # '5': (scale x, y, shift x, y, rotate)
        self.z_where_type = z_where_type
        z_where_loc, z_where_std, self.z_where_dim = util.init_z_where(
                                                            self.z_where_type)
        self.register_buffer("z_where_loc", z_where_loc.expand(self.max_strks, 
                                                            self.z_where_dim))
        self.register_buffer("z_where_std", z_where_std.expand(self.max_strks,
                                                            self.z_where_dim))
        self.imgs_dist_std = torch.nn.Parameter(torch.ones(1, res, res), 
                                                            requires_grad=True)
        # Image renderer, and its parameters
        self.res = res
        self.bezier = Bezier(res=self.res, steps=500, method='bounded')
        self.norm_pixel_method = 'tanh' # maxnorm or tanh
        self.register_buffer("dilation_kernel", torch.ones(2,2))
        # Whether the sigma for renderer or per-stroke-slope depends on the input
        self.input_dependent_param = input_dependent_param
        if self.input_dependent_param:
            self.sigma, self.tanh_norm_slope_stroke, self.tanh_norm_slope\
                                                            = None, None, None
        else:
            self.sigma = torch.nn.Parameter(torch.tensor(6.), 
                                                            requires_grad=True)
            self.tanh_norm_slope_stroke = torch.nn.Parameter(torch.tensor(6.), 
                                                            requires_grad=True)
            self.tanh_norm_slope = torch.nn.Parameter(torch.tensor(6.), 
                                                            requires_grad=True)
        self.transform_z_what = transform_z_what


    def get_sigma(self): 
        return util.constrain_parameter(self.sigma, min=.01, max=.04)
    def get_tanh_slope(self):
        return util.constrain_parameter(self.tanh_norm_slope, min=.1,max=.7)
    def get_tanh_slope_strk(self):
        return util.constrain_parameter(self.tanh_norm_slope_stroke, min=.1, 
                                                                     max=.7)
    def get_imgs_dist_std(self):
        return F.softplus(self.imgs_dist_std)
        
    def control_points_dist(self, bs=[1, 3]):
        '''(z_what Prior) Batched control points distribution
        Return: dist of
            bs: [bs, max_strks]
            event_shape: [pts_per_strk, 2]
        '''
        dist =  Independent(
            Normal(self.pts_loc, self.pts_std), reinterpreted_batch_ndims=2,
        ).expand(bs)
        assert (dist.event_shape == torch.Size([self.pts_per_strk, 2]) and 
                dist.batch_shape == torch.Size([*bs]))
        return dist
        
    def presence_dist(self, bs=[1, 3]):
        '''(z_pres Prior) Batched presence distribution 
        Return: dist of
            bs [bs]
            event_shape [max_strks]
        '''
        dist = Independent(
            Bernoulli(self.z_pres_prob), reinterpreted_batch_ndims=0,
        ).expand(bs)

        assert (dist.event_shape == torch.Size([]) and 
                dist.batch_shape == torch.Size([*bs]))
        return dist

    def transformation_dist(self, bs=[1, 3]):
        '''(z_where Prior) Batched transformation distribution
        Return: dist of
            bs [bs]
            event_shape [max_strks, z_where_dim (3-5)]
        '''
        dist = Independent(
            Normal(self.z_where_loc, self.z_where_std), 
            reinterpreted_batch_ndims=1,
        ).expand(bs)
        assert (dist.event_shape == torch.Size([self.z_where_dim]) and 
                dist.batch_shape == torch.Size([*bs]))
        return dist

    def img_dist(self, latents=None, canvas=None):
        '''Batched `Likelihood distribution` of `image` conditioned on `latent
        parameters`.
        Args:
            latents: 
                z_pres: [bs, n_strks] 
                z_what: [bs, n_strks, pts_per_strk, 2 (x, y)]
                z_where:[bs, n_strks, z_where_dim]
        Return:
            Dist over images: [bs, 1 (channel), H, W]
        '''
        assert latents is not None or canvas is not None
        if canvas is None:
            imgs_dist_loc = self.renders_imgs(latents)
        else:
            imgs_dist_loc = canvas
        bs = imgs_dist_loc.shape[0]

        # imgs_dist_std = torch.ones_like(imgs_dist_loc) 
        imgs_dist_std = self.get_imgs_dist_std().repeat(bs, 1, 1, 1)
        dist = Independent(Laplace(imgs_dist_loc, imgs_dist_std), 
                            reinterpreted_batch_ndims=3
                        )
        assert (dist.event_shape == torch.Size([1, self.res, self.res]) and 
                dist.batch_shape == torch.Size([bs]))
        return dist

    def renders_imgs(self, latents):
        '''Batched img rendering
        Args:
            latents: 
                z_pres: [bs, n_strks] 
                z_what: [bs, n_strks, pts_per_strk, 2 (x, y)]
                z_where:[bs, n_strks, z_where_dim]
        Return:
            images: [bs, 1 (channel), H, W]
        '''
        z_pres, z_what, z_where = latents
        bs, n_strks = z_pres.shape

        # Get affine matrix: [bs * n_strk, 2, 3]
        z_where_mtrx = util.get_affine_matrix_from_param(
                                    z_where.view(bs*n_strks, -1), 
                                    self.z_where_type)
        
        # todo: transform the pts then render
        if self.transform_z_what:

            z_where = z_where.view(bs*n_strks, -1)
            z_what = z_what.view(bs*n_strks, 
                                            self.pts_per_strk, 2)
            # Using manual
            # # Scale
            # transformed_z_what = z_what*z_where[:, 0:1].unsqueeze(-1)
            # # Rotate
            # rotate_ang = z_where[:, 3:4]
            # first_rot = torch.cat([torch.cos(rotate_ang), 
            #                        -torch.sin(rotate_ang)], dim=1).unsqueeze(1)
            # second_rot = torch.cat([torch.sin(rotate_ang), 
            #                         torch.cos(rotate_ang)], dim=1).unsqueeze(1)
            # transformed_z_what = torch.cat([
            #             (transformed_z_what*first_rot).sum(-1, keepdim=True),
            #             (transformed_z_what*second_rot).sum(-1, keepdim=True)
            #             ], dim=2)
            # # Shift
            # transformed_z_what = transformed_z_what+z_where[:, 1:3].unsqueeze(1)
            # transformed_z_what = transformed_z_what.view(bs, self.n_strks, 
            #                                             self.pts_per_strk, 2)

            # Using Homo
            homo_coord = torch.cat([z_what, 
                                    torch.ones(bs*n_strks, self.pts_per_strk,1)
                                    .to(z_what.device)], dim=2)
            transformed_z_what = torch.matmul(z_where_mtrx,homo_coord.transpose(1,2)
                                                                ).transpose(1,2)
            transformed_z_what = transformed_z_what.view(bs, n_strks, 
                                                            self.pts_per_strk, 2)
            z_what = transformed_z_what

        # Get rendered image: [bs, n_strk, n_channel (1), H, W]
        if self.input_dependent_param:
            sigma = self.sigma
        else:
            sigma = self.get_sigma()

        if self.execution_guided:
            imgs = self.bezier(z_what, sigma=sigma.clone(), keep_strk_dim=True)  
            imgs = imgs * z_pres[:, :, None, None, None].clone()
        else:
            imgs = self.bezier(z_what, sigma=sigma, keep_strk_dim=True)  
            imgs = imgs * z_pres[:, :, None, None, None]

        # reshape image for further processing
        imgs = imgs.view(bs*n_strks, 1, self.res, self.res)

        # todo Transform back. z_where goes from a standardized img to the observed.
        if not self.transform_z_what:
            imgs = util.inverse_spatial_transformation(imgs, z_where_mtrx)

        # For small attention windows
        # imgs = dilation(imgs, self.dilation_kernel, 
        #                                         max_val=1.)

        # max normalized so each image has pixel values [0, 1]
        # size: [bs*n_strk, n_channel (1), H, W]
        imgs = util.normalize_pixel_values(imgs, method="maxnorm",)

        # Change back to [bs, n_strk, n_channel (1), H, W]
        imgs = imgs.view(bs, n_strks, 1, self.res, self.res)

        # Normalize per stroke
        if self.input_dependent_param:
            slope = self.tanh_norm_slope_stroke
        else:
            slope = self.get_tanh_slope_strk()
        imgs = util.normalize_pixel_values(imgs, 
                                method=self.norm_pixel_method,
                                slope=slope)

        # Change to [bs, n_channel (1), H, W] through `sum`
        imgs = imgs.sum(1) 

        if n_strks > 1:
            # only normalize again if there were more then 1 stroke
            if self.input_dependent_param:
                slope = self.tanh_norm_slope
            else:
                slope = self.get_tanh_slope()
            imgs = util.normalize_pixel_values(imgs, 
                            method=self.norm_pixel_method,
                            slope=self.get_tanh_slope())
        
        assert not imgs.isnan().any()
        assert imgs.max() <= 1.
        return imgs

    def renders_glimpses(self, z_what):
        '''Get glimpse reconstruction from z_what control points
        Args:
            z_what: [bs, n_strk, n_strks, 2]
        Return:
            recon: [bs, n_strks, 1, res, res]
        '''
        assert len(z_what.shape) == 4, f"z_what shape: {z_what.shape} isn't right"
        bs, n_strks, n_pts = z_what.shape[:3]
        res = self.res
        # Get rendered image: [bs, n_strk, n_channel (1), H, W]
        recon = self.bezier(z_what, sigma=self.get_sigma(), keep_strk_dim=True)
        recon = recon.view(bs*n_strks, 1, res, res)
        recon = util.normalize_pixel_values(recon, method='maxnorm')
        recon = recon.view(bs, n_strks, 1, res, res)
        if self.input_dependent_param:
            slope = self.tanh_norm_slope_stroke
        else:
            slope = self.get_tanh_slope_strk()

        recon = util.normalize_pixel_values(recon, method='tanh', 
                                            slope=self.get_tanh_slope_strk())
        return recon

    def log_prob(self, latents, imgs, z_pres_mask, canvas, decoder_param=None):
        '''
        Args:
            latents: 
                z_pres: [bs, max_strks] 
                z_what: [bs, max_strks, pts_per_strk, 2 (x, y)]
                z_where:[bs, max_strks, z_where_dim]
            imgs: [bs, 1, res, res]
            z_pres_mask: [bs, max_strks]
            canvas: [bs, 1, res, res] the renders from guide's internal decoder
            decoder_param:
                sigma, slope: [bs, max_strks]
        Return:
            Joint log probability
        '''
        z_pres, z_what, z_where = latents
        shape = imgs.shape[:-3]
        bs = torch.Size([*shape, self.max_strks])

        # assuming z_pres here are in the right format, i.e. no 1s following 0s
        # log_prob output: [bs, max_strokes]
        # z_pres_mask: [bs, max_strokes]
        log_prior =  ZLogProb(
                    z_what=(self.control_points_dist(bs).log_prob(z_what) * 
                                                                z_pres),
                    z_pres=(self.presence_dist(bs).log_prob(z_pres) * 
                                                                z_pres_mask),
                    z_where=(self.transformation_dist(bs).log_prob(z_where) * 
                                                                z_pres),
                    )

        # Likelihood
        # self.sigma = decoder_param.sigma
        # self.tanh_norm_slope_stroke = decoder_param.slope[0]
        log_likelihood = self.img_dist(latents=latents, 
                                       canvas=canvas).log_prob(imgs)
        return log_prior, log_likelihood

    def sample(self, bs=[1]):
        # todo 2: with the guide, z_pres are in the right format, but the sampled 
        # todo 2: ones are not
        # todo although sample is not used at this moment
        raise NotImplementedError("Haven't made sure the sampled z_pres are legal")
        z_pres = self.control_points_dist(bs).sample()
        z_what = self.presence_dist(bs).sample()
        z_where = self.transformation_dist(bs).sample()
        latents = ZSample(z_pres, z_what, z_where)
        imgs = self.img_dist(latents).sample()

        return imgs, latents


class Guide(nn.Module):
    def __init__(self, max_strks=2, pts_per_strk=5, img_dim=[1,28,28],
                                                    hidden_dim=256, 
                                                    z_where_type='3', 
                                                    execution_guided=False,
                                                    exec_guid_type=None,
                                                    transform_z_what=False, 
                                                    input_dependent_param=True,
                                                    prior_dist='Independent'):
        super().__init__()

        # Parameters
        self.max_strks = max_strks
        self.pts_per_strk = pts_per_strk
        self.img_dim = img_dim
        self.img_numel = np.prod(img_dim)
        self.hidden_dim = hidden_dim
        self.z_pres_dim = 1
        self.z_what_dim = self.pts_per_strk * 2
        self.z_where_type = z_where_type
        self.z_where_dim = util.init_z_where(self.z_where_type).dim

        # Internal renderer
        self.execution_guided = execution_guided
        self.exec_guid_type = exec_guid_type
        self.prior_dist = prior_dist
        self.internal_decoder = GenerativeModel(z_where_type=self.z_where_type,
                                                pts_per_strk=self.pts_per_strk,
                                                max_strks=self.max_strks,
                                                res=img_dim[-1],
                                                execution_guided=execution_guided,
                                                transform_z_what=transform_z_what,
                                                input_dependent_param=\
                                                        input_dependent_param,
                                                prior_dist=prior_dist)
        # Inference networks
        # Module 1: front_cnn and style_rnn
        #   image -> `cnn_out_dim`-dim hidden representation
        self.mlp_out_dim = 256
        self.cnn_out_dim = 16928 if self.img_dim[-1] == 50 else 4608
        n_in_channels = 2 if (self.execution_guided and self.exec_guid_type == 
                                                        'canvas') else 1
        self.front_cnn = util.init_cnn(mlp_out_dim=self.mlp_out_dim,
                                       cnn_out_dim=self.cnn_out_dim,
                                       num_mlp_layers=0, # this is important
                                       mlp_hidden_dim=256,
                                       n_in_channels=n_in_channels, 
                                       n_mid_channels=16, 
                                       n_out_channels=32,)
        #   Style RNN:
        #   (image encoding; previous_z) -> `rnn_hid_dim`-dim hidden state
        self.style_rnn_in_dim = (self.mlp_out_dim + 
                                 self.z_pres_dim + 
                                 self.z_where_dim)
        self.style_rnn_hid_dim = 256
        self.style_rnn = torch.nn.GRUCell(self.style_rnn_in_dim, 
                                          self.style_rnn_hid_dim)
        # style_mlp:
        #   rnn hidden state -> (z_pres, z_where dist parameters)
        self.style_mlp = PresWhereMLP(in_dim=self.style_rnn_hid_dim, 
                                             z_where_type=self.z_where_type,
                                             z_where_dim=self.z_where_dim)

        # Module 2: z_what_cnn, z_what_rnn, z_what_mlp
        # stn transformed image -> (`pts_per_strk` control points)
        self.z_what_cnn = util.ConvolutionNetwork(n_in_channels=1, 
                                                  n_mid_channels=8, 
                                                  n_out_channels=12,)
        self.z_what_cnn_out_dim = 6348 if self.img_dim[-1] == 50 else 1728#300
        self.z_what_rnn_hid_dim = 256
        self.z_what_rnn = torch.nn.GRUCell((self.z_what_cnn_out_dim + 
                                            self.z_what_dim), 
                                           self.z_what_rnn_hid_dim)
        self.z_what_mlp = WhatMLP(in_dim=self.z_what_rnn_hid_dim,
                                               pts_per_strk=self.pts_per_strk,
                                               hid_dim=self.z_what_rnn_hid_dim,
                                               num_layers=1)

        # Module 3: Baseline (bl) rnn and regressor
        self.bl_hid_dim = 256
        self.bl_rnn = torch.nn.GRUCell(self.style_rnn_in_dim, self.bl_hid_dim)
        self.bl_regressor = nn.Sequential(
            nn.Linear(self.bl_hid_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
        

    def forward(self, imgs):
        '''
        Args: 
            img: [bs, 1, H, W]
        Returns:
            latents:
                z_pres:
                z_what:
                z_where:
            log_prob:
                log_prob_pres:
                log_prob_what:
                log_prob_where:
            latents_distributions:
                z_pres_dist:
                z_what_dist:
                z_where_dist:
        '''
        bs = imgs.size(0)

        # Init model state for performing inference
        state = GuideState(
            h=torch.zeros(bs, self.style_rnn_hid_dim, device=imgs.device),
            z_what_h=torch.zeros(bs, self.z_what_rnn_hid_dim, 
                                                            device=imgs.device),
            bl_h=torch.zeros(bs, self.bl_hid_dim, device=imgs.device),
            z_pres=torch.ones(bs, 1, device=imgs.device),
            z_where=torch.zeros(bs, self.z_where_dim, device=imgs.device),
            z_what=torch.zeros(bs, self.z_what_dim, device=imgs.device),
        )

        # z samples for each step
        z_pres_smpl = torch.ones(bs, self.max_strks, device=imgs.device)
        z_what_smpl = torch.zeros(bs, self.max_strks, self.pts_per_strk, 
                                                         2, device=imgs.device)
        z_where_smpl = torch.ones(bs, self.max_strks, self.z_where_dim, 
                                                            device=imgs.device)
        # z distribution parameters for each step
        z_pres_pms = torch.ones(bs, self.max_strks, device=imgs.device)
        z_what_pms = torch.zeros(bs, self.max_strks, self.pts_per_strk, 
                                                      2, 2, device=imgs.device)
        z_where_pms = torch.ones(bs, self.max_strks, self.z_where_dim, 
                                                         2, device=imgs.device)
        # z log-prob (lprb) for each step
        z_pres_lprb = torch.zeros(bs, self.max_strks, device=imgs.device)
        z_what_lprb = torch.zeros(bs, self.max_strks, device=imgs.device)
        z_where_lprb = torch.zeros(bs, self.max_strks, device=imgs.device)

        # baseline_value
        baseline_value = torch.zeros(bs, self.max_strks, device=imgs.device)

        # sigma and slope: for rendering
        sigmas = torch.zeros(bs, self.max_strks, device=imgs.device)
        strk_slopes = torch.zeros(bs, self.max_strks, device=imgs.device)

        '''Signal mask for each step
        At time t, `mask_prev` stroes whether prev.z_pres==0, `mask_curr` 
            stores whether current.z_pres==0 after an `inference_step`.
        The first element of `mask_prev` is always 1, while the first element of 
            `mask_curr` depends on the outcome of the `inference_step`.
        `mask_prev` can be used to mask out the KL of z_pres, because the first
            appearence z_pres==0 is also accounted.
        `mask_curr` can be used to mask out the KL of z_what, z_where, and
            reconstruction, because they are ignored since the first z_pres==0
        '''
        mask_prev = torch.ones(bs, self.max_strks, device=imgs.device)

        if self.execution_guided:
            # if exec_guid_type == 'residual', canvas stores the difference
            # if exec_guid_type == 'canvas-so-far', canvas stores the cummulative
            canvas = torch.zeros(bs, *self.img_dim, device=imgs.device)
            if self.exec_guid_type == 'residual':
                residual = torch.zeros(bs, *self.img_dim, device=imgs.device)
            else:
                residual = None
        else:
            canvas, residual = None, None

        for t in range(self.max_strks):
            # following the online example
            mask_prev[:, t] = state.z_pres.squeeze()

            # Do one inference step and save results
            result = self.inference_step(state, imgs, canvas, residual)

            state = result['state']
            assert (state.z_pres.shape == torch.Size([bs, 1]) and
                    state.z_what.shape == 
                            torch.Size([bs, self.pts_per_strk, 2]) and
                    state.z_where.shape == 
                            torch.Size([bs, self.z_where_dim]))
            # z_pres: [bs, 1]
            # z_what: [bs, pts_per_strk, 2];
            # z_where: [bs, z_where_dim]
            z_pres_smpl[:, t] = state.z_pres.squeeze(-1)
            z_what_smpl[:, t] = state.z_what
            z_where_smpl[:, t] = state.z_where

            assert (result['z_pres_pms'].shape == 
                                              torch.Size([bs, 1]) and
                    result['z_what_pms'].shape == 
                        torch.Size([bs, self.pts_per_strk, 2, 2]) and
                    result['z_where_pms'].shape == 
                            torch.Size([bs, self.z_where_dim, 2]))
            z_pres_pms[:, t] = result['z_pres_pms'].squeeze(-1)
            z_what_pms[:, t] = result['z_what_pms']
            z_where_pms[:, t] = result['z_where_pms']

            assert (result['z_pres_lprb'].shape == torch.Size([bs, 1]) and
                    result['z_what_lprb'].shape == torch.Size([bs, 1]) and
                    result['z_where_lprb'].shape == torch.Size([bs, 1]))
            z_pres_lprb[:, t] = result['z_pres_lprb'].squeeze(-1)
            z_what_lprb[:, t] = result['z_what_lprb'].squeeze(-1)
            z_where_lprb[:, t] = result['z_where_lprb'].squeeze(-1)
            baseline_value[:, t] = result['baseline_value'].squeeze(-1)

            sigmas[:, t] = result['sigma'].squeeze(-1)
            strk_slopes[:, t] = result['slope'][0].squeeze(-1)
            # add_slopes is shared across strokes in non-execution-guided models
            add_slopes = result['slope'][1].squeeze(-1)

            # Update the canvas
            if self.execution_guided:
                self.internal_decoder.sigma = sigmas[:, t:t+1]
                self.internal_decoder.tanh_norm_slope_stroke = strk_slopes[:, t:t+1]
                canvas_step = self.internal_decoder.renders_imgs(
                                            (z_pres_smpl[:, t:t+1],
                                                z_what_smpl[:, t:t+1],
                                                z_where_smpl[:, t:t+1]))
                canvas = canvas + canvas_step
                canvas = util.normalize_pixel_values(canvas, 
                            method=self.internal_decoder.norm_pixel_method,
                            slope=add_slopes)
                if self.exec_guid_type == "residual":
                    # compute the residual
                    residual = torch.clamp(imgs - canvas, min=0.)

        # todo 1: init the distributions
        # z_pres_dist = None
        # z_what_dist = None
        # z_where_dist = None

        data = GuideReturn(z_smpl=ZSample(
                                z_pres=z_pres_smpl, 
                                z_what=z_what_smpl,
                                z_where=z_where_smpl),
                                z_pres_pms=z_pres_pms,
                        #    z_what_dist=z_what_dist,
                        #    z_where_dist=z_where_dist, 
                           z_lprb=ZLogProb(
                                z_pres=z_pres_lprb,
                                z_what=z_what_lprb,
                                z_where=z_where_lprb),
                           baseline_value=baseline_value,
                           mask_prev=mask_prev,
                           decoder_param=DecoderParam(
                               sigma=sigmas,
                               slope=(strk_slopes, add_slopes)),
                           canvas=canvas,
                           residual=residual,
                           )    
        return data
        
    def inference_step(self, p_state, imgs, canvas, residual):
        '''Given previous (initial) state and input image, predict the current
        step latent distribution
        Args:
            p_state::GuideState
            imgs [bs, 1, res, res]
            canvas [bs, 1, res, res]
        '''
        bs = imgs.size(0)

        # embed image, Input embedding, previous states, Output rnn encoding hid
        if self.execution_guided:
            if self.exec_guid_type == 'canvas':
            # concat at channel dim
                cnn_embed = self.front_cnn(torch.cat([imgs, canvas], dim=1)
                                                                ).view(bs, -1)
            elif self.exec_guid_type == 'residual':
                cnn_embed = self.front_cnn(residual).view(bs, -1)
            else:
                raise NotImplementedError
        else:
            cnn_embed = self.front_cnn(imgs).view(bs, -1)
        rnn_input = torch.cat([cnn_embed, p_state.z_pres, p_state.z_where], dim=1)

        hid = self.style_rnn(rnn_input, p_state.h)

        # Predict presence and location from h
        z_pres_p, z_where_loc, z_where_scale, sigma, strk_slope, add_slope \
                                                        = self.style_mlp(hid)

        # If previous z_pres is 0, force z_pres to 0
        z_pres_p = z_pres_p * p_state.z_pres

        # Numerical stability
        eps = 1e-12
        z_pres_p = z_pres_p.clamp(min=eps, max=1.0-eps)

        # Sample z_pres
        assert z_pres_p.shape == torch.Size([bs, 1])
        z_pres_post = Independent(Bernoulli(z_pres_p), 
                                        reinterpreted_batch_ndims=1)
        assert (z_pres_post.event_shape == torch.Size([1]) and
                z_pres_post.batch_shape == torch.Size([bs]))
        z_pres = z_pres_post.sample()

        # If previous z_pres is 0, then this z_pres should also be 0.
        # However, this is sampled from a Bernoulli whose probability is at
        # least eps. In the unlucky event that the sample is 1, we force this
        # to 0 as well.
        z_pres = z_pres * p_state.z_pres

        # log prob: log q(z_pres[i] | x, z_{<i}) if z_pres[i-1]=1, else 0
        # Mask with p_state.z_pres instead of z_pres. 
        # Keep if prev == 1, curr == 0 or 1; remove if prev == 0
        z_pres_lprb = z_pres_post.log_prob(z_pres).unsqueeze(-1) * p_state.z_pres
        # z_pres_lprb = z_pres_lprb.squeeze()
        
        # Sample z_where, get log_prob
        assert z_where_loc.shape == torch.Size([bs, self.z_where_dim])
        z_where_post = Independent(Normal(z_where_loc, z_where_scale),
                                        reinterpreted_batch_ndims=1)
        assert (z_where_post.event_shape == torch.Size([self.z_where_dim]) and
                z_where_post.batch_shape == torch.Size([bs]))        
        z_where = z_where_post.rsample()
        z_where_lprb = z_where_post.log_prob(z_where).unsqueeze(-1) * z_pres
        # z_where_lprb = z_where_lprb.squeeze()

        # Get spatial transformed "crop" from input image
        if self.exec_guid_type == 'residual' and False:
            trans_imgs = util.spatial_transform(residual, 
                        util.get_affine_matrix_from_param(z_where, 
                                                z_where_type=self.z_where_type))
        else:
            trans_imgs = util.spatial_transform(imgs, 
                        util.get_affine_matrix_from_param(z_where, 
                                                z_where_type=self.z_where_type))
        
        # Sample z_what, get log_prob
        # [bs, pts_per_strk, 2, 1]
        z_what_rnn_in = torch.cat([self.z_what_cnn(trans_imgs), 
                                   p_state.z_what.view(bs, -1)], dim=1)
        z_what_h = self.z_what_rnn(z_what_rnn_in, p_state.z_what_h)
        z_what_loc, z_what_scale = (
            self.z_what_mlp(z_what_h)
            ).view([bs, self.pts_per_strk, 2, 2]).chunk(2, -1)
            
        # [bs, pts_per_strk, 2]
        z_what_loc = z_what_loc.squeeze(-1)
        z_what_scale = z_what_scale.squeeze(-1)
        z_what_post = Independent(Normal(z_what_loc, z_what_scale), 
                                        reinterpreted_batch_ndims=2)
        assert (z_what_post.event_shape == torch.Size([self.pts_per_strk, 2]) and
                z_what_post.batch_shape == torch.Size([bs]))

        # [bs, pts_per_strk, 2] 
        z_what = z_what_post.rsample()
        # log_prob(z_what): [bs, 1]
        # z_pres: [bs, 1]
        z_what_lprb = z_what_post.log_prob(z_what).unsqueeze(-1) * z_pres
        # z_what_lprb = z_what_lprb.squeeze()

        # Compute baseline for z_pres
        # depending on previous latent variables only
        bl_h = self.bl_rnn(rnn_input.detach(), p_state.bl_h)
        baseline_value = self.bl_regressor(bl_h).squeeze() # shape (B,)
        
        new_state = GuideState(
            z_pres=z_pres,
            z_what=z_what,
            z_where=z_where,
            h=hid,
            z_what_h=z_what_h,
            bl_h=bl_h,
        )
        out = {
            'state': new_state,
            'z_pres_smpl': z_pres,
            'z_what_smpl': z_what,
            'z_where_smpl': z_where,
            'z_pres_pms': z_pres_p,
            'z_what_pms': torch.cat((z_what_loc.unsqueeze(-1), 
                                     z_what_scale.unsqueeze(-1)), dim=-1),
            'z_where_pms': torch.cat((z_where_loc.unsqueeze(-1), 
                                      z_where_scale.unsqueeze(-1)), dim=-1),
            'z_pres_lprb': z_pres_lprb,
            'z_what_lprb': z_what_lprb,
            'z_where_lprb': z_where_lprb,
            'baseline_value': baseline_value,
            'sigma': sigma,
            'slope': (strk_slope, add_slope),
        }
        return out

    def named_parameters(self, prefix='', recurse=True):
        for n, p in super().named_parameters(prefix=prefix, recurse=recurse):
            if n.split(".")[0] != 'internal_decoder':
                yield n, p

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
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1 + z_where_dim * 2 + 3),
                # 1: z_pres + ... + 2 for sigma, strk_slope, add_slope
            )
        if z_where_type == '3':
            # Initialize the weight/bias with identity transformation
            self.seq[4].weight.data.zero_()
            # [pres, scale_loc, shift_loc, scale_std, shift_std
            #  10  , 4 for normal digits
            self.seq[4].bias = torch.nn.Parameter(
                torch.tensor([4,4,0,0,-4,-4,-4], dtype=torch.float))
        elif z_where_type == '4_rotate':
            # Initialize the weight/bias with identity transformation
            self.seq[4].weight.data.zero_()
            # [pres, scale_loc, shift_loc, rot_loc, scale_std, shift_std, rot_std
            #  10  , 4 for normal digits
            self.seq[4].bias = torch.nn.Parameter(torch.tensor(
                [4, 4,0,0,0, -4,-4,-4,-4, 6,1,1], dtype=torch.float)) 
        else:
            raise NotImplementedError
        
    def forward(self, h):
        # todo make capacible with other z_where_types
        z = self.seq(h)
        # z_pres_p = torch.sigmoid(z[:, :1])
        z_pres_p = util.constrain_parameter(z[:, :1], min=0, max=1.)
        if self.type == '3':
            z_where_loc_scale = util.constrain_parameter(z[:, 1:1+1], min=0, max=1)
            z_where_loc_shift = util.constrain_parameter(z[:, 1+1:1+self.z_where_dim], 
                                                                 min=-1, max=1)
            z_where_loc = torch.cat([z_where_loc_scale, z_where_loc_shift], 
                                                                 dim=1)
            # z_where_scale = constrain_parameter(z[:, (1+self.z_where_dim):], 
            #                                                      min=.1, max=.2)
            z_where_scale = F.softplus(z[:, (1+self.z_where_dim):])
        elif self.type == '4_rotate':
            z_where_scale_loc = util.constrain_parameter(z[:, 1:2], min=0, max=1)
            z_where_shift_loc = util.constrain_parameter(z[:, 2:4], min=-1, max=1)
            z_where_ang_loc = util.constrain_parameter(z[:, 4:5], 
                                                                min=-45, max=45)
            z_where_loc = torch.cat(
                [z_where_scale_loc, z_where_shift_loc, z_where_ang_loc], dim=1)
            # z_where_scale = constrain_parameter(z[:, (1+self.z_where_dim):], 
            #                                                      min=.1, max=.2)
            z_where_scale = F.softplus(z[:, 5:9])
            sigma = util.constrain_parameter(z[:, 9:10], min=.02, max=.04)
            strk_slope = util.constrain_parameter(z[:, 10:11], min=.1, max=.9)
            add_slope = util.constrain_parameter(z[:, 11:12], min=.1, max=.9)
            return z_pres_p, z_where_loc, z_where_scale, sigma, strk_slope, add_slope

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
        # z_what_loc = torch.sigmoid(out[:, 0:(self.out_dim/2)])
        # z_what_scale = torch.softplus(out[:, (self.out_dim/2):])
        # out = torch.cat([z_what_loc, z_what_scale])
        out = torch.sigmoid(out)
        return out