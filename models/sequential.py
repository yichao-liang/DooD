'''
Attend, Infer, Repeat-style model
Sequential model with spline z_what latent variables
'''
import pdb
from collections import namedtuple

import numpy as np
from numpy import prod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, Laplace, Bernoulli
from einops import rearrange
from kornia.morphology import dilation, erosion

import util
from splinesketch.code.bezier import Bezier
from models.sequential_mlp import *
from models import air_mlp

# latent variable tuple
ZSample = namedtuple("ZSample", "z_pres z_what z_where")
ZLogProb = namedtuple("ZLogProb", "z_pres z_what z_where")
DecoderParam = namedtuple("DecoderParam", "sigma slope")
GuideState = namedtuple('GuideState', 'h_l h_c bl_h z_pres z_where z_what')
GenState = namedtuple('GenState', 'h_l h_c z_pres z_where z_what')
GuideReturn = namedtuple('GuideReturn', ['z_smpl', 
                                         'z_lprb', 
                                         'mask_prev',
                                         # 'z_pres_dist', 'z_what_dist','z_where_dist', 
                                         'baseline_value', 
                                         'z_pms',
                                         'decoder_param',
                                         'canvas',
                                         'residual',
                                         'z_prior',
                                         'hidden_states'])
GenReturn = namedtuple('GenReturn', ['z_smpl',
                                     'canvas'])

def schedule_model_parameters(gen, guide, iteration, loss, device):
    pass


class GenerativeModel(nn.Module):
    def __init__(self, max_strks=2, pts_per_strk=5, res=28, z_where_type='3',
                                                    execution_guided=False, 
                                                    transform_z_what=True,
                                                    input_dependent_param=True,
                                                    prior_dist='Independent',
                                                    hidden_dim=256,
                                                    num_mlp_layers=2,
                                                    maxnorm=True,
                                                    strk_tanh=True,
                                                    constrain_param=True,
                                                    fixed_prior=True,
                                                    spline_decoder=True,
                                                    render_method='bounded',
                                                    intermediate_likelihood=None,
                                                    ):
        super().__init__()
        self.max_strks = max_strks
        self.pts_per_strk = pts_per_strk
        self.execution_guided = execution_guided
        self.maxnorm = maxnorm
        self.strk_tanh = strk_tanh
        self.constrain_param = constrain_param
        self.intr_ll = intermediate_likelihood
        if self.intr_ll == "Geom":
            self.intr_ll_geo_p = torch.nn.Parameter(torch.tensor(-10.), 
                                                            requires_grad=True)

        if self.intr_ll is not None:
            assert self.execution_guided, "intermediate likelihood needs" + \
                                        "execution_guided = True"

        # Prior parameters
        self.prior_dist = prior_dist
        self.z_where_type = z_where_type
        # todo
        self.fixed_prior = fixed_prior
        if prior_dist == 'Independent':
            if self.fixed_prior:
                # z_what
                self.register_buffer("pts_loc", torch.zeros(self.pts_per_strk, 2)+.5)
                self.register_buffer("pts_std", torch.ones(self.pts_per_strk, 2)/5)
                # z_pres
                self.register_buffer("z_pres_prob", torch.zeros(self.max_strks)+.5)
                # z_where: default '3'
                z_where_loc, z_where_std, self.z_where_dim = \
                                            util.init_z_where(self.z_where_type)
                self.register_buffer("z_where_loc", 
                        z_where_loc.expand(self.max_strks, self.z_where_dim))
                self.register_buffer("z_where_std", 
                        z_where_std.expand(self.max_strks, self.z_where_dim))
            else:
                self.pts_loc = torch.nn.Parameter(
                                torch.zeros(self.pts_per_strk, 2)+.5, 
                                requires_grad=True)
                self.pts_std = torch.nn.Parameter(
                                torch.ones(self.pts_per_strk, 2)/5, 
                                requires_grad=True)
                self.z_pres_prob = torch.nn.Parameter(
                                torch.zeros(self.max_strks)+.5, 
                                requires_grad=True)
                z_where_loc, z_where_std, self.z_where_dim = \
                                            util.init_z_where(self.z_where_type)
                self.z_where_loc = torch.nn.Parameter(
                                z_where_loc, requires_grad=True)
                self.z_where_std = torch.nn.Parameter(
                                z_where_std, requires_grad=True)
        elif prior_dist == 'Sequential':
            self.h_dim = hidden_dim
            self.z_where_dim = util.init_z_where(self.z_where_type).dim

            self.gen_style_mlp = PresWherePriorMLP(
                                                in_dim=self.h_dim,
                                                z_where_type=z_where_type,
                                                z_where_dim=self.z_where_dim,
                                                hidden_dim=hidden_dim,
                                                num_layers=num_mlp_layers,
                                                constrain_param=constrain_param)
            # self.renderer_param_mlp = RendererParamMLP(in_dim=self.h_dim,)
            self.gen_zhwat_mlp = WhatPriorMLP(
                                                in_dim=hidden_dim,
                                                pts_per_strk=self.pts_per_strk,
                                                hid_dim=hidden_dim,
                                                num_layers=num_mlp_layers,
                                                constrain_param=constrain_param)
        else:
            raise NotImplementedError
        self.imgs_dist_std = torch.nn.Parameter(torch.ones(1, res, res), 
                                                            requires_grad=True)

        
        # Image renderer, and its parameters
        self.res = res
        # todo
        self.spline_decoder = spline_decoder
        if self.spline_decoder:
            # self.decoder = Bezier(res=self.res, steps=500, method='base')
            self.decoder = Bezier(res=self.res, steps=100, method=render_method)
        else:
            z_what_dim = self.pts_per_strk * 2
            self.decoder = air_mlp.Decoder(z_what_dim=z_what_dim, 
                                img_dim=[1, res, res],
                                hidden_dim=hidden_dim,
                                num_layers=2)
        self.norm_pixel_method = 'tanh' 
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


    def get_intr_ll_geo_p(self):
        return util.constrain_parameter(self.intr_ll_geo_p, min=1e-6, 
                                                            max=1.-1e-6)
    def get_sigma(self): 
        return util.constrain_parameter(self.sigma, min=.01, max=.04)
    def get_tanh_slope(self):
        return util.constrain_parameter(self.tanh_norm_slope, min=.1,max=.7)
    def get_tanh_slope_strk(self):
        return util.constrain_parameter(self.tanh_norm_slope_stroke, min=.1, 
                                                                     max=.7)
    def get_imgs_dist_std(self):
        # return F.softplus(self.imgs_dist_std) + 1e-6
        return util.constrain_parameter(self.imgs_dist_std, min=.01, max=1)
        
    def control_points_dist(self, h_c=None, bs=[1, 3]):
        '''(z_what Prior) Batched control points distribution
        Args: dist of
            bs: [ptcs, bs]
            h_c [ptcs, bs, h_dim]: hidden-states for computing sequential prior 
        Return:
            dist event_shape: [pts_per_strk, 2]
        '''
        if self.prior_dist == "Sequential" and h_c is not None:
            loc, std = self.gen_zhwat_mlp(h_c.view(prod(bs), -1))
            # [bs, pts_per_strk, 2]
            loc = loc.view([*bs, self.pts_per_strk, 2])
            std = std.view([*bs, self.pts_per_strk, 2])

            # if not self.constrain_param:
            #     loc, std = constrain_z_what(loc, std)

        elif self.prior_dist == "Independent":
            loc, std = self.pts_loc.expand(*bs, self.pts_per_strk, 2), \
                       self.pts_std.expand(*bs, self.pts_per_strk, 2)
            if not self.fixed_prior:
                loc = constrain_z_what(loc)
                std = torch.sigmoid(std) + 1e-12
        else:
            raise NotImplementedError

        dist =  Independent(Normal(loc, std), reinterpreted_batch_ndims=2)
        self.z_what_loc = loc
        self.z_what_std = std

        assert (dist.event_shape == torch.Size([self.pts_per_strk, 2]) and 
                dist.batch_shape == torch.Size([*bs]))
        return dist
        
    def presence_dist(self, h_l=None, bs=[1, 3]):
        '''(z_pres Prior) Batched presence distribution 
        Args: dist of
            bs [n_particles, batch_size]
            h_l [n_particlestcs, bs, h_dim]: hidden-states for computing 
            sequential prior dist event_shape [max_strks]
        Return:
            dist: batch_shape=[ptcs, bs]; event_shape=[]
        '''
        if self.prior_dist == "Sequential" and h_l is not None:
            z_pres_p, _, _ = self.gen_style_mlp(h_l.view(prod(bs), -1))
            z_pres_p = z_pres_p.squeeze(-1)
        elif self.prior_dist == "Independent":
            z_pres_p = self.z_pres_prob.expand(*bs)
            if not self.fixed_prior:
                z_pres_p = util.constrain_parameter(z_pres_p, min=1e-12, 
                                                              max=1-(1e-12))
        else:
            raise NotImplementedError
        self.z_pres_p = z_pres_p
        dist = Independent(
            Bernoulli(z_pres_p.view(*bs)), reinterpreted_batch_ndims=0,
        )
        assert (dist.event_shape == torch.Size([]) and 
                dist.batch_shape == torch.Size([*bs]))

        return dist

    def transformation_dist(self, h_l=None, bs=[1, 3]):
        '''(z_where Prior) Batched transformation distribution
        Args:
            bs [ptcs, bs]
            h_l [ptcs, bs, h_dim]: hidden-states for computing sequential prior 
            dist event_shape [max_strks, z_where_dim (3-5)]
        '''
        if self.prior_dist == "Sequential" and h_l is not None:
            _, loc, std = self.gen_style_mlp(h_l.view(prod(bs), -1))
            loc, std = loc.squeeze(-1), std.squeeze(-1)
            # if not self.constrain_param:
            #     loc, std = constrain_z_where(self.z_where_type, loc, std)

            self.z_where_loc = loc
            self.z_where_std = std
        elif self.prior_dist == "Independent":
            loc, std = self.z_where_loc.expand(*bs, self.z_where_dim), \
                       self.z_where_std.expand(*bs, self.z_where_dim)
            if not self.fixed_prior:
                loc = constrain_z_where(z_where_type=self.z_where_type,
                                        z_where_loc=loc.unsqueeze(0))
                std = util.constrain_parameter(std, min=1e-6, max=1)
        else:
            raise NotImplementedError

        dist = Independent(
                        Normal(loc.view(*bs, -1), std.view(*bs, -1)), 
                                            reinterpreted_batch_ndims=1,)
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
            if self.intr_ll is None:
                imgs_dist_loc = canvas
            else:
                # if intermediate likelihood, use all the canvas steps
                imgs_dist_loc = canvas[:, 1:]

        ptcs, bs = shp = imgs_dist_loc.shape[:2]
        # imgs_dist_std = torch.ones_like(imgs_dist_loc) 
        if canvas is None or self.intr_ll is None:
            imgs_dist_std = self.get_imgs_dist_std().repeat(*shp, 1, 1, 1)
        else:
            # [bs, n_canvas, n_channel, res, res]
            imgs_dist_std = self.get_imgs_dist_std().repeat(*shp, 1, 1, 1)

        try:
            dist = Independent(Laplace(imgs_dist_loc, imgs_dist_std), 
                            reinterpreted_batch_ndims=3
                        )
        except ValueError as e:
            print(e, "Invalid scale parameters {imgs_dist_std}")
            breakpoint()

        if canvas is not None and self.intr_ll is not None:
            batch_size = [*shp, self.max_strks]
        else:
            batch_size = [*shp]
        assert (dist.event_shape == torch.Size([1, self.res, self.res]) and 
                dist.batch_shape == torch.Size(batch_size))
        return dist

    def renders_imgs(self, latents):
        '''Batched img rendering
        Args:
            latents: 
                z_pres: [ptcs, bs, n_strks] 
                z_what: [ptcs, bs, n_strks, pts_per_strk, 2 (x, y)]
                z_where:[ptcs, bs, n_strks, z_where_dim]
        Return:
            images: [ptcs, bs, 1 (channel), H, W]
        '''
        z_pres, z_what, z_where = latents
        ptcs, bs, n_strks, pts_per_strk, _ = z_what.shape
        shp = z_pres.shape[:2]
        
        # todo: transform the pts then render
        # if self.transform_z_what:

        #     z_where = z_where.view(bs*n_strks, -1)
        #     z_what = z_what.view(bs*n_strks, 
        #                                     self.pts_per_strk, 2)
        #     transformed_z_what = util.transform_z_what(
        #                             z_what=z_what, 
        #                             z_where=z_where,
        #                             z_where_type=self.z_where_type,
        #                             res=self.res)

        # Get rendered image: [bs, n_strk, n_channel (1), H, W]
        if self.input_dependent_param:
            sigma = self.sigma
        else:
            sigma = self.get_sigma()

        if self.spline_decoder:
            # imgs [bs, n_strk, 1, res, res]
            try:
                imgs = self.decoder(z_what.view(prod(shp), n_strks, pts_per_strk, 2), 
                                            sigma=sigma.view(prod(shp), -1), 
                                            keep_strk_dim=True)  
            except:
                breakpoint()
        else:
            imgs = self.decoder(z_what.view(prod(shp), n_strks, -1))
        try:
            imgs = imgs * z_pres.reshape(prod(shp), -1)[:, :, None, None, None]
        except:
            breakpoint()

        # reshape image for further processing
        imgs = imgs.view(ptcs*bs*n_strks, 1, self.res, self.res)
        if not self.transform_z_what:
            # Get affine matrix: [bs * n_strk, 2, 3]
            z_where_mtrx = util.get_affine_matrix_from_param(
                                    z_where.view(ptcs*bs*n_strks, -1), 
                                    self.z_where_type)
            imgs = util.inverse_spatial_transformation(imgs, z_where_mtrx)

        # For small attention windows
        # imgs = dilation(imgs, self.dilation_kernel, max_val=1.)

        # max normalized so each image has pixel values [0, 1]
        # size: [bs*n_strk, n_channel (1), H, W]
        if self.maxnorm and self.spline_decoder:
            imgs = util.normalize_pixel_values(imgs, method="maxnorm",)

        # Change back to [ptcs*bs, n_strk, n_channel (1), H, W]
        imgs = imgs.view(ptcs*bs, n_strks, 1, self.res, self.res)

        # Normalize per stroke
        if self.input_dependent_param:
            slope = self.tanh_norm_slope_stroke
        else:
            slope = self.get_tanh_slope_strk()

        if self.strk_tanh and self.spline_decoder:
            # output imgs: [prod(shp), n_strks, 1, res, res]
            imgs = util.normalize_pixel_values(imgs, 
                                            method=self.norm_pixel_method,
                                            slope=slope.view(prod(shp), -1))

        # Change to [bs, n_channel (1), H, W] through `sum`
        imgs = imgs.sum(1) 

        if n_strks > 1:
            # only normalize again if there were more then 1 stroke
            # if not self.spline_decoder:
            #     slope = 0.6
            if self.input_dependent_param:
                slope = self.tanh_norm_slope.squeeze().view(prod(shp))
            else:
                slope = self.get_tanh_slope().view(prod(shp))
            imgs = util.normalize_pixel_values(imgs, 
                            method=self.norm_pixel_method,
                            slope=slope).view(ptcs, bs, 1, self.res, self.res)
        
        assert not imgs.isnan().any()
        if self.strk_tanh or n_strks > 1:
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
        if self.spline_decoder:
            recon = self.decoder(z_what, sigma=self.get_sigma(), keep_strk_dim=True)
        else:
            recon = self.decoder(z_what.view(bs, n_strks, -1))
        recon = recon.view(bs*n_strks, 1, res, res)
        if self.maxnorm and self.spline_decoder:
            recon = util.normalize_pixel_values(recon, method='maxnorm')
        recon = recon.view(bs, n_strks, 1, res, res)
        if self.input_dependent_param:
            slope = self.tanh_norm_slope_stroke
        else:
            slope = self.get_tanh_slope_strk()

        if self.strk_tanh and self.spline_decoder:
            recon = util.normalize_pixel_values(recon, method='tanh', 
                                                       slope=slope)
        return recon

    def log_prob(self, latents, imgs, z_pres_mask, canvas, decoder_param=None,
                                                           z_prior=None):
        '''
        Args:
            latents: 
                z_pres: [ptcs, bs, max_strks] 
                z_what: [ptcs, bs, max_strks, pts_per_strk, 2 (x, y)]
                z_where:[ptcs, bs, max_strks, z_where_dim]
            imgs: [bs, 1, res, res]
            z_pres_mask: [bs, max_strks]
            canvas: [bs, 1, res, res] the renders from guide's internal decoder
            decoder_param:
                sigma, slope: [bs, max_strks]
        Return:
            Joint log probability
        '''
        z_pres, z_what, z_where = latents
        ptcs, _ = shape = z_pres.shape[:2]
        img_shape = imgs.shape[-3:]
        imgs = imgs.unsqueeze(0).repeat(ptcs, 1, 1, 1, 1)
        bs = torch.Size([*shape, self.max_strks])

        # assuming z_pres here are in the right format, i.e. no 1s following 0s
        # log_prob output: [bs, max_strokes]
        # z_pres_mask: [bs, max_strokes]
        if z_prior is not None and z_prior[0] is not None:
            log_prior = z_prior
        else:
            log_prior =  ZLogProb(
                    z_pres=(self.presence_dist(bs=bs).log_prob(z_pres) * 
                                                                z_pres_mask),
                    z_what=(self.control_points_dist(bs=bs).log_prob(z_what) * 
                                                                z_pres),
                    z_where=(self.transformation_dist(bs=bs).log_prob(z_where) * 
                                                                z_pres),
                    )

        # Likelihood
        # self.sigma = decoder_param.sigma
        # self.tanh_norm_slope_stroke = decoder_param.slope[0]
        if self.intr_ll:
            imgs = imgs.unsqueeze(1).expand(*bs, *img_shape)
        
        # if self.intr_ll: log_likelihood [bs, max_steps]
        # else: [bs]
        log_likelihood = self.img_dist(latents=latents, 
                                       canvas=canvas).log_prob(imgs)

        if self.intr_ll is not None:
            num_steps = z_pres.sum(1)
            assert log_likelihood.shape == z_pres.shape
            log_likelihood = log_likelihood * z_pres
        if self.intr_ll == 'Mean':
            log_likelihood_ = torch.zeros(shape).to(z_pres.device)
            # Divide by the number of steps if the number is not 0
            log_likelihood_[num_steps != 0] = (
                                    log_likelihood[num_steps != 0] / 
                                    num_steps[num_steps != 0].unsqueeze(1)
                                ).sum(1)
            log_likelihood = log_likelihood_
        elif self.intr_ll == 'Geom':
            weights = util.geom_weights_from_z_pres(z_pres, p=self.get_intr_ll_geo_p())
            log_likelihood = (log_likelihood * weights.exp()).sum(1)

        return log_prior, log_likelihood

    def sample(self, canvas=None, hs=None, z_pms=None, bs=[1], in_img=None, 
                decoder_param=None):
        '''
        Args:
            in_img [bs, 1, res, res]:
                For unconditioned sampling: in_img is some random inputs for the
                    sigma, strk_tanh network;
                For ... (update later)
        Return:
            imgs [bs, 1, res, res]
            latent:
                z_pres [ptcs, bs, strks, ],
                z_what [ptcs, bs, strks, pts, 2]
                z_where [ptcs, bs, strks, 3],
        '''
        self.device = next(self.parameters()).device

        if self.prior_dist == 'Sequential':
            # z samples for each step
            z_pres_smpl = torch.ones(*bs, self.max_strks, device=self.device)
            z_what_smpl = torch.zeros(*bs, self.max_strks, self.pts_per_strk, 
                                                         2, device=self.device)
            z_where_smpl = torch.ones(*bs, self.max_strks, self.z_where_dim, 
                                                            device=self.device)
            if canvas is None:
                canvas = torch.zeros(*bs, 1, self.res, self.res,
                                                            device=self.device)

            if z_pms is not None:
                char_cond_gen = True
            h_l = torch.zeros(*bs, self.h_dim, device=self.device)
            h_c = torch.zeros(*bs, self.h_dim, device=self.device)

            # if latents is not None:
            #     z_pres, z_what, z_where = latents
            # else:
            # todo: the initial states are hard to control
            z_pres = torch.ones(*bs, 1, device=self.device)
            # z_where = util.init_z_where(self.z_where_type)[0].unsqueeze(0
            #             ).expand(bs[0], self.z_where_dim).to(self.device)
            z_where = torch.zeros(*bs, self.z_where_dim, device=self.device)
            z_what = torch.zeros(*bs, self.pts_per_strk, 2, 
                                                        device=self.device)

            state = GenState(h_l=h_l,
                             h_c=h_c,
                             z_pres=z_pres,
                             z_where=z_where,
                             z_what=z_what)

            for t in range(self.max_strks):
                if char_cond_gen:#and t==0:
                    result = self.generation_step(state, canvas, in_img, 
                        z_pms=ZLogProb(
                            z_pres=z_pms.z_pres[:, :, t: t+1].squeeze(0),
                            z_where=z_pms.z_where[:, :, t].squeeze(0),
                            z_what=z_pms.z_what[:, :, t].squeeze(0)),
                        decoder_param=DecoderParam(
                            sigma=decoder_param[0][:, :, t: t+1].squeeze(0),
                            slope=(decoder_param[1][0][:, :, t: t+1].squeeze(0),
                                   decoder_param[1][1][:, :, t: t+1].squeeze(0)))
                               )
                else:
                    result = self.generation_step(state, canvas, in_img)
                state = result['state']

                # if char_cond_gen:
                #     state = GenState(
                #                 h_l=hs[0][:, :, t].squeeze(0),
                #                 h_c=hs[1][:, :, t].squeeze(0),
                #                 z_pres=state.z_pres,
                #                 z_where=state.z_where,
                #                 z_what=state.z_what
                # #                 # z_pres=smpl_latents.z_pres[:, :, t: t+1].squeeze(0),
                # #                 # z_where=smpl_latents.z_where[:, :, t].squeeze(0),
                # #                 # z_what=smpl_latents.z_what[:, :, t].squeeze(0)
                #                     )


                # z_pres: [bs, 1]
                z_pres_smpl[:, t] = state.z_pres.squeeze(-1)
                # z_what: [bs, pts_per_strk, 2];
                z_what_smpl[:, t] = state.z_what
                # z_where: [bs, z_where_dim]
                z_where_smpl[:, t] = state.z_where

                self.sigma = result['sigma']
                self.tanh_norm_slope_stroke = result['slope'][0]
                # [bs]
                add_slope = result['slope'][1].squeeze(-1)

                # z_pres_smpl, etc has shape [bs, ...]. Thus to use renders_imgs
                # we need to unsqueeze(0) for the n_particle dimension
                latents = (z_pres_smpl[:, t:t+1].unsqueeze(0),
                            z_what_smpl[:, t:t+1].unsqueeze(0),
                            z_where_smpl[:, t:t+1].unsqueeze(0))
                canvas_step = self.renders_imgs(latents)
                canvas = canvas + canvas_step
                if not self.spline_decoder:
                    raise NotImplementedError
                canvas = util.normalize_pixel_values(canvas, 
                                                method=self.norm_pixel_method,
                                                slope=add_slope)
            imgs = canvas
        else:
            # todo 2: with the guide, z_pres are in the right format, but the sampled 
            # todo 2: ones are not
            # todo although sample is not used at this moment
            raise NotImplementedError("Haven't made sure the sampled z_pres are legal")
            z_pres_smpl = self.control_points_dist(bs).sample()
            z_what_smpl = self.presence_dist(bs).sample()
            z_where_smpl = self.transformation_dist(bs).sample()
            latents = ZSample(z_pres, z_what, z_where)
            imgs = self.img_dist(latents).sample()

        return GenReturn(z_smpl=ZSample(
                                    z_pres=z_pres_smpl,
                                    z_what=z_what_smpl,
                                    z_where=z_where_smpl,),
                             canvas=canvas)

    def generation_step(self, p_state, canvas, in_img, z_pms=None, 
                                                       decoder_param=None):
        '''Given previous state and input image, predict the next based on prior
        distributions
        Args:
            state::GenState
            canvas [bs, 1, res, res]
            in_img [bs, 1, res, res]
        '''
        bs = canvas.size(0)
        # todo add mod
        canvas_embed = self.img_feature_extractor(canvas).view(bs, -1)
        in_img_embed = self.img_feature_extractor(in_img).view(bs, -1)

        # Sample z_pres and z_where
        rnn_input = torch.cat([canvas_embed, p_state.z_pres, p_state.z_where], 
                               dim=1)
        # todo add mod
        h_l = self.style_rnn(rnn_input, p_state.h_l)

        mlp_in = [h_l, in_img_embed]
        mlp_in = torch.cat(mlp_in, dim=1)

        z_pres_p, z_where_loc, z_where_std = self.gen_style_mlp(h_l)
        sigma, strk_slope, add_slope = self.renderer_param_mlp(mlp_in)
        # [bs, 1]
        z_pres_p = z_pres_p
        z_where_loc = z_where_loc.squeeze(-1)
        z_where_std = z_where_std.squeeze(-1)

        if z_pms is not None:
            z_pres_p = z_pms.z_pres
            sigma, (strk_slope, add_slope) = decoder_param

        z_pres_dist = Independent(
            Bernoulli(z_pres_p), reinterpreted_batch_ndims=1,
        )#.expand(bs)
        assert (z_pres_dist.event_shape == torch.Size([1]) and 
                z_pres_dist.batch_shape == torch.Size([bs]))

        if z_pms is not None:
            z_where_loc = z_pms.z_where[:, :, 0]
            z_where_std = z_pms.z_where[:, :, 1]

        z_where_dist = Independent(
            Normal(z_where_loc, z_where_std), reinterpreted_batch_ndims=1,
        )#.expand(bs)
        assert (z_where_dist.event_shape == torch.Size([self.z_where_dim]) and 
                z_where_dist.batch_shape == torch.Size([bs]))

        z_pres = z_pres_dist.sample()
        z_pres = z_pres * p_state.z_pres
        z_where = z_where_dist.sample()

        # Sample z_what
        z_what_rnn_in = torch.cat([canvas_embed,
                                        p_state.z_what.view(bs, -1)], dim=1)
        h_c = self.z_what_rnn(z_what_rnn_in, p_state.h_c)

        z_what_loc, z_what_std = self.gen_zhwat_mlp(h_c)
        # [bs, pts_per_strk, 2]
        z_what_loc = z_what_loc.view([bs, self.pts_per_strk, 2])
        z_what_std = z_what_std.view([bs, self.pts_per_strk, 2])

        if z_pms is not None:
            z_what_loc = z_pms.z_what[:, :, :, 0]
            z_what_std = z_pms.z_what[:, :, :, 1]

        z_what_dist = Independent(Normal(z_what_loc, z_what_std), 
                                        reinterpreted_batch_ndims=2)
        assert (z_what_dist.event_shape == torch.Size([self.pts_per_strk, 2]) and
                z_what_dist.batch_shape == torch.Size([bs]))
        
        z_what = z_what_dist.sample()

        new_state = GenState(
                    h_l=h_l,
                    h_c=h_c,
                    z_pres=z_pres,
                    z_what=z_what,
                    z_where=z_where)
        
        return {'state': new_state,
                'sigma': sigma,
                'slope': (strk_slope, add_slope)}

    def no_img_dist_named_params(self):
        for n, p in self.named_parameters():
            if n in ['imgs_dist_std', 'decoder.c', 'decoder.d']:
                continue
            else:
                yield n, p
                
    def img_dist_named_params(self):
        for n, p in self.named_parameters():
            if n in ['imgs_dist_std']:
                yield n, p
            else:
                continue
    
    def learnable_named_parameters(self):
        for n, p in self.named_parameters():
            if n in ['decoder.c', 'decoder.d']:
                continue
            else:
                yield n, p

class Guide(nn.Module):
    def __init__(self, max_strks=2, pts_per_strk=5, img_dim=[1,28,28],
                                                hidden_dim=256, 
                                                z_where_type='3', 
                                                execution_guided=False,
                                                exec_guid_type=None,
                                                transform_z_what=False, 
                                                input_dependent_param=True,
                                                prior_dist='Independent',
                                                target_in_pos=None,
                                                feature_extractor_sharing=True,
                                                num_mlp_layers=2,
                                                num_bl_layers=2,
                                                bl_mlp_hid_dim=512,
                                                bl_rnn_hid_dim=256,
                                                maxnorm=True,
                                                strk_tanh=True,
                                                z_what_in_pos=None,
                                                constrain_param=True,
                                                render_method='bounded',
                                                intermediate_likelihood=None,
                                                ):
        '''
        Args:
            intermediate_likelihood:str: [None, 'Mean', 'Geom' (for Geometric 
                distribution like averaging)]
        '''
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
        self.maxnorm = maxnorm
        self.strk_tanh = strk_tanh
        self.z_what_in_pos = z_what_in_pos
        self.constrain_param = constrain_param
        self.intr_ll = intermediate_likelihood
        if self.intr_ll is not None:
            assert execution_guided, "intermediate likelihood needs" + \
                                            "execution_guided = True"

        # Internal renderer
        self.execution_guided = execution_guided
        self.exec_guid_type = exec_guid_type
        self.prior_dist = prior_dist
        self.target_in_pos = target_in_pos
        # if prior_dist == "Independent":
        #     # If Independent, we have the option to input the target img to the
        #     # {style, what}_RNN or the {style, what}_MLP
        #     self.target_in_pos = target_in_pos
        # else:
        #     self.target_in_pos = 'MLP'

        if self.execution_guided or self.prior_dist == 'Sequential':
            self.internal_decoder = GenerativeModel(z_where_type=self.z_where_type,
                                                pts_per_strk=self.pts_per_strk,
                                                max_strks=self.max_strks,
                                                res=img_dim[-1],
                                                execution_guided=execution_guided,
                                                transform_z_what=transform_z_what,
                                                input_dependent_param=\
                                                        input_dependent_param,
                                                prior_dist=prior_dist,
                                                num_mlp_layers=num_mlp_layers,
                                                maxnorm=maxnorm,
                                                strk_tanh=strk_tanh,
                                                constrain_param=constrain_param,
                                                render_method=render_method
                                                )
        # Inference networks
        # Module 1: front_cnn and style_rnn
        #   image -> `cnn_out_dim`-dim hidden representation
        # -> res=50, 33856 when [1, 32, 64]; 16928 when [1, 16, 32]
        # -> res=28, 4608 when
        self.feature_extractor_sharing = feature_extractor_sharing
        self.cnn_out_dim = 16928 if self.img_dim[-1] == 50 else 4608
        self.feature_extractor_out_dim = 256
        self.img_feature_extractor = util.init_cnn(
                                            n_in_channels=1,
                                            n_mid_channels=16,#32, 
                                            n_out_channels=32,#64,
                                            cnn_out_dim=self.cnn_out_dim,
                                            mlp_out_dim=
                                                self.feature_extractor_out_dim,
                                            mlp_hidden_dim=hidden_dim,
                                            num_mlp_layers=1)
        # self.cnn_out_dim = 2500 if self.img_dim[-1] == 50 else 784
        # self.feature_extractor_out_dim = 2500 if self.img_dim[-1] == 50 else 784
        # self.img_feature_extractor = lambda x: torch.reshape(x, (x.shape[0], -1)
        #                                                     )

        # Style RNN:
        #   (image encoding; prev_z) -> `rnn_hid_dim`-dim hidden state
        self.style_rnn_in_dim = self.z_pres_dim + self.z_where_dim
        # Target image
        # if self.prior_dist == 'Independent' and self.target_in_pos == "RNN":
        #     self.style_rnn_in_dim += self.feature_extractor_out_dim
        if self.target_in_pos == 'RNN':
            self.style_rnn_in_dim += self.feature_extractor_out_dim
        # Canvas
        if self.execution_guided:
            self.style_rnn_in_dim += self.feature_extractor_out_dim
        # z_what
        if self.z_what_in_pos == 'z_where_rnn':
            self.style_rnn_in_dim += self.z_what_dim
        # if (self.prior_dist == "Independent" and self.target_in_pos == "RNN" 
        #     and self.execution_guided):
        #     self.style_rnn_in_dim = (self.feature_extractor_out_dim*2 + 
        #                                 self.z_pres_dim + 
        #                                 self.z_where_dim)
        # else:
        #     self.style_rnn_in_dim = (self.feature_extractor_out_dim + 
        #                                 self.z_pres_dim + 
        #                                 self.z_where_dim)
        
        self.style_rnn_hid_dim = 256
        self.style_rnn = torch.nn.GRUCell(self.style_rnn_in_dim, 
                                          self.style_rnn_hid_dim)

        # style_mlp:
        #   rnn hidden state -> (z_pres, z_where dist parameters)
        self.style_mlp_in_dim = self.style_rnn_hid_dim
        # if (self.prior_dist == 'Sequential' or
        #     (self.prior_dist == 'Independent' and self.target_in_pos == 'MLP')
        #    ):
        if self.target_in_pos == 'MLP':
            # Add target_img dim
            self.style_mlp_in_dim += self.feature_extractor_out_dim
        if self.prior_dist == 'Sequential':
            self.style_mlp_in = 'h+target'
        else:
            self.style_mlp_in = 'h+target'
        '''
            if (self.style_mlp_in == 'h+target' or 
                self.style_mlp_in == 'h+residual'):
                self.style_mlp_in_dim = (self.feature_extractor_out_dim + 
                                            self.style_rnn_hid_dim)
            else: raise NotImplementedError
        elif self.prior_dist == 'Independent':
            if self.target_in_pos == 'RNN':
                self.style_mlp_in_dim = self.style_rnn_hid_dim
            elif self.target_in_pos == 'MLP':
                # If we only pass the target to the posterior inf net
                self.style_mlp_in_dim = (self.feature_extractor_out_dim + 
                                            self.style_rnn_hid_dim)
            else: raise NotImplementedError
        else:
            raise NotImplementedError
        '''
        self.style_mlp = PresWhereMLP(in_dim=self.style_mlp_in_dim, 
                                      z_where_type=self.z_where_type,
                                      z_where_dim=self.z_where_dim,
                                      hidden_dim=hidden_dim,
                                      num_layers=num_mlp_layers,
                                      constrain_param=constrain_param
                                      )
        
        self.renderer_param_mlp = RendererParamMLP(
                                      in_dim=self.style_mlp_in_dim,
                                      hidden_dim=hidden_dim,
                                      num_layers=num_mlp_layers,
                                      maxnorm=self.maxnorm,
                                      strk_tanh=self.strk_tanh)

        # Module 2:
        # z_what RNN
        # stn transformed image -> (`pts_per_strk` control points)
        # full model
        #   rnn: canvas + prev.z_what
        #   mlp: target or residual + h
        # air-like
        #   rnn: prev.z_what + target
        #   mlp: h
        self.z_what_rnn_in_dim = self.z_what_dim

        # Target (transformed)
        # if self.prior_dist == 'Independent' and self.target_in_pos == "RNN":
        if self.target_in_pos == "RNN":
            self.z_what_rnn_in_dim += self.feature_extractor_out_dim

        # Canvas
        # If prior_dist == 'Independent' and self.execution_guided then the 
        # canvas is only used for the style net and not z_what net.
        if self.prior_dist == 'Sequential' and self.execution_guided:
            self.z_what_rnn_in_dim += self.feature_extractor_out_dim

        # minus z_what
        if self.z_what_in_pos == 'z_where_rnn':
            self.z_what_rnn_in_dim -= self.z_what_dim
        # self.z_what_rnn_in_dim = self.features_extractor_out_dim
        
        # if self.execution_guided and self.target_in_pos == 'RNN':
        #     self.z_what_rnn_in_dim = (self.feature_extractor_out_dim*2 +
        #                               self.z_what_dim)
        # else:
        #     self.z_what_rnn_in_dim = (self.feature_extractor_out_dim +
        #                               self.z_what_dim)
                        
        # if self.z_what_in_pos == 'z_where_rnn':
        #     self.z_what_rnn_in_dim -= self.z_what_dim

        self.z_what_rnn_hid_dim = hidden_dim
        self.z_what_rnn = torch.nn.GRUCell(self.z_what_rnn_in_dim, 
                                            self.z_what_rnn_hid_dim)
        
        # z_what MLP
        self.what_mlp_in_dim = self.z_what_rnn_hid_dim

        # if (self.prior_dist == 'Sequential' or 
        #     (self.prior_dist == 'Independent' and self.target_in_pos == 'MLP')):
        if self.target_in_pos == 'MLP':
            self.what_mlp_in_dim += self.feature_extractor_out_dim
        self.z_what_mlp = WhatMLP(in_dim=self.what_mlp_in_dim,
                                  pts_per_strk=self.pts_per_strk,
                                  hid_dim=self.z_what_rnn_hid_dim,
                                  num_layers=num_mlp_layers,
                                  constrain_param=constrain_param
                                  )

        # Use a seperate feature extractor for glimpse
        if not self.feature_extractor_sharing:
            self.glimpse_feature_extractor = util.init_cnn(
                                            n_in_channels=1,
                                            n_mid_channels=16,#32, 
                                            n_out_channels=32,#64,
                                            cnn_out_dim=self.cnn_out_dim,
                                            mlp_out_dim=
                                                self.feature_extractor_out_dim,
                                            mlp_hidden_dim=256,
                                            num_mlp_layers=1)

        # Module 3: Baseline (bl) rnn and regressor
        self.bl_hid_dim = bl_rnn_hid_dim
        self.bl_in_dim = (self.feature_extractor_out_dim  + 
                          self.z_pres_dim + 
                          self.z_where_dim +
                          self.z_what_dim)
        if self.execution_guided:
            self.bl_in_dim += self.feature_extractor_out_dim
        # if self.prior_dist == 'Sequential':
        #     self.bl_rnn = torch.nn.GRUCell((self.feature_extractor_out_dim * 2 + 
        #                                     self.z_pres_dim + 
        #                                     self.z_where_dim +
        #                                     self.z_what_dim),
        #                                     self.bl_hid_dim)
        # else:
        #     self.bl_rnn = torch.nn.GRUCell((self.style_rnn_in_dim + 
        #                                     self.z_what_dim),
        #                                     self.bl_hid_dim)
        self.bl_rnn = torch.nn.GRUCell(self.bl_in_dim, self.bl_hid_dim)
        self.bl_regressor = util.init_mlp(in_dim=self.bl_hid_dim,
                                          out_dim=1,
                                          hidden_dim=bl_mlp_hid_dim,
                                          num_layers=num_bl_layers)

    def forward(self, imgs, num_particles=1):
        '''
        Args: 
            img: [bs, 1, H, W]
            num_particles: int
        Returns:
            z_smpl:
                z_pres [ptcs, bs, strks]
                z_what [ptcs, bs, strks, pts, 2]
                z_where [ptcs, bs, strks, 3]
            z_lprb:
                z_pres [ptcs, bs, strks]
                z_what [ptcs, bs, strks, pts, 2]
                z_where [ptcs, bs, strks, 3]
            z_pms:
                z_pres [ptcs, bs, strks]
                z_what [ptcs, bs, strks, pts, 2, 2 for (loc, std)]
                z_where [ptcs, bs, strks, 3, 2 for (loc, std)]
            baseline_value [ptcs, bs, strks]
            decoder_param:
                sigma, strk_slope [ptcs, bs, strks]
                add_slope [ptcs, bs]
            canvas [ptcs, bs, 1, res, res]
            z_prior:
                z_pres [ptcs, bs, strks]
                z_what [ptcs, bs, strks, pts, 2]
                z_where [ptcs, bs, strks, 3]
            h_ls, h_cs: [ptcs, bs, strks, h_dim]
        '''
        bs = imgs.size(0)
        ptcs = num_particles
        shp = [ptcs, bs]
        img_dim = imgs.shape[1:]

        imgs = imgs.unsqueeze(0).repeat(ptcs, 1, 1, 1, 1)
        
        # canvas [bs, max_strk, img_dim]
        (state, baseline_value, mask_prev,
         z_pres_pms, z_where_pms, z_what_pms,
         z_pres_smpl, z_where_smpl, z_what_smpl,
         z_pres_lprb, z_where_lprb, z_what_lprb,
         z_pres_prir, z_where_prir, z_what_prir,
         sigmas, h_cs, h_ls, strk_slopes, add_slopes, 
         canvas, residual) = self.initialize_state(imgs, ptcs)

        for t in range(self.max_strks):
            # following the online example
            # state.z_pres: [ptcs, bs, 1]
            mask_prev[:, :, t] = state.z_pres.squeeze()

            # Do one inference step and save results
            if self.intr_ll is None:
                result = self.inference_step(state, imgs, canvas, residual)
            else:
                # only pass in the most updated canvas
                result = self.inference_step(state, imgs, canvas[:, :, t], 
                                                                residual)
            state = result['state']
            assert (state.z_pres.shape == torch.Size([ptcs, bs, 1]) and
                    state.z_what.shape == 
                            torch.Size([ptcs, bs, self.pts_per_strk, 2]) and
                    state.z_where.shape == 
                            torch.Size([ptcs, bs, self.z_where_dim]))

            # Update and store the information
            # z_pres: [ptcs, bs, 1]
            z_pres_smpl[:, :, t] = state.z_pres.squeeze(-1)
            # z_what: [ptcs * bs, pts_per_strk, 2];
            z_what_smpl[:, :, t] = state.z_what
            # z_where: [ptcs, bs, z_where_dim]
            z_where_smpl[:, :, t] = state.z_where

            assert (result['z_pres_pms'].shape == torch.Size([ptcs, bs, 1])
                and result['z_what_pms'].shape == torch.Size([ptcs, bs, 
                                                    self.pts_per_strk, 2, 2]) 
                and result['z_where_pms'].shape == torch.Size([ptcs, bs, 
                                                    self.z_where_dim, 2]))
            z_pres_pms[:, :, t] = result['z_pres_pms'].squeeze(-1)
            z_what_pms[:, :, t] = result['z_what_pms']
            z_where_pms[:, :, t] = result['z_where_pms']

            assert (result['z_pres_lprb'].shape == torch.Size([ptcs, bs, 1]) and
                    result['z_what_lprb'].shape == torch.Size([ptcs, bs, 1]) and
                    result['z_where_lprb'].shape == torch.Size([ptcs, bs, 1]))
            z_pres_lprb[:, :, t] = result['z_pres_lprb'].squeeze(-1)
            z_what_lprb[:, :, t] = result['z_what_lprb'].squeeze(-1)
            z_where_lprb[:, :, t] = result['z_where_lprb'].squeeze(-1)
            baseline_value[:, :, t] = result['baseline_value'].squeeze(-1)

            sigmas[:, :, t] = result['sigma'].squeeze(-1)
            strk_slopes[:, :, t] = result['slope'][0].squeeze(-1)
            # add_slopes is shared across strokes in non-execution-guided models
            add_slopes[:, :, t] = result['slope'][1].squeeze(-1)

            # Update the canvas
            if self.execution_guided:
                self.internal_decoder.sigma = sigmas[:, :, t:t+1].clone()
                # tanh_slope [ptcs, bs, 1]
                self.internal_decoder.tanh_norm_slope_stroke = \
                                                strk_slopes[:, :, t:t+1]
                # todo
                canvas_step = self.internal_decoder.renders_imgs((
                                                z_pres_smpl[:, :, t:t+1].clone(),
                                                z_what_smpl[:, :, t:t+1],
                                                z_where_smpl[:, :, t:t+1]))
                canvas_step = canvas_step.view(*shp, *img_dim)
                if self.intr_ll is None:
                    canvas = canvas + canvas_step
                    canvas = canvas_so_far = util.normalize_pixel_values(
                                                canvas, 
                                                method='tanh', 
                                                slope=add_slopes[:, :, t])
                else:
                    canvas_so_far = canvas[:, :, t] + canvas_step
                    canvas[:, :, t+1] = util.normalize_pixel_values(
                                                            canvas_so_far, 
                                                method='tanh',
                                                slope=add_slopes[:, :, t:t+1])
                if self.exec_guid_type == "residual":
                    # compute the residual
                    residual = torch.clamp(imgs - canvas_so_far, min=0.)

            # Calculate the prior with the hidden states.
            if self.prior_dist == 'Sequential':
                h_l, h_c = state.h_l, state.h_c
                h_ls[:, :, t], h_cs[:, :, t] = h_l, h_c
                z_pres_prir[:, :, t] = self.internal_decoder.presence_dist(
                                        h_l, [*shp]
                                        ).log_prob(z_pres_smpl[:, :, t].clone()
                                        ) * mask_prev[:, :, t].clone()
                z_where_prir[:, :, t] = self.internal_decoder.transformation_dist(
                                        h_l.clone(), [*shp]
                                        ).log_prob(z_where_smpl[:, :, t].clone()
                                        ) * z_pres_smpl[:, :, t].clone()
                z_what_prir[:, :, t] = self.internal_decoder.control_points_dist(
                                        h_c.clone(), [*shp]
                                        ).log_prob(z_what_smpl[:, :, t].clone()
                                        ) * z_pres_smpl[:, :, t].clone()

        # todo 1: init the distributions which can be returned; can be useful
        data = GuideReturn(z_smpl=ZSample(
                                z_pres=z_pres_smpl, 
                                z_what=z_what_smpl,
                                z_where=z_where_smpl),
                           z_pms=ZLogProb(
                               z_pres=z_pres_pms,
                               z_where=z_where_pms,
                               z_what=z_what_pms),
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
                           z_prior=ZLogProb(
                               z_pres=z_pres_prir,
                               z_what=z_what_prir,
                               z_where=z_where_prir),
                            hidden_states=(h_ls, 
                                           h_cs)
                           )
        return data
        
    def inference_step(self, p_state, imgs, canvas, residual):

        '''Given previous (initial) state and input image, predict the current
        step latent distribution.
        Args:
            p_state::GuideState
            imgs [ptcs, bs, 1, res, res]
            canvas [ptcs, bs, 1, res, res] or None
            residual [ptcs, bs, 1, res, res]
        '''
        ptcs, bs = shp = imgs.shape[:2]
        img_dim = imgs.shape[2:]

        # embed image
        try:
            img_embed = self.img_feature_extractor(imgs.view(prod(shp), *img_dim
                                                            )).view(*shp, -1)
        except:
            breakpoint()
        canvas_embed = self.img_feature_extractor(canvas.view(prod(shp), *img_dim
                                                            )).view(*shp, -1) \
                                                if canvas is not None else None

        # Predict z_pres, z_where from target and canvas
        style_rnn_in, h_l, style_mlp_in = self.get_style_mlp_in(img_embed, 
                                                                canvas_embed, 
                                                                residual, 
                                                                p_state)
        (z_pres, 
        z_where, 
        z_pres_lprb, 
        z_where_lprb, 
        z_pres_p, 
        z_where_pms, 
        sigma, 
        strk_slope, 
        add_slope) = self.get_z_l(style_mlp_in, p_state)

        # Get spatial transformed "crop" from input image
        # imgs [bs, *img_dim]
        # trans_imgs [ptcs * bs, *img_dim]
        trans_imgs = util.spatial_transform(
                                    imgs.view(prod(shp), *img_dim), 
                                    util.get_affine_matrix_from_param(
                                            z_where.view(prod(shp), -1), 
                                            z_where_type=self.z_where_type))
        
        zwhat_mlp_in, h_c = self.get_zwhat_mlp_in(
                    trans_imgs.view(*shp, *img_dim), 
                    canvas_embed.view(*shp, -1) if canvas is not None else None, 
                    p_state)
        z_what, z_what_lprb, z_what_pms = self.get_z_c(zwhat_mlp_in, 
                                                       p_state, 
                                                       z_pres)

        # Compute baseline for z_pres
        # depending on previous latent variables only
        # if self.prior_dist == 'Sequential':
        bl_input = [
                    img_embed.detach().view(prod(shp), -1), 
                    p_state.z_pres.detach().view(prod(shp), -1), 
                    p_state.z_where.detach().view(prod(shp), -1), 
                    p_state.z_what.detach().view(prod(shp), -1),
                    ]
        if self.execution_guided:
            bl_input.append(canvas_embed.detach().view(prod(shp), -1)) 
        bl_input = torch.cat(bl_input, dim=1)
        # else:
        #     bl_input = torch.cat([
        #                         style_rnn_in.detach(),
        #                         p_state.z_what.detach().view(bs, -1)
        #                         ], dim=1)
        bl_h = self.bl_rnn(bl_input, p_state.bl_h.view(prod(shp), -1))
        # bl_h = self.bl_rnn(style_rnn_in.detach(), p_state.bl_h)
        baseline_value = self.bl_regressor(bl_h) # shape (prod(shp),)
        baseline_value = baseline_value.view(*shp, -1) * p_state.z_pres
        
        new_state = GuideState(
            z_pres=z_pres,
            z_what=z_what,
            z_where=z_where,
            h_l=h_l,
            h_c=h_c,
            bl_h=bl_h,
        )
        out = {
            'state': new_state,
            'z_pres_pms': z_pres_p,
            # [bs, pts_per_strk, 2, 2]
            'z_what_pms': torch.cat((z_what_pms[0].unsqueeze(-1), 
                                     z_what_pms[1].unsqueeze(-1)), dim=-1),
            # [bs, z_where_dim, 2]
            'z_where_pms': torch.cat((z_where_pms[0].unsqueeze(-1), 
                                     z_where_pms[1].unsqueeze(-1)), dim=-1),
            'z_pres_lprb': z_pres_lprb,
            'z_what_lprb': z_what_lprb,
            'z_where_lprb': z_where_lprb,
            'baseline_value': baseline_value,
            'sigma': sigma,
            'slope': (strk_slope, add_slope),
        }
        return out

    def get_style_mlp_in(self, img_embed, canvas_embed, residual, p_state):
        '''Get the input for `style_mlp` from the current img and p_state
        Args:
            img_embed [ptcs, bs, embed_dim]
            canvas_embed [ptcs, bs, embed_dim] if self.execution_guided or None 
            p_state GuideState
        Return:
            style_mlp_in [ptcs, bs, style_mlp_in_dim]
            h_l [ptcs, bs, h_dim]
            style_rnn_in [ptcs, bs, style_rnn_in_dim]
        '''
        ptcs, bs = shp = img_embed.shape[:2]
        if self.exec_guid_type == 'residual':
            residual_embed = self.img_feature_extractor(residual.view(prod(shp), 
                                    1, self.res, self.res)).view(prod(shp), -1)

        # Style RNN input
        rnn_in = [p_state.z_pres.view(prod(shp), -1), 
                  p_state.z_where.view(prod(shp), -1)]
        # z_what
        if self.z_what_in_pos == 'z_where_rnn':
            rnn_in.append(p_state.z_what.view(prod(shp), -1))
            
        # canvas
        if (self.execution_guided and self.exec_guid_type == 'canvas'):
            rnn_in.append(canvas_embed.view(prod(shp), -1)) 

        # target / residual
        if (self.target_in_pos == 'RNN' and
            self.execution_guided and self.exec_guid_type == 'residual'):
            rnn_in.append(residual_embed.view(prod(shp), -1))
        elif self.target_in_pos == 'RNN':
            rnn_in.append(img_embed.view(prod(shp), -1))

        # if (self.prior_dist == 'Independent' and self.target_in_pos == 'RNN' and
        #     self.execution_guided and self.exec_guid_type == 'residual'):
        #     rnn_in.append(residual_embed)
        # elif (self.prior_dist == 'Independent' and self.target_in_pos == 'RNN' and
        #     not self.execution_guided):
        #     rnn_in.append(img_embed)
        
        style_rnn_in = torch.cat(rnn_in, dim=1)
        h_l = self.style_rnn(style_rnn_in, p_state.h_l.view(prod(shp), -1))

        # Style MLP input
        mlp_in = [h_l]

        # target / residual
        if (self.target_in_pos == "MLP" and self.style_mlp_in == 'h+target'):
            mlp_in.append(img_embed.view(prod(shp), -1))
        elif self.target_in_pos == "MLP" and self.style_mlp_in == 'h+residual':
            mlp_in.append(residual_embed.view(prod(shp), -1))
        style_mlp_in = torch.cat(mlp_in, dim=1)

        return (style_rnn_in.view(*shp, -1), h_l.view(*shp, -1),
                style_mlp_in.view(*shp, -1))

    def get_z_l(self, style_mlp_in, p_state):
        """Predict z_pres and z_where from `style_mlp_in`
        Args:
            style_mlp_in [ptcs, bs, in_dim]: input based on input types
            p_state: GuideState
        Return:
            z_pres [ptcs, bs, 1]
            z_where [ptcs, bs, z_where_dim]
            z_pres_lprb [ptcs, bs, 1]
            z_where_lprb [ptcs, bs, 1]
            z_pres_p [ptcs, bs, 1]
            z_where_pms
                z_where_loc [ptcs, bs, z_where_dim]
                z_where_scale [ptcs, bs, z_where_dim]
            sigma [ptcs, bs, 1]
            strk_slope [ptcs, bs, 1]
            add_slope [ptcs, bs, 1]
        """
        ptcs, bs = shp = style_mlp_in.shape[:2]

        # Predict presence and location from h
        # z_pres [prod(shp), 1]
        z_pres_p, z_where_loc, z_where_scale = self.style_mlp(
                                            style_mlp_in.view(prod(shp), -1))
        z_pres_p = z_pres_p.view(*shp, -1)
        z_where_loc = z_where_loc.view(*shp, -1)
        z_where_scale = z_where_scale.view(*shp, -1)

        sigma, strk_slope, add_slope = self.renderer_param_mlp(
                                            style_mlp_in.view(prod(shp), -1))
        sigma = sigma.view(*shp, -1)
        strk_slope = strk_slope.view(*shp, -1)
        add_slope = add_slope.view(*shp, -1)

        # If previous z_pres is 0, force z_pres to 0
        z_pres_p = z_pres_p * p_state.z_pres

        # Numerical stability -> added to net output
        eps = 1e-12
        z_pres_p = z_pres_p.clamp(min=eps, max=1.0-eps)

        # Sample z_pres
        assert z_pres_p.shape == torch.Size([*shp, 1])
        z_pres_post = Independent(Bernoulli(z_pres_p), 
                                        reinterpreted_batch_ndims=1)
        assert (z_pres_post.event_shape == torch.Size([1]) and
                z_pres_post.batch_shape == torch.Size([*shp]))
        # z_pres: [ptcs, bs, 1]
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
        
        # Sample z_where, get log_prob
        assert z_where_loc.shape == torch.Size([*shp, self.z_where_dim])
        z_where_post = Independent(Normal(z_where_loc, z_where_scale),
                                                    reinterpreted_batch_ndims=1)
        assert (z_where_post.event_shape == torch.Size([self.z_where_dim]) and
                z_where_post.batch_shape == torch.Size([*shp]))        
        # z_where: [ptcs, bs, z_where_dim]
        z_where = z_where_post.rsample()

        # constrain sample
        if not self.constrain_param:
            z_where = constrain_z_where(self.z_where_type, 
                                        z_where.view(prod(shp), -1), 
                                        clamp=True)
            z_where = z_where.view(*shp, -1)
                                                            
        z_where_lprb = z_where_post.log_prob(z_where).unsqueeze(-1) * z_pres
        # z_where_lprb = z_where_lprb.squeeze()

        return (z_pres, z_where, z_pres_lprb, z_where_lprb, z_pres_p, 
                (z_where_loc, z_where_scale), sigma, strk_slope, add_slope)

    def get_zwhat_mlp_in(self, trans_imgs, canvas_embed, p_state):
        '''Get the input for the zwhat_mlp
        Args:
            trans_imgs: [ptcs, bs, *img_dim]
            canvas_embed: [ptcs, bs, em_dim]
            p_state
        Return:
            z_what_mlp_in: [ptcs, bs, in_dim]
            h_c: [ptcs, bs, in_dim]
        '''
        # Sample z_what, get log_prob
        ptcs, bs = shp = trans_imgs.shape[:2]
        img_dim = trans_imgs.shape[2:]

        # [bs, pts_per_strk, 2, 1]
        if self.feature_extractor_sharing:
            trans_embed = self.img_feature_extractor(trans_imgs.view(prod(shp), 
                                                                    *img_dim))
        else:
            trans_embed = self.glimpse_feature_extractor(trans_imgs.view(prod(
                                                                        shp, 
                                                                    *img_dim)))
            
        # z_what RNN input
        z_what_rnn_in = []
        # prev.z_what
        if self.z_what_in_pos == 'z_what_rnn':
            z_what_rnn_in.append(p_state.z_what.view(prod(shp), -1))

        # Canvas
        if (self.prior_dist == 'Sequential' and self.execution_guided and 
            self.exec_guid_type == 'canvas'):
            z_what_rnn_in.append(canvas_embed.view(prod(shp), -1))
        
        # Target
        if self.target_in_pos == 'RNN':
            z_what_rnn_in.append(trans_embed.view(prod(shp), -1))
        
        z_what_rnn_in = torch.cat(z_what_rnn_in, dim=1)
        h_c = self.z_what_rnn(z_what_rnn_in, p_state.h_c.view(prod(shp), -1))

        # z_what MLP input
        z_what_mlp_in = [h_c]
        if self.target_in_pos == 'MLP':
            z_what_mlp_in.append(trans_embed.view(prod(shp), -1))
        z_what_mlp_in = torch.cat(z_what_mlp_in, dim=1)

        return z_what_mlp_in.view(*shp, -1), h_c.view(*shp, -1)

    def get_z_c(self, zwhat_mlp_in, p_state, z_pres):
        '''
        Args:
            zwhat_mlp_in [ptcs, bs, mlp_in_dim]
            z_pres [ptcs, bs, 1]
            ptcs::int
        Return:
            z_what [ptcs, bs, pts_per_strk, 2]
            z_what_lprb [ptcs, bs, 1]
            z_what_loc = [ptcs, bs, pts_per_strk, 2]
            z_what_std = [ptcs, bs, pts_per_strk, 2]
        '''
        # bs here is actually ptcs * bs
        ptcs, bs = shp = zwhat_mlp_in.shape[:2]
        z_what_loc, z_what_std = self.z_what_mlp(zwhat_mlp_in.view(prod(shp), -1))

        # [bs, pts_per_strk, 2]
        z_what_loc = z_what_loc.view([*(shp), self.pts_per_strk, 2])
        z_what_std = z_what_std.view([*(shp), self.pts_per_strk, 2])
            
        z_what_post = Independent(Normal(z_what_loc, z_what_std), 
                                                    reinterpreted_batch_ndims=2)
        assert (z_what_post.event_shape == torch.Size([self.pts_per_strk, 2]) and
                z_what_post.batch_shape == torch.Size([*(shp)]))

        # [ptcs, bs, pts_per_strk, 2] 
        z_what = z_what_post.rsample()
        # constrain samples
        # if not self.constrain_param:
        #     z_what = constrain_z_what(z_what, clamp=True)

        # log_prob(z_what): [ptcs, bs, 1]
        # z_pres: [ptcs, bs, 1]
        z_what_lprb = z_what_post.log_prob(z_what).unsqueeze(-1) * z_pres
        # z_what_lprb = z_what_lprb.squeeze()
        return z_what, z_what_lprb, (z_what_loc, z_what_std)

    def initialize_state(self, imgs, ptcs):
        '''
        Args:
            ptcs::int: number of particles
        Return:
            state::GuideState
            baseline_value [ptcs, bs, ..]
            mask_prev [ptcs, bs, max_strks]
            z_pres_pms [ptcs, bs, ..]
            z_where_pms [ptcs, bs, ..]
            z_what_pm [ptcs, bs, ..]
            z_pres_smpl [ptcs, bs, ..]
            z_where_smpl [ptcs, bs, ..]
            z_what_smpl [ptcs, bs, ..]
            z_pres_lprb [ptcs, bs, ..]
            z_where_lprb [ptcs, bs, ..]
            z_what_lpr [ptcs, bs, ..]
            z_pres_prir [ptcs, bs, ..]
            z_where_prir [ptcs, bs, ..]
            z_what_prir [ptcs, bs, ..]
            sigmas [ptcs, bs, ..]
            strk_slopes [ptcs, bs, ..]
            canvas [ptcs, bs, ..]
            residual [ptcs, bs, ..]
        '''
        ptcs, bs = imgs.shape[:2]

        # Init model state for performing inference
        state = GuideState(
            h_l=torch.zeros(ptcs, bs, self.style_rnn_hid_dim, device=imgs.device),
            h_c=torch.zeros(ptcs, bs, self.z_what_rnn_hid_dim, device=imgs.device),
            bl_h=torch.zeros(ptcs, bs, self.bl_hid_dim, device=imgs.device),
            z_pres=torch.ones(ptcs, bs, 1, device=imgs.device),
            z_where=torch.zeros(ptcs, bs, self.z_where_dim, device=imgs.device),
            z_what=torch.zeros(ptcs, bs, self.z_what_dim, device=imgs.device),
        )

        # z samples for each step
        z_pres_smpl = torch.ones(ptcs, bs, self.max_strks, device=imgs.device)
        z_what_smpl = torch.zeros(ptcs, bs, self.max_strks, self.pts_per_strk, 
                                                         2, device=imgs.device)
        z_where_smpl = torch.ones(ptcs, bs, self.max_strks, self.z_where_dim, 
                                                            device=imgs.device)
        # z distribution parameters for each step
        z_pres_pms = torch.ones(ptcs, bs, self.max_strks, device=imgs.device)
        z_what_pms = torch.zeros(ptcs, bs, self.max_strks, self.pts_per_strk, 2, 2, 
                                                            device=imgs.device)
        z_where_pms = torch.ones(ptcs, bs, self.max_strks, self.z_where_dim, 2, 
                                                            device=imgs.device)
        # z log posterior probability (lprb) for each step
        z_pres_lprb = torch.zeros(ptcs, bs, self.max_strks, 
                                                            device=imgs.device)
        z_what_lprb = torch.zeros(ptcs, bs, self.max_strks, 
                                                            device=imgs.device)
        z_where_lprb = torch.zeros(ptcs, bs, self.max_strks, 
                                                            device=imgs.device)

        # z log prior probability (lprb) for each step;
        #   computed when `prior_dist` = Sequential
        if self.prior_dist == 'Sequential':
            z_pres_prir = torch.zeros(ptcs, bs, self.max_strks, 
                                                            device=imgs.device)
            z_what_prir = torch.zeros(ptcs, bs, self.max_strks, 
                                                            device=imgs.device)
            z_where_prir = torch.zeros(ptcs, bs, self.max_strks, 
                                                            device=imgs.device)
        elif self.prior_dist == 'Independent':
            z_pres_prir, z_what_prir, z_where_prir = [None] * 3

        # baseline_value
        baseline_value = torch.zeros(ptcs, bs, self.max_strks, 
                                                            device=imgs.device)

        # sigma and slope: for rendering
        sigmas = torch.zeros(ptcs, bs, self.max_strks, device=imgs.device)
        strk_slopes = torch.zeros(ptcs, bs, self.max_strks, device=imgs.device)
        add_slopes = torch.zeros(ptcs, bs, self.max_strks, device=imgs.device)

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
        mask_prev = torch.ones(ptcs, bs, self.max_strks, device=imgs.device)

        if self.execution_guided:
            # if exec_guid_type == 'residual', canvas stores the difference
            # if exec_guid_type == 'canvas-so-far', canvas stores the cummulative
            if self.intr_ll:
                canvas = torch.zeros(ptcs, bs, self.max_strks + 1, *self.img_dim, 
                                                        device=imgs.device)
            else:
                canvas = torch.zeros(ptcs, bs, *self.img_dim, device=imgs.device)
            if self.exec_guid_type == 'residual':
                residual = torch.zeros(ptcs, bs, *self.img_dim, device=imgs.device)
            else:
                residual = None
        else:
            canvas, residual = None, None
        
        h_cs = torch.zeros(ptcs, bs, self.max_strks, self.style_rnn_hid_dim, device=imgs.device)
        h_ls = torch.zeros(ptcs, bs, self.max_strks, self.style_rnn_hid_dim, device=imgs.device)
        
        return (state, baseline_value, mask_prev, 
                z_pres_pms, z_where_pms, z_what_pms,
                z_pres_smpl, z_where_smpl, z_what_smpl, 
                z_pres_lprb, z_where_lprb, z_what_lprb,
                z_pres_prir, z_where_prir, z_what_prir, 
                sigmas, h_cs, h_ls, strk_slopes, add_slopes, canvas, residual)
    
    def character_conditioned_sampling(self, cond_img, n_samples=7):
        '''Only useful when it has an internal render, o/w it returns None and 
        one should use the param returned from `forward` to draw new samples
        for the generative model to render the image.
        Args:
            cond_img [bs, 1, res, res]: imgs to condition on
            num_samples::int: number of samples for each conditioned image
        Return:
            sampled_img [n_samples, bs, 1, res, res]
        '''
        if self.execution_guided:
            out = self.forward(cond_img)
            cond_img = character_conditioned_sampling(guide_out=out, 
                                            decoder=self.internal_decoder)
        else:
            return None

    def named_parameters(self, prefix='', recurse=True):
        for n, p in super().named_parameters(prefix=prefix, recurse=recurse):
            if n.split(".")[0] != 'internal_decoder':
                yield n, p
    
    def baseline_params(self):
        for n, p in self.named_parameters():
            if n.split("_")[0] == 'bl':
                yield p

    def air_params(self):
        for n, p in self.named_parameters():
            if n.split("_")[0] != 'bl':
                yield p

    def get_z_what_net_params(self):
        for n, p in self.named_parameters():
            if n.split("_")[1] == 'what':
                yield p

    def no_style_mlp_air_parameters(self):
        for n, p in self.named_parameters():
            if n.split("_")[0] != 'bl' and n.split('.')[0] != 'style_mlp':
                yield p