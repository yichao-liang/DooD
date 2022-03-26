'''
Attend, Infer, Repeat-style Sequential Spline (SSP) model:
Sequential model with spline z_what latent variables
'''
import pdb
from collections import namedtuple
import itertools

import numpy as np
from numpy import prod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, Laplace, Bernoulli,\
    ContinuousBernoulli
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal
from einops import rearrange
from kornia.morphology import dilation, erosion

import util
from splinesketch.code.bezier import Bezier
from models.ssp_mlp import *
from models import air_mlp, template
from models.template import ZSample, ZLogProb, GuideState,GenState

# latent variable tuple
DecoderParam = namedtuple("DecoderParam", "sigma slope")
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
                                         'hidden_states',])
GenReturn = namedtuple('GenReturn', ['z_smpl',
                                     'canvas'])

def schedule_model_parameters(gen, guide, iteration, loss, device):
    pass

class GenerativeModel(nn.Module):
    def __init__(self, max_strks=2, pts_per_strk=5, res=28, z_where_type='3',
                                                use_canvas=False, 
                                                transform_z_what=True,
                                                input_dependent_param=True,
                                                prior_dist='Independent',
                                                hidden_dim=512,
                                                num_mlp_layers=2,
                                                maxnorm=True,
                                                sgl_strk_tanh=True,
                                                add_strk_tanh=True,
                                                constrain_param=True,
                                                fixed_prior=True,
                                                spline_decoder=True,
                                                render_method='bounded',
                                                intermediate_likelihood=None,
                                                dependent_prior=False,
                                                # feature_extractor_out_dim=256,
                                                sep_where_pres_net=False,
                                                no_pres_rnn=False,
                                                no_rnn=False,
                                                prior_dependency='wr|wt',
                                                bern_img_dist=True,
                                                linear_sum=True,
                                                n_comp=4,
                                                correlated_latent=True,
                                                use_bezier_rnn=False,
                                                condition_by_img=False,
                                                constrain_var=False,
                                                    ):
        super().__init__()
        self.max_strks = max_strks
        self.pts_per_strk = pts_per_strk
        self.use_canvas = use_canvas
        self.maxnorm = maxnorm
        self.sgl_strk_tanh = sgl_strk_tanh
        self.add_strk_tanh = add_strk_tanh
        self.n_comp = n_comp
        self.correlated_latent = correlated_latent
        self.use_bezier_rnn = use_bezier_rnn
        self.constrain_var = constrain_var
        self.condition_by_img = condition_by_img
        # self.constrain_param = constrain_param
        self.constrain_smpl = not constrain_param
        self.intr_ll = intermediate_likelihood
        if self.intr_ll == "Geom":
            self.intr_ll_geo_p = torch.nn.Parameter(torch.tensor(-10.), 
                                                            requires_grad=True)

        if self.intr_ll is not None:
            assert self.use_canvas, "intermediate likelihood needs" + \
                                        "use_canvas = True"
        self.dependent_prior = dependent_prior
        self.sep_where_pres_net = sep_where_pres_net
        self.no_pres_rnn = no_pres_rnn
        self.no_rnn = no_rnn
        self.prior_dependency = prior_dependency
        self.bern_img_dist = bern_img_dist

        # Prior parameters
        self.prior_dist = prior_dist
        self.z_where_type = z_where_type
        # todo
        self.fixed_prior = fixed_prior
        if prior_dist == 'Independent':
            if self.fixed_prior:
                # z_what
                self.register_buffer("pts_loc", 
                                        torch.zeros(self.pts_per_strk, 2)+.5)
                self.register_buffer("pts_std", 
                                        torch.zeros(self.pts_per_strk, 2)+.2)
                # z_pres
                self.register_buffer("z_pres_prob", 
                                        torch.zeros(self.max_strks)+.5)
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
                                torch.zeros(self.max_strks)+.99, 
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

            # exp finds that sharing the pr_wr_mlp when sep_where_pres_net
            # leads to better results
            # if self.sep_where_pres_net:
            #     self.pr_mlp_in_dim = self.h_dim
            #     self.wr_mlp_in_dim = self.h_dim
            # else:
            self.pr_wr_mlp_in_dim = self.h_dim
            self.wt_mlp_in_dim = self.h_dim

            if self.dependent_prior: 
                if condition_by_img:
                    feature_extractor_out_dim = hidden_dim
                    self.wt_mlp_in_dim += feature_extractor_out_dim

                if self.prior_dependency == 'wr|wt':
                    # if self.sep_where_pres_net:
                    #     self.wr_mlp_in_dim += 10
                    # else:
                    self.pr_wr_mlp_in_dim += pts_per_strk * 2
                elif self.prior_dependency == 'wt|wr':
                    self.wt_mlp_in_dim += self.z_where_dim
                else: raise NotImplementedError

            if self.no_pres_rnn or self.no_rnn:
                self.z_pres_prob = torch.nn.Parameter(
                                torch.zeros(self.max_strks)+.99, 
                                requires_grad=True)

            # if self.sep_where_pres_net:
            #     self.gen_pr_mlp = PresPriorMLP(
            #                         in_dim=self.pr_mlp_in_dim,
            #                         hidden_dim=hidden_dim,
            #                         num_layers=1)
            #     self.gen_wr_mlp = WherePriorMLP(
            #                         in_dim=self.wr_mlp_in_dim, 
            #                         z_where_type=self.z_where_type,
            #                         z_where_dim=self.z_where_dim,
            #                         hidden_dim=hidden_dim,
            #                         num_layers=num_mlp_layers)
            # else:
            pri_mlp_hid_dim = 256
            self.gen_pr_wr_mlp = PresWherePriorMLP(
                                                in_dim=self.pr_wr_mlp_in_dim,
                                                z_where_type=z_where_type,
                                                z_where_dim=self.z_where_dim,
                                                hidden_dim=pri_mlp_hid_dim,
                                                num_layers=num_mlp_layers,
                                                # constrain_param=constrain_param,
                                                n_comp=n_comp
                                            )
            # self.renderer_param_mlp = RendererParamMLP(in_dim=self.h_dim,)
            if use_bezier_rnn:
                self.bezier_rnn = ControlPointPriorRNN(
                                        # + 2 for prev point sample
                                        in_dim=self.wt_mlp_in_dim+2,
                                        pts_per_strk=self.pts_per_strk,
                                        hid_dim=pri_mlp_hid_dim,
                                        n_comp=n_comp,
                                        correlated_latent=correlated_latent,
                                    )
            else:
                self.gen_wt_mlp = WhatPriorMLP(
                                                in_dim=self.wt_mlp_in_dim,
                                                pts_per_strk=self.pts_per_strk,
                                                hid_dim=pri_mlp_hid_dim,
                                                num_layers=num_mlp_layers,
                                                n_comp=n_comp
                                                )
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
            self.sigma, self.sgl_strk_tanh_slope, self.add_strk_tanh_slope\
                                                            = None, None, None
        else:
            self.sigma = torch.nn.Parameter(torch.tensor(6.), 
                                                            requires_grad=True)
            self.sgl_strk_tanh_slope = torch.nn.Parameter(torch.tensor(6.), 
                                                            requires_grad=True)
            self.add_strk_tanh_slope = torch.nn.Parameter(torch.tensor(6.), 
                                                            requires_grad=True)
        self.transform_z_what = transform_z_what



    def get_intr_ll_geo_p(self):
        return util.constrain_parameter(self.intr_ll_geo_p, min=.5, 
                                                            max=1.-1e-6)
    def get_sigma(self): 
        return util.constrain_parameter(self.sigma, min=.01, max=.04)
    def get_add_strk_tanh_slope(self):
        return util.constrain_parameter(self.add_strk_tanh_slope, min=.1,max=.7)
    def get_sgl_strk_tanh_slope(self):
        return util.constrain_parameter(self.sgl_strk_tanh_slope, min=.1, 
                                                                     max=.7)
    def get_imgs_dist_std(self):
        # return F.softplus(self.imgs_dist_std) + 1e-6
        return util.constrain_parameter(self.imgs_dist_std, min=1e-2, max=1)
        # return util.constrain_parameter(self.imgs_dist_std, min=1e-4, max=1)
        
    def presence_dist(self, h_l=None, bs=[1, 3], pri_wt=None, t=0, 
                      sample=False):
        '''
        (z_pres Prior) Batched presence distribution 
        It can be | sequential where it has to have h_l, or
                  | independent | with fixed prior, or
                                | with learned prior
        On the other hand, it can either condition on z_what or not
        Args:
            bs [n_particles, batch_size]
            h_l [n_particlestcs, bs, h_dim]: hidden-states for computing 
            sequential prior dist event_shape [max_strks]
            glmp_eb [n_ptcs, bs, feature_dim]
        Return:
            dist: batch_shape=[ptcs, bs]; event_shape=[]
        '''
        if self.sep_where_pres_net and self.prior_dist == 'Sequential':
            h_l = h_l[0]

        if self.prior_dist == "Sequential":
            assert h_l != None, "need hidden states!"
            
            if self.no_pres_rnn or self.no_rnn:
                z_pres_p = self.z_pres_prob[t].expand(*bs)
                z_pres_p = util.constrain_parameter(z_pres_p, min=1e-12, 
                                                              max=1-(1e-12))
            else:
                mlp_in = h_l.view(prod(bs), -1).clone()
                if pri_wt is not None and self.prior_dependency == 'wr|wt':
                    mlp_in = torch.cat([h_l.view(prod(bs), -1),
                                        pri_wt.view(prod(bs), -1)], dim=-1)
                # if self.sep_where_pres_net:
                #     z_pres_p = self.gen_pr_mlp(mlp_in)
                # else:
                z_pres_p, _, _, _, _ = self.gen_pr_wr_mlp(mlp_in)
                z_pres_p = z_pres_p.squeeze(-1)
                if sample: 
                    print(f"z_pres param at {t}", z_pres_p)
                    print("cons pres")
                    z_pres_p[z_pres_p < .4] = 0.
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

    def transformation_dist(self, h_l=None, bs=[1, 3], pri_wt=None, 
                            sample=False):
        '''(z_where Prior) Batched transformation distribution.
        It can be | sequential where it has to have h_l, or
                  | independent | with fixed prior, or
                                | with learned prior
        On the other hand, it can either condition on z_what or not
        Args:
            bs [ptcs, bs]
            h_l [ptcs, bs, h_dim]: hidden-states for computing sequential prior 
            dist event_shape [max_strks, z_where_dim (3-5)]
            glmp_eb [n_ptcs, bs, feature_dim]

        '''
        if self.sep_where_pres_net and self.prior_dist == 'Sequential':
            h_l = h_l[1]

        if self.prior_dist == "Sequential":
            assert h_l != None, "need hidden states!"
            mlp_in = h_l.view(prod(bs), -1).clone()
            if pri_wt is not None and self.prior_dependency == 'wr|wt':
                mlp_in = torch.cat([h_l.view(prod(bs), -1),
                                       pri_wt.view(prod(bs), -1)], dim=-1)
            # if self.sep_where_pres_net:
            #     loc, std = self.gen_wr_mlp(mlp_in)
            # else:
            _, pi, loc, std, cor = self.gen_pr_wr_mlp(mlp_in)
            n_comp = pi.shape[1]
            wr_dim = self.z_where_dim
            loc = loc.view(*bs, n_comp, -1)
            std = std.view(*bs, n_comp, -1)
            cor = cor.view(*bs, n_comp)
            self.z_where_loc = loc
            self.z_where_std = std
            self.z_where_cor = cor

            mix = Categorical(logits=pi.view(*bs, -1))
            if self.correlated_latent:
                # [ptcs, bs , n_comp, where_dim, where_dim]
                cor = cor.view(prod(bs), n_comp)
                tril = torch.diag_embed(std.view(prod(bs)*n_comp, -1)).view(
                                            prod(bs), n_comp, wr_dim, wr_dim)
                tril[:, :, 1, 0] = cor
                tril = tril.view(*bs, n_comp, wr_dim, wr_dim)

                if sample or self.constrain_var:
                    print("cons where")
                    cov = tril @ tril.transpose(-1,-2)
                    # print("where cov", cov)
                    cov[:,:,:2] = cov[:,:,:2] * 0.01 # shift
                    cov[:,:,2] = cov[:,:,2] * 0.7 # scale
                    cov[:,:,3] = cov[:,:,3] * 0.7 # rot
                    tril = torch.linalg.cholesky(cov, upper=False)

                comp = MultivariateNormal(loc=loc, scale_tril=tril)
            else:
                comp = Independent(Normal(loc, std), 
                        reinterpreted_batch_ndims=1)
            dist = MixtureSameFamily(mix, comp)

        elif self.prior_dist == "Independent":
            loc, std = self.z_where_loc.expand(*bs, self.z_where_dim), \
                       self.z_where_std.expand(*bs, self.z_where_dim)
            if not self.fixed_prior:
                loc = constrain_z_where(z_where_type=self.z_where_type,
                                        z_where_loc=loc.squeeze(0))
                std = util.constrain_parameter(std, min=1e-6, max=1)      
            dist = Independent(Normal(
                        loc.view(*bs, -1), std.view(*bs, -1)
                    ), reinterpreted_batch_ndims=1,)
        else:
            raise NotImplementedError

        assert (dist.event_shape == torch.Size([self.z_where_dim]) and 
                dist.batch_shape == torch.Size([*bs]))

        return dist
    
    def control_points_dist(self, h_c=None, bs=[1, 3], pri_wr=None, 
                            
                            sample=False):
        '''(z_what Prior) Batched control points distribution
        It can be | sequential where it has to have h_l, or
                  | independent | with fixed prior, or
                                | with learned prior
        Args: dist of
            bs: [ptcs, bs]
            h_c [ptcs, bs, h_dim]: hidden-states for computing sequential prior 
        Return:
            dist event_shape: [pts_per_strk, 2]
        '''
        n_comp, pts_per_strk = self.n_comp, self.pts_per_strk
        if self.prior_dist == "Sequential" and h_c is not None:
            mlp_in = h_c.view(prod(bs), -1).clone()
            if pri_wr != None and self.prior_dependency == 'wt|wr':
                mlp_in = torch.cat([h_c.view(prod(bs), -1),
                                    pri_wr.view(prod(bs), -1)], dim=-1)
            if self.use_bezier_rnn:
                pi, loc, std, cor = self.bezier_rnn(mlp_in)
            else:
                pi, loc, std, cor = self.gen_wt_mlp(mlp_in)

            pi = pi.view(*bs, pts_per_strk, n_comp)
            loc = loc.view([*bs, pts_per_strk, n_comp, 2])

            if self.correlated_latent:
                cor = cor.view([prod(bs), pts_per_strk, n_comp])
                tril = torch.diag_embed(std)
                try:
                    tril[:, :, :, 1, 0] = cor
                except:
                    breakpoint()
                tril = tril.view([*bs, pts_per_strk, n_comp, 2, 2])
                
                if sample or self.constrain_var:
                    print("cons what")
                    cov = tril @ tril.transpose(-1,-2)
                    # cov = cov * .01
                    cov[:, 2] = cov[:, 2] * .01
                    cov[:, :2] = cov[:, :2] * .01
                    cov[:, 3:] = cov[:, 3:] * .01
                    # print("what cov", cov)
                    tril = torch.linalg.cholesky(cov, upper=False)

                comp = MultivariateNormal(loc=loc, scale_tril=tril)
                mix = Categorical(logits=pi)
                dist = Independent(MixtureSameFamily(mix, comp),
                                reinterpreted_batch_ndims=1)
            else:
                std = std.view([*bs, pts_per_strk, n_comp, 2])
                comp = Independent(Normal(loc,std),
                                reinterpreted_batch_ndims=1)
                mix = Categorical(logits=pi)
                dist = Independent(MixtureSameFamily(mix, comp),
                                reinterpreted_batch_ndims=1)
            # else:
            #     pi, loc, std, cor = self.gen_wt_mlp(mlp_in)

            #     # [bs, pts_per_strk, 2]
            #     loc = loc.view([*bs, n_comp, pts_per_strk, 2])
            #     std = std.view([*bs, n_comp, pts_per_strk, 2])
            #     # [prod(bs), pts_per_strk, 2, 2]

            #     if self.correlated_latent:
            #         tril = torch.diag_embed(std.view(
            #                                 prod(bs)*n_comp, pts_per_strk, -1))
            #         tril[:, :, 1, 0] = cor.reshape(prod(bs)*n_comp, 
            #                                        pts_per_strk,)
            #         tril = tril.view([*bs, n_comp, pts_per_strk, 2, 2])


            #         if sample or self.constrain_var:
            #             print("cons what")
            #             cov = tril @ tril.transpose(-1,-2)
            #             print("what cov", cov)
            #             cov = cov * .01
            #             tril = torch.linalg.cholesky(cov, upper=False)

            #         comp = Independent(
            #                     MultivariateNormal(loc=loc, scale_tril=tril), 
            #                 reinterpreted_batch_ndims=1)
            #     else:
            #         comp = Independent(Normal(
            #                 loc, std),
            #             reinterpreted_batch_ndims=2)

            #     mix = Categorical(logits=pi.view(*bs, -1))
            #     dist = MixtureSameFamily(mix, comp)

        elif self.prior_dist == "Independent":
            loc, std = self.pts_loc.expand(*bs, self.pts_per_strk, 2), \
                       self.pts_std.expand(*bs, self.pts_per_strk, 2)

            if not self.fixed_prior:
                loc = constrain_z_what(loc)
                std = torch.sigmoid(std) + 1e-12
            dist =  Independent(Normal(loc, std), reinterpreted_batch_ndims=2)
        else:
            raise NotImplementedError

        self.z_what_loc = loc
        self.z_what_std = std
        self.z_what_cor = cor.view(*bs, n_comp, -1)
        assert (dist.event_shape == torch.Size([self.pts_per_strk, 2]) and 
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
                # From 1 to 5
                imgs_dist_loc = canvas[:, :, 1:]

        ptcs, bs = shp = imgs_dist_loc.shape[:2]
        # imgs_dist_std = torch.ones_like(imgs_dist_loc) 
        if canvas is None or self.intr_ll is None:
            imgs_dist_std = self.get_imgs_dist_std().repeat(*shp, 1, 1, 1)
        else:
            # [bs, n_canvas, n_channel, res, res]
            imgs_dist_std = self.get_imgs_dist_std().repeat(*shp, 1, 1, 1, 1)

        try:
            if self.bern_img_dist:
                dist = Independent(ContinuousBernoulli(imgs_dist_loc),
                                                reinterpreted_batch_ndims=3)
            else:
                dist = Independent(Laplace(imgs_dist_loc, imgs_dist_std), 
                                                reinterpreted_batch_ndims=3)
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
        '''Batched img rendering. Decode z_what then transform accroding to
        z_where with inverse spatial transform.
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
        
        # Get rendered image: [bs, n_strk, n_channel (1), H, W]
        if self.input_dependent_param:
            sigma = self.sigma
        else:
            sigma = self.get_sigma()

        if self.transform_z_what:
            trans_what = util.transform_z_what(
                                z_what.view(prod(shp), n_strks,pts_per_strk, 2),
                                z_where.view(prod(shp), n_strks, -1),
                                z_where_type=self.z_where_type)
            imgs = self.decoder(trans_what, sigma.view(prod(shp), n_strks, -1),
                                keep_strk_dim=True)

            imgs = imgs * z_pres.reshape(prod(shp), -1)[:, :, None, None, None]
            # reshape image for further processing
            imgs = imgs.view(ptcs*bs*n_strks, 1, self.res, self.res)
        else:
            if self.spline_decoder:
                # imgs [ptcs*bs, n_strk, 1, res, res]
                imgs = self.decoder(z_what.view(prod(shp), n_strks, 
                                                    pts_per_strk, 2), 
                            sigma=sigma.view(prod(shp), -1), keep_strk_dim=True)  
            else:
                imgs = self.decoder(z_what.view(prod(shp), n_strks, -1))

            imgs = imgs * z_pres.reshape(prod(shp), -1)[:, :, None, None, None]
            # reshape image for further processing
            imgs = imgs.view(ptcs*bs*n_strks, 1, self.res, self.res)

            # Get affine matrix: [ptcs*bs*n_strk, 2, 3]
            z_where_mtrx = util.get_affine_matrix_from_param(
                                    z_where.view(ptcs*bs*n_strks, -1), 
                                    self.z_where_type)
            imgs = util.inverse_spatial_transformation(imgs, z_where_mtrx)

        # max normalized so each image has pixel values [0, 1]
        # size: [ptcs*bs*n_strk, n_channel (1), H, W]
        if self.maxnorm:
            imgs = util.normalize_pixel_values(imgs, method="maxnorm",)

        # Change back to [ptcs, bs, n_strk, n_channel (1), H, W]
        imgs = imgs.view(ptcs, bs, n_strks, 1, self.res, self.res)

        if self.sgl_strk_tanh:
            # Normalize per stroke
            if self.input_dependent_param:
                slope = self.sgl_strk_tanh_slope
            else:
                slope = self.get_sgl_strk_tanh_slope()
            # output imgs: [prod(shp), n_strks, 1, res, res]
            imgs = util.normalize_pixel_values(imgs, 
                                            method=self.norm_pixel_method,
                                            slope=slope)
        # Change to [ptcs, bs, n_channel (1), H, W] through `sum`
        imgs = imgs.sum(2) 

        if n_strks > 1 and self.add_strk_tanh:
            # only normalize again if there were more then 1 stroke
            if self.input_dependent_param:
                # should have shape [ptcs, bs]
                slope = self.add_strk_tanh_slope
            else:
                slope = self.get_add_strk_tanh_slope().view(prod(shp))
            imgs = util.normalize_pixel_values(imgs, 
                            method=self.norm_pixel_method,
                            slope=slope)

        imgs = imgs.view(ptcs, bs, 1, self.res, self.res)

        return imgs 
        
    def renders_glimpses(self, z_what):
        '''Get glimpse reconstruction from z_what control points without being
        transformed by z_where
        Args:
            z_what: [ptcs, bs, n_strk, n_pts, 2]
        Return:
            recon: [ptcs, bs, n_strks, 1, res, res]
        '''
        try:
            assert len(z_what.shape) == 5
        except:
            print(f"z_what shape: {z_what.shape} isn't right")
            breakpoint()
        ptcs, bs, n_strks, n_pts = z_what.shape[:4]
        res = self.res

        # Get rendered image: [ptcs*bs, n_strk, n_channel (1), H, W]
        if self.spline_decoder:
            recon = self.decoder(z_what.view(ptcs*bs, n_strks, n_pts, 2), 
                                sigma=self.get_sigma().view(prod(ptcs*bs), -1), 
                                keep_strk_dim=True)  
        else:
            recon = self.decoder(z_what.view(ptcs*bs, n_strks, -1))

        if self.maxnorm:
            recon = util.normalize_pixel_values(
                    recon.view(ptcs*bs*n_strks, 1, res, res), method='maxnorm')

        recon = recon.view(ptcs, bs, n_strks, 1, res, res)

        if self.sgl_strk_tanh:
            if self.input_dependent_param:
                slope = self.sgl_strk_tanh_slope
            else:
                slope = self.get_sgl_strk_tanh_slope()
            recon = util.normalize_pixel_values(recon, method='tanh', 
                                                slope=slope)
        recon = recon.view(ptcs, bs, n_strks, 1, res, res)
        return recon

    def renders_cum_imgs(self, latents):
        '''Batched img rendering. Decode z_what then transform accroding to
        z_where with inverse spatial transform.
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
        
        # Get rendered image: [bs, n_strk, n_channel (1), H, W]
        if self.input_dependent_param:
            sigma = self.sigma
        else:
            sigma = self.get_sigma()

        if self.spline_decoder:
            # imgs [ptcs*bs, n_strk, 1, res, res]
            imgs = self.decoder(z_what.view(prod(shp), n_strks, 
                                                pts_per_strk, 2), 
                        sigma=sigma.view(prod(shp), -1), keep_strk_dim=True)  
        else:
            imgs = self.decoder(z_what.view(prod(shp), n_strks, -1))

        imgs = imgs * z_pres.reshape(prod(shp), -1)[:, :, None, None, None]

        # reshape image for further processing
        imgs = imgs.view(ptcs*bs*n_strks, 1, self.res, self.res)

        # Get affine matrix: [ptcs*bs*n_strk, 2, 3]
        z_where_mtrx = util.get_affine_matrix_from_param(
                                z_where.view(ptcs*bs*n_strks, -1), 
                                self.z_where_type)
        imgs = util.inverse_spatial_transformation(imgs, z_where_mtrx)

        # max normalized so each image has pixel values [0, 1]
        # size: [ptcs*bs*n_strk, n_channel (1), H, W]
        if self.maxnorm:
            imgs = util.normalize_pixel_values(imgs, method="maxnorm",)

        # Change back to [ptcs, bs, n_strk, n_channel (1), H, W]
        imgs = imgs.view(ptcs, bs, n_strks, 1, self.res, self.res)

        if self.sgl_strk_tanh:
            # Normalize per stroke
            if self.input_dependent_param:
                slope = self.sgl_strk_tanh_slope
            else:
                slope = self.get_sgl_strk_tanh_slope()
            # output imgs: [prod(shp), n_strks, 1, res, res]
            imgs = util.normalize_pixel_values(imgs, 
                                            method=self.norm_pixel_method,
                                            slope=slope)
        # [ptcs, bs, n_strk, n_channel, res, res]
        imgs = imgs.cumsum(2)
        imgs = imgs * z_pres.reshape(*shp, -1)[:, :, :, None, None, None]
        # imgs = imgs.sum(2) 

        if n_strks > 1 and self.add_strk_tanh:
            # only normalize again if there were more then 1 stroke
            if self.input_dependent_param:
                # should have shape [ptcs, bs]
                slope = self.add_strk_tanh_slope
                slope = slope.unsqueeze(-1).repeat(1, 1, n_strks)
            else:
                slope = self.get_add_strk_tanh_slope().view(prod(shp))
            imgs = util.normalize_pixel_values(imgs, 
                            method=self.norm_pixel_method,
                            slope=slope)

        imgs = imgs.view(ptcs, bs, n_strks, 1, self.res, self.res)

        return imgs 
    
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
            log_likelihood [ptcs, bs]
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
        # self.sgl_strk_tanh_slope = decoder_param.slope[0]
        if self.intr_ll:
            imgs = imgs.unsqueeze(2).expand(*bs, *img_shape)
        
        # if self.intr_ll: log_likelihood [bs, max_steps]
        # else: [bs]
        # if self.bern_img_dist:
        #     imgs = imgs.round()
        # if self.bern_img_dist:
        #     rec = self.img_dist(latents=latents, canvas=canvas).mean
        #     log_likelihood = F.binary_cross_entropy(rec, imgs,reduction='none'
        #                                             ).sum([2,3,4])
        # else:
        imgs = torch.clamp(imgs, min=0., max=1.)
        log_likelihood = self.img_dist(latents=latents, 
                                       canvas=canvas).log_prob(imgs)

        # with multiple particles, we need to compute likelihood in shape
        # [ptcs*bs, ...] and reshape back to [ptcs, bs], as different
        # particles might have different number of steps
        if self.intr_ll is not None:
            # reshape to [ptcs * bs, n_strks]
            log_likelihood = log_likelihood.view(prod(shape), -1)
            z_pres = z_pres.view(prod(shape), -1)
            num_steps = z_pres.sum(1)
            assert log_likelihood.shape == z_pres.shape
            log_likelihood = log_likelihood * z_pres
        if self.intr_ll == 'Mean':
            log_likelihood_ = torch.zeros(prod(shape)).to(z_pres.device)
            # Divide by the number of steps if the number is not 0
            log_likelihood_[num_steps != 0] = (
                                    log_likelihood[num_steps != 0] / 
                                    num_steps[num_steps != 0].unsqueeze(1)
                                ).sum(1)
            log_likelihood = log_likelihood_
        elif self.intr_ll == 'Geom':
            weights = util.geom_weights_from_z_pres(z_pres, 
                                                    p=self.get_intr_ll_geo_p())
            log_likelihood = (log_likelihood * weights).sum(1)
        if self.intr_ll is not None:
            log_likelihood = log_likelihood.view(*shape)


        return log_prior, log_likelihood

    def sample(self, canvas=None, 
                    hs=None, 
                    z_pms=None, 
                    bs=[1], 
                    in_img=None, 
                    decoder_param=None,
                    char_cond_gen=False,
                    linear_sum=True):
        '''
        Args:
            bs::list: representing the shape of generated image, e.g. [bs]
            in_img [bs, 1, res, res]:
                For unconditioned sampling: in_img is some random inputs for the
                    sigma, sgl_strk_tanh network;
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
                strk_img = None
                if linear_sum:
                    strk_img = torch.zeros(*bs, self.max_strks, 1, self.res, 
                                                self.res, device=self.device)

            if z_pms is not None:
                char_cond_gen = True
            if self.sep_where_pres_net:
                h_l = (torch.zeros(*bs, self.h_dim, device=self.device),
                       torch.zeros(*bs, self.h_dim, device=self.device))
            else:
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
                print(f"sampling time {t}")
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
                    result = self.generation_step(state, canvas, 
                                                  decoder_param=decoder_param,
                                                  t=t)
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

                # self.sigma = result['sigma']
                self.sigma = decoder_param.sigma[:, :, t:t+1]
                self.sgl_strk_tanh_slope = decoder_param.slope[0][:, :, t:t+1]

                # z_pres_smpl, etc has shape [bs, ...]. Thus to use renders_imgs
                # we need to unsqueeze(0) for the n_particle dimension
                latents = (z_pres_smpl[:, t:t+1].unsqueeze(0),
                            z_what_smpl[:, t:t+1].unsqueeze(0),
                            z_where_smpl[:, t:t+1].unsqueeze(0))
                canvas_step = self.renders_imgs(latents).squeeze(0)
                
                if linear_sum:
                    # breakpoint()
                    strk_img[:, t] = canvas_step
                    canvas = torch.sum(strk_img[:, :t+1], dim=1)
                    add_slope = decoder_param.slope[1][0, :, 0]
                    canvas = util.normalize_pixel_values(canvas,
                                                         method='tanh',
                                                         slope=add_slope)
                else:
                    canvas = canvas + canvas_step
                    if t > 0:
                        add_slope = decoder_param.slope[1][0, :, t-1]
                        canvas = util.normalize_pixel_values(canvas, 
                                                method='tanh',
                                                slope=add_slope)
            if not linear_sum:
                canvas = None
            # imgs = canvas
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
                                    z_pres=z_pres_smpl.unsqueeze(0),
                                    z_what=z_what_smpl.unsqueeze(0),
                                    z_where=z_where_smpl.unsqueeze(0),),
                        canvas=canvas)

    def generation_step(self, p_state, canvas, z_pms=None, 
                        decoder_param=None, t=0):
        '''Given previous state and input image, predict the next based on prior
        distributions
        Args:
            state::GenState
            canvas [bs, 1, res, res]
            z_pms: prev latent states for completion style sampling
        Return:
            z_pres: [bs, 1]
            z_where: [bs, z_where_dim]
            z_what: [bs, num_pts, 2]
        '''
        bs, img_dim = canvas.shape[0], canvas.shape[1:]
        
        canvas_embed = self.img_feature_extractor(canvas).view(bs, -1)

        # z_what hidden states 
        wt_rnn_in = torch.cat([
                                p_state.z_what.view(bs, -1),
                                canvas_embed,
                            ], dim=1)
        h_c = self.wt_rnn(wt_rnn_in, p_state.h_c)
        
        # z_where, z_pres hidden states
        if self.sep_where_pres_net:
            pr_rnn_in = torch.cat([
                                p_state.z_pres.view(bs, -1),
                                canvas_embed,
                            ], dim=1)
            h_pr = self.pr_rnn(pr_rnn_in, p_state.h_l[0])

            wr_rnn_in = torch.cat([
                                p_state.z_where.view(bs, -1),
                                canvas_embed,
                            ], dim=1)
            h_wr = self.wr_rnn(wr_rnn_in, p_state.h_l[1])
            h_l = [h_pr, h_wr]
        else:
            raise NotImplementedError
            pr_wr_rnn_input = torch.cat([
                                p_state.z_where,
                                canvas_embed, 
                            ], dim=1)
            h_l = self.pr_wr_rnn(pr_wr_rnn_input, p_state.h_l)

        if self.dependent_prior:
            if self.prior_dependency == 'wt|wr':
                # [bs]
                z_pres = self.presence_dist(h_l, [bs], pri_wt=None, t=t,
                                                sample=True).sample()
                # [bs, where_dim]
                z_where = self.transformation_dist(h_l, [bs], pri_wt=None,
                                                sample=True).sample()
                pri_wr = z_where
                if self.condition_by_img:
                    glmps = util.spatial_transform(
                        canvas.view(bs, *img_dim),
                        util.get_affine_matrix_from_param(
                            z_where.view(bs, -1),
                            z_where_type=self.z_where_type))
                    glmps_em = self.img_feature_extractor(glmps
                                            ).view(bs, -1).detach()
                    pri_wr = torch.cat([pri_wr, glmps_em], -1)
                # [bs, num_pts, 2]
                z_what = self.control_points_dist(h_c, [bs], pri_wr=pri_wr,
                                                sample=True).sample()
            elif self.prior_dependency == 'wr|wt':
                z_what = self.control_points_dist(h_c, [bs], pri_wr=None,
                                                sample=True).sample()
                z_pres = self.presence_dist(h_l, [bs], pri_wt=z_what, t=t,
                                                sample=True).sample()
                z_where = self.transformation_dist(h_l, [bs], pri_wt=z_what,
                                                sample=True).sample()
            else:
                raise NotImplementedError
        else:
            z_pres = self.presence_dict(h_l, [bs], pri_wt=None, t=t
                                            ).sample()
            z_where = self.transformation_dict(h_l, [bs], pri_wt=None
                                            ).sample()
            z_what = self.control_points_dist(h_c, [bs], pri_wr=None
                                            ).sample()
        z_pres = z_pres.view(bs, 1)
        z_pres = z_pres * p_state.z_pres
        # mlp_in = h_c
        # if self.dependent_prior and self.prior_dependency == 'wt|wr':
        #     mlp_in = torch.cat([mlp_in, z_where.view(bs, -1)], dim=-1)
        # z_what_loc, z_what_std = self.gen_zhwat_mlp(h_c)
        # # [bs, pts_per_strk, 2]
        # z_what_loc = z_what_loc.view([bs, self.pts_per_strk, 2])
        # z_what_std = z_what_std.view([bs, self.pts_per_strk, 2])
        # if z_pms is not None:
        #     z_what_loc = z_pms.z_what[:, :, :, 0]
        #     z_what_std = z_pms.z_what[:, :, :, 1]
        # z_what_dist = Independent(Normal(z_what_loc, z_what_std), 
        #                                 reinterpreted_batch_ndims=2)
        # assert (z_what_dist.event_shape == torch.Size([self.pts_per_strk, 2])and
        #         z_what_dist.batch_shape == torch.Size([bs]))
        # z_what = z_what_dist.sample()

        # # Sample z_pres and z_where
        # mlp_in = h_l
        
        # if self.dependent_prior and self.prior_dependency == 'wr|wt':
        #     mlp_in = torch.cat([mlp_in, z_what.view(bs, -1)], dim=-1)
        # if not self.no_pres_rnn:
        #     z_pres_p, z_where_loc, z_where_std = self.gen_pr_mr_mlp(mlp_in)
        # else:
        #     _, z_where_loc, z_where_std = self.gen_pr_mr_mlp(mlp_in)
        #     z_pres_p = self.z_pres_prob[t].expand(bs, 1)
        #     z_pres_p = util.constrain_parameter(z_pres_p, min=1e-12,
        #                                                   max=1-(1e-12))
        # # [bs, 1]
        # z_where_loc = z_where_loc.squeeze(-1)
        # z_where_std = z_where_std.squeeze(-1)

        # if z_pms is not None:
        #     z_pres_p = z_pms.z_pres
        #     # sigma, (strk_slope, add_slope) = decoder_param

        # z_pres_dist = Independent(
        #     Bernoulli(z_pres_p), reinterpreted_batch_ndims=1,
        # )
        # assert (z_pres_dist.event_shape == torch.Size([1]) and 
        #         z_pres_dist.batch_shape == torch.Size([bs]))

        # if z_pms is not None:
        #     z_where_loc = z_pms.z_where[:, :, 0]
        #     z_where_std = z_pms.z_where[:, :, 1]

        # z_where_dist = Independent(
        #     Normal(z_where_loc, z_where_std), reinterpreted_batch_ndims=1,
        # )
        # assert (z_where_dist.event_shape == torch.Size([self.z_where_dim]) and 
        #         z_where_dist.batch_shape == torch.Size([bs]))

        # z_pres = z_pres_dist.sample()
        # z_pres = z_pres * p_state.z_pres
        # z_where = z_where_dist.sample()


        new_state = GenState(
                            h_l=h_l,
                            h_c=h_c,
                            z_pres=z_pres,
                            z_what=z_what,
                            z_where=z_where
                        )
        
        return {'state': new_state,
                # 'sigma': sigma,
                # 'slope': (strk_slope, add_slope)
                }

    
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

class Guide(template.Guide):
    def __init__(self, 
                        max_strks:int=2, 
                        pts_per_strk:int=5, 
                        img_dim:list=[1,28,28],
                        hidden_dim:int=512, 
                        z_where_type:str='3', 
                        use_canvas:bool=False,
                        use_residual:bool=None,
                        transform_z_what:bool=False, 
                        input_dependent_param:bool=True,
                        prior_dist:str='Independent',
                        target_in_pos:str=None,
                        feature_extractor_sharing:bool=True,
                        num_mlp_layers:int=2,
                        num_bl_layers:int=2,
                        bl_mlp_hid_dim:int=512,
                        bl_rnn_hid_dim:int=256,
                        maxnorm:bool=True,
                        sgl_strk_tanh:bool=True,
                        add_strk_tanh:bool=True,
                        z_what_in_pos:str=None,
                        constrain_param:bool=True,
                        render_method:str='bounded',
                        intermediate_likelihood:str=None,
                        dependent_prior:bool=False,
                        prior_dependency:str="wr|wt",
                        spline_decoder:bool=True,
                        residual_pixel_count:bool=False,
                        sep_where_pres_net:bool=False,
                        render_at_the_end:bool=False,
                        simple_pres:bool=False,
                        residual_no_target:bool=False,
                        canvas_only_to_zwhere:bool=False,
                        detach_canvas_so_far:bool=True,
                        detach_canvas_embed:bool=True,
                        detach_rsd:bool=True,
                        detach_rsd_embed:bool=True,
                        no_post_rnn:bool=False,
                        no_pres_rnn:bool=False,
                        no_rnn:bool=False,
                        only_rsd_ratio_pres:bool=False,
                        no_what_post_rnn:bool=False,
                        no_pres_post_rnn:bool=False,
                        bern_img_dist=False,
                        dataset=None,
                        linear_sum=True,
                        n_comp=4,
                        correlated_latent=False,
                        use_bezier_rnn=False,
                        condition_by_img=True,
                ):
        '''
        Args:
            intermediate_likelihood:str: [None, 'Mean', 'Geom' (for Geometric 
                distribution like averaging)]
        '''
        self.pts_per_strk = pts_per_strk
        self.z_what_dim = self.pts_per_strk * 2
        self.linear_sum = linear_sum
        super().__init__(
                max_strks=max_strks, 
                img_dim=img_dim,
                hidden_dim=hidden_dim, 
                z_where_type=z_where_type, 
                use_canvas=use_canvas,
                use_residual=use_residual,
                feature_extractor_sharing=feature_extractor_sharing,
                z_what_in_pos=z_what_in_pos,
                prior_dist=prior_dist,
                target_in_pos=target_in_pos,
                intermediate_likelihood=intermediate_likelihood,
                num_bl_layers=num_bl_layers,
                bl_mlp_hid_dim=bl_mlp_hid_dim,
                bl_rnn_hid_dim=bl_rnn_hid_dim,
                maxnorm=maxnorm,
                dependent_prior=dependent_prior,
                spline_decoder=spline_decoder,
                residual_pixel_count=residual_pixel_count,
                sep_where_pres_net=sep_where_pres_net,
                simple_pres=simple_pres,
                no_post_rnn=no_post_rnn,
                residual_no_target=residual_no_target,
                canvas_only_to_zwhere=canvas_only_to_zwhere,
                detach_canvas_so_far=detach_canvas_so_far,
                detach_canvas_embed=detach_canvas_embed,
                detach_rsd=detach_rsd,
                detach_rsd_embed=detach_rsd_embed,
                no_pres_rnn=no_pres_rnn,
                no_rnn=no_rnn,
                only_rsd_ratio_pres=only_rsd_ratio_pres,
                no_what_post_rnn=no_what_post_rnn,
                no_pres_post_rnn=no_pres_post_rnn,
                )
        # Parameters
        # self.constrain_param = constrain_param
        self.constrain_smpl = not constrain_param
        self.sgl_strk_tanh = sgl_strk_tanh
        self.add_strk_tanh = add_strk_tanh
        self.render_at_the_end = render_at_the_end
        self.prior_dependency = prior_dependency
        self.condition_by_img = condition_by_img

        # Internal renderer
        if self.use_canvas or self.prior_dist == 'Sequential':
            self.internal_decoder = GenerativeModel(
                                            z_where_type=self.z_where_type,
                                            pts_per_strk=self.pts_per_strk,
                                            max_strks=self.max_strks,
                                            res=img_dim[-1],
                                            use_canvas=use_canvas,
                                            transform_z_what=transform_z_what,
                                            input_dependent_param=\
                                                    input_dependent_param,
                                            prior_dist=prior_dist,
                                            num_mlp_layers=num_mlp_layers,
                                            maxnorm=maxnorm,
                                            sgl_strk_tanh=sgl_strk_tanh,
                                            add_strk_tanh = add_strk_tanh,
                                            constrain_param=constrain_param,
                                            render_method=render_method,
                                            dependent_prior=dependent_prior,
                                            spline_decoder=spline_decoder,
                                            sep_where_pres_net=sep_where_pres_net,
                                            no_pres_rnn=no_pres_rnn,
                                            no_rnn=no_rnn,
                                            prior_dependency=prior_dependency,
                                            hidden_dim=hidden_dim,
                                            bern_img_dist=bern_img_dist,
                                            linear_sum=linear_sum,
                                            n_comp=n_comp,
                                            correlated_latent=correlated_latent,
                                            use_bezier_rnn=use_bezier_rnn,
                                            condition_by_img=condition_by_img,
                                        )
        # Inference networks
        # Style_mlp:
        #   rnn hidden state -> (z_pres, z_where dist parameters)
        post_mlp_hid_dim = 256
        if self.sep_where_pres_net:
            self.where_mlp = WhereMLP(in_dim=self.wr_mlp_in_dim,
                                  z_where_type=self.z_where_type,
                                  z_where_dim=self.z_where_dim,
                                  hidden_dim=post_mlp_hid_dim,
                                  num_layers=num_mlp_layers,
                                  dataset=dataset,
                                  constrain_param=constrain_param,
                                  ) 
            self.pres_mlp = PresMLP(in_dim=self.pr_mlp_in_dim,
                                hidden_dim=post_mlp_hid_dim,
                                num_layers=num_mlp_layers,
                                dataset=dataset)
        else:
            self.pr_wr_mlp = PresWhereMLP(in_dim=self.pr_wr_mlp_in_dim, 
                                      z_where_type=self.z_where_type,
                                      z_where_dim=self.z_where_dim,
                                      hidden_dim=post_mlp_hid_dim,
                                      num_layers=num_mlp_layers,
                                      constrain_param=constrain_param,
                                      spline_decoder=spline_decoder,
                                      )
        if self.sep_where_pres_net:
            render_mlp_in_dim = self.wr_mlp_in_dim
        else:
            render_mlp_in_dim = self.pr_wr_mlp_in_dim
        self.renderer_param_mlp = RendererParamMLP(
                                      in_dim=render_mlp_in_dim,
                                      hidden_dim=post_mlp_hid_dim,
                                      num_layers=num_mlp_layers,
                                      maxnorm=self.maxnorm,
                                      sgl_strk_tanh=self.sgl_strk_tanh,
                                      spline_decoder=spline_decoder,
                                      dataset=dataset,
                                      )

        self.wt_mlp = WhatMLP(in_dim=self.wt_mlp_in_dim,
                                  pts_per_strk=self.pts_per_strk,
                                  hid_dim=post_mlp_hid_dim,
                                  num_layers=num_mlp_layers,
                                  constrain_param=constrain_param,
                                  dataset=dataset,
                                  )
        self.init_h_wt = torch.nn.Parameter(torch.zeros(self.wt_rnn_hid_dim), 
                                                        requires_grad=True)
        if self.sep_where_pres_net:
            self.init_h_pr = torch.nn.Parameter(torch.zeros(
                                    self.pr_wr_rnn_hid_dim), requires_grad=True)
            self.init_h_wr = torch.nn.Parameter(torch.zeros(
                                    self.pr_wr_rnn_hid_dim), requires_grad=True)
        else:
            self.init_h_prwr = torch.nn.Parameter(torch.zeros(
                                    self.pr_wr_rnn_hid_dim), requires_grad=True)
        # self.init_h_wt = torch.zeros(self.wt_rnn_hid_dim).cuda()
        # self.init_h_pr = torch.zeros(self.pr_wr_rnn_hid_dim).cuda()
        # self.init_h_wr = torch.zeros(self.pr_wr_rnn_hid_dim).cuda()


    def forward(self, imgs, num_particles=1):#, writer=None):
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
         sigmas, h_ls, h_cs, sgl_strk_tanh_slope, add_strk_tanh_slope, 
         canvas, residual, strk_img) = self.initialize_state(imgs, ptcs)
        for t in range(self.max_strks):
            # following the online example
            # state.z_pres: [ptcs, bs, 1]
            mask_prev[:, :, t] = state.z_pres.squeeze()

            if self.constrain_z_pres_param_this_ite:
            # if t==0:
                self.constrain_z_pres_param_this_step = True
                # some experimental condition
            else: self.constrain_z_pres_param_this_step = False

            # Do one inference step and save results
            if self.intr_ll is None:
                result = self.inference_step(p_state=state, imgs=imgs, 
                                             canvas=canvas, residual=residual)
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
            sgl_strk_tanh_slope[:, :, t] = result['slope'][0].squeeze(-1)
            add_strk_tanh_slope[:, :, t] = result['slope'][1].squeeze(-1)

            # Update the canvas
            if self.use_canvas:
                self.internal_decoder.sigma = sigmas[:, :, t:t+1].clone()
                # tanh_slope [ptcs, bs, 1]
                self.internal_decoder.sgl_strk_tanh_slope = \
                                              sgl_strk_tanh_slope[:, :, t:t+1]
                canvas_step = self.internal_decoder.renders_imgs((
                                        z_pres_smpl[:, :, t:t+1].clone(),
                                        z_what_smpl[:, :, t:t+1],
                                        z_where_smpl[:, :, t:t+1].clone()))
                canvas_step = canvas_step.view(*shp, *img_dim)
                
                
                if self.intr_ll is None:
                    prev_canv = canvas
                    
                    if self.linear_sum:
                        # save canvas step
                        strk_img[:, :, t] = canvas_step
                        canvas = torch.sum(strk_img[:, :, :t+1], dim=2)
                        if self.add_strk_tanh:
                            canvas = util.normalize_pixel_values(
                                            canvas, 
                                            method='tanh', 
                                            slope=add_strk_tanh_slope[:, :, 0])
                        if (self.detach_canvas_so_far and\
                            t+1 != self.max_strks):
                            # when t+1 = max_strks it doesn't have to detached
                            canvas = canvas.detach()
                    else:
                        canvas = canvas + canvas_step
                        # not using detach_canvas is disencouraged
                            
                        if self.add_strk_tanh and not self.detach_canvas_so_far:
                            canvas = util.normalize_pixel_values(
                                            canvas, 
                                            method='tanh', 
                                            slope=add_strk_tanh_slope[:, :, t])
                        # in case of detach_canvas_so_far, the return canvas is
                        # None. The last add_strk_slope is thus used twice, 1 for 
                        # the last step, one for rendering the whole recon. We can
                        # potentially improve it by the following modification. 
                        # This isn't good when canvas is not detached because it
                        # complicates the gradient graph
                        if (self.add_strk_tanh and t > 0 and\
                            self.detach_canvas_so_far):
                            canvas = util.normalize_pixel_values(
                                            canvas, 
                                            method='tanh', 
                                            slope=add_strk_tanh_slope[:, :, t-1])
                        if self.detach_canvas_so_far:
                            canvas = canvas.detach()
                    # from plot import debug_plot as dp
                    # dp(imgs, canvas, writer, t)
                else:
                    canvas_so_far = (canvas[:, :, t:t+1] +
                                     canvas_step.unsqueeze(2))
                    canvas[:, :, t+1] = util.normalize_pixel_values(
                                        canvas_so_far, 
                                        method='tanh',
                                        slope=add_strk_tanh_slope[:, :, t:t+1]
                                        ).squeeze(2)
                if self.use_residual or self.residual_pixel_count:
                    # compute the residual
                    residual = torch.clamp(imgs - canvas, min=0.)
                    if self.detach_rsd:
                        residual = residual.detach()

            # Calculate the prior with the hidden states.
            if self.prior_dist == 'Sequential':
                pri_wt, pri_wr = None, None
                if self.dependent_prior:
                    # condition_by_img = False
                    # if condition_by_img:
                    #     self.internal_decoder.sigma = sigmas[:, :, t:t+1].clone()
                    #     self.internal_decoder.sgl_strk_tanh_slope = \
                    #                 sgl_strk_tanh_slope[:, :, t:t+1]
                    #     glmps = self.internal_decoder.renders_glimpses(
                    #                     state.z_what.unsqueeze(2)
                    #                     ).view(prod(shp), *img_dim)
                    #     pri_wt = self.img_feature_extractor(glmps).view(*shp, 
                    #                                                 -1).detach()
                    # else:
                    if self.prior_dependency == 'wr|wt':
                        pri_wt = state.z_what.view(*shp, -1).detach()
                        pri_wr = None
                        if self.condition_by_img:
                            self.internal_decoder.sigma = sigmas[:, :, t:t+1
                                                                    ].clone()
                            self.internal_decoder.sgl_strk_tanh_slope = \
                                    sgl_strk_tanh_slope[:, :, t:t+1]
                            glmps = self.internal_decoder.renders_glimpses(
                                            state.z_what.unsqueeze(2)
                                            ).view(prod(shp), *img_dim)
                            glmps_em = self.img_feature_extractor(glmps
                                                    ).view(*shp, -1)#.detach()
                            pri_wt = torch.cat([pri_wt, glmps_em], -1)

                    elif self.prior_dependency == 'wt|wr':
                        pri_wt = None
                        pri_wr = state.z_where.view(*shp, -1)
                        if self.condition_by_img:
                            # if not detach, the gradients are passed from
                            # prior to post nets
                            glmps = util.spatial_transform(
                                prev_canv.view(prod(shp), *img_dim),
                                util.get_affine_matrix_from_param(
                                    state.z_where.view(prod(shp), -1),
                                    z_where_type=self.z_where_type))
                            glmps_em = self.img_feature_extractor(glmps
                                                    ).view(*shp, -1)#.detach()
                            pri_wr = torch.cat([pri_wr, glmps_em], -1).detach()
                    else:
                        raise NotImplementedError
                    
                # log the hidden states
                h_l, h_c = state.h_l, state.h_c
                if self.sep_where_pres_net:
                    h_prs, h_wrs = h_ls
                    h_prs[:, :, t], h_wrs[:, :, t] = h_l[0], h_l[1]
                    h_cs[:, :, t] = h_c
                    h_ls = h_prs, h_wrs
                else:
                    h_ls[:, :, t], h_cs[:, :, t] = h_l, h_c
                
                if self.no_rnn:
                    # use feature as hidden states
                    h_l = h_c = self.img_feature_extractor(
                            prev_canv.view(prod(shp), *img_dim)).view(*shp, -1)
                    h_l = [h_l, h_l]

                z_pres_prir[:, :, t] = self.internal_decoder.presence_dist(
                                        h_l, [*shp], pri_wt, t=t
                                        ).log_prob(z_pres_smpl[:, :, t].clone()
                                        ) * mask_prev[:, :, t].clone()
                z_where_prir[:, :, t] = self.internal_decoder.transformation_dist(
                                        h_l, [*shp], pri_wt
                                        ).log_prob(z_where_smpl[:, :, t].clone()
                                        ) * z_pres_smpl[:, :, t].clone()
                z_what_prir[:, :, t] = self.internal_decoder.control_points_dist(
                                        h_c.clone(), [*shp], pri_wr,
                                        ).log_prob(z_what_smpl[:, :, t].clone()
                                        ) * z_pres_smpl[:, :, t].clone()

        # todo 1: init the distributions which can be returned; can be useful
        if self.detach_canvas_so_far and not self.linear_sum:
            canvas = None
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
                               slope=(sgl_strk_tanh_slope, add_strk_tanh_slope)),
                           canvas=canvas,
                           residual=residual,
                           z_prior=ZLogProb(
                               z_pres=z_pres_prir,
                               z_what=z_what_prir,
                               z_where=z_where_prir),
                            hidden_states=(h_ls, h_cs)
                           )
        return data
        
    # @profile
    def inference_step(self, p_state, imgs, canvas, residual):

        '''Given previous (initial) state and input image, predict the current
        step latent distribution.
        Args:
            p_state::GuideState
            imgs [ptcs, bs, 1, res, res]
            canvas [ptcs, bs, 1, res, res] or None
            residual [ptcs, bs, 1, res, res] or None
        '''
        shp = imgs.shape[:2]
        img_dim = imgs.shape[2:]

        img_embed, canvas_embed, residual_embed, rsd_ratio =\
                                self.get_img_features(imgs, canvas, residual)

        # Predict z_pres, z_where from target and canvas
        pr_wr_mlp_in, h_l = self.get_pr_wr_mlp_in(img_embed, 
                                                                canvas_embed, 
                                                                residual_embed, 
                                                                rsd_ratio,
                                                                p_state)

        (z_pres, z_where, 
         z_pres_lprb, z_where_lprb, 
         z_pres_p, z_where_pms, 
         sigma, strk_slope, add_slope) =\
                                self.get_z_l(pr_wr_mlp_in, p_state, rsd_ratio)

        # Get spatial transformed "crop" from input image
        # imgs [bs, *img_dim]
        # trans_imgs [ptcs * bs, *img_dim]
        trans_imgs = util.spatial_transform(
                            imgs.view(prod(shp), *img_dim), 
                            util.get_affine_matrix_from_param(
                            z_where.view(prod(shp), -1), 
                            z_where_type=self.z_where_type)
                        ).view(*shp, *img_dim)

        if self.use_residual:
            trans_rsd = util.spatial_transform(
                            residual.view(prod(shp), *img_dim), 
                            util.get_affine_matrix_from_param(
                            z_where.view(prod(shp), -1), 
                            z_where_type=self.z_where_type)
                        ).view(*shp, *img_dim)
        else: trans_rsd = None
        
        wt_mlp_in, h_c = self.get_wt_mlp_in(trans_imgs, 
                                            trans_rsd,
                                            canvas_embed,
                                            p_state)
        z_what, z_what_lprb, z_what_pms = self.get_z_c(wt_mlp_in, 
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
        if 'canvas' in self.bl_in:
            bl_input.append(canvas_embed.detach().view(prod(shp), -1)) 
        bl_input = torch.cat(bl_input, dim=1)
        bl_h = self.bl_rnn(bl_input, p_state.bl_h.view(prod(shp), -1))
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

    # @profile
    def get_z_l(self, pr_wr_mlp_in, p_state, rsd_ratio=None):
        """Predict z_pres and z_where from `pr_wr_mlp_in`
        Args:
            pr_wr_mlp_in [ptcs, bs, in_dim]: input based on input types
                or (pr_mlp_in, wr_mlp_in)
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
        if self.sep_where_pres_net:
            pr_mlp_in, wr_mlp_in = pr_wr_mlp_in
            shp = pr_mlp_in.shape[:2]
        else:
            shp = pr_wr_mlp_in.shape[:2]

        # Predict presence and location from h
        # z_pres [prod(shp), 1]
        if self.sep_where_pres_net:
            z_pres_p = self.pres_mlp(pr_mlp_in.view(prod(shp), -1))
            z_where_loc, z_where_scale = self.where_mlp(
                                     wr_mlp_in.view(prod(shp), -1))
        else:
            z_pres_p, z_where_loc, z_where_scale = self.pr_wr_mlp(
                                            pr_wr_mlp_in.view(prod(shp), -1))
        if self.simple_pres:
            # in this case the predictions above are ignored
            assert rsd_ratio is not None
            z_pres_p = rsd_ratio.detach() ** self.get_pr_rsd_power()


        z_pres_p = z_pres_p.view(*shp, -1)
        z_where_loc = z_where_loc.view(*shp, -1)
        z_where_scale = z_where_scale.view(*shp, -1)


        if self.sep_where_pres_net:
            sigma, strk_slope, add_slope = self.renderer_param_mlp(
                                            wr_mlp_in.view(prod(shp), -1))
        else:
            sigma, strk_slope, add_slope = self.renderer_param_mlp(
                                            pr_wr_mlp_in.view(prod(shp), -1))
        sigma = sigma.view(*shp, -1)
        strk_slope = strk_slope.view(*shp, -1)
        add_slope = add_slope.view(*shp, -1)

        z_pres, z_where, z_pres_lprb, z_where_lprb = self.sample_pr_wr(p_state,
                                          z_pres_p, z_where_loc, z_where_scale)

        return (z_pres, z_where, z_pres_lprb, z_where_lprb, z_pres_p, 
                (z_where_loc, z_where_scale), sigma, strk_slope, add_slope)


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
        z_what_loc, z_what_std = self.wt_mlp(zwhat_mlp_in.view(prod(shp), -1))

        # [bs, pts_per_strk, 2]
        z_what_loc = z_what_loc.view([*(shp), self.pts_per_strk, 2])
        z_what_std = z_what_std.view([*(shp), self.pts_per_strk, 2])
        
        z_what, z_what_lprb = self.sample_wt(z_what_loc, z_what_std, z_pres) 
        
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
            sgl_strk_tanh_slope [ptcs, bs, ..]
            canvas [ptcs, bs, ..]
            residual [ptcs, bs, ..]
        '''
        ptcs, bs = imgs.shape[:2]

        # Init model state for performing inference
        if self.sep_where_pres_net:
            # h_l = (torch.zeros(ptcs, bs, self.pr_wr_rnn_hid_dim, 
            #                    device=imgs.device),
            #        torch.zeros(ptcs, bs, self.pr_wr_rnn_hid_dim, 
            #                    device=imgs.device))
            h_l = (self.init_h_pr.expand(ptcs, bs, self.pr_wr_rnn_hid_dim),
                   self.init_h_wr.expand(ptcs, bs, self.pr_wr_rnn_hid_dim))
        else:
            # h_l = torch.zeros(ptcs, bs, self.pr_wr_rnn_hid_dim, 
            #                    device=imgs.device)
            h_l = self.init_h_prwr.expand(ptcs, bs, self.pr_wr_rnn_hid_dim)

        state = GuideState(
            h_l=h_l,
            # h_c=torch.zeros(ptcs, bs, self.wt_rnn_hid_dim, device=imgs.device),
            h_c=self.init_h_wt.expand(ptcs, bs, self.wt_rnn_hid_dim),
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
        sgl_strk_tanh_slope = torch.zeros(ptcs, bs, self.max_strks, device=imgs.device)
        add_strk_tanh_slope = torch.zeros(ptcs, bs, self.max_strks, device=imgs.device)

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

        strk_img = None
        if self.linear_sum:
            strk_img = torch.zeros(ptcs, bs, self.max_strks, *self.img_dim,
                                                        device=imgs.device)
        if self.use_canvas:
            if self.intr_ll:
                canvas = torch.zeros(ptcs, bs, self.max_strks + 1, *self.img_dim, 
                                                        device=imgs.device)
            else:
                canvas = torch.zeros(ptcs, bs, *self.img_dim, device=imgs.device)
            if self.use_residual:
                residual = imgs.detach().clone()
            else:
                residual = None
        else:
            canvas, residual = None, None
        
        if self.sep_where_pres_net:
            h_prs = torch.zeros(ptcs, bs, self.max_strks, self.pr_wr_rnn_hid_dim, 
                                                            device=imgs.device)
            h_wrs = torch.zeros(ptcs, bs, self.max_strks, self.pr_wr_rnn_hid_dim, 
                                                            device=imgs.device)
            h_ls = (h_prs, h_wrs)
        else:
            h_ls = torch.zeros(ptcs, bs, self.max_strks, self.pr_wr_rnn_hid_dim, 
                                                            device=imgs.device)
        h_cs = torch.zeros(ptcs, bs, self.max_strks, self.pr_wr_rnn_hid_dim, 
                                                            device=imgs.device)
        
        return (state, baseline_value, mask_prev, 
                z_pres_pms, z_where_pms, z_what_pms,
                z_pres_smpl, z_where_smpl, z_what_smpl, 
                z_pres_lprb, z_where_lprb, z_what_lprb,
                z_pres_prir, z_where_prir, z_what_prir, 
                sigmas, h_ls, h_cs, sgl_strk_tanh_slope, add_strk_tanh_slope, 
                canvas, residual, strk_img)
    
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
        if self.use_canvas:
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

    def pr_net_param(self):
        for n, p in self.named_parameters():
            w1, w2 = n.split("_")[:2]
            if w1 == 'pr' and w2 != 'wr':
                yield p
    
    def non_pr_net_air_param(self):
        for n, p in self.named_parameters():
            word = n.split("_")[0]
            if word != 'pr' and word != 'bl':
                yield p