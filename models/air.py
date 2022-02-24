'''
Attend, Infer, Repeat model
Sequential model with distributed z_what latent variables
'''
import pdb
from collections import namedtuple
import itertools

import numpy as np
from numpy import prod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, Laplace, Bernoulli
from einops import rearrange
from kornia.morphology import dilation, erosion

import util
from models.air_mlp import *
from models import template
from models.template import ZSample,ZLogProb, GuideState, GenState

# latent variable tuple
GuideReturn = namedtuple('GuideReturn', ['z_smpl', 
                                         'z_lprb', 
                                         'z_prior',
                                         'mask_prev',
                                         'baseline_value', 
                                         'z_pms',
                                         'canvas',
                                         'residual'])
GenReturn = namedtuple('GenReturn', ['z_smpl',
                                     'canvas'])

def schedule_model_parameters(gen, guide, iteration, args):
    # schdule the success prob of z_pres distribution
    p = util.anneal_weight(init_val=0.99, final_val=args.final_bern,
                           cur_ite=iteration, anneal_step=1e5)
    gen.z_pres_prob = torch.zeros(args.strokes_per_img).cuda() + p
        
    
class GenerativeModel(nn.Module):
    def __init__(self, max_strks=2, res=28, z_where_type='3',
                                                    use_canvas=False, 
                                                    transform_z_what=True,
                                                    hidden_dim=256,
                                                    z_what_dim=50,
                                                    prior_dist='Independent',
                                                    ):
        super().__init__()
        self.max_strks = max_strks
        self.use_canvas=use_canvas
        self.h_dim = hidden_dim

        # Prior parameters
        self.prior_dist = prior_dist
        self.z_what_dim = z_what_dim
        self.z_where_type = z_where_type
        z_where_loc, z_where_std, self.z_where_dim = util.init_z_where(
                                                                self.z_where_type)
        if prior_dist == 'Independent':
            self.register_buffer("z_what_loc", torch.zeros(z_what_dim))
            self.register_buffer("z_what_std", torch.ones(z_what_dim))

            # z_pres
            self.register_buffer("z_pres_prob", torch.zeros(self.max_strks)+.5)

            # z_where: default '3'
            self.register_buffer("z_where_loc", z_where_loc.expand(self.max_strks, 
                                                                self.z_where_dim))
            self.register_buffer("z_where_std", z_where_std.expand(self.max_strks,
                                                                self.z_where_dim))
        elif prior_dist == 'Sequential':
            self.gen_style_mlp = PresWhereMLP(
                                                in_dim=self.h_dim,
                                                z_where_type=z_where_type,
                                                z_where_dim=self.z_where_dim,
                                                )
            # self.renderer_param_mlp = RendererParamMLP(in_dim=self.h_dim,)
            self.gen_zhwat_mlp = WhatMLP(
                                                in_dim=self.h_dim,
                                                z_what_dim=self.z_what_dim,
                                                hid_dim=hidden_dim,
                                                num_layers=2)

        
        # img
        self.imgs_dist_std = torch.nn.Parameter(torch.ones(1, res, res), 
                                                            requires_grad=True)        
        # Decoder
        self.decoder = Decoder(z_what_dim=z_what_dim, 
                                img_dim=[1, res, res],
                                hidden_dim=hidden_dim,
                                num_layers=2)
        
        # Image renderer, and its parameters
        self.res = res

    def get_imgs_dist_std(self):
        # return F.softplus(self.imgs_dist_std) + 1e-6
        return util.constrain_parameter(self.imgs_dist_std, min=.01, max=1)

    def control_points_dist(self, h_c=None, bs=[1, 3]):
        '''(z_what Prior) Batched control points distribution
        Return: dist of
            bs: [bs, max_strks]
        '''
        if self.prior_dist == "Sequential" and h_c is not None:
            loc, std = self.gen_zhwat_mlp(h_c.view(prod(bs), -1))
            # [bs, pts_per_strk, 2]
            loc = loc.view([*bs, self.z_what_dim])
            std = std.view([*bs, self.z_what_dim])
            self.z_what_loc = loc.expand(*bs, self.z_what_dim)
            self.z_what_std = std.expand(*bs, self.z_what_dim)
        elif self.prior_dist == "Independent":
            loc, std = self.z_what_loc.expand(*bs, self.z_what_dim), \
                       self.z_what_std.expand(*bs, self.z_what_dim)
        else:
            raise NotImplementedError

        dist =  Independent(
                    Normal(loc, std), reinterpreted_batch_ndims=1)

        assert (dist.event_shape == torch.Size([self.z_what_dim]) and 
                dist.batch_shape == torch.Size([*bs]))
        return dist
        
    def presence_dist(self, h_l=None, bs=[1, 3]):
        '''(z_pres Prior) Batched presence distribution 
        Return: dist of
            bs [bs]
        '''
        if self.prior_dist == "Sequential" and h_l is not None:
            z_pres_p, _, _ = self.gen_style_mlp(h_l.view(prod(bs), -1))
            z_pres_p = z_pres_p.squeeze(-1)
            self.z_pres_p = z_pres_p
        elif self.prior_dist == "Independent":
            z_pres_p = self.z_pres_prob.expand(*bs)
        else:
            raise NotImplementedError
        
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
            self.z_where_loc = loc
            self.z_where_std = std
        elif self.prior_dist == "Independent":
            loc, std = self.z_where_loc.expand(*bs, self.z_where_dim), \
                       self.z_where_std.expand(*bs, self.z_where_dim)
        else:
            raise NotImplementedError

        dist = Independent(
            Normal(loc.view(*bs, -1), std.view(*bs, -1)), 
                    reinterpreted_batch_ndims=1,
        )
        
        assert (dist.event_shape == torch.Size([self.z_where_dim]) and 
                dist.batch_shape == torch.Size([*bs]))
        return dist

    def img_dist(self, latents=None, canvas=None):
        '''Batched `Likelihood distribution` of `image` conditioned on `latent
        parameters`.
        Args:
            latents: 
                z_pres: [bs, n_strks] 
                z_what: [bs, n_strks, z_what_dim, 2 (x, y)]
                z_where:[bs, n_strks, z_where_dim]
        Return:
            Dist over images: [bs, 1 (channel), H, W]
        '''
        assert latents is not None or canvas is not None
        if canvas is None:
            imgs_dist_loc = self.renders_imgs(latents)
        else:
            imgs_dist_loc = canvas

        ptcs, bs = shp = imgs_dist_loc.shape[:2]

        imgs_dist_std = self.get_imgs_dist_std()
        dist = Independent(Laplace(imgs_dist_loc, imgs_dist_std), 
                            reinterpreted_batch_ndims=3)

        assert (dist.event_shape == torch.Size([1, self.res, self.res]) and 
                dist.batch_shape == torch.Size([*shp]))
        return dist

    def renders_imgs(self, latents):
        '''Batched img rendering
        Args:
            latents: 
                z_pres: [ptcs, bs, n_strks] 
                z_what: [ptcs, bs, n_strks, z_what_dim]
                z_where:[ptcs, bs, n_strks, z_where_dim]
        Return:
            images: [ptcs, bs, 1 (channel), H, W]
        '''
        z_pres, z_what, z_where = latents
        ptcs, bs, n_strks, z_what_dim = z_what.shape
        shp = z_pres.shape[:2]
        
        # Get rendered image: [bs, n_strk, n_channel (1), H, W]
        imgs = self.decoder(z_what.view(prod(shp), n_strks, -1))  
        imgs = imgs * z_pres.view(prod(shp), -1)[:, :, None, None, None]

        # reshape image for further processing
        imgs = imgs.view(ptcs*bs*n_strks, 1, self.res, self.res)

        # Get affine matrix: [bs * n_strk, 2, 3]
        z_where_mtrx = util.get_affine_matrix_from_param(
                                z_where.view(ptcs*bs*n_strks, -1), 
                                self.z_where_type)
        imgs = util.inverse_spatial_transformation(imgs, z_where_mtrx)

        # max normalized so each image has pixel values [0, 1]
        # [bs*n_strk, n_channel (1), H, W]

        # Change back to [bs, n_strk, n_channel (1), H, W]
        imgs = imgs.view(ptcs*bs, n_strks, 1, self.res, self.res)

        # Change to [bs, n_channel (1), H, W] through `sum`
        imgs = imgs.sum(1) 
        imgs = util.normalize_pixel_values(imgs, method="tanh", slope=0.6) # tanh works


        try:
            assert not imgs.isnan().any()
        except:
            breakpoint()
        return imgs.view(*shp, 1, self.res, self.res)

    def renders_glimpses(self, z_what):
        '''Get glimpse reconstruction from z_what control points
        Args:
            z_what: [ptcs, bs, n_strk, z_what_dim]
        Return:
            recon: [ptcs, bs, n_strks, 1, res, res]
        '''
        assert len(z_what.shape) == 4, \
                                    f"z_what shape: {z_what.shape} isn't right"
        ptcs, bs, n_strks, z_what_dim = z_what.shape
        res = self.res
        # Get rendered image: [bs, n_strk, n_channel (1), H, W]
        recon = self.decoder(z_what.view(ptcs*bs, n_strks, z_what_dim))
        recon = recon.view(ptcs*bs*n_strks, 1, self.res, self.res)
        # recon = util.normalize_pixel_values(recon, method="maxnorm",)
        recon = recon.view(ptcs, bs, n_strks, 1, self.res, self.res)

        return recon

    def log_prob(self, latents, imgs, z_pres_mask, canvas, z_prior=None):
        '''
        Args:
            latents: 
                z_pres: [ptcs, bs, max_strks] 
                z_what: [ptcs, bs, max_strks, z_what_dim, 2 (x, y)]
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
                    z_where=(self.transformation_dist(bs=bs).log_prob(z_where)* 
                        z_pres),
                    )

        # Likelihood
        # self.sigma = decoder_param.sigma
        # self.tanh_norm_slope_stroke = decoder_param.slope[0]
        img_dist = self.img_dist(latents=latents, canvas=canvas)
        log_likelihood = img_dist.log_prob(imgs)
        return log_prior, log_likelihood

    def sample(self, canvas, hs, latents, bs=[1]):
        # todo 2: with the guide, z_pres are in the right format, but the sampled 
        # todo 2: ones are not
        # todo although sample is not used at this moment
        raise NotImplementedError(
            "Haven't made sure the sampled z_pres are legal")
        z_pres = self.control_points_dist(bs).sample()
        z_what = self.presence_dist(bs).sample()
        z_where = self.transformation_dist(bs).sample()
        latents = ZSample(z_pres, z_what, z_where)
        imgs = self.img_dist(latents).sample()

        return imgs, latents

    def decoder_named_params(self):
        for n, p in self.named_parameters():
            if n.split(".")[0] == 'decoder':
                yield n, p

    def no_img_dist_named_params(self):
        for n, p in self.named_parameters():
            if n in ['imgs_dist_std']:
                continue
            else:
                yield n, p

class Guide(template.Guide):
    def __init__(self, max_strks=2, 
                    img_dim=[1,28,28],
                    hidden_dim=256, 
                    z_where_type='3', 
                    use_canvas=False,
                    use_residual=None,
                    z_what_dim=50,
                    feature_extractor_sharing=True,
                    z_what_in_pos='z_where_rnn',
                    prior_dist='Independent',
                    target_in_pos="RNN",
                    intermediate_likelihood=None,
                    sep_where_pres_mlp=True,
                                            ):
        # Parameters
        self.z_what_dim = z_what_dim
        super().__init__(max_strks=max_strks, 
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
                    sep_where_pres_mlp=sep_where_pres_mlp,
                    )

        # Internal renderer
        if self.use_canvas or self.prior_dist == 'Sequential':
            self.internal_decoder = GenerativeModel(
                                            z_where_type=self.z_where_type,
                                            z_what_dim=self.z_what_dim,
                                            max_strks=self.max_strks,
                                            res=img_dim[-1],
                                            use_canvas=use_canvas,
                                            prior_dist=self.prior_dist,
                                            )
        # Inference networks
        if self.sep_where_pres_mlp:
            self.pr_mlp = PresMLP(in_dim=self.pr_wr_mlp_in_dim)
            self.wr_mlp = WhereMLP(in_dim=self.pr_wr_mlp_in_dim, 
                                      z_where_type=self.z_where_type,
                                      z_where_dim=self.z_where_dim)
        else:
            self.pr_wr_mlp = PresWhereMLP(in_dim=self.pr_wr_mlp_in_dim, 
                                      z_where_type=self.z_where_type,
                                      z_where_dim=self.z_where_dim)

        # Module 2: z_what_cnn, wt_rnn, wt_mlp
        # stn transformed image -> (`z_what_dim` control points)
        # self.wt_rnn_in_dim = (self.feature_extractor_out_dim)
        # z_what MLP
        self.wt_mlp = WhatMLP(in_dim=self.wt_mlp_in_dim,
                                  z_what_dim=self.z_what_dim,
                                  hid_dim=hidden_dim,
                                  num_layers=2)

    def forward(self, imgs, num_particles=1):
        '''
        Args: 
            img: [bs, 1, H, W]
            num_particles: int
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
        ptcs = num_particles
        shp = [ptcs, bs]
        img_dim = imgs.shape[1:]

        imgs = imgs.unsqueeze(0).repeat(ptcs, 1, 1, 1, 1)

        (state, baseline_value, mask_prev, 
         z_pres_pms, z_where_pms, z_what_pms,
         z_pres_smpl, z_where_smpl, z_what_smpl, 
         z_pres_lprb, z_where_lprb, z_what_lprb,
         z_pres_prir, z_where_prir, z_what_prir, 
         canvas, residual) = self.initialize_state(imgs, ptcs)

        for t in range(self.max_strks):
            # following the online example
            mask_prev[:, :, t] = state.z_pres.squeeze()

            # Do one inference step and save results
            result = self.inference_step(state, imgs, canvas, residual)
            state = result['state']
            assert (state.z_pres.shape == torch.Size([ptcs, bs, 1]) and
                    state.z_what.shape == 
                            torch.Size([ptcs, bs, self.z_what_dim]) and
                    state.z_where.shape == 
                            torch.Size([ptcs, bs, self.z_where_dim]))

            # Update and store the information
            # z_pres: [bs, 1]
            z_pres_smpl[:, :, t] = state.z_pres.squeeze(-1)
            # z_what: [bs, z_what_dim, 2];
            z_what_smpl[:, :, t] = state.z_what
            # z_where: [bs, z_where_dim]
            z_where_smpl[:, :, t] = state.z_where

            assert (result['z_pres_pms'].shape == torch.Size([ptcs, bs, 1])
                and  result['z_what_pms'].shape == 
                        torch.Size([ptcs, bs, self.z_what_dim, 2]) 
                and  result['z_where_pms'].shape == 
                            torch.Size([ptcs, bs, self.z_where_dim, 2]))

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

            # Update the canvas
            if self.use_canvas:
                canvas_step = self.internal_decoder.renders_imgs((
                                            z_pres_smpl[:, :, t:t+1].clone(),
                                            z_what_smpl[:, :, t:t+1],
                                            z_where_smpl[:, :, t:t+1]))
                canvas_step = canvas_step.view(*shp, *img_dim)
                canvas = canvas + canvas_step
                if self.use_residual == "residual":
                    # compute the residual
                    residual = torch.clamp(imgs - canvas, min=0.)

            # Calculate the prior with the hidden states.
            if self.prior_dist == 'Sequential':
                h_l, h_c = state.h_l, state.h_c
                z_pres_prir[:, :, t] = self.internal_decoder.presence_dist(
                                        h_l, [*shp]
                                        ).log_prob(z_pres_smpl[:, :, t].clone()
                                        ) * mask_prev[:, :, t].clone()
                z_where_prir[:, :, t] = (self.internal_decoder
                                        .transformation_dist(
                                        h_l.clone(), [*shp]
                                        ).log_prob(z_where_smpl[:, :, t].clone()
                                        ) * z_pres_smpl[:, :, t].clone())
                z_what_prir[:, :, t] = (self.internal_decoder
                                        .control_points_dist(
                                        h_c.clone(), [*shp]
                                        ).log_prob(z_what_smpl[:, :, t].clone()
                                        ) * z_pres_smpl[:, :, t].clone())

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
                           canvas=canvas,
                           residual=residual,
                           z_prior=ZLogProb(
                               z_pres=z_pres_prir,
                               z_what=z_what_prir,
                               z_where=z_where_prir),
                           )    
        return data
        
    def inference_step(self, p_state, imgs, canvas, residual):
        '''Given previous (initial) state and input image, predict the current
        step latent distribution
        Args:
            p_state::GuideState
            imgs [ptcs, bs, 1, res, res]
            canvas [ptcs, bs, 1, res, res] or None
            residual [ptcs, bs, 1, res, res]
        '''
        ptcs, bs = shp = imgs.shape[:2]
        img_dim = imgs.shape[2:]

        img_embed, canvas_embed, residual_embed = self.get_img_features(
                                                        imgs, canvas, residual)
        # Predict z_pres, z_where from target and canvas
        pr_wr_mlp_in, h_l = self.get_pr_wr_mlp_in(img_embed, 
                                                                canvas_embed,
                                                                residual_embed,
                                                                residual, 
                                                                p_state)
        (z_pres, 
         z_where, 
         z_pres_lprb, 
         z_where_lprb, 
         z_pres_p, 
         z_where_pms)  = self.get_z_l(pr_wr_mlp_in, p_state)

        # Get spatial transformed "crop" from input image
        trans_imgs = util.spatial_transform(
                                    imgs.view(prod(shp), *img_dim), 
                                    util.get_affine_matrix_from_param(
                                            z_where.view(prod(shp), -1), 
                                            z_where_type=self.z_where_type))
        wt_mlp_in, h_c = self.get_wt_mlp_in(
                    trans_imgs.view(*shp, *img_dim), 
                    canvas_embed,
                    residual_embed,
                    p_state)
        z_what, z_what_lprb, z_what_pms = self.get_z_c(wt_mlp_in, 
                                                            p_state, z_pres)

        # Compute baseline for z_pres
        # depending on previous latent variables only
        bl_input = [
                    img_embed.detach().view(prod(shp), -1), 
                    p_state.z_pres.detach().view(prod(shp), -1), 
                    p_state.z_where.detach().view(prod(shp), -1), 
                    p_state.z_what.detach().view(prod(shp), -1)
                    ]
        if self.use_canvas:
            bl_input.append(canvas_embed.detach().view(prod(shp), -1)) 
        bl_input = torch.cat(bl_input, dim=1)
        bl_h = self.bl_rnn(bl_input.detach(), p_state.bl_h.view(prod(shp), -1))
        baseline_value = self.bl_regressor(bl_h) # shape (B,)
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
            # [bs, z_what_dim, 2, 2]
            'z_what_pms': torch.cat((z_what_pms[0].unsqueeze(-1), 
                                     z_what_pms[1].unsqueeze(-1)), dim=-1),
            # [bs, z_where_dim, 2]
            'z_where_pms': torch.cat((z_where_pms[0].unsqueeze(-1), 
                                     z_where_pms[1].unsqueeze(-1)), dim=-1),
            'z_pres_lprb': z_pres_lprb,
            'z_what_lprb': z_what_lprb,
            'z_where_lprb': z_where_lprb,
            'baseline_value': baseline_value,
        }
        return out

    def get_z_l(self, pr_wr_mlp_in, p_state):
        """Predict z_pres and z_where from `pr_wr_mlp_in`
        Args:
            pr_wr_mlp_in [ptcs, bs, in_dim]: input based on input types
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
        """
        ptcs, bs = shp = pr_wr_mlp_in.shape[:2]

        # Predict presence and location from h
        if self.sep_where_pres_mlp:
            z_pres_p = self.pr_mlp(pr_wr_mlp_in.view(prod(shp), -1))
            z_where_loc, z_where_scale =\
                       self.wr_mlp(pr_wr_mlp_in.view(prod(shp), -1))
        else:
            z_pres_p, z_where_loc, z_where_scale = self.pr_wr_mlp(
                                            pr_wr_mlp_in.view(prod(shp), -1))
        z_pres_p = z_pres_p.view(*shp, -1)
        z_where_loc = z_where_loc.view(*shp, -1)
        z_where_scale = z_where_scale.view(*shp, -1)
        # If previous z_pres is 0, force z_pres to 0
        z_pres_p = z_pres_p * p_state.z_pres

        # Numerical stability
        eps = 1e-12
        z_pres_p = z_pres_p.clamp(min=eps, max=1.0-eps)

        # Sample z_pres
        assert z_pres_p.shape == torch.Size([*shp, 1])
        z_pres_post = Independent(Bernoulli(z_pres_p), 
                                        reinterpreted_batch_ndims=1)
        assert (z_pres_post.event_shape == torch.Size([1]) and
                z_pres_post.batch_shape == torch.Size([*shp]))
        z_pres = z_pres_post.sample()

        # If previous z_pres is 0, this z_pres should also be 0.
        # However, this is sampled from a Bernoulli whose probability is at
        # least eps. In the unlucky event that the sample is 1, we force this
        # to 0 as well.
        z_pres = z_pres * p_state.z_pres

        # log prob: log q(z_pres[i] | x, z_{<i}) if z_pres[i-1]=1, else 0
        # Mask with p_state.z_pres instead of z_pres. 
        # Keep if prev == 1, curr == 0 or 1; remove if prev == 0
        z_pres_lprb = z_pres_post.log_prob(z_pres).unsqueeze(-1) * p_state.z_pres
        assert z_pres_lprb.shape == torch.Size([*shp, 1])
        
        # Sample z_where, get log_prob
        assert z_where_loc.shape == torch.Size([*shp, self.z_where_dim])
        z_where_post = Independent(Normal(z_where_loc, z_where_scale),
                                                    reinterpreted_batch_ndims=1)
        assert (z_where_post.event_shape == torch.Size([self.z_where_dim]) and
                z_where_post.batch_shape == torch.Size([*shp]))        

        z_where = z_where_post.rsample()
        z_where_lprb = z_where_post.log_prob(z_where).unsqueeze(-1) * z_pres
        assert z_where_lprb.shape == torch.Size([*shp, 1])

        return (z_pres, z_where, z_pres_lprb, z_where_lprb, z_pres_p, 
                (z_where_loc, z_where_scale))

    def get_z_c(self, wt_mlp_in, p_state, z_pres):
        '''
        Args:
            wt_mlp_in [ptcs, bs, mlp_in_dim]
            z_pres [ptcs, bs, 1]
            ptcs::int
        Return:
            z_what [ptcs, bs, pts_per_strk, 2]
            z_what_lprb [ptcs, bs, 1]
            z_what_loc = [ptcs, bs, pts_per_strk, 2]
            z_what_std = [ptcs, bs, pts_per_strk, 2]
        '''
        ptcs, bs = shp = wt_mlp_in.shape[:2]
        z_what_loc, z_what_std = self.wt_mlp(wt_mlp_in.view(prod(shp), -1))

        # [bs, z_dim]
        z_what_loc = z_what_loc.view([*shp, self.z_what_dim])
        z_what_std = z_what_std.view([*shp, self.z_what_dim])
            
        z_what_post = Independent(Normal(z_what_loc, z_what_std), 
                                        reinterpreted_batch_ndims=1)
        assert (z_what_post.event_shape == torch.Size([self.z_what_dim]) and
                z_what_post.batch_shape == torch.Size([*shp]))

        # [bs, z_what_dim, 2] 
        z_what = z_what_post.rsample()

        # log_prob(z_what): [bs, 1]
        # z_pres: [bs, 1]
        z_what_lprb = z_what_post.log_prob(z_what).unsqueeze(-1) * z_pres
        return z_what, z_what_lprb, (z_what_loc, z_what_std)

    def initialize_state(self, imgs, ptcs):
        '''
        Args:
            imgs [ptcs, bs, 1, res, res]
            ptcs::int
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
            h_l=torch.zeros(ptcs, bs, self.pr_wr_rnn_hid_dim, 
                                                            device=imgs.device),
            h_c=torch.zeros(ptcs, bs, self.wt_rnn_hid_dim, 
                                                            device=imgs.device),
            bl_h=torch.zeros(ptcs, bs, self.bl_hid_dim, device=imgs.device),
            z_pres=torch.ones(ptcs, bs, 1, device=imgs.device),
            z_where=torch.zeros(ptcs, bs, self.z_where_dim, device=imgs.device),
            z_what=torch.zeros(ptcs, bs, self.z_what_dim, device=imgs.device),
        )

        # z samples for each step
        z_pres_smpl = torch.ones(ptcs, bs, self.max_strks, device=imgs.device)
        z_what_smpl = torch.zeros(ptcs, bs, self.max_strks, self.z_what_dim, 
                                                            device=imgs.device)
        z_where_smpl = torch.ones(ptcs, bs, self.max_strks, self.z_where_dim, 
                                                            device=imgs.device)
        # z distribution parameters for each step
        z_pres_pms = torch.ones(ptcs, bs, self.max_strks, device=imgs.device)
        z_what_pms = torch.zeros(ptcs, bs, self.max_strks, self.z_what_dim, 2, 
                                                            device=imgs.device)
        z_where_pms = torch.ones(ptcs, bs, self.max_strks, self.z_where_dim, 2, 
                                                            device=imgs.device)
        # z log posterior probability (lprb) for each step
        z_pres_lprb = torch.zeros(ptcs, bs, self.max_strks, device=imgs.device)
        z_what_lprb = torch.zeros(ptcs, bs, self.max_strks, device=imgs.device)
        z_where_lprb = torch.zeros(ptcs, bs, self.max_strks, device=imgs.device)

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

        if self.use_canvas:
            # if use_residual == 'residual', canvas stores the difference
            # if use_residual == 'canvas-so-far', canvas stores the cummulative
            canvas = torch.zeros(ptcs, bs, *self.img_dim, device=imgs.device)
            if self.use_residual:
                residual = torch.zeros(ptcs, bs, *self.img_dim, 
                                                            device=imgs.device)
            else:
                residual = None
        else:
            canvas, residual = None, None
        
        return (state, baseline_value, mask_prev, 
                z_pres_pms, z_where_pms, z_what_pms,
                z_pres_smpl, z_where_smpl, z_what_smpl, 
                z_pres_lprb, z_where_lprb, z_what_lprb,
                z_pres_prir, z_where_prir, z_what_prir, 
                canvas, residual)

    # def named_parameters(self, prefix='', recurse=True):
    #     for n, p in super().named_parameters(prefix=prefix, recurse=recurse):
    #         if n.split(".")[0] != 'internal_decoder':
    #             yield n, p
    
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


    def non_decoder_named_params(self):
        for n, p in self.named_parameters():
            if n.split(".")[1] != 'internal_decoder':
                yield n, p