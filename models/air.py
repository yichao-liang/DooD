'''
Attend, Infer, Repeat-pr_wr model
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
ZSample = namedtuple("ZSample", "z_pres z_what z_where")
ZLogProb = namedtuple("ZLogProb", "z_pres z_what z_where")
GuideState = namedtuple('GuideState', 'h_l h_c bl_h z_pres z_where z_what')
GenState = namedtuple('GenState', 'h_l h_c z_pres z_where z_what')
GuideReturn = namedtuple('GuideReturn', ['z_smpl', 
                                         'z_lprb', 
                                         'mask_prev',
                                         'baseline_value', 
                                         'z_pms',
                                         'canvas',
                                         'residual'])
GenReturn = namedtuple('GenReturn', ['z_smpl',
                                     'canvas'])

class GenerativeModel(nn.Module):
    def __init__(self, max_strks=2, res=28, z_where_type='3',
                                                    execution_guided=False, 
                                                    transform_z_what=True,
                                                    hidden_dim=200,
                                                    z_what_dim=50):
        super().__init__()
        self.max_strks = max_strks
        self.execution_guided=execution_guided

        # Prior parameters
        
        # z_what
        self.z_what_dim = z_what_dim
        self.register_buffer("z_what_loc", torch.zeros(z_what_dim))
        self.register_buffer("z_what_std", torch.ones(z_what_dim))

        # z_pres
        self.register_buffer("z_pres_prob", torch.zeros(self.max_strks)+.5)

        # z_where: default '3'
        self.z_where_type = z_where_type
        z_where_loc, z_where_std, self.z_where_dim = util.init_z_where(
                                                            self.z_where_type)
        self.register_buffer("z_where_loc", z_where_loc.expand(self.max_strks, 
                                                            self.z_where_dim))
        self.register_buffer("z_where_std", z_where_std.expand(self.max_strks,
                                                            self.z_where_dim))
        
        # img
        self.register_buffer("imgs_dist_std", torch.ones(1, res, res) / 3)
        
        # Decoder
        self.decoder = Decoder(z_what_dim=z_what_dim, 
                                img_dim=[1, res, res],
                                hidden_dim=hidden_dim,
                                num_layers=2)
        
        # Image renderer, and its parameters
        self.res = res

    def control_points_dist(self, bs=[1, 3]):
        '''(z_what Prior) Batched control points distribution
        Return: dist of
            bs: [bs, max_strks]
        '''
        loc, std = self.z_what_loc, self.z_what_std

        dist =  Independent(
                    Normal(loc, std), reinterpreted_batch_ndims=1).expand(bs)

        assert (dist.event_shape == torch.Size([self.z_what_dim]) and 
                dist.batch_shape == torch.Size([*bs]))
        return dist
        
    def presence_dist(self, bs=[1, 3]):
        '''(z_pres Prior) Batched presence distribution 
        Return: dist of
            bs [bs]
        '''
        z_pres_p = self.z_pres_prob
        
        dist = Independent(
            Bernoulli(z_pres_p), reinterpreted_batch_ndims=0,
        ).expand(bs)

        assert (dist.event_shape == torch.Size([]) and 
                dist.batch_shape == torch.Size([*bs]))
        return dist

    def transformation_dist(self, bs=[1, 3]):
        '''(z_where Prior) Batched transformation distribution
        Return: dist of
            bs [bs]
        '''
        loc, std = self.z_where_loc, self.z_where_std

        dist = Independent(
            Normal(loc, std), reinterpreted_batch_ndims=1,
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
        bs = imgs_dist_loc.shape[0]

        dist = Independent(Laplace(imgs_dist_loc, self.imgs_dist_std), 
                            reinterpreted_batch_ndims=3)
        assert (dist.event_shape == torch.Size([1, self.res, self.res]) and 
                dist.batch_shape == torch.Size([bs]))
        return dist

    def renders_imgs(self, latents):
        '''Batched img rendering
        Args:
            latents: 
                z_pres: [bs, n_strks] 
                z_what: [bs, n_strks, z_what_dim]
                z_where:[bs, n_strks, z_where_dim]
        Return:
            images: [bs, 1 (channel), H, W]
        '''
        z_pres, z_what, z_where = latents
        bs, n_strks = z_pres.shape
        
        # Get rendered image: [bs, n_strk, n_channel (1), H, W]
        imgs = self.decoder(z_what)  
        imgs = imgs * z_pres[:, :, None, None, None]

        # reshape image for further processing
        imgs = imgs.view(bs*n_strks, 1, self.res, self.res)

        # Get affine matrix: [bs * n_strk, 2, 3]
        z_where_mtrx = util.get_affine_matrix_from_param(
                                z_where.view(bs*n_strks, -1), 
                                self.z_where_type)
        imgs = util.inverse_spatial_transformation(imgs, z_where_mtrx)

        # max normalized so each image has pixel values [0, 1]
        # [bs*n_strk, n_channel (1), H, W]
        # imgs = util.normalize_pixel_values(imgs, method="maxnorm",)

        # Change back to [bs, n_strk, n_channel (1), H, W]
        imgs = imgs.view(bs, n_strks, 1, self.res, self.res)

        # Change to [bs, n_channel (1), H, W] through `sum`
        imgs = imgs.sum(1) 
        # imgs = util.normalize_pixel_values(imgs, method="maxnorm") # maxnorm doesn' work
        imgs = util.normalize_pixel_values(imgs, method="tanh", slope=0.6) # tanh works


        try:
            assert not imgs.isnan().any()
        except:
            breakpoint()
        # assert imgs.max() <= 1.
        return imgs

    def renders_glimpses(self, z_what):
        '''Get glimpse reconstruction from z_what control points
        Args:
            z_what: [bs, n_strk, n_strks, 2]
        Return:
            recon: [bs, n_strks, 1, res, res]
        '''
        assert len(z_what.shape) == 3, \
                                    f"z_what shape: {z_what.shape} isn't right"
        bs, n_strks, z_what_dim = z_what.shape[:3]
        res = self.res
        # Get rendered image: [bs, n_strk, n_channel (1), H, W]
        recon = self.decoder(z_what)
        recon = recon.view(bs*n_strks, 1, self.res, self.res)
        # recon = util.normalize_pixel_values(recon, method="maxnorm",)
        recon = recon.view(bs, n_strks, 1, self.res, self.res)

        return recon

    def log_prob(self, latents, imgs, z_pres_mask, canvas, decoder_param=None,
                    z_prior=None):
        '''
        Args:
            latents: 
                z_pres: [bs, max_strks] 
                z_what: [bs, max_strks, z_what_dim, 2 (x, y)]
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
        log_likelihood = self.img_dist(latents=latents, 
                                       canvas=canvas).log_prob(imgs)
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

class Guide(nn.Module):
    def __init__(self, max_strks=2, img_dim=[1,28,28],
                                            hidden_dim=256, 
                                            z_where_type='3', 
                                            execution_guided=False,
                                            exec_guid_type=None,
                                            z_what_dim=50,
                                            feature_extractor_sharing=True):
        super().__init__()
        # Parameters
        self.max_strks = max_strks
        self.img_dim = img_dim
        self.img_numel = np.prod(img_dim)
        self.hidden_dim = hidden_dim
        self.z_pres_dim = 1
        self.z_what_dim = z_what_dim
        self.z_where_type = z_where_type
        self.z_where_dim = util.init_z_where(self.z_where_type).dim

        # Internal renderer
        self.execution_guided = execution_guided
        self.exec_guid_type = exec_guid_type
        self.target_in_pos = 'RNN'

        if self.execution_guided:
            self.internal_decoder = GenerativeModel(
                                            z_where_type=self.z_where_type,
                                            z_what_dim=self.z_what_dim,
                                            max_strks=self.max_strks,
                                            res=img_dim[-1],
                                            execution_guided=execution_guided)
        # Inference networks
        # Module 1: front_cnn and pr_wr_rnn
        self.feature_extractor_sharing = feature_extractor_sharing
        # self.cnn_out_dim = 16928 if self.img_dim[-1] == 50 else 4608
        # self.feature_extractor_out_dim = 256
        # self.img_feature_extractor = util.init_cnn(
        #                                     n_in_channels=1,
        #                                     n_mid_channels=16,#32, 
        #                                     n_out_channels=32,#64,
        #                                     cnn_out_dim=self.cnn_out_dim,
        #                                     mlp_out_dim=
    #                                         self.feature_extractor_out_dim,
        #                                     mlp_hidden_dim=256,
        #                                     num_mlp_layers=1)
        self.cnn_out_dim = 2500 if self.img_dim[-1] == 50 else 784
        self.feature_extractor_out_dim = 2500 if self.img_dim[-1] == 50 else 784
        self.img_feature_extractor = lambda x: torch.reshape(x, (x.shape[0], -1)
                                                            )

        self.pr_wr_rnn_in_dim = (self.feature_extractor_out_dim + 
                                    self.z_pres_dim + 
                                    self.z_where_dim + 
                                    self.z_what_dim)

        self.pr_wr_rnn_hid_dim = 256
        self.pr_wr_rnn = torch.nn.GRUCell(self.pr_wr_rnn_in_dim, 
                                          self.pr_wr_rnn_hid_dim)

        # pr_wr_mlp:
        #   rnn hidden state -> (z_pres, z_where dist parameters)
        self.pr_wr_mlp_in_dim = self.pr_wr_rnn_hid_dim

        self.pr_wr_mlp = PresWhereMLP(in_dim=self.pr_wr_mlp_in_dim, 
                                      z_where_type=self.z_where_type,
                                      z_where_dim=self.z_where_dim)

        # Module 2: z_what_cnn, z_what_rnn, z_what_mlp
        # stn transformed image -> (`z_what_dim` control points)
        self.z_what_rnn_in_dim = (self.feature_extractor_out_dim)

        self.z_what_rnn_hid_dim = 256
        self.z_what_rnn = torch.nn.GRUCell(self.z_what_rnn_in_dim, 
                                            self.z_what_rnn_hid_dim)

        self.what_mlp_in_dim = self.z_what_rnn_hid_dim
        self.z_what_mlp = WhatMLP(in_dim=self.what_mlp_in_dim,
                                  z_what_dim=self.z_what_dim,
                                  hid_dim=hidden_dim,
                                  num_layers=2)

        # Module 3: Baseline (bl) rnn and regressor
        self.bl_hid_dim = 256
        self.bl_rnn = torch.nn.GRUCell((self.pr_wr_rnn_in_dim),
                                        self.bl_hid_dim)
        self.bl_regressor = nn.Sequential(
            nn.Linear(self.bl_hid_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
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
        (state, baseline_value, mask_prev, 
         z_pres_pms, z_where_pms, z_what_pms,
         z_pres_smpl, z_where_smpl, z_what_smpl, 
         z_pres_lprb, z_where_lprb, z_what_lprb,
         canvas, residual) = self.initialize_state(imgs)

        for t in range(self.max_strks):
            # following the online example
            mask_prev[:, t] = state.z_pres.squeeze()

            # Do one inference step and save results
            result = self.inference_step(state, imgs, canvas, residual)
            state = result['state']
            assert (state.z_pres.shape == torch.Size([bs, 1]) and
                    state.z_what.shape == 
                            torch.Size([bs, self.z_what_dim]) and
                    state.z_where.shape == 
                            torch.Size([bs, self.z_where_dim]))

            # Update and store the information
            # z_pres: [bs, 1]
            z_pres_smpl[:, t] = state.z_pres.squeeze(-1)
            # z_what: [bs, z_what_dim, 2];
            z_what_smpl[:, t] = state.z_what
            # z_where: [bs, z_where_dim]
            z_where_smpl[:, t] = state.z_where

            assert (result['z_pres_pms'].shape == 
                                              torch.Size([bs, 1]))
            #    and  result['z_what_pms'].shape == 
            #             torch.Size([bs, self.z_what_dim, 2, 2]) and
            #         result['z_where_pms'].shape == 
            #                 torch.Size([bs, self.z_where_dim, 2]))
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

            # Update the canvas
            if self.execution_guided:
                canvas_step = self.internal_decoder.renders_imgs((
                                                z_pres_smpl[:, t:t+1].clone(),
                                                z_what_smpl[:, t:t+1],
                                                z_where_smpl[:, t:t+1]))
                canvas = canvas + canvas_step
                if self.exec_guid_type == "residual":
                    # compute the residual
                    residual = torch.clamp(imgs - canvas, min=0.)

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
                           )    
        return data
        
    def inference_step(self, p_state, imgs, canvas, residual):
        '''Given previous (initial) state and input image, predict the current
        step latent distribution
        Args:
            p_state::GuideState
            imgs [bs, 1, res, res]
            canvas [bs, 1, res, res] or None
            residual [bs, 1, res, res]
        '''
        bs = imgs.size(0)

        # embed image
        img_embed = self.img_feature_extractor(imgs)

        # Predict z_pres, z_where from target and canvas
        pr_wr_mlp_in, pr_wr_rnn_in, h_l = self.get_pr_wr_mlp_in(img_embed, 
                                                                residual, 
                                                                p_state)
        (z_pres, 
         z_where, 
         z_pres_lprb, 
         z_where_lprb, 
         z_pres_p, 
         z_where_pms)  = self.process_z_l(pr_wr_mlp_in, p_state)


        # Get spatial transformed "crop" from input image
        trans_imgs = util.spatial_transform(
                                    imgs, 
                                    util.get_affine_matrix_from_param(
                                                z_where, 
                                                z_where_type=self.z_where_type))
        
        z_what_mlp_in, h_c = self.get_z_what_mlp_in(trans_imgs, p_state)
        z_what, z_what_lprb, z_what_pms = self.process_z_c(z_what_mlp_in, 
                                                            p_state, z_pres)

        # Compute baseline for z_pres
        # depending on previous latent variables only
        bl_h = self.bl_rnn(pr_wr_rnn_in.detach(), p_state.bl_h)
        baseline_value = self.bl_regressor(bl_h).squeeze() # shape (B,)
        baseline_value = baseline_value * p_state.z_pres.squeeze()
        
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

    def get_pr_wr_mlp_in(self, img_embed, residual, p_state):
        '''Get the input for `pr_wr_mlp` from the current img and p_state
        Args:
            img_embed [bs, embed_dim]
            canvas_embed [bs, 1, res, res] if self.execution_guided or None 
            p_state GuideState
        Return:
            pr_wr_mlp_in [bs, pr_wr_mlp_in_dim]
            pr_wr_rnn_in [bs, pr_wr_rnn_in_dim]
        '''
        bs = img_embed.shape[0]
        if self.execution_guided:
            # This is not exactly the same as before, previously we would 
            # concat the img and canvas and get 1 embedding.

            if self.exec_guid_type == 'residual':
                residual_embed = self.img_feature_extractor(residual
                                                            ).view(bs, -1)
                rnn_input = torch.cat([residual_embed, 
                            p_state.z_pres, p_state.z_where, p_state.z_what], 
                            dim=1)         
            else: raise NotImplementedError                                   
        else:
            rnn_input = torch.cat([img_embed, p_state.z_pres, p_state.z_where, 
                                                        p_state.z_what], dim=1)
        # Get the new h_l
        h_l = self.pr_wr_rnn(rnn_input, p_state.h_l)
        pr_wr_mlp_in = h_l
        return pr_wr_mlp_in, rnn_input, h_l

    def process_z_l(self, pr_wr_mlp_in, p_state):
        """Predict z_pres and z_where from `pr_wr_mlp_in`
        Args:
            pr_wr_mlp_in [bs, in_dim]: input based on input types
            p_state: GuideState
        Return:

        """
        bs = pr_wr_mlp_in.shape[0]

        # Predict presence and location from h
        z_pres_p, z_where_loc, z_where_scale = self.pr_wr_mlp(pr_wr_mlp_in)
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

        # If previous z_pres is 0, this z_pres should also be 0.
        # However, this is sampled from a Bernoulli whose probability is at
        # least eps. In the unlucky event that the sample is 1, we force this
        # to 0 as well.
        z_pres = z_pres * p_state.z_pres

        # log prob: log q(z_pres[i] | x, z_{<i}) if z_pres[i-1]=1, else 0
        # Mask with p_state.z_pres instead of z_pres. 
        # Keep if prev == 1, curr == 0 or 1; remove if prev == 0
        z_pres_lprb = z_pres_post.log_prob(z_pres).unsqueeze(-1) * p_state.z_pres
        assert z_pres_lprb.shape == torch.Size([bs, 1])
        
        # Sample z_where, get log_prob
        assert z_where_loc.shape == torch.Size([bs, self.z_where_dim])
        z_where_post = Independent(Normal(z_where_loc, z_where_scale),
                                                    reinterpreted_batch_ndims=1)
        assert (z_where_post.event_shape == torch.Size([self.z_where_dim]) and
                z_where_post.batch_shape == torch.Size([bs]))        

        z_where = z_where_post.rsample()
        z_where_lprb = z_where_post.log_prob(z_where).unsqueeze(-1) * z_pres
        assert z_where_lprb.shape == torch.Size([bs, 1])
        # z_where_lprb = z_where_lprb.squeeze()

        return (z_pres, z_where, z_pres_lprb, z_where_lprb, z_pres_p, 
                (z_where_loc, z_where_scale))

    def get_z_what_mlp_in(self, trans_imgs, p_state):
        '''Get the input for the z_what_mlp
        '''
        # Sample z_what, get log_prob
        bs = trans_imgs.shape[0]
        # [bs, z_what_dim, 2, 1]
            
        z_what_rnn_in = self.img_feature_extractor(trans_imgs) 
        h_c = self.z_what_rnn(z_what_rnn_in, p_state.h_c)
        mlp_in = h_c
        
        return mlp_in, h_c

    def process_z_c(self, z_what_mlp_in, p_state, z_pres):
        bs = z_what_mlp_in.shape[0]
        z_what_loc, z_what_scale = self.z_what_mlp(z_what_mlp_in)
        # [bs, z_dim]
        z_what_loc = z_what_loc.view([bs, self.z_what_dim])
        z_what_scale = z_what_scale.view([bs, self.z_what_dim])
            
        z_what_post = Independent(Normal(z_what_loc, z_what_scale), 
                                        reinterpreted_batch_ndims=1)
        assert (z_what_post.event_shape == torch.Size([self.z_what_dim]) and
                z_what_post.batch_shape == torch.Size([bs]))

        # [bs, z_what_dim, 2] 
        z_what = z_what_post.rsample()

        # log_prob(z_what): [bs, 1]
        # z_pres: [bs, 1]
        z_what_lprb = z_what_post.log_prob(z_what).unsqueeze(-1) * z_pres
        return z_what, z_what_lprb, (z_what_loc, z_what_scale)

    def initialize_state(self, imgs):
        bs = imgs.size(0)

        # Init model state for performing inference
        state = GuideState(
            h_l=torch.zeros(bs, self.pr_wr_rnn_hid_dim, device=imgs.device),
            h_c=torch.zeros(bs, self.z_what_rnn_hid_dim, device=imgs.device),
            bl_h=torch.zeros(bs, self.bl_hid_dim, device=imgs.device),
            z_pres=torch.ones(bs, 1, device=imgs.device),
            z_where=torch.zeros(bs, self.z_where_dim, device=imgs.device),
            z_what=torch.zeros(bs, self.z_what_dim, device=imgs.device),
        )

        # z samples for each step
        z_pres_smpl = torch.ones(bs, self.max_strks, device=imgs.device)
        z_what_smpl = torch.zeros(bs, self.max_strks, self.z_what_dim, 
                                                            device=imgs.device)
        z_where_smpl = torch.ones(bs, self.max_strks, self.z_where_dim, 
                                                            device=imgs.device)
        # z distribution parameters for each step
        z_pres_pms = torch.ones(bs, self.max_strks, device=imgs.device)
        z_what_pms = torch.zeros(bs, self.max_strks, self.z_what_dim, 2, 
                                                            device=imgs.device)
        z_where_pms = torch.ones(bs, self.max_strks, self.z_where_dim, 2, 
                                                            device=imgs.device)
        # z log posterior probability (lprb) for each step
        z_pres_lprb = torch.zeros(bs, self.max_strks, device=imgs.device)
        z_what_lprb = torch.zeros(bs, self.max_strks, device=imgs.device)
        z_where_lprb = torch.zeros(bs, self.max_strks, device=imgs.device)

        # z log prior probability (lprb) for each step;

        # baseline_value
        baseline_value = torch.zeros(bs, self.max_strks, device=imgs.device)

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
        
        return (state, baseline_value, mask_prev, 
                z_pres_pms, z_where_pms, z_what_pms,
                z_pres_smpl, z_where_smpl, z_what_smpl, 
                z_pres_lprb, z_where_lprb, z_what_lprb,
                canvas, residual)

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
                nn.Linear(256, 1 + z_where_dim * 2),
            )
    
    def forward(self, h):
        # todo make capacible with other z_where_types
        z = self.seq(h)
        z_pres_p = torch.sigmoid(z[:, :1])
        # z_pres_p = util.constrain_parameter(z[:, :1], min=0, max=1.) + 1e-6
        if self.type == '3':
            # z_where_loc_scale = z[:, 1:2]
            # z_where_loc_shift = z[:, 2:4]
            z_where_loc = z[:, 1:4]
            z_where_scale = F.softplus(z[:, 4:])
        # elif self.type == '4_rotate':
        #     z_where_scale_loc = util.constrain_parameter(z[:, 1:2], min=0, 
        #                                                                 max=1)
        #     z_where_shift_loc = util.constrain_parameter(z[:, 2:4], min=-1, 
        #                                                                 max=1)
        #     z_where_ang_loc = util.constrain_parameter(z[:, 4:5], min=-45, 
        #                                                                 max=45)
        #     z_where_loc = torch.cat(
        #         [z_where_scale_loc, z_where_shift_loc, z_where_ang_loc], dim=1)
                
        #     z_where_scale = F.softplus(z[:, 5:9]) + 1e-6
        #     return z_pres_p, z_where_loc, z_where_scale

        else: 
            raise NotImplementedError
        return z_pres_p, z_where_loc, z_where_scale
    

class WhatMLP(nn.Module):
    def __init__(self, in_dim=256, z_what_dim=50, hid_dim=512, num_layers=2):
        super().__init__()
        self.z_what_dim = z_what_dim
        self.out_dim = z_what_dim * 2
        self.mlp = util.init_mlp(in_dim=in_dim, 
                                out_dim=self.out_dim,
                                hidden_dim=hid_dim,
                                num_layers=num_layers)
    def forward(self, x):
        # out = constrain_parameter(self.mlp(x), min=.3, max=.7)
        out = self.mlp(x)
        z_what_loc = F.tanh(out[:, 0:self.z_what_dim])
        z_what_scale = F.softplus(out[:, self.z_what_dim:]) + 1e-6
        # out = torch.cat([z_what_loc, z_what_scale])
        # out = torch.sigmoid(out)
        # z_what_loc = out[:, 0:(int(self.out_dim/2))]
        # z_what_scale = out[:, (int(self.out_dim/2)):] + 1e-6
        return z_what_loc, z_what_scale

class Decoder(nn.Module):
    def __init__(self, z_what_dim=50, 
                       img_dim=[1, 50, 50], 
                       hidden_dim=256, 
                       num_layers=2,
                       bias=-2.0):
        super().__init__()
        self.img_dim = img_dim
        self.z_what_dim = z_what_dim
        self.out_dim = np.prod(img_dim)
        self.net = util.init_mlp(in_dim=z_what_dim,
                                 out_dim=self.out_dim, 
                                 hidden_dim=hidden_dim, 
                                 num_layers=num_layers)
        self.bias = bias
    
    def forward(self, z_what):
        out = self.net(z_what)
        out_loc = torch.sigmoid(out + self.bias
                                        ).view(*z_what.shape[:2], *self.img_dim)
        return out_loc