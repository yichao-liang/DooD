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
GuideState = namedtuple('GuideState', 'h bl_h z_pres z_where z_what')
GuideReturn = namedtuple('GuideReturn', ['z_smpl', 'z_lprb', 'mask_prev',
                                # 'z_pres_dist', 'z_what_dist','z_where_dist', 
                                'baseline_value'])

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
    def __init__(self, max_strks=2, pts_per_strk=5):
        super().__init__()
        self.max_strks = max_strks
        self.pts_per_strk = pts_per_strk


        # Prior parameters
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

    def control_points_dist(self, batch_shape=[1, 3]):
        '''(z_what Prior) Batched control points distribution
        Return: dist of
            batch_shape: [batch_shape, max_strks]
            event_shape: [pts_per_strk, 2]
        '''
        dist =  Independent(
            Normal(self.pts_loc, self.pts_std), reinterpreted_batch_ndims=2,
        ).expand(batch_shape)
        assert (dist.event_shape == torch.Size([self.pts_per_strk, 2]) and 
                dist.batch_shape == torch.Size([*batch_shape]))
        return dist
        
    def presence_dist(self, batch_shape=[1, 3]):
        '''(z_pres Prior) Batched presence distribution 
        Return: dist of
            batch_shape [batch_shape]
            event_shape [max_strks]
        '''
        dist = Independent(
            Bernoulli(self.z_pres_prob), reinterpreted_batch_ndims=0,
        ).expand(batch_shape)

        assert (dist.event_shape == torch.Size([]) and 
                dist.batch_shape == torch.Size([*batch_shape]))
        return dist

    def transformation_dist(self, batch_shape=[1, 3]):
        '''(z_where Prior) Batched transformation distribution
        Return: dist of
            batch_shape [batch_shape]
            event_shape [max_strks, z_where_dim (3-5)]
        '''
        dist = Independent(
            Normal(self.z_where_loc, self.z_where_std), 
            reinterpreted_batch_ndims=1,
        ).expand(batch_shape)
        assert (dist.event_shape == torch.Size([self.z_where_dim]) and 
                dist.batch_shape == torch.Size([*batch_shape]))
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
        z_where = util.get_affine_matrix_from_param(z_where.view(
                                np.prod(b_strk_shape), -1), self.z_where_type)

        # [batch_size, n_strk, n_channel (1), H, W]
        imgs_dist_loc = self.bezier(z_what, sigma=util.constrain_parameter(
                                                self.sigma, min=.01, max=.05),
                                            keep_strk_dim=True)  
        imgs_dist_loc = imgs_dist_loc * z_pres[:, :, None, None, None]

        # max normalized so each image has pixel values [0, 1]
        # size: [batch_size*n_strk, n_channel (1), H, W]
        imgs_dist_loc = util.normalize_pixel_values(imgs_dist_loc.view(
                                np.prod(b_strk_shape), 1, self.res, self.res), 
                                method="maxnorm")

        # Transform back. z_where goes from a standardized img to the observed.
        imgs_dist_loc = util.inverse_spatial_transformation(imgs_dist_loc, 
                                                            z_where)

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
                dist.batch_shape == torch.Size([b_strk_shape[0]]))
        return dist

    def log_prob(self, latents, imgs, z_pres_mask):
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
        batch_shape = torch.Size([*shape, self.max_strks])

        # assuming z_pres here are in the right format, i.e. no 1s following 0s
        # log_prob output: [batch_size, max_strokes]
        # z_pres_mask: [batch_size, max_strokes]
        log_prior =  ZLogProb(
                    z_what=(self.control_points_dist(batch_shape).log_prob(z_what) * 
                                                                z_pres_mask),
                    z_pres=(self.presence_dist(batch_shape).log_prob(z_pres) * 
                                                                z_pres_mask),
                    z_where=(self.transformation_dist(batch_shape).log_prob(z_where) * 
                                                                z_pres),
                    )

        # Likelihood
        log_likelihood = self.img_dist(latents).log_prob(imgs)
        return log_prior, log_likelihood

    def sample(self, batch_shape=[1]):
        # todo 2: with the guide, z_pres are in the right format, but the sampled 
        # todo 2: ones are not
        # todo although sample is not used at this moment
        raise NotImplementedError("Haven't made sure the sampled z_pres are legal")
        z_pres = self.control_points_dist(batch_shape).sample()
        z_what = self.presence_dist(batch_shape).sample()
        z_where = self.transformation_dist(batch_shape).sample()
        latents = ZSample(z_pres, z_what, z_where)
        imgs = self.img_dist(latents).sample()

        return imgs, latents


class Guide(nn.Module):
    def __init__(self, max_strks=2, pts_per_strk=5, img_dim=[1,28,28],
                hidden_dim=256):
        super().__init__()

        # Parameters
        self.max_strks = max_strks
        self.pts_per_strk = pts_per_strk
        self.img_numel = np.prod(img_dim)
        self.hidden_dim = hidden_dim
        self.z_pres_dim = 1
        self.z_what_dim = self.pts_per_strk * 2
        self.z_where_type = '3'
        self.z_where_dim = util.init_z_where(self.z_where_type).dim

        # Inference distribution
        # front_cnn:
        #   image -> `cnn_out_dim`-dim hidden representation
        self.cnn_out_dim = 256
        self.front_cnn = util.init_cnn(in_dim=img_dim, out_dim=self.cnn_out_dim,
                        num_mlp_layers=1, mlp_hidden_dim=self.cnn_out_dim,
                        n_in_channels=1, n_mid_channels=8, n_out_channels=10,)
        # rnn:
        #   (image encoding; previous_z) -> `rnn_hid_dim`-dim hidden state
        self.rnn_in_dim = (self.cnn_out_dim + self.z_pres_dim + self.z_what_dim
                                                             + self.z_where_dim)
        self.rnn_hid_dim = 256
        self.rnn = torch.nn.GRUCell(self.rnn_in_dim, self.rnn_hid_dim)

        # pres_where_mlp:
        #   rnn hidden state -> (z_pres, z_where dist parameters)
        self.pres_where_mlp = util.Predictor(in_dim=self.rnn_hid_dim, 
                                             z_where_type=self.z_where_type,
                                             z_where_dim=self.z_where_dim)

        # z_what_cnn
        # stn transformed image -> (`pts_per_strk` control points)
        self.z_what_cnn = util.init_cnn(in_dim=img_dim, 
                                        out_dim=self.pts_per_strk*2*2,
                                        num_mlp_layers=1, mlp_hidden_dim=256,
                                        n_in_channels=1, n_mid_channels=8, 
                                        n_out_channels=10,)

        # Baseline (bl) rnn and regressor
        self.bl_hid_dim = 256
        self.bl_rnn = torch.nn.GRUCell(self.rnn_in_dim, self.bl_hid_dim)
        self.bl_regressor = nn.Sequential(
            nn.Linear(self.bl_hid_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, imgs):
        '''
        Args: 
            img: [batch_size, 1, H, W]
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
        batch_size = imgs.size(0)

        # Init model state for performing inference
        state = GuideState(
            h=torch.zeros(batch_size, self.rnn_hid_dim, device=imgs.device),
            bl_h=torch.zeros(batch_size, self.bl_hid_dim, device=imgs.device),
            z_pres=torch.ones(batch_size, 1, device=imgs.device),
            z_where=torch.zeros(batch_size, self.z_where_dim, device=imgs.device),
            z_what=torch.zeros(batch_size, self.z_what_dim, device=imgs.device),
        )

        # z samples for each step
        z_pres_smpl = torch.ones(batch_size, self.max_strks, device=imgs.device)
        z_what_smpl = torch.zeros(batch_size, self.max_strks, self.pts_per_strk, 
                                                         2, device=imgs.device)
        z_where_smpl = torch.ones(batch_size, self.max_strks, self.z_where_dim, 
                                                            device=imgs.device)
        # z distribution parameters for each step
        z_pres_pms = torch.ones(batch_size, self.max_strks, device=imgs.device)
        z_what_pms = torch.zeros(batch_size, self.max_strks, self.pts_per_strk, 
                                                      2, 2, device=imgs.device)
        z_where_pms = torch.ones(batch_size, self.max_strks, self.z_where_dim, 
                                                         2, device=imgs.device)
        # z log-prob (lprb) for each step
        z_pres_lprb = torch.zeros(batch_size, self.max_strks, device=imgs.device)
        z_what_lprb = torch.zeros(batch_size, self.max_strks, device=imgs.device)
        z_where_lprb = torch.zeros(batch_size, self.max_strks, device=imgs.device)

        # baseline_value
        baseline_value = torch.zeros(batch_size, self.max_strks, device=imgs.device)

        '''Signal mask for each step
        At time t, `mask_prev` stroes whether prev.z_pres==0, `mask_curr` 
            stores whether z_pres==0 after an `inference_step`.
        The first element of `mask_prev` is also 1, while the first element of 
            `mask_curr` depends on the outcome of the `inference_step`.
        `mask_prev` can be used to mask out the KL of z_pres, because the first
            appearence z_pres==0 is also accounted.
        `mask_curr` can be used to mask out the KL of z_what, z_where, and
            reconstruction, because they are ignored since the first z_pres==0
        '''
        mask_prev = torch.ones(batch_size, self.max_strks, device=imgs.device)

        for t in range(self.max_strks):
            # following the online example
            mask_prev[:, t] = state.z_pres.squeeze()

            # Do one inference step and save results
            result = self.inference_step(state, imgs)

            state = result['state']

            assert (state.z_pres.shape == torch.Size([batch_size, 1]) and
                    state.z_what.shape == 
                            torch.Size([batch_size, self.pts_per_strk, 2]) and
                    state.z_where.shape == 
                            torch.Size([batch_size, self.z_where_dim]))
            #z_what: [bs, pts_per_strk, 2];
            # z_where: [bs, z_where_dim]
            z_pres_smpl[:, t] = state.z_pres.squeeze(-1)
            z_what_smpl[:, t] = state.z_what
            z_where_smpl[:, t] = state.z_where

            assert (result['z_pres_pms'].shape == 
                                              torch.Size([batch_size, 1]) and
                    result['z_what_pms'].shape == 
                        torch.Size([batch_size, self.pts_per_strk, 2, 2]) and
                    result['z_where_pms'].shape == 
                            torch.Size([batch_size, self.z_where_dim, 2]))
            z_pres_pms[:, t] = result['z_pres_pms'].squeeze(-1)
            z_what_pms[:, t] = result['z_what_pms']
            z_where_pms[:, t] = result['z_where_pms']

            assert (result['z_pres_lprb'].shape == torch.Size([batch_size, 1]) and
                    result['z_what_lprb'].shape == torch.Size([batch_size, 1]) and
                    result['z_where_lprb'].shape == torch.Size([batch_size, 1]))
            z_pres_lprb[:, t] = result['z_pres_lprb'].squeeze(-1)
            z_what_lprb[:, t] = result['z_what_lprb'].squeeze(-1)
            z_where_lprb[:, t] = result['z_where_lprb'].squeeze(-1)
            baseline_value[:, t] = result['baseline_value'].squeeze(-1)

        # todo 1: init the distributions
        # z_pres_dist = None
        # z_what_dist = None
        # z_where_dist = None

        data = GuideReturn(z_smpl=ZSample(
                                z_pres=z_pres_smpl, 
                                z_what=z_what_smpl,
                                z_where=z_where_smpl),
                        #    z_pres_dist=z_pres_dist,
                        #    z_what_dist=z_what_dist,
                        #    z_where_dist=z_where_dist, 
                           z_lprb=ZLogProb(
                                z_pres=z_pres_lprb,
                                z_what=z_what_lprb,
                                z_where=z_where_lprb),
                           baseline_value=baseline_value,
                           mask_prev=mask_prev)    
        return data
        
    def inference_step(self, p_state, imgs):
        '''Given previous (initial) state and input image, predict the current
        step latent distribution
        '''
        batch_size = imgs.size(0)

        # embed image, Input embedding, previous states, Output rnn encoding hid
        cnn_embed = self.front_cnn(imgs).view(batch_size, -1)
        rnn_input = torch.cat([cnn_embed, p_state.z_pres, 
                    p_state.z_what.view(batch_size,-1), p_state.z_where], dim=1)

        hid = self.rnn(rnn_input, p_state.h)

        # Predict presence and location from h
        z_pres_p, z_where_loc, z_where_scale = self.pres_where_mlp(hid)

        # If previous z_pres is 0, force z_pres to 0
        z_pres_p = z_pres_p * p_state.z_pres

        # Numerical stability
        eps = 1e-12
        z_pres_p = z_pres_p.clamp(min=eps, max=1.0-eps)

        # Sample z_pres
        assert z_pres_p.shape == torch.Size([batch_size, 1])
        z_pres_post = Independent(Bernoulli(z_pres_p), 
                                        reinterpreted_batch_ndims=1)
        assert (z_pres_post.event_shape == torch.Size([1]) and
                z_pres_post.batch_shape == torch.Size([batch_size]))
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
        assert z_where_loc.shape == torch.Size([batch_size, self.z_where_dim])
        z_where_post = Independent(Normal(z_where_loc, z_where_scale),
                                        reinterpreted_batch_ndims=1)
        assert (z_where_post.event_shape == torch.Size([self.z_where_dim]) and
                z_where_post.batch_shape == torch.Size([batch_size]))        
        z_where = z_where_post.rsample()
        z_where_lprb = z_where_post.log_prob(z_where).unsqueeze(-1) * z_pres
        # z_where_lprb = z_where_lprb.squeeze()

        # Get spatial transformed "crop" from input image
        trans_imgs = util.spatial_transform(imgs, 
                        util.get_affine_matrix_from_param(z_where, 
                                                z_where_type=self.z_where_type))
        
        # Sample z_what, get log_prob
        # todo 1: make a specialized network that return this directly
        # [batch_size, pts_per_strk, 2, 1]
        z_what_loc, z_what_scale = (
            torch.sigmoid(self.z_what_cnn(trans_imgs)
            ).view([batch_size, self.pts_per_strk, 2, 2]).chunk(2, -1)
            )
        # [batch_size, pts_per_strk, 2]
        z_what_loc = z_what_loc.squeeze(-1)
        z_what_scale = z_what_scale.squeeze(-1) + 1e-3
        z_what_post = Independent(Normal(z_what_loc, z_what_scale), 
                                        reinterpreted_batch_ndims=2)
        assert (z_what_post.event_shape == torch.Size([self.pts_per_strk, 2]) and
                z_what_post.batch_shape == torch.Size([batch_size]))

        # [batch_size, pts_per_strk, 2] 
        z_what = z_what_post.rsample()
        # log_prob(z_what): [batch_size, 1]
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
            bl_h=bl_h
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
            'baseline_value': baseline_value
        }
        return out
