import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Laplace, Dirichlet
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

class GenerativeModel(nn.Module):
    def __init__(self, ctrl_pts_per_strk=5, 
                        prior_dist='Normal', 
                        likelihood_dist='Normal',
                        strks_per_img=1):
        """in the base model we pre-specify the number of control points and 
        work with data that has 1 stroke.

        Args:
            ctrl_pts_per_strk (int): number of control points per stroke
            prior_dist (str): currently support ['Normal', 'Dirichlet']
        """

        super().__init__()
        self.strks_per_img = strks_per_img
        self.ctrl_pts_per_strk = ctrl_pts_per_strk

        # num_stroke set to 1
        # Prior and likelihood distribution.
        self.prior_dist = prior_dist
        self.likelihood_dist = likelihood_dist
        if prior_dist == 'Normal':
            self.register_buffer("control_points_loc", 
                                    torch.zeros(self.strks_per_img, 
                                    self.ctrl_pts_per_strk, 2) + 0.5)
            self.register_buffer("control_points_scale", 
                                    torch.ones(self.strks_per_img, 
                                    self.ctrl_pts_per_strk, 2) / 5)
                # divide by 5 works
                # shouldn't use value too small, e.g. .1, which leads to no mass
                # on point around 1.
        elif prior_dist == 'Dirichlet':
            # ctrl_pts_per_strk * 2 for x, y coordinates. the last dim can't be
            # used for x, y because it has to add up to 1.
            self.register_buffer("concentration",
                                torch.ones(self.strks_per_img, self.ctrl_pts_per_strk * 2, 2))
        elif prior_dist == 'Uniform':
            self.register_buffer("uniform_low",
                    torch.zeros(self.strks_per_img, self.ctrl_pts_per_strk, 2) - 10)
            self.register_buffer("uniform_high",
                    torch.ones(self.strks_per_img, self.ctrl_pts_per_strk, 2) + 9)
        else:
            raise NotImplementedError

        # Image renderer
        self.bezier = Bezier(res=28, steps=500, method='bounded')

        """For thicker stroke, and blur control
        `sigma` is for passing in the renderer, should make sure it's positive.
        `sigma` Controls the bandwidth of the Gaussian kernel for rendering. The
        higher, the larger range of curve points that it takes into 
        consideration.
        """
        # To make it learnable:
        self.sigma = torch.nn.Parameter(torch.tensor(6.), requires_grad=True)
        # self.sigma = torch.log(torch.tensor(.04))
        self.register_buffer("dilation_kernel", torch.ones(2,2))
        self.register_buffer("erosion_kernel", torch.ones(3,3))
        self.norm_pixel_method = 'tanh' # maxnorm or tanh
        self.tanh_norm_slope = torch.nn.Parameter(torch.tensor(6.), 
                                                            requires_grad=True)
        # self.norm_pixel_method = 'maxnorm'
        # self.gauss = kornia.filters.GaussianBlur2d((7, 7), (5.5, 5.5))
    
    def control_points_dist(self):
        if self.prior_dist == 'Normal':
            return torch.distributions.Independent(
                torch.distributions.Normal(self.control_points_loc,
                                            self.control_points_scale),
                reinterpreted_batch_ndims=3,
            )
        elif self.prior_dist == 'Dirichlet':
            return torch.distributions.Independent(
                torch.distributions.Dirichlet(self.concentration),
                reinterpreted_batch_ndims=2,
            )
        elif self.prior_dist == 'Uniform':
            return torch.distributions.Independent(
                torch.distributions.Uniform(self.uniform_low, 
                                            self.uniform_high),
                reinterpreted_batch_ndims=2,
            )
        else:
            raise NotImplementedError
                                    
    def control_points_dist_b(self, batch_shape):
        '''batch of control points where each member corresponds to an image
        '''
        # batch_size = torch.Size([batch_shape])
        return self.control_points_dist().expand(batch_shape)

    def imgs_dist(self, control_points):
        # todo
        pass

    def img_dist_b(self, control_points_b,):
        '''
        Args:
            control_points_b: tensor of shape:
                [batch, 1 stroke, num_control_points, 2]
        Return:
            distribution of shape: [*shape, H, W]
        '''
        batch_dim = control_points_b.shape[0]

        if self.prior_dist == "Dirichlet":
            # chunk, reshape from batch_dim, 1, 2*nControlPoints, 2 to
            # batch_dim, 1, nControlPoints, 2
            control_points_b = control_points_b.chunk(2, -1)[0].\
                                                reshape(batch_dim, 1, -1 , 2)

        if self.norm_pixel_method == 'maxnorm':
            sigma_min, sigma_max = .01, .05
        elif self.norm_pixel_method == 'tanh':
            sigma_min, sigma_max = .01, .05

        imgs_dist_loc = self.bezier(control_points_b, 
                                # sigma=torch.exp(self.sigma)) # positive
                                  sigma=util.constrain_parameter(self.sigma, 
                                         min=sigma_min, max=sigma_max))  
        
        # max normalized so each image has pixel values [0, 1]
        imgs_dist_loc = util.normalize_pixel_values(imgs_dist_loc, 
                                                            method="maxnorm")
        # useful with maxnorm normalize
        if self.norm_pixel_method == 'maxnorm':
            imgs_dist_loc = dilation(imgs_dist_loc, self.dilation_kernel, 
                                                max_val=1.)
        elif self.norm_pixel_method == 'tanh':
            # the slope of the tanh experiment can be further tuned.
            imgs_dist_loc = util.normalize_pixel_values(imgs_dist_loc, 
                            method=self.norm_pixel_method,
                            slope=util.constrain_parameter(self.tanh_norm_slope,
                                                            min=.1,max=.7))
                                        
        # useful with sigmoid norm
        # imgs_dist_loc = erosion(imgs_dist_loc, self.erosion_kernel, max_val=1.)

        if self.stn_transform is not None:
            imgs_dist_loc = util.inverse_stn_transformation(imgs_dist_loc, 
                                                            self.stn_transform)
            imgs_dist_loc = util.normalize_pixel_values(imgs_dist_loc,
                                                method=self.norm_pixel_method)

        assert not imgs_dist_loc.isnan().any()
        assert imgs_dist_loc.max() <= 1.

        if self.likelihood_dist=='Laplace':
            event_dist = Laplace
            imgs_dist_std = torch.ones_like(imgs_dist_loc) 
        elif self.likelihood_dist=='Normal':
            event_dist = Normal
            imgs_dist_std = torch.ones_like(imgs_dist_loc)/ 100
        else:
            raise NotImplementedError

        dist = torch.distributions.Independent(
                event_dist(imgs_dist_loc, imgs_dist_std), 
                reinterpreted_batch_ndims=3
            )
        return dist
    
    def log_prob(self, latents, imgs):
        """Evaluates log p(z, x)
        Args:
            latent:
                control_points_b: [*shape, 1 stroke, num_points, 2]
            imgs: [*shape, img_H, img_W]
        """
        shape = imgs.shape[:-3]

        # rescale the target img to be of the same as the output
        # batch_dim = latents.shape[0]
        # max_per_recon = self.bezier(latents.clone().detach()).reshape(batch_dim, -1).max(1)[0]
        # max_per_recon = max_per_recon.reshape(batch_dim, 1, 1, 1)
        # imgs = imgs * max_per_recon

        log_prior, log_likelihood = 0, 0 
        if ((self.prior_dist == 'Dirichlet') and (latents.shape[-2] * 2 == 
                        self.control_points_dist_b(shape).event_shape[-2])):
            '''This happens when e.g. we use a *Normal inference dist* with output
            size [b, strks_per_img, n_points, 2 (for x, y)] with a *Dir prior
            dist* which has shape [b, strks_per_img, n_points*2 (for x, y), 2
            (which adds up to 1)].'''
            latents = latents.view([*shape, self.strks_per_img, 
                                                self.ctrl_pts_per_strk * 2, 1])
            latents = torch.cat([latents, torch.ones_like(latents)-latents], 
                                                                        dim=-1)

        log_prior = self.control_points_dist_b(shape).log_prob(latents)
        log_likelihood = self.img_dist_b(latents).log_prob(imgs)
        # log_joint = log_prior + log_likelihood
        return log_prior, log_likelihood
        
    def sample(self, batch_shape=[1]):
        """Sample from p(z, x)
        """
        control_points_b = self.control_points_dist_b(batch_shape).sample()
        imgs = self.img_dist_b(control_points_b).sample()
        return imgs, control_points_b

class Guide(nn.Module):
    def __init__(self, strks_per_img=1, ctrl_pts_per_strk=5, img_dim=[1, 28, 28], 
                hidden_dim=256, num_layers=2, dist="Normal", net_type="CNN"):
        super().__init__()

        # Parameters
        self.img_dim = np.prod(img_dim)
        self.strks_per_img = strks_per_img
        self.ctrl_pts_per_strk = ctrl_pts_per_strk
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.stn_transform = None

        # Inference distribution
        self.dist = dist
        if dist == 'Dirichlet':
            # _ * 2 (for x, y) * 2 (for concentration, 2 values for an value)
            self.output_dim = self.strks_per_img*self.ctrl_pts_per_strk*2*2
        elif dist == 'Normal':
            # _ * 2 (for x, y) * 2 (for loc, std)
            self.output_dim = self.strks_per_img*self.ctrl_pts_per_strk*2*2
        else: raise NotImplementedError

        self.net_type = net_type
        if net_type == 'MLP':
            self.control_points_net = util.init_mlp(
                in_dim=self.img_dim,
                out_dim=self.output_dim,            
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,)
        elif net_type == 'CNN':
            self.control_points_net = util.init_cnn(
                in_dim=self.img_dim,
                out_dim=self.output_dim,
                num_mlp_layers=num_layers,)
        elif net_type == 'STN':
            self.stn, self.control_points_net = util.init_stn(
                                        in_dim=self.img_dim,
                                        out_dim=self.output_dim,
                                        num_mlp_layers=num_layers,
                                        end_cnn=True)    
        else: raise NotImplementedError
    
    def get_control_points_dist(self, imgs):
        '''q(control_points | img) = Normal( ; f(MLP(imgs)))
        Args:
            imgs: [*shape, channels, h, w]
        Return:
            dist: [*shape, strks_per_img, n_control_points, 2 (for x, y)]
        '''
        shape = imgs.shape[:-3]
        if self.net_type == "MLP":
            # flatten the images
            imgs = imgs.view(-1, self.img_dim)
        elif self.net_type == "STN":
            imgs, self.stn_transform = self.stn(imgs, output_theta=True)
            if self.keep_stn_out:
                self.stn_out_imgs = imgs.detach().clone()

        if self.dist == 'Normal':
            # raw_loc shape: [batch, strokes, points, (x,y), 1]
            raw_loc, raw_scale = (
                torch.sigmoid(
                self.control_points_net(imgs)
                # 1: num_stroke, _: num_cont_points, 2: (x, y), 2: mean, std
                ).view(*[*shape, self.strks_per_img, self.ctrl_pts_per_strk, 2, 2]) 
                .chunk(2, -1)
                )
            raw_loc = raw_loc.squeeze(-1)
            raw_scale = raw_scale.squeeze(-1) + 0.001 # to ensure non-0 std

            return torch.distributions.Independent(
                    torch.distributions.Normal(raw_loc, raw_scale),
                    reinterpreted_batch_ndims=3,
                )
        elif self.dist == 'Dirichlet':
            concentration = (
                torch.sigmoid(
                    self.control_points_net(imgs)
                ).view(*[*shape, self.strks_per_img, self.ctrl_pts_per_strk * 2, 2])
            )
            return torch.distributions.Independent(
                    torch.distributions.Dirichlet(concentration),
                    reinterpreted_batch_ndims=2,
            )

    def log_prob(self, imgs, latent):
        '''Evaluate log q(control_points | imgs)

        Args:
            imgs: [*batch_shape, channel, h, w]
            latent: [*shape, num_strokes, num_points, 2 (for x, y)]
        '''
        return self.get_control_points_dist(imgs).log_prob(latent)

    def rsample(self, imgs, sample_shape=[], stn_out=False):
        '''Sample from q(control_points | imgs)

        Args:
            imgs: [*batch_shape, channels, h, w]
            sample_shape: list-like object (default [])

        Returns: latent
            control_points: [*sample_shape, *batch_shape, num_strokes, 
                                                    num_points, 2 (for x, y)]
        '''
        self.keep_stn_out = stn_out
        control_points = self.get_control_points_dist(imgs).rsample(sample_shape)
        if stn_out and self.net_type == "STN":
            return control_points, self.stn_out_imgs
        else:
            return control_points