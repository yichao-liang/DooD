import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Laplace, Dirichlet
from einops import rearrange
from kornia.morphology import dilation

import util
from splinesketch.code.bezier import Bezier

class GenerativeModel(nn.Module):
    def __init__(self, control_points_dim=20, 
                        prior_dist='Normal', 
                        likelihood_dist='Normal',
                        device=None):
        """in the base model we pre-specify the number of control points and 
        work with data that has 1 stroke.

        Args:
            control_points_dim (int): number of control points per stroke
            prior_dist (str): currently support ['Normal', 'Dirichlet']
        """

        super().__init__()
        self.supported_prior_dist = ["Normal", "Dirichlet"]
        assert prior_dist in self.supported_prior_dist
        self.control_points_dim = control_points_dim

        # num_stroke set to 1
        self.prior_dist = prior_dist
        self.likelihood_dist = likelihood_dist
        if prior_dist == 'Normal':
            self.register_buffer("control_points_loc", 
                                torch.zeros(1, self.control_points_dim, 2) + 0.5)
            self.register_buffer("control_points_scale", 
                                torch.ones(1, self.control_points_dim, 2))
        elif prior_dist == 'Dirichlet':
            # control_points_dim * 2 for x, y coordinates. the last dim can't be
            # used for x, y because it has to add up to 1.
            self.register_buffer("concentration",
                                torch.ones(1, self.control_points_dim * 2, 2))
        else:
            raise NotImplementedError

        self.bezier = Bezier(res=28, steps=500, method='bounded')
        self.dilation_kernel = torch.ones(3,3).to(device)
    
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
        # [b c h w] # grad is fine here
        imgs_dist_loc = self.bezier(control_points_b)

        max_per_recon = imgs_dist_loc.detach().clone().reshape(
                                                    batch_dim, -1).max(1)[0]
        max_per_recon = max_per_recon.reshape(batch_dim, 1, 1, 1)
        
        # don't divide by 0, or by a really small number
        imgs_dist_loc = util.safe_div(imgs_dist_loc, max_per_recon)
        imgs_dist_loc = dilation(imgs_dist_loc, self.dilation_kernel, max_val=1.)

        assert not imgs_dist_loc.isnan().any()
        assert imgs_dist_loc.max() <= 1.

        if self.likelihood_dist=='Laplace':
            event_dist = Laplace
            imgs_dist_std = torch.ones_like(imgs_dist_loc)/ 10 
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
    def __init__(self, control_points_dim=20, img_dim=[1, 28, 28], 
                hidden_dim=256, num_layers=3, dist="Normal", net_type="CNN"):
        super().__init__()

        self.img_dim = np.prod(img_dim)
        self.control_points_dim = control_points_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.dist = dist
        if dist == 'Dirichlet':
            # _ * 2 (for x, y) * 2 (for concentration, 2 values for an value)
            self.output_dim = self.control_points_dim * 2 * 2
        elif dist == 'Normal':
            # _ * 2 (for x, y) * 2 (for loc, std)
            self.output_dim = self.control_points_dim * 2 * 2
        else: raise NotImplementedError

        self.net_type = net_type
        if net_type == 'MLP':
            self.control_points_net = util.init_mlp(
                in_dim=self.img_dim,
                out_dim=self.output_dim,            
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
            )
        elif net_type == 'CNN':
            self.control_points_net = util.init_cnn(
                in_dim=self.img_dim,
                out_dim=self.output_dim,
                num_mlp_layers=num_layers,
            )
        else: raise NotImplementedError

    
    def get_control_points_dist(self, imgs):
        '''q(control_points | img) = Normal( ; f(MLP(imgs)))
        Args:
            imgs: [*shape, channels, h, w]
        Return:
            dist: [*shape, n_strokes, n_control_points, 2 (for x, y)]
        '''
        shape = imgs.shape[:-3]
        if self.net_type == "MLP":
            # flatten the images
            imgs = imgs.view(-1, self.img_dim)

        if self.dist == 'Normal':
            # raw_loc shape: [batch, strokes, points, (x,y), 1]
            raw_loc, raw_scale = (
                torch.sigmoid(
                self.control_points_net(imgs)
                # 1: num_stroke, _: num_cont_points, 2: (x, y), 2: mean, std
                ).view(*[*shape, 1, self.control_points_dim, 2, 2]) 
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
                ).view(*[*shape, 1, self.control_points_dim * 2, 2])
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

    def rsample(self, imgs, sample_shape=[]):
        '''Sample from q(control_points | imgs)

        Args:
            imgs: [*batch_shape, channels, h, w]
            sample_shape: list-like object (default [])

        Returns: latent
            control_points: [*sample_shape, *batch_shape, num_strokes (1), 
                                                    num_points, 2 (for x, y)]
        '''
        control_points = self.get_control_points_dist(imgs).rsample(sample_shape)
        return control_points