import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import util
from splinesketch.code.bezier import Bezier


class GenerativeModel(nn.Module):
    def __init__(self, control_points_dim=5, prior_dist='Dirichlet'):
        """in the base model we pre-specify the number of control points and 
        work with data that has 1 stroke.

        Args:
            control_points_dim (int): number of control points per stroke
            prior_dist (str): currently support ['Normal', 'Dirichlet']
        """

        super().__init__()
        self.control_points_dim = control_points_dim

        # num_stroke set to 1
        self.prior = prior_dist
        if prior_dist == 'Normal':
            self.register_buffer("control_points_loc", 
                                torch.zeros(1, self.control_points_dim, 2))
            self.register_buffer("control_points_scale", 
                                torch.ones(1, self.control_points_dim, 2))
        elif prior_dist == 'Dirichlet':
            self.register_buffer("concentration",
                                torch.ones(1, self.control_points_dim, 2))
        else:
            raise NotImplementedError

        self.bezier = Bezier(res=28, steps=400, method='bounded')
    
    def control_points_dist(self):
        if self.prior == 'Normal':
            return torch.distributions.Independent(
                torch.distributions.Normal(self.control_points_loc,
                                            self.control_points_scale),
                reinterpreted_batch_ndims=3,
            )
        elif self.prior == 'Dirichlet':
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

    def img_dist_b(self, control_points_b):
        '''
        Args:
            control_points_b: tensor of shape:
                [batch, 1 stroke, num_control_points, 2]
        Return:
            distribution of shape: [*shape, H, W]
        '''
        batch_dim = control_points_b.shape[0]
        recon_imgs = self.bezier(control_points_b) # [b c h w] # grad is fine here
        
        imgs_dist_std = torch.ones_like(recon_imgs) / 100

        # in the dirichlet case, the grad has no nan up to here, and only has 
        # nan after computing log_likelihood with log_prob.
        dist = torch.distributions.Independent(
                torch.distributions.Normal(recon_imgs, imgs_dist_std), 
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
        # max_per_recon = self.bezier(latents).clone().detach().reshape(batch_dim, -1).max(1)[0]
        # max_per_recon = max_per_recon.reshape(batch_dim, 1, 1, 1)
        # imgs = imgs * max_per_recon
        log_prior, log_likelihood = 0, 0 
        log_prior = self.control_points_dist_b(shape).log_prob(latents)
        log_likelihood = self.img_dist_b(latents).log_prob(imgs)
        log_joint = log_prior + log_likelihood
        return log_joint
        
    def sample(self, batch_shape):
        """Sample from p(z, x)
        """
        control_points_b = self.control_points_dist_b(batch_shape).sample()
        imgs = self.img_dist_b(control_points_b).sample()
        return imgs

class Guide(nn.Module):
    def __init__(self, control_points_dim=5, img_dim=[1, 28, 28], hidden_dim=512,
                num_layers=5, dist="Normal"):
        super().__init__()

        self.img_dim = np.prod(img_dim)
        self.control_points_dim = control_points_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.dist = dist
        if dist == 'Dirichlet':
            # _ * 2 (for x, y) * 1 (for concentration)
            self.output_dim = self.control_points_dim * 2
        elif dist == 'Normal':
            # _ * 2 (for x, y) * 2 (for loc, std)
            self.output_dim = self.control_points_dim * 2 * 2
        else: raise NotImplementedError

        self.control_points_mlp = util.init_mlp(
            in_dim=self.img_dim,
            out_dim=self.output_dim,            
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
    
    def get_control_points_dist(self, imgs):
        '''q(control_points | img) = Normal( ; f(MLP(imgs)))
        Args:
            imgs: [*shape, channels, h, w]
        '''
        shape = imgs.shape[:-3]

        if self.dist == 'Normal':
            # raw_loc shape: [batch, strokes, points, (x,y), 1]
            raw_loc, raw_scale = (
                F.sigmoid(
                self.control_points_mlp(imgs.view(-1, self.img_dim))
                # 1: num_stroke, _: num_cont_points, 2: (x, y), 2: mean, std
                ).view(*[*shape, 1, self.control_points_dim, 2, 2]) 
                .chunk(2, -1)
                )
            raw_loc = raw_loc.squeeze(-1)
            raw_scale = raw_scale.squeeze(-1)

            return torch.distributions.Independent(
                    torch.distributions.Normal(*util.normal_raw_params_transform(
                                                            raw_loc, raw_scale)),
                    reinterpreted_batch_ndims=3,
            )
        elif self.dist == 'Dirichlet':
            concentration = (
                torch.sigmoid(
                    self.control_points_mlp(imgs.view(-1, self.img_dim))
                ).view(*[*shape, 1, self.control_points_dim, 2])
            )
            return torch.distributions.Independent(
                    torch.distributions.Dirichlet(concentration),
                    reinterpreted_batch_ndims=3,
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