import torch
import torch.nn.functional as F
import numpy as np

def get_elbo_loss(generative_model, guide, imgs, loss="nll", iteration=None):
    '''
    Args:
        loss (str): "nll": negative log likelihood, "l1": L1 loss, "elbo": -ELBO
    '''

    if torch.is_tensor(imgs):
        if loss == 'l1':
            if iteration == 0:
                # this loss doesn't use a prior
                generative_model.sigma = torch.log(torch.tensor(.02))
        if loss == 'elbo':
            if iteration == 0:
                generative_model.sigma = torch.log(torch.tensor(.04))
                generative_model.control_points_scale = (torch.ones(
                                        generative_model.n_strokes, 
                                        generative_model.control_points_dim, 2
                                    )/5).cuda()
            if iteration == 100:
                generative_model.sigma = torch.log(torch.tensor(.03))
        elif loss == 'nll':
            # working with σ in renderer set to >=.02, σ for image Gaussian <=.2
            if iteration == 0:
                generative_model.sigma = torch.log(torch.tensor(.02))
                generative_model.control_points_scale = (torch.ones(
                                        generative_model.n_strokes, 
                                        generative_model.control_points_dim, 2
                                    )/5).cuda()

        latent = guide.rsample(imgs)

        guide_log_prob = guide.log_prob(imgs, latent)
        log_prior, log_likelihood = generative_model.log_prob(latent, imgs)
        # average across batch, and average ll across dim
        log_likelihood = log_likelihood #/ np.prod(imgs.shape[-3:])
        generative_model_log_prob = log_likelihood + log_prior                                
        recon_img = generative_model.img_dist_b(latent).mean

        if loss == 'l1':
            loss = F.l1_loss(recon_img, imgs,reduction='none').sum((1,2,3)) -\
                                                        recon_img.sum((1,2,3))
        elif loss == 'elbo':
            loss = -generative_model_log_prob + guide_log_prob
        elif loss == "nll":
            # working
            # loss = -(log_likelihood.sum() + recon_img.sum((1,2,3)).sum())

            # working; almost -ELBO but I have to use mean() for the last term for it to work
            # loss = -(log_likelihood.sum() + recon_img.sum((1,2,3)).sum() - guide_log_prob.mean())
            loss = -generative_model_log_prob  
        else:
            raise NotImplementedError

        return loss, -generative_model_log_prob, guide_log_prob
    else:
        raise NotImplementedError("not implemented")
