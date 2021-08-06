import torch
import torch.nn.functional as F
import numpy as np

def get_elbo_loss(generative_model, guide, imgs, loss="nll"):
    '''
    Args:
        loss (str): "nll": negative log likelihood, "l1": L1 loss, "elbo": -ELBO
    '''

    if torch.is_tensor(imgs):
        latent = guide.rsample(imgs)
        guide_log_prob = guide.log_prob(imgs, latent)
        log_prior, log_likelihood = generative_model.log_prob(latent, imgs)
        # average across batch, and average ll across dim
        log_likelihood = log_likelihood / np.prod(imgs.shape[-3:])
        generative_model_log_prob = log_likelihood + log_prior                                

        if loss == 'l1':
            recon_img = generative_model.img_dist_b(latent).mean
            loss = F.l1_loss(recon_img, imgs, reduction='sum')/recon_img.sum()
        elif loss == 'elbo':
            loss = -generative_model_log_prob + guide_log_prob
        elif loss == "nll":
            loss = -log_likelihood
        else:
            raise NotImplementedError

        return loss, -generative_model_log_prob, guide_log_prob
    else:
        raise NotImplementedError("not implemented")
