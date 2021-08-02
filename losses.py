import torch
import numpy as np

def get_elbo_loss(generative_model, guide, imgs):
    '''
    '''

    if torch.is_tensor(imgs):
        latent = guide.rsample(imgs)
        guide_log_prob = guide.log_prob(imgs, latent)
        generative_model_log_prob = (generative_model.log_prob(latent, imgs)/
                                        np.prod(imgs.shape[-3:]))

        loss = -(generative_model_log_prob - guide_log_prob)
        # loss = -(generative_model_log_prob)
        # loss = -( guide_log_prob) # This loss passes

        return loss
    else:
        raise NotImplementedError("not implemented")