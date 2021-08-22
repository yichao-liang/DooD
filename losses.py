from collections import namedtuple

import torch
import torch.nn.functional as F
import numpy as np

BaseLoss = namedtuple('BaseLoss', ['overall_loss', 
                                   'neg_generative_log_joint_prob',
                                   'log_posterior'])

SequentialLoss = namedtuple('SequentialLoss', ['overall_loss', 
                                                'model_loss',
                                                'baseline_loss', 
                                                'neg_reinforce_term',
                                                'neg_elbo',
                                                'neg_log_likelihood',
                                                'neg_log_prior', 
                                                'log_posterior'])

def get_loss_sequential(generative_model, guide, imgs, loss='elbo'):
    '''Get loss for sequential model, e.g. AIR
    Args:
        loss (str): "nll": negative log likelihood, "l1": L1 loss, "elbo": -ELBO
    '''
    if torch.is_tensor(imgs):
        # Guide output
        guide_out = guide(imgs)
        # wheater to mask current value based on prev.z_pres; more doc in model
        latents, log_post, bl_value, mask_prev = (guide_out.z_smpl,
                guide_out.z_lprb, guide_out.baseline_value, guide_out.mask_prev)

        if (not guide.execution_guided and 
                                generative_model.input_dependent_param):
            # Then we will render the random variables
            generative_model.sigma, (generative_model.tanh_norm_slope_stroke, \
                generative_model.tanh_norm_slope) = guide_out.decoder_param
        
        # Posterior log probability: [batch_size, max_strks] (before summing)
        log_post_z = torch.cat(
                            [prob.sum(-1, keepdim=True) for prob in log_post], 
                            dim=-1).sum(-1)

        # Prior and Likelihood probability
        # Prior: [batch_size, max_strks]
        # Likelihood: [batch_size]
        log_prior, log_likelihood = generative_model.log_prob(
                                                        latents=latents, 
                                                        imgs=imgs,
                                                        z_pres_mask=mask_prev,
                                                        canvas=guide_out.canvas)
        # z_pres_prior_lprb, z_what_post_lprb, z_where_post_lprb = log_prior
        log_prior_z = torch.cat(
                            [prob.sum(-1, keepdim=True) for prob in log_prior],
                            dim=-1).sum(-1)
        generative_joint_log_prob = (log_likelihood + log_prior_z)

        if loss == 'elbo':
            # Compute ELBO: [bs, ]
            elbo = - log_post_z + generative_joint_log_prob

            # bl_target size: [batch_size, max_strks]
            # sum_{i=t}^T [ KL[i] - log p(x | z) ] 
            # =sum_{i=t}^T [ log_post - log_prior - log p(x | z)]
            # for all steps up to (and including) the first z_pres=0
            # (flip -> cumsum -> flip) so that it cumulate on to the left
            bl_target = torch.cat([prob.flip(1).cumsum(-1).flip(1).unsqueeze(-1)
                            for prob in log_post], dim=-1).sum(-1)
            bl_target -= torch.cat([prob.flip(1).cumsum(-1).flip(1).unsqueeze(-1)
                            for prob in log_prior], dim=-1).sum(-1)
            bl_target = bl_target - log_likelihood[:, None] # this is like -ELBO
            bl_target = bl_target * mask_prev

            # The "REINFORCE"  term in the gradient is: [bs,]; 
            # bl)target is the negative elbo
            # bl_value: [bs, max_strks]; z_pres [bs, max_strks]; 
            # (bl_target - bl_value) * gradient[z_pres_likelihood]
            neg_reinforce_term = (bl_target - bl_value).detach()*log_post.z_pres
            neg_reinforce_term = neg_reinforce_term * mask_prev
            neg_reinforce_term = neg_reinforce_term.sum(1)   # [bs, ]

            # [bs, ]
            # Q: shouldn't reinforce be negative? 
            # A: it's already negative from (KL - likelihood)
            model_loss = neg_reinforce_term - elbo

            # MSE as baseline loss: [bs, n_strks]
            baseline_loss = F.mse_loss(bl_value, bl_target.detach(), 
                                                            reduction='none')
            baseline_loss = baseline_loss * mask_prev # [bs, n_strks]
            baseline_loss = baseline_loss.sum(1) # [bs, ]

            loss = model_loss + baseline_loss # [bs, ]
        else:
            raise NotImplementedError                             
        return SequentialLoss(overall_loss=loss, 
                                model_loss=model_loss,
                                baseline_loss=baseline_loss,
                                neg_reinforce_term=neg_reinforce_term,
                                neg_elbo=-elbo,
                                neg_log_likelihood=-log_likelihood,
                                neg_log_prior=-log_prior_z, 
                                log_posterior=log_post_z)

    else:
        raise NotImplementedError

def get_loss_base(generative_model, guide, imgs, loss="nll"):
    '''
    Args:
        loss (str): "nll": negative log likelihood, "l1": L1 loss, "elbo": -ELBO
    '''

    if torch.is_tensor(imgs):


        latent = guide.rsample(imgs)
        generative_model.stn_transform = guide.stn_transform

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

        return BaseLoss(overall_loss=loss, 
                    neg_generative_log_joint_prob=-generative_model_log_prob, 
                    log_posterior=guide_log_prob)
    else:
        raise NotImplementedError("not implemented")
