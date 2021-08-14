import torch
import torch.nn.functional as F
import numpy as np

def get_loss_sequential(generative_model, guide, imgs, loss='elbo'):
    '''Get loss for sequential model, e.g. AIR
    Args:
        loss (str): "nll": negative log likelihood, "l1": L1 loss, "elbo": -ELBO
    '''
    if torch.is_tensor(imgs):
        # Guide output
        guide_out = guide(imgs)
        latents = guide_out.z_smpl
        log_post = guide_out.z_lprb
        bl_value = guide_out.baseline_value
        # wheater to mask current value based on prev.z_pres; more doc in model
        mask_prev = guide_out.mask_prev 

        # Posterior log probability: [batch_size, max_strks] (before summing)
        z_pres_post_lprb = log_post.z_pres.sum(-1)
        z_what_post_lprb = log_post.z_what.sum(-1)
        z_where_post_lprb = log_post.z_where.sum(-1)
        # [bs]
        z_post_lprob = z_pres_post_lprb + z_what_post_lprb + z_where_post_lprb

        # Prior and Likelihood probability
        # Prior: [batch_size, max_strks]
        # Likelihood: [batch_size]
        log_prior, log_likelihood = generative_model.log_prob(latents, imgs,
                                                                    mask_prev)
        # z_pres_prior_lprb, z_what_post_lprb, z_where_post_lprb = log_prior
        
        generative_joint_log_prob = (log_likelihood + torch.cat(
                            [prob.sum(-1, keepdim=True) for prob in log_prior],
                            dim=-1).sum(-1))

        if loss == 'elbo':
            # Compute ELBO: [bs, ]
            elbo = -z_post_lprob + generative_joint_log_prob

            # bl_target size: [batch_size, max_strks]
            # sum_{i=t}^T [ KL[i] - log p(x | z) ] 
            # for all steps up to (and including) the first z_pres=0
            # flip -> cumsum -> flip so that it cumulate on to the left
            bl_target = torch.cat([prob.flip(1).cumsum(-1).flip(1).unsqueeze(-1)
                           for prob in [*log_post, *log_prior]], dim=-1).sum(-1)
            bl_target = bl_target - log_likelihood[:, None] # this is like -ELBO
            bl_target = bl_target * mask_prev

            # The "REINFORCE"  term in the gradient is: [bs,]; 
            # bl_value: [bs, max_strks]; z_pres [bs, max_strks]; 
            # (bl_target - bl_value) * gradient[z_pres_likelihood]
            reinforce_term = (bl_target - bl_value).detach() * log_post.z_pres
            reinforce_term = reinforce_term * mask_prev
            reinforce_term = reinforce_term.sum(1)   # [bs, ]

            # [bs, ]
            # Q: shouldn't reinforce be negative? 
            # A: it's already negative from (KL - likelihood)
            model_loss = reinforce_term - elbo

            # MSE as baseline loss: [bs, n_strks]
            baseline_loss = F.mse_loss(bl_value, bl_target.detach(), 
                                                            reduction='none')
            baseline_loss = baseline_loss * mask_prev # [bs, n_strks]
            baseline_loss = baseline_loss.sum(1) # [bs, ]

            loss = model_loss + baseline_loss # [bs, ]
        else:
            raise NotImplementedError                             
        return loss, -generative_joint_log_prob, z_post_lprob
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

        return loss, -generative_model_log_prob, guide_log_prob
    else:
        raise NotImplementedError("not implemented")
