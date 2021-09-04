from collections import namedtuple

import torch
import torch.nn.functional as F
import numpy as np
from util import incremental_average

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

def get_loss_sequential(generative_model, guide, imgs, loss_type='elbo', k=1,
                                                    iteration=0, writer=None):
    '''Get loss for sequential model, e.g. AIR
    Args:
        loss_type (str): "nll": negative log likelihood, "l1": L1 loss, "elbo": -ELBO
        k (int): the number of samples to compute to compute the loss.
    '''
    (k_overall_loss, k_model_loss, k_baseline_loss, k_neg_reinforce_term,
     k_neg_elbo, k_neg_log_likelihood, k_neg_log_prior, k_log_posterior) = [0]*8

    for i in range(k):
        if torch.is_tensor(imgs):
            # Guide output
            guide_out = guide(imgs)
            # wheater to mask current value based on prev.z_pres; more doc in model
            latents, log_post, bl_value, mask_prev, canvas, z_prior = (
                guide_out.z_smpl, guide_out.z_lprb, guide_out.baseline_value, 
                    guide_out.mask_prev, guide_out.canvas, guide_out.z_prior)

            if (not guide.execution_guided and 
                                    generative_model.input_dependent_param):
                # If we haven't rendered the final reconstruction AND we have
                # predicted render parameters
                generative_model.sigma, (generative_model.tanh_norm_slope_stroke,\
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
                                                        canvas=canvas,
                                                        z_prior=z_prior)

            # z_pres_prior_lprb, z_what_post_lprb, z_where_post_lprb = log_prior
            log_prior_z = torch.cat(
                                [prob.sum(-1, keepdim=True) for prob in log_prior],
                                dim=-1).sum(-1)
            generative_joint_log_prob = (log_likelihood + log_prior_z)

            if loss_type == 'elbo':
                # Compute ELBO: [bs, ]
                elbo = - log_post_z + generative_joint_log_prob
                assert elbo.shape == imgs.shape[:-3]

                # bl_target size: [batch_size, max_strks]
                # sum_{i=t}^T [ KL[i] - log p(x | z) ] 
                # =sum_{i=t}^T [ log_post - log_prior - log p(x | z)]
                # for all steps up to (and including) the first z_pres=0
                # (flip -> cumsum -> flip) so that it cumulate on to the left
                bl_target = torch.cat([prob.flip(1).cumsum(-1).flip(1).unsqueeze(-1)
                                for prob in log_post], dim=-1).sum(-1)
                bl_target -= torch.cat([prob.flip(1).cumsum(-1).flip(1).unsqueeze(-1)
                                for prob in log_prior], dim=-1).sum(-1)
                 # this is like -ELBO
                bl_target = bl_target - log_likelihood[:, None]
                bl_target = (bl_target * mask_prev)

                # The "REINFORCE"  term in the gradient is: [bs,]; 
                # bl)target is the negative elbo
                # bl_value: [bs, max_strks]; z_pres [bs, max_strks]; 
                # (bl_target - bl_value) * gradient[z_pres_likelihood]
                neg_reinforce_term = (bl_target - bl_value).detach()*log_post.z_pres
                neg_reinforce_term = neg_reinforce_term * mask_prev
                neg_reinforce_term = neg_reinforce_term.sum(1) # [bs, ]

                # [bs, ]
                # Q: shouldn't reinforce be negative? 
                # A: it's already negative from (KL - likelihood)
                model_loss = neg_reinforce_term - elbo

                # MSE as baseline loss: [bs, n_strks]
                # div by img_dim and clip grad works for independent prior
                # but not for sequential
                baseline_loss = F.mse_loss(bl_value, bl_target.detach(), 
                                    reduction='none')
                baseline_loss = baseline_loss * mask_prev # [bs, n_strks]
                baseline_loss = baseline_loss.sum(1)

                loss = model_loss + baseline_loss # [bs, ]

                # Append to the k_samples list
                k_overall_loss = incremental_average(k_overall_loss, loss, n=i+1)
                k_neg_elbo = incremental_average(k_neg_elbo, -elbo, n=i+1)
            else:
                raise NotImplementedError 
        else:
            raise NotImplementedError

    # Log the scale parameters
    if writer is not None:
        if generative_model.input_dependent_param:
            writer.add_histogram("Parameters/gen.sigma",
                        guide_out.decoder_param.sigma.detach(), iteration)
            writer.add_histogram("Parameters/tanh.stroke_slopes",
                        guide_out.decoder_param.slope[0].detach(), iteration)
            writer.add_histogram("Parameters/tanh.add_slopes",
                        guide_out.decoder_param.slope[1].detach(), iteration)
        if generative_model.prior_dist == 'Sequential':
            writer.add_histogram("Parameters/z_pres_prior.p",
                        guide.internal_decoder.z_pres_p.detach(), iteration)
            writer.add_histogram("Parameters/z_what_prior.loc",
                        guide.internal_decoder.z_what_loc.detach(), iteration)
            writer.add_histogram("Parameters/z_what_prior.std",
                        guide.internal_decoder.z_what_std.detach(), iteration)
            writer.add_histogram("Parameters/z_where_prior.loc.scale",
                        guide.internal_decoder.z_where_loc.detach()[:, 0], iteration)
            writer.add_histogram("Parameters/z_where_prior.loc.shift",
                        guide.internal_decoder.z_where_loc.detach()[:, 1:3], iteration)
            writer.add_histogram("Parameters/z_where_prior.loc.rotate",
                        guide.internal_decoder.z_where_loc.detach()[:, 3], iteration)
            writer.add_histogram("Parameters/z_where_prior.std",
                        guide.internal_decoder.z_where_std.detach(), iteration)
                        
        writer.add_histogram("Parameters/z_pres_posterior",
                        guide_out.z_pms.z_pres.detach(), iteration)
        writer.add_histogram("Parameters/z_where_posterior.loc.scale",
                        guide_out.z_pms.z_where.detach()[:, :, 0, 0], iteration)
        writer.add_histogram("Parameters/z_where_posterior.loc.shift",
                        guide_out.z_pms.z_where.detach()[:, :, 1:3, 0], iteration)
        writer.add_histogram("Parameters/z_where_posterior.loc.rotate",
                        guide_out.z_pms.z_where.detach()[:, :, 3, 0], iteration)
        writer.add_histogram("Parameters/z_where_posterior.std",
                        guide_out.z_pms.z_where.detach()[:, :, :, 1], iteration)
        writer.add_histogram("Parameters/z_what_posterior.loc",
                        guide_out.z_pms.z_what.detach()[:, :, :, :, 0], iteration)
        writer.add_histogram("Parameters/z_what_posterior.scale",
                        guide_out.z_pms.z_what.detach()[:, :, :, :, 1], iteration)
    return SequentialLoss(overall_loss=k_overall_loss, 
                                model_loss=model_loss,
                                baseline_loss=baseline_loss,
                                neg_reinforce_term=neg_reinforce_term,
                                neg_elbo=k_neg_elbo,
                                neg_log_likelihood=-log_likelihood,
                                neg_log_prior=-log_prior_z, 
                                log_posterior=log_post_z)


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

def get_loss_air(generative_model, guide, imgs, loss_type='elbo',
                                                    iteration=0, 
                                                    writer=None):
    '''Get loss for sequential model, e.g. AIR
    Args:
        loss_type (str): "nll": negative log likelihood, "l1": L1 loss, "elbo": -ELBO
        k (int): the number of samples to compute to compute the loss.
    '''
    (k_overall_loss, k_model_loss, k_baseline_loss, k_neg_reinforce_term,
     k_neg_elbo, k_neg_log_likelihood, k_neg_log_prior, k_log_posterior) = [0]*8

    # Fixed 1 sample
    for i in range(1):
        if torch.is_tensor(imgs):
            # Guide output
            guide_out = guide(imgs)
            # wheater to mask current value based on prev.z_pres; 
            # more doc in model
            latents, log_post, bl_value, mask_prev, canvas = (
                guide_out.z_smpl, guide_out.z_lprb, guide_out.baseline_value, 
                guide_out.mask_prev, guide_out.canvas)

            # Posterior log probability: [batch_size, max_strks] (before summing)
            log_post_z = torch.cat(
                                [prob.sum(-1, keepdim=True) for prob in 
                                                    log_post], dim=-1).sum(-1)

            # Prior and Likelihood probability
            # Prior: [batch_size, max_strks]
            # Likelihood: [batch_size]
            log_prior, log_likelihood = generative_model.log_prob(
                                                        latents=latents, 
                                                        imgs=imgs,
                                                        z_pres_mask=mask_prev,
                                                        canvas=canvas)

            # z_pres_prior_lprb, z_what_post_lprb, z_where_post_lprb = log_prior
            log_prior_z = torch.cat(
                                [prob.sum(-1, keepdim=True) for prob in 
                                    log_prior], dim=-1).sum(-1)
            generative_joint_log_prob = (log_likelihood + log_prior_z)

            if loss_type == 'elbo':
                # Compute ELBO: [bs, ]
                elbo = - log_post_z + generative_joint_log_prob

                # bl_target size: [batch_size, max_strks]
                # sum_{i=t}^T [ KL[i] - log p(x | z) ] 
                # =sum_{i=t}^T [ log_post - log_prior - log p(x | z)]
                # for all steps up to (and including) the first z_pres=0
                # (flip -> cumsum -> flip) so that it cumulate on to the left
                bl_target = torch.cat([prob.flip(1).cumsum(-1).flip(1
                        ).unsqueeze(-1) for prob in log_post], dim=-1).sum(-1)
                bl_target -= torch.cat([prob.flip(1).cumsum(-1).flip(1
                        ).unsqueeze(-1) for prob in log_prior], dim=-1).sum(-1)
                 # this is like -ELBO
                bl_target = bl_target - log_likelihood[:, None]
                bl_target = (bl_target * mask_prev)

                # The "REINFORCE"  term in the gradient is: [bs,]; 
                # bl)target is the negative elbo
                # bl_value: [bs, max_strks]; z_pres [bs, max_strks]; 
                # (bl_target - bl_value) * gradient[z_pres_likelihood]
                neg_reinforce_term = (bl_target - bl_value
                                                    ).detach() * log_post.z_pres
                neg_reinforce_term = neg_reinforce_term * mask_prev
                neg_reinforce_term = neg_reinforce_term.sum(1) # [bs, ]

                # [bs, ]
                # Q: shouldn't reinforce be negative? 
                # A: it's already negative from (KL - likelihood)
                model_loss = neg_reinforce_term - elbo

                # MSE as baseline loss: [bs, n_strks]
                # div by img_dim and clip grad works for independent prior
                # but not for sequential
                baseline_loss = F.mse_loss(bl_value, bl_target.detach(), 
                                    reduction='none')
                baseline_loss = baseline_loss * mask_prev # [bs, n_strks]
                baseline_loss = baseline_loss.sum(1)

                loss = model_loss + baseline_loss # [bs, ]

                # Append to the k_samples list
                k_overall_loss = incremental_average(k_overall_loss, loss, 
                                                     n=i+1)
                k_neg_elbo = incremental_average(k_neg_elbo, -elbo, n=i+1)
            else:
                raise NotImplementedError 
        else:
            raise NotImplementedError

    # Log the scale parameters
    
    if writer is not None:
        with torch.no_grad():
            writer.add_histogram("Parameters/z_pres",
                        guide_out.z_pms.z_pres.detach(), iteration)
            writer.add_histogram("Parameters/z_where_posterior.loc.scale",
                        guide_out.z_pms.z_where.detach()[:, :, 0, 0], iteration)
            writer.add_histogram("Parameters/z_where_posterior.loc.shift",
                        guide_out.z_pms.z_where.detach()[:, :, 1:, 0,], iteration)
            writer.add_histogram("Parameters/z_where_posterior.scale",
                        guide_out.z_pms.z_where.detach()[:, :, :, 1], iteration)
            writer.add_histogram("Parameters/z_what_posterior.loc",
                        guide_out.z_pms.z_what.detach()[:, :, :, 0], iteration)
            writer.add_histogram("Parameters/z_what_posterior.scale",
                        guide_out.z_pms.z_what.detach()[:, :, :, 1], iteration)
        
            writer.add_histogram("Parameters/img.loc", 
                    generative_model.renders_imgs(latents), iteration)
    return SequentialLoss(overall_loss=k_overall_loss, 
                                model_loss=model_loss,
                                baseline_loss=baseline_loss,
                                neg_reinforce_term=neg_reinforce_term,
                                neg_elbo=k_neg_elbo,
                                neg_log_likelihood=-log_likelihood,
                                neg_log_prior=-log_prior_z, 
                                log_posterior=log_post_z)

def get_loss_vae(generative_model, 
                  guide, 
                  imgs, 
                  iteration=None, 
                  writer=None):
    '''
    Args:
        loss (str): "nll": negative log likelihood, "l1": L1 loss, "elbo": -ELBO
    '''

    if torch.is_tensor(imgs):


        guide_out = guide(imgs)
        z_smpl, (z_loc, z_std), guide_log_prob = (guide_out.z_smpl,
                                                  guide_out.z_pms, 
                                                  guide_out.z_lprb)

        log_prior, log_likelihood = generative_model.log_prob(z_smpl, imgs)
        # average across batch, and average ll across dim
        generative_model_log_prob = log_likelihood + log_prior                                

        loss = -generative_model_log_prob + guide_log_prob

        if writer is not None:
            writer.add_histogram("Parameters/z_loc", z_loc.detach(), iteration)
            writer.add_histogram("Parameters/z_std", z_std.detach(), iteration)

        return BaseLoss(overall_loss=loss, 
                    neg_generative_log_joint_prob=-generative_model_log_prob, 
                    log_posterior=guide_log_prob)
    else:
        raise NotImplementedError("not implemented")