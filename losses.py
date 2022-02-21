from collections import namedtuple

import torch
import torch.nn.functional as F
import numpy as np
from util import incremental_average
from models.air import ZLogProb

BaseLoss = namedtuple('BaseLoss', ['overall_loss', 
                                   'neg_log_prior',
                                   'neg_log_likelihood',
                                   'log_posterior',
                                   'neg_elbo'])

SequentialLoss = namedtuple('SequentialLoss', ['overall_loss', 
                                                'model_loss',
                                                'baseline_loss', 
                                                'neg_reinforce_term',
                                                'neg_elbo',
                                                'neg_log_likelihood',
                                                'neg_log_prior', 
                                                'log_posterior'])

def get_loss_sequential(generative_model, guide, imgs, loss_type='elbo', k=1,
                            iteration=0, writer=None, writer_tag=None, beta=1,
                            args=None):
    '''Get loss for sequential model, e.g. AIR
    Args:
        loss_type (str): "nll": negative log likelihood, "l1": L1 loss, "elbo": -ELBO
        k (int): the number of samples to compute to compute the loss.
        beta (float): beta term as in beta-VAE
    '''
    if args.constrain_z_pres_param and iteration > 9001 and iteration < 14000:
        # the second clause is some experimental condition
        guide.constrain_z_pres_param_this_ite = True
    else: guide.constrain_z_pres_param_this_ite = False

    # Guide output
    guide_out = guide(imgs, k)
    
    # wheater to mask current value based on prev.z_pres; more doc in model
    latents, log_post, bl_value, mask_prev, canvas, z_prior = (
        guide_out.z_smpl, guide_out.z_lprb, guide_out.baseline_value, 
        guide_out.mask_prev, guide_out.canvas, guide_out.z_prior)

    # if ((not guide.use_canvas and generative_model.input_dependent_param)
    if ((canvas is None and generative_model.input_dependent_param)
        or args.render_at_the_end):
        # If we haven't rendered the final reconstruction AND we have
        # predicted render parameters
        generative_model.sigma = guide_out.decoder_param.sigma
        generative_model.sgl_strk_tanh_slope = \
                                    guide_out.decoder_param.slope[0]
        generative_model.add_strk_tanh_slope = \
                                    guide_out.decoder_param.slope[1][:, :, -1]
        if args.render_at_the_end: canvas = None
    # multi by beta
    # log_post_ = {}
    # for i, z in enumerate(log_post._fields):
    #     log_post_[z] = log_post[i] * beta
    # log_post = ZLogProb(**log_post_)

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

    # multiply by beta
    # log_prior_ = {}
    # for i, z in enumerate(log_prior._fields):
    #     log_prior_[z] = log_prior[i] * beta
    # log_prior = ZLogProb(**log_prior_)

    # z_pres_prior_lprb, z_what_post_lprb, z_where_post_lprb = log_prior
    log_prior_z = torch.cat(
                        [prob.sum(-1, keepdim=True) for prob in 
                            log_prior], dim=-1).sum(-1)
    generative_joint_log_prob = (log_likelihood + log_prior_z)

    # Compute ELBO: [ptcs, bs, ]
    elbo = - log_post_z + generative_joint_log_prob
    assert elbo.shape == torch.Size([k, *imgs.shape[:-3]])

    # bl_target size: [batch_size, max_strks]
    # sum_{i=t}^T [ KL[i] - log p(x | z) ] 
    # =sum_{i=t}^T [ log_post - log_prior - log p(x | z)]
    # for all steps up to (and including) the first z_pres=0
    # (flip -> cumsum -> flip) so that it cumulate on to the left

    # multiply by beta
    # log_post_ = {}
    # for i, z in enumerate(log_post._fields):
    #     log_post_[z] = log_post[i] * beta
    # log_post = ZLogProb(**log_post_)
    
    # log_prior_ = {}
    # for i, z in enumerate(log_prior._fields):
    #     log_prior_[z] = log_prior[i] * beta
    # log_prior = ZLogProb(**log_prior_)
    # divide by beta

    reinforce_ll = log_likelihood.detach()

    # sign correction for the loss in reinforce -- keep it positive
    correct_reinforce_loss = True
    if correct_reinforce_loss:
        # mean_ll = reinforce_ll.mean() # only when <0 else commented out
        # if mean_ll < 0: # only when <0 or commented out
        if iteration == 0:
            # use generative_model to store min_ll
            generative_model.min_ll = reinforce_ll.mean()
        # correct version that should be always positive, hence the negative
        # version should be always negative.
        reinforce_ll = reinforce_ll + generative_model.min_ll.abs()
        

    reinforce_ll = reinforce_ll / beta

    if args.no_baseline:
        bl_value = 0.
    bl_target = torch.cat([prob.flip(-1).cumsum(-1).flip(-1).unsqueeze(-1)
                    for prob in log_post], dim=-1).sum(-1)
    bl_target -= torch.cat([prob.flip(-1).cumsum(-1).flip(-1).unsqueeze(-1)
                    for prob in log_prior], dim=-1).sum(-1)
    # this is like -ELBO
    bl_target = bl_target - reinforce_ll[:, :, None]
    bl_target = (bl_target * mask_prev)
    reinforce_loss = bl_target - bl_value

    # The "REINFORCE"  term in the gradient is: [ptcs, bs,]; 
    # bl_target is the negative elbo
    # bl_value: [ptcs, bs, max_strks]; z_pres [ptcs, bs, max_strks]; 
    # (bl_target - bl_value) * gradient[z_pres_posterior]
    neg_reinforce_term = (reinforce_loss).detach() * log_post.z_pres
    neg_reinforce_term = neg_reinforce_term * mask_prev
    neg_reinforce_term = neg_reinforce_term.sum(2) # [ptcs, bs, ]

    # [ptcs, bs, ]
    # Q: shouldn't reinforce be negative? 
    # A: it's already negative from (KL - likelihood)
    model_loss = neg_reinforce_term - elbo


    if args.no_baseline:
        baseline_loss = torch.tensor(0.)
        loss = model_loss
    else:
        # MSE as baseline loss: [ptcs, bs, n_strks]
        # div by img_dim and clip grad works for independent prior
        # but not for sequential
        baseline_loss = F.mse_loss(bl_value, bl_target.detach(), 
                                   reduction='none')
        baseline_loss = baseline_loss * mask_prev # [ptcs, bs, n_strks]
        baseline_loss = baseline_loss.sum(2)
        loss = model_loss + baseline_loss # [bs, ]

    # Log the scale parameters
    if writer is not None:
        with torch.no_grad():
            if writer_tag is None:
                writer_tag = ''
            writer.add_scalar(
                f"{writer_tag}Train curves/neg_log_likelihood in reinforce",
                -reinforce_ll.mean(), iteration)
            writer.add_scalar(f"{writer_tag}Train curves/beta", beta, 
                              iteration)
            writer.add_scalar(f"{writer_tag}Train curves/reinforce variance",
                              reinforce_loss.detach().var(), iteration)
            # loss
            for n, log_prob in zip(log_post._fields, log_post):
                writer.add_scalar(f"{writer_tag}Train curves/log_posterior/"+n, 
                                    log_prob.detach().sum(-1).mean(), 
                                    iteration)
            for n, log_prob in zip(['z_pres', 'z_where', 'z_what'], log_prior):
                writer.add_scalar(f"{writer_tag}Train curves/log_prior/"+n, 
                                    log_prob.detach().sum(-1).mean(), 
                                    iteration)

            if args.log_param:
                # this includes: renderer parameters, prior parameters,
                # posterior parameters and posterior samples
                # renderer parameters
                writer.add_histogram(f"{writer_tag}Parameters/img_dist_std",
                        generative_model.get_imgs_dist_std().detach(), 
                        iteration)
                if generative_model.input_dependent_param:
                    writer.add_histogram(f"{writer_tag}Parameters/gen.sigma",
                                guide_out.decoder_param.sigma.detach(), 
                                iteration)
                    writer.add_histogram(
                                f"{writer_tag}Parameters/tanh.add_slopes",
                                guide_out.decoder_param.slope[1].detach(), 
                                iteration)
                    if generative_model.sgl_strk_tanh:
                        writer.add_histogram(
                                f"{writer_tag}Parameters/tanh.stroke_slopes",
                                guide_out.decoder_param.slope[0].detach(), 
                                iteration)
                
                # z prior parameters
                if generative_model.prior_dist == 'Sequential':
                    writer.add_histogram(
                                f"{writer_tag}Parameters/z_pres_prior.p",
                                guide.internal_decoder.z_pres_p.detach(), 
                                iteration)
                    writer.add_histogram(
                                f"{writer_tag}Parameters/z_what_prior.loc",
                                guide.internal_decoder.z_what_loc.detach(), 
                                iteration)
                    writer.add_histogram(
                                f"{writer_tag}Parameters/z_what_prior.std",
                                guide.internal_decoder.z_what_std.detach(), 
                                iteration)
                    writer.add_histogram(
                            f"{writer_tag}Parameters/z_where_prior.loc.scale",
                            guide.internal_decoder.z_where_loc.detach()[:, 0], 
                            iteration)
                    writer.add_histogram(
                            f"{writer_tag}Parameters/z_where_prior.loc.shift",
                            guide.internal_decoder.z_where_loc.detach()[:, 1:3], 
                            iteration)
                    if guide.z_where_type == '4_rotate':
                        writer.add_histogram(
                            f"{writer_tag}Parameters/z_where_prior.loc.rotate",
                            guide.internal_decoder.z_where_loc.detach()[:, 3], 
                            iteration)
                    writer.add_histogram(
                            f"{writer_tag}Parameters/z_where_prior.std",
                            guide.internal_decoder.z_where_std.detach(), 
                            iteration)
                elif (not generative_model.fixed_prior and 
                    generative_model.prior_dist == 'Independent'):
                    writer.add_histogram(
                            f"{writer_tag}Parameters/z_pres_prior.p",
                            generative_model.z_pres_prob.detach(), 
                            iteration)
                    writer.add_histogram(
                            f"{writer_tag}Parameters/z_what_prior.loc",
                            generative_model.pts_loc.detach(), iteration)
                    writer.add_histogram(
                            f"{writer_tag}Parameters/z_what_prior.std",
                            generative_model.pts_std.detach(), iteration)
                    writer.add_histogram(
                            f"{writer_tag}Parameters/z_where_prior.loc.scale",
                            generative_model.z_where_loc.detach()[0], iteration)
                    writer.add_histogram(
                            f"{writer_tag}Parameters/z_where_prior.loc.shift",
                            generative_model.z_where_loc.detach()[1:3], 
                            iteration)
                    if guide.z_where_type == '4_rotate':
                        writer.add_histogram(
                            f"{writer_tag}Parameters/z_where_prior.loc.rotate",
                            generative_model.z_where_loc.detach()[3], iteration)
                    writer.add_histogram(
                            f"{writer_tag}Parameters/z_where_prior.std",
                            generative_model.z_where_std.detach(), iteration)
                    
                                
                # z posterior parameters
                z_pres_pms = guide_out.z_pms.z_pres.detach()
                writer.add_scalar(f"{writer_tag}Train curves/minimal z_pres.p",
                                z_pres_pms.min(), iteration)
                writer.add_histogram(
                            f"{writer_tag}Parameters/z_pres_posterior.p",
                            z_pres_pms, iteration)
                writer.add_histogram(
                        f"{writer_tag}Parameters/z_where_posterior.loc.scale",
                        guide_out.z_pms.z_where.detach()[:, :, :, 0, 0], 
                        iteration)
                writer.add_histogram(
                        f"{writer_tag}Parameters/z_where_posterior.loc.shift",
                        guide_out.z_pms.z_where.detach()[:, :, :, 1:3, 0], 
                        iteration)
                if guide.z_where_type == '4_rotate':
                    writer.add_histogram(
                        "f{writer_tag}Parameters/z_where_posterior.loc.rotate",
                        guide_out.z_pms.z_where.detach()[:, :, :, 3, 0], 
                        iteration)
                writer.add_histogram(
                        f"{writer_tag}Parameters/z_where_posterior.std",
                        guide_out.z_pms.z_where.detach()[:, :, :, :, 1], 
                        iteration)
                writer.add_histogram(
                        f"{writer_tag}Parameters/z_what_posterior.loc",
                        guide_out.z_pms.z_what.detach()[:, :, :, :, :, 0], 
                        iteration)
                writer.add_histogram(
                        f"{writer_tag}Parameters/z_what_posterior.std",
                        guide_out.z_pms.z_what.detach()[:, :, :, :, :, 1], 
                        iteration)
                writer.add_scalar(
                        f"{writer_tag}Train curves/z_what_posterior.std_sum",
                        guide_out.z_pms.z_what.detach()[:, :, :, :, :, 1].sum(), 
                        iteration)

                # z posterior samples
                z_pres_smpls = guide_out.z_smpl.z_pres.detach()
                writer.add_scalar(f"{writer_tag}Train curves/# of 0s in z_pres",
                    np.prod(z_pres_smpls.shape) - z_pres_smpls.sum(),
                    iteration)
                writer.add_histogram(f"{writer_tag}Samples/z_pres",
                                z_pres_smpls, iteration)
                writer.add_histogram(f"{writer_tag}Samples/z_where.scale",
                                guide_out.z_smpl.z_where.detach()[:, :, :, 0], 
                                iteration)
                writer.add_histogram(f"{writer_tag}Samples/z_where.shift",
                                guide_out.z_smpl.z_where.detach()[:, :, :, 1:3], 
                                iteration)
                if guide.z_where_type == '4_rotate':
                    writer.add_histogram(f"{writer_tag}Samples/z_where.rotate",
                                guide_out.z_smpl.z_where.detach()[:, :, :, 3], 
                                iteration)
                writer.add_histogram(f"{writer_tag}Samples/z_what",
                                guide_out.z_smpl.z_what.detach(), iteration)
    loss = torch.logsumexp(loss, dim=0) - torch.log(torch.tensor(k))
    elbo = torch.logsumexp(elbo, dim=0) - torch.log(torch.tensor(k))
    return SequentialLoss(overall_loss=loss, 
                            model_loss=model_loss,
                            baseline_loss=baseline_loss,
                            neg_reinforce_term=neg_reinforce_term,
                            neg_elbo=-elbo,
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

def get_loss_air(generative_model, guide, imgs, loss_type='elbo', k=1,
                                                    iteration=0, 
                                                    writer=None,
                                                    writer_tag=None,
                                                    beta=1):
    '''Get loss for sequential model, e.g. AIR
    Args:
        loss_type (str): "nll": negative log likelihood, "l1": L1 loss, "elbo": -ELBO
        k (int): the number of samples to compute to compute the loss.
    '''
    # Guide output
    guide_out = guide(imgs, k)
    # wheater to mask current value based on prev.z_pres; 
    # more doc in model
    latents, log_post, bl_value, mask_prev, canvas, z_prior = (
        guide_out.z_smpl, guide_out.z_lprb, guide_out.baseline_value, 
        guide_out.mask_prev, guide_out.canvas, guide_out.z_prior)
    
    # multi by beta
    log_post_ = {}
    for i, z in enumerate(log_post._fields):
        log_post_[z] = log_post[i] * beta
    log_post = ZLogProb(**log_post_)
    
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
                                                z_prior=z_prior,
                                                )

    # multiply by beta
    log_prior_ = {}
    for i, z in enumerate(log_prior._fields):
        log_prior_[z] = log_prior[i] * beta
    log_prior = ZLogProb(**log_prior_)

    # z_pres_prior_lprb, z_what_post_lprb, z_where_post_lprb = log_prior
    log_prior_z = torch.cat(
                        [prob.sum(-1, keepdim=True) for prob in 
                            log_prior], dim=-1).sum(-1)
    generative_joint_log_prob = (log_likelihood + log_prior_z)

    # Compute ELBO: [bs, ]
    elbo = - log_post_z + generative_joint_log_prob

    # bl_target size: [batch_size, max_strks]
    # sum_{i=t}^T [ KL[i] - log p(x | z) ] 
    # =sum_{i=t}^T [ log_post - log_prior - log p(x | z)]
    # for all steps up to (and including) the first z_pres=0
    # (flip -> cumsum -> flip) so that it cumulate on to the left
    bl_target = torch.cat([prob.flip(-1).cumsum(-1).flip(-1
            ).unsqueeze(-1) for prob in log_post], dim=-1).sum(-1)
    bl_target -= torch.cat([prob.flip(-1).cumsum(-1).flip(-1
            ).unsqueeze(-1) for prob in log_prior], dim=-1).sum(-1)
        # this is like -ELBO
    bl_target = bl_target - log_likelihood[:, :, None]
    bl_target = (bl_target * mask_prev)

    # The "REINFORCE"  term in the gradient is: [bs,]; 
    # bl)target is the negative elbo
    # bl_value: [bs, max_strks]; z_pres [bs, max_strks]; 
    # (bl_target - bl_value) * gradient[z_pres_likelihood]
    neg_reinforce_term = (bl_target - bl_value).detach() * log_post.z_pres
    neg_reinforce_term = neg_reinforce_term * mask_prev
    neg_reinforce_term = neg_reinforce_term.sum(2) # [bs, ]

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
    baseline_loss = baseline_loss.sum(2)

    loss = model_loss + baseline_loss # [bs, ]
    
    if writer is not None:
        with torch.no_grad():
            if writer_tag is None:
                writer_tag = ''
            # loss
            for n, log_prob in zip(log_post._fields, log_post):
                writer.add_scalar(f"{writer_tag}Train curves/log_posterior/"+n, 
                                    log_prob.detach().sum(-1).mean(), 
                                    iteration)
            for n, log_prob in zip(['z_pres', 'z_where', 'z_what'], log_prior):
                writer.add_scalar(f"{writer_tag}Train curves/log_prior/"+n, 
                                    log_prob.detach().sum(-1).mean(), 
                                    iteration)
            # z prior parameters
            if generative_model.prior_dist == 'Sequential':
                writer.add_histogram(f"{writer_tag}Parameters/z_pres_prior.p",
                            guide.internal_decoder.z_pres_p.detach(), 
                            iteration)
                writer.add_histogram(f"{writer_tag}Parameters/z_what_prior.loc",
                            guide.internal_decoder.z_what_loc.detach(), 
                            iteration)
                writer.add_histogram(f"{writer_tag}Parameters/z_what_prior.std",
                            guide.internal_decoder.z_what_std.detach(), 
                            iteration)
                writer.add_histogram(
                            f"{writer_tag}Parameters/z_where_prior.loc.scale",
                            guide.internal_decoder.z_where_loc.detach()[:, 0], 
                            iteration)
                writer.add_histogram(
                            f"{writer_tag}Parameters/z_where_prior.loc.shift",
                            guide.internal_decoder.z_where_loc.detach()[:, 1:3], 
                            iteration)
                if guide.z_where_type == '4_rotate':
                    writer.add_histogram(
                            f"{writer_tag}Parameters/z_where_prior.loc.rotate",
                            guide.internal_decoder.z_where_loc.detach()[:, 3], 
                            iteration)
                writer.add_histogram(
                            f"{writer_tag}Parameters/z_where_prior.std",
                            guide.internal_decoder.z_where_std.detach(), 
                            iteration)

            writer.add_histogram("Parameters/z_pres",
                        guide_out.z_pms.z_pres.detach(), 
                        iteration)
            writer.add_histogram("Parameters/z_where_posterior.loc.scale",
                        guide_out.z_pms.z_where.detach()[:, :, 0, 0], 
                        iteration)
            writer.add_histogram("Parameters/z_where_posterior.loc.shift",
                        guide_out.z_pms.z_where.detach()[:, :, 1:, 0,], 
                        iteration)
            writer.add_histogram("Parameters/z_where_posterior.scale",
                        guide_out.z_pms.z_where.detach()[:, :, :, 1], 
                        iteration)
            writer.add_histogram("Parameters/z_what_posterior.loc",
                        guide_out.z_pms.z_what.detach()[:, :, :, 0], 
                        iteration)
            writer.add_histogram("Parameters/z_what_posterior.scale",
                        guide_out.z_pms.z_what.detach()[:, :, :, 1], 
                        iteration)
        
            writer.add_histogram("Parameters/img.loc", 
                    generative_model.renders_imgs(latents), iteration)

    loss = torch.logsumexp(loss, dim=0) - torch.log(torch.tensor(k))
    elbo = torch.logsumexp(elbo, dim=0) - torch.log(torch.tensor(k))

    return SequentialLoss(overall_loss=loss, 
                            model_loss=model_loss,
                            baseline_loss=baseline_loss,
                            neg_reinforce_term=neg_reinforce_term,
                            neg_elbo=-elbo,
                            neg_log_likelihood=-log_likelihood,
                            neg_log_prior=-log_prior_z, 
                            log_posterior=log_post_z)

def get_loss_vae(generative_model, 
                  guide, 
                  imgs, 
                  iteration=None, 
                  writer=None, k=1):
    '''
    Args:
        loss (str): "nll": negative log likelihood, "l1": L1 loss, "elbo": -ELBO
    '''
    guide_out = guide(imgs, k)
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

    loss = torch.logsumexp(loss, dim=0) - torch.log(torch.tensor(k))

    return BaseLoss(overall_loss=loss, 
                neg_log_prior=-log_prior,
                neg_log_likelihood=-log_likelihood,
                log_posterior=guide_log_prob,
                neg_elbo=loss.clone())