from collections import namedtuple

import torch
import torch.nn.functional as F
import numpy as np
from numpy import prod
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

def get_loss_sequential(generative_model, 
                        guide, 
                        imgs, 
                        k=1,
                        iteration=0, writer=None, writer_tag=None, beta=1,
                        args=None, alpha=0.8, train=True, 
                        average_particle_score=True):
    '''Get loss for sequential model, e.g. AIR
    Args:
        loss_type (str): "nll": negative log likelihood, "l1": L1 loss, "elbo": -ELBO
        k (int): the number of samples to compute to compute the loss.
        beta (float): beta term as in beta-VAE
        c, v, alpha: parameters for NVIL algorithm for centering, normalizing
            the REINFORCE learning signals.
    '''
    if args.no_spline_renderer:
        generative_model = guide.internal_decoder
    # beta=3
    # if iteration >= 60000:
    #     guide.wt_mlp.more_range, guide.where_mlp.more_range = True, True
    if args.constrain_z_pres_param and iteration < 10000:
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

    modify_reparam_likelihood = False
    if modify_reparam_likelihood:
        beta2 = 1.5
        log_likelihood = log_likelihood * beta2
    # log_likelihood = log_likelihood/beta
    # z_pres_prior_lprb, z_what_post_lprb, z_where_post_lprb = log_prior
    log_prior_z = torch.cat(
                        [prob.sum(-1, keepdim=True) for prob in 
                            log_prior], dim=-1).sum(-1)
    generative_joint_log_prob = (log_likelihood + log_prior_z)

    # Compute ELBO: [ptcs, bs, ]
    elbo = - log_post_z + generative_joint_log_prob
    assert elbo.shape == torch.Size([k, *imgs.shape[:-3]])

    # reinforce_sgnl size: [batch_size, max_strks]
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
    # deprecated; now using update_reinforce_loss;
    # we have shown that using this help with initialize to good reconstruction
    if args.update_reinforce_ll:
        if iteration == 0:
            # use generative_model to store min_ll
            generative_model.mean_ll = reinforce_ll.mean()
        # centering
        reinforce_ll = reinforce_ll - generative_model.mean_ll

    reinforce_ll = reinforce_ll / beta

    if modify_reparam_likelihood:
        # divide twice to cancel out the effect from multiply
        reinforce_ll = reinforce_ll / beta2

    if args.global_reinforce_signal:
        # ptcs, bs, n_strks
        bl_target = torch.cat([prob.sum(-1, keepdim=True) 
                        for prob in log_post], dim=-1).sum(-1, keepdim=True)
        bl_target -= torch.cat([prob.sum(-1, keepdim=True)
                        for prob in log_prior], dim=-1).sum(-1, keepdim=True)
        bl_target = bl_target.repeat(1, 1, args.strokes_per_img)
        
        last_t = mask_prev.sum(-1).int().squeeze(0) - 1
        bl_value = bl_value.index_select(-1, last_t)[:, :, 0:1].repeat(
                                                    1, 1, args.strokes_per_img)
        # bl_value = bl_value[:, :, -1].repeat(1, 1, args.strokes_per_img)
    else:
        bl_target = torch.cat([prob.flip(-1).cumsum(-1).flip(-1).unsqueeze(-1)
                        for prob in log_post], dim=-1).sum(-1)
        bl_target -= torch.cat([prob.flip(-1).cumsum(-1).flip(-1).unsqueeze(-1)
                        for prob in log_prior], dim=-1).sum(-1)

    # this is like -ELBO
    bl_target = bl_target - reinforce_ll[:, :, None]

    # Q: check whether to keep masking; masking result in z_pres 0 has no gradient
    # A: this is used to train the baseline, so we'd keep the learning signal
    bl_target = (bl_target * mask_prev)

    # this is still updated reinforce_sgnl; purely created for later convenience
    reinforce_sgnl = bl_target - bl_value

    # addition:
    # Q: should this be before or after reinforce_sgnl masking?
    # A: Before, because if masking then we want no signal for such term and 
    # centering can cause it to be nonezero, defeating the purpose of centering
    if args.update_reinforce_loss:
        reinforce_sgnl = reinforce_sgnl.detach()
        num_nonzero = mask_prev.sum()
        signal_mean = (reinforce_sgnl * mask_prev).sum() / num_nonzero
        signal_var = ( ( (reinforce_sgnl - signal_mean) * mask_prev) ** 2
                       ).sum() / num_nonzero
        if iteration == 0: 
            guide.c, guide.v = signal_mean, signal_var
        guide.c = alpha * guide.c + (1 - alpha) * signal_mean
        guide.v = alpha * guide.v + (1 - alpha) * signal_var
        # Q: should this be for each time step or for all elements?
        # A: For all elements, as the gradient for the param is the sum of the
        # gradients for each step.
        # Q: should this be for unmasked elements or including the masked?
        reinforce_sgnl = (reinforce_sgnl - guide.c) / max(
                                                        1, torch.sqrt(guide.v))

    # The "REINFORCE"  term in the gradient is: [ptcs, bs,]; 
    # bl_target is the negative elbo
    # bl_value: [ptcs, bs, max_strks]; z_pres [ptcs, bs, max_strks]; 
    # (bl_target - bl_value) * gradient[z_pres_posterior]
    neg_reinforce_term = reinforce_sgnl.detach() * log_post.z_pres
    # Q: check whether to keep masking
    # A:
    # when not masked all 0s to have learning signals
    # when masked with z_pres, no 0 has gradients
    # when masked with mask_prev, the first pres=0 has gradients
    neg_reinforce_term = neg_reinforce_term * mask_prev
    neg_reinforce_term = neg_reinforce_term.sum(2) # [ptcs, bs, ]

    # [ptcs, bs, ]
    # Q: shouldn't reinforce be negative? 
    # A: it's already negative from (KL - likelihood)
    model_loss = neg_reinforce_term - elbo


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
            # z posterior samples
            z_pres_smpls = guide_out.z_smpl.z_pres.detach()
            shp = z_pres_smpls.shape[:3]
            if train:
                writer.add_scalar(
                    f"{writer_tag}Train curves/neg_log_likelihood in reinforce",
                    -reinforce_ll.mean(), iteration)
                writer.add_scalar(f"{writer_tag}Train curves/beta", beta, 
                                iteration)
                writer.add_scalar(f"{writer_tag}Train curves/reinforce variance",
                                reinforce_sgnl.detach().var(), iteration)
                # loss
                for n, log_prob in zip(log_post._fields, log_post):
                    writer.add_scalar(f"{writer_tag}Train curves/log_posterior/"+n, 
                                        log_prob.detach().sum(-1).mean(), 
                                        iteration)
                for n, log_prob in zip(['z_pres', 'z_where', 'z_what'], log_prior):
                    writer.add_scalar(f"{writer_tag}Train curves/log_prior/"+n, 
                                        log_prob.detach().sum(-1).mean(), 
                                        iteration)
                writer.add_scalar(f"{writer_tag}Train curves/# of 1s in z_pres",
                    z_pres_smpls.sum(),
                    iteration)

            if args.log_param and iteration % 50 == 0:
                log_z_post_samples = True
                z_pres = z_pres_smpls.view(prod(shp)).detach().cpu().numpy()
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
                # if generative_model.prior_dist == 'Sequential':
                #     pr_pri_p = guide.internal_decoder.z_pres_p.detach().cpu()
                #     # pr_pri_p = pr_pri_p.view(prod(shp), -1).numpy()[z_pres==1]
                #     writer.add_histogram(
                #                 f"{writer_tag}Parameters/z_pres_prior.p",
                #                 pr_pri_p, iteration)
                #     writer.add_histogram(
                #                 f"{writer_tag}Parameters/z_what_prior.loc",
                #                 guide.internal_decoder.z_what_loc.detach(), 
                #                 iteration)
                #     writer.add_histogram(
                #                 f"{writer_tag}Parameters/z_what_prior.std",
                #                 guide.internal_decoder.z_what_std.detach(), 
                #                 iteration)
                    # wr_pri_scale_mean = guide.internal_decoder.z_where_loc.\
                    #                                 detach()[:, :, :, 0].cpu()
                    # wr_pri_scale_mean = wr_pri_scale_mean.view(prod(shp), -1
                    #                                     ).numpy()[z_pres==1]
                    # writer.add_histogram(
                    #         f"{writer_tag}Parameters/z_where_prior.scale.mean",
                    #         wr_pri_scale_mean, iteration)
                #     writer.add_histogram(
                #             f"{writer_tag}Parameters/z_where_prior.loc.shift",
                #             guide.internal_decoder.z_where_loc.detach()[:, 1:3], 
                #             iteration)
                #     if guide.z_where_type == '4_rotate':
                #         writer.add_histogram(
                #             f"{writer_tag}Parameters/z_where_prior.loc.rotate",
                #             guide.internal_decoder.z_where_loc.detach()[:, 3], 
                #             iteration)
                #     writer.add_histogram(
                #             f"{writer_tag}Parameters/z_where_prior.std",
                #             guide.internal_decoder.z_where_std.detach(), 
                #             iteration)
                # elif (not generative_model.fixed_prior and 
                #     generative_model.prior_dist == 'Independent'):
                #     writer.add_histogram(
                #             f"{writer_tag}Parameters/z_pres_prior.p",
                #             generative_model.z_pres_prob.detach(), 
                #             iteration)
                #     writer.add_histogram(
                #             f"{writer_tag}Parameters/z_what_prior.loc",
                #             generative_model.pts_loc.detach(), iteration)
                #     writer.add_histogram(
                #             f"{writer_tag}Parameters/z_what_prior.std",
                #             generative_model.pts_std.detach(), iteration)
                #     writer.add_histogram(
                #             f"{writer_tag}Parameters/z_where_prior.loc.scale",
                #             generative_model.z_where_loc.detach()[0], iteration)
                #     writer.add_histogram(
                #             f"{writer_tag}Parameters/z_where_prior.loc.shift",
                #             generative_model.z_where_loc.detach()[1:3], 
                #             iteration)
                #     if guide.z_where_type == '4_rotate':
                #         writer.add_histogram(
                #             f"{writer_tag}Parameters/z_where_prior.loc.rotate",
                #             generative_model.z_where_loc.detach()[3], iteration)
                #     writer.add_histogram(
                #             f"{writer_tag}Parameters/z_where_prior.std",
                #             generative_model.z_where_std.detach(), iteration)
                    
                                
                # z posterior parameters
                # if args.simple_pres:
                #     writer.add_scalar(f"{writer_tag}Train curves/pr_rsd_power",
                #                 guide.get_pr_rsd_power().detach(), iteration)
                # z_pres_pms = guide_out.z_pms.z_pres.detach()
                # writer.add_scalar(f"{writer_tag}Train curves/minimal z_pres.p",
                #                 z_pres_pms.min(), iteration)
                # writer.add_histogram(
                #             f"{writer_tag}Parameters/z_pres_posterior.p",
                #             z_pres_pms, iteration)
                # writer.add_histogram(
                #         f"{writer_tag}Parameters/z_where_posterior.loc.scale",
                #         guide_out.z_pms.z_where.detach()[:, :, :, 0, 0], 
                #         iteration)
                # writer.add_histogram(
                #         f"{writer_tag}Parameters/z_where_posterior.loc.shift",
                #         guide_out.z_pms.z_where.detach()[:, :, :, 1:3, 0], 
                #         iteration)
                # if guide.z_where_type == '4_rotate':
                #     writer.add_histogram(
                #         f"{writer_tag}Parameters/z_where_posterior.loc.rotate",
                #         guide_out.z_pms.z_where.detach()[:, :, :, 3, 0], 
                #         iteration)
                # writer.add_histogram(
                #         f"{writer_tag}Parameters/z_where_posterior.std",
                #         guide_out.z_pms.z_where.detach()[:, :, :, :, 1], 
                #         iteration)
                # writer.add_histogram(
                #         f"{writer_tag}Parameters/z_what_posterior.loc",
                #         guide_out.z_pms.z_what.detach()[:, :, :, :, :, 0], 
                #         iteration)
                # writer.add_histogram(
                #         f"{writer_tag}Parameters/z_what_posterior.std",
                #         guide_out.z_pms.z_what.detach()[:, :, :, :, :, 1], 
                #         iteration)
                # writer.add_scalar(
                #         f"{writer_tag}Train curves/z_what_posterior.std_sum",
                #         guide_out.z_pms.z_what.detach()[:, :, :, :, :, 1].sum(), 
                #         iteration)

                # z posterior samples
                if log_z_post_samples:
                    # writer.add_histogram(f"{writer_tag}Samples/z_pres",
                    #                 z_pres_smpls, iteration)
                    z_where = guide_out.z_smpl.z_where.detach().cpu()
                    if guide.z_where_type == '5':
                        z_where_shift = z_where[:, :, :, 0:2].view(prod(shp),-1
                                                            ).numpy()[z_pres==1]
                        z_where_scale = z_where[:, :, :, 2:4].view(prod(shp),-1
                                                            ).numpy()[z_pres==1]
                        z_where_rot = z_where[:, :, :, 4].view(prod(shp),-1
                                                            ).numpy()[z_pres==1]
                    else:
                        z_where_shift = z_where[:, :, :, 0:2].view(prod(shp),-1
                                                            ).numpy()[z_pres==1]
                        z_where_scale = z_where[:, :, :, 2:3].view(prod(shp),-1
                                                            ).numpy()[z_pres==1]
                        if guide.z_where_type == '4_rotate':
                            z_where_rot = z_where[:, :, :, 3].view(prod(shp),-1
                                                            ).numpy()[z_pres==1]   
                    z_what = guide_out.z_smpl.z_what.detach().cpu()
                    z_what = z_what.view(prod(shp),-1).numpy()[z_pres==1]
                    if z_pres.sum() > 0:
                        writer.add_histogram(f"{writer_tag}Samples/z_where.scale",
                                                    z_where_scale, iteration)
                        writer.add_histogram(f"{writer_tag}Samples/z_where.shift",
                                                    z_where_shift, iteration)
                        if guide.z_where_type in ['5', '4_rotate']:
                            writer.add_histogram(f"{writer_tag}Samples/z_where.rotate",
                                                    z_where_rot, iteration)
                        writer.add_histogram(f"{writer_tag}Samples/z_what",
                                                            z_what, iteration)
    if average_particle_score:
        loss = torch.logsumexp(loss, dim=0) - torch.log(torch.tensor(k))
        elbo = torch.logsumexp(elbo, dim=0) - torch.log(torch.tensor(k))
        log_likelihood = (torch.logsumexp(log_likelihood, dim=0) - 
                            torch.log(torch.tensor(k)))
        log_prior_z = (torch.logsumexp(log_prior_z, dim=0) - 
                        torch.log(torch.tensor(k)))
        log_post_z = (torch.logsumexp(log_post_z, dim=0) - 
                        torch.log(torch.tensor(k)))
    return SequentialLoss(overall_loss=loss, 
                            model_loss=model_loss,
                            baseline_loss=baseline_loss,
                            neg_reinforce_term=neg_reinforce_term,
                            neg_elbo=-elbo,
                            neg_log_likelihood=-log_likelihood,
                            neg_log_prior=-log_prior_z, 
                            log_posterior=log_post_z)


def get_loss_base(generative_model, guide, imgs, loss="elbo"):
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

def get_loss_air(guide, imgs, k=1,
                                                    iteration=0, 
                                                    writer=None,
                                                    writer_tag='',
                                                    beta=1, args=None, 
                                                    alpha=0.8):
    '''Get loss for sequential model, e.g. AIR
    Args:
        loss_type (str): "nll": negative log likelihood, "l1": L1 loss, "elbo": -ELBO
        k (int): the number of samples to compute to compute the loss.
    '''
    # Guide output
    guide_out = guide(imgs, k)
    generative_model = guide.internal_decoder
    # wheater to mask current value based on prev.z_pres; 
    # more doc in model
    latents, log_post, bl_value, mask_prev, canvas, z_prior = (
        guide_out.z_smpl, guide_out.z_lprb, guide_out.baseline_value, 
        guide_out.mask_prev, guide_out.canvas, guide_out.z_prior)
    
    # multi by beta
    log_post_ = {}
    for i, z in enumerate(log_post._fields):
        log_post_[z] = log_post[i]
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
        log_prior_[z] = log_prior[i]
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
    
    reinforce_ll = log_likelihood.detach()
    reinforce_ll = reinforce_ll / beta
    # this is like -ELBO
    bl_target = bl_target - reinforce_ll[:, :, None]

    bl_target = (bl_target * mask_prev)

    # this is still updated reinforce_sgnl; purely created for later convenience
    reinforce_sgnl = bl_target - bl_value

    if True:
        reinforce_sgnl = reinforce_sgnl.detach()
        num_nonzero = mask_prev.sum()
        signal_mean = (reinforce_sgnl * mask_prev).sum() / num_nonzero
        signal_var = ( ( (reinforce_sgnl - signal_mean) * mask_prev) ** 2
                        ).sum() / num_nonzero
        if iteration == 0: 
            guide.c, guide.v = signal_mean, signal_var
        guide.c = alpha * guide.c + (1 - alpha) * signal_mean
        guide.v = alpha * guide.v + (1 - alpha) * signal_var
        # Q: should this be for each time step or for all elements?
        # A: For all elements, as the gradient for the param is the sum of the
        # gradients for each step.
        # Q: should this be for unmasked elements or including the masked?
        reinforce_sgnl = (reinforce_sgnl - guide.c) / max(
                                                        1, torch.sqrt(guide.v))

    # The "REINFORCE"  term in the gradient is: [bs,]; 
    # bl)target is the negative elbo
    # bl_value: [bs, max_strks]; z_pres [bs, max_strks]; 
    # (bl_target - bl_value) * gradient[z_pres_likelihood]
    neg_reinforce_term = reinforce_sgnl.detach() * log_post.z_pres
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
    
    z_pres_smpls = guide_out.z_smpl.z_pres.detach()

    if writer != None:
        with torch.no_grad():
            if writer_tag is None:
                writer_tag = ''

            writer.add_scalar(f"{writer_tag}Train curves/# of 1s in z_pres",
                z_pres_smpls.sum(),
                iteration)

            # loss
            for n, log_prob in zip(log_post._fields, log_post):
                writer.add_scalar(f"{writer_tag}Train curves/log_posterior/"+n, 
                                    log_prob.detach().sum(-1).mean(), 
                                    iteration)
            for n, log_prob in zip(['z_pres', 'z_where', 'z_what'], log_prior):
                writer.add_scalar(f"{writer_tag}Train curves/log_prior/"+n, 
                                    log_prob.detach().sum(-1).mean(), 
                                    iteration)
            if args.log_param and iteration % 50 == 0:
            # if iteration % 50 == 0:
                writer.add_histogram(f"{writer_tag}Parameters/img_dist_std",
                        generative_model.get_imgs_dist_std().detach(), 
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
            
                # writer.add_histogram("Parameters/img.loc", 
                #         generative_model.renders_imgs(latents), iteration)

    loss = torch.logsumexp(loss, dim=0) - torch.log(torch.tensor(k))
    elbo = torch.logsumexp(elbo, dim=0) - torch.log(torch.tensor(k))
    log_likelihood = (torch.logsumexp(log_likelihood, dim=0) - 
                        torch.log(torch.tensor(k)))
    log_prior_z = (torch.logsumexp(log_prior_z, dim=0) - 
                    torch.log(torch.tensor(k)))
    log_post_z = (torch.logsumexp(log_post_z, dim=0) - 
                    torch.log(torch.tensor(k)))

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