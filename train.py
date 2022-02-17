import itertools
from collections import namedtuple
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import util
import plot
import losses
import test
from models import base, air
from models.mws.handwritten_characters.losses import get_mws_loss

def anneal_lr(args, model, iteration):
    if args.model_type == 'Sequential':
        lr = util.anneal_weight(init_val=1e-3, final_val=1e-3,
                                cur_ite=iteration, anneal_step=2e4,
                                init_ite=1e4)
        args.lr = lr
        new_optimizer = util.init_optimizer(args, model)
        return args, new_optimizer
    if args.model_type == 'AIR':
        lr = util.anneal_weight(init_val=1e-3, final_val=1e-3,
                                cur_ite=iteration, anneal_step=2e4,
                                init_ite=1e4)
        args.lr = lr
        new_optimizer = util.init_optimizer(args, model)
        return args, new_optimizer

def increase_beta(args, model, iteration):
    if args.increase_beta:
        args.beta = util.heat_weight(init_val=1, final_val=args.final_beta,
                                        cur_ite=iteration, heat_step=3e4,
                                        init_ite=2e4)
        return args
        
def train(model, optimizer, stats, data_loader, args, writer, 
            dataset_name=None):

    if args.model_type == 'MWS':
        num_iterations_so_far = len(stats.theta_losses)
        num_epochs_so_far = 0
    else:
        num_iterations_so_far = len(stats.trn_losses)
        num_epochs_so_far = len(stats.tst_losses)
    iteration = num_iterations_so_far
    epoch = num_epochs_so_far

    if args.model_type != 'MWS':
        generative_model, guide = model
    else:
        mws_transform = transforms.Resize([args.img_res, args.img_res], 
                                            antialias=True)
        generative_model, guide, memory = model
        if memory is not None:
            util.logging.info(
            f"Size of MWS's memory of shape {memory.shape}: "
            f"{memory.element_size() * memory.nelement() / 1e6:.2f} MB"
            )
    guide.train()
    generative_model.train()
    train_loader, val_loader = data_loader
    
    # For ploting first
    imgs, target = next(iter(train_loader))
    if args.model_type == 'MWS':
        obs_id = imgs.type(torch.int64)            
        imgs = mws_transform(target)
    imgs = util.transform(imgs.to(args.device))
    if dataset_name is not None and dataset_name == "Omniglot":
        imgs = 1 - imgs

    while iteration < args.num_iterations:
        # Log training reconstruction in Tensorboard
        with torch.no_grad():
            plot.plot_reconstructions(imgs=imgs, guide=guide, 
                                      generative_model=generative_model, 
                                      args=args, writer=writer, epoch=epoch,
                                      writer_tag='Train', 
                                      dataset_name=args.dataset)
            # test.stroke_mll_plot(model, val_loader, args, writer, epoch)

        for imgs, target in train_loader:
            if args.anneal_lr:
                args, optimizer = anneal_lr(args, model, iteration)
            if args.increase_beta:
                args = increase_beta(args, model, iteration)

            # Special data pre-processing for MWS
            obs_id = None
            if args.model_type == 'MWS':
                obs_id = imgs.type(torch.int64)            
                imgs = mws_transform(target)
            # pre-processing for Omniglot
            if dataset_name is not None and dataset_name == "Omniglot":
                imgs = 1 - imgs

            # prepare the data
            if iteration < np.inf:
                imgs = util.transform(imgs.to(args.device))
            else:
                imgs = imgs.to(args.device)
            # fit on only 1 batch

            optimizer.zero_grad()
            loss_tuple = get_loss_tuple(args, generative_model, guide, 
                                    iteration, imgs, writer, obs_id)
            loss = loss_tuple.overall_loss.mean()
            loss.backward()

            # Constrain the baseline gradients
            # torch.nn.utils.clip_grad_norm_(guide.parameters(), max_norm=1e5)
                
            # Log loss, gradients and some parameters
            for n, l in zip(loss_tuple._fields, loss_tuple):
                writer.add_scalar("Train curves/"+n, l.detach().mean(), 
                                                                    iteration)
            # Check for nans gradients, parameters
            if args.log_grad:
                named_params = get_model_named_params(args, guide, 
                                                        generative_model)
                for name, parameter in named_params:
                    writer.add_scalar(f"Grad_norm/{name}", 
                                        parameter.grad.norm(2), iteration)
                #     try:
                    ## if (name == 'style_mlp.seq.linear_modules.2.weight' and
                    ##     (parameter.grad.norm(2) > 6e4)):
                    ##     print(f'{name} has grad_norm = {parameter.grad.norm(2)}')
                    ##     util.debug_gradient(name, parameter, imgs, 
                    ##                         guide, generative_model, optimizer,
                    ##                         iteration, writer, args)

                    # if torch.isnan(parameter).any() or torch.isnan(parameter.grad).any():
                    #     print(f"{name}.grad has {parameter.grad.isnan().sum()}"
                    #           f"/{np.prod(parameter.shape)} nan parameters")
                    #     breakpoint()
                # except Exception as e:
                #     print(e)
                #     breakpoint()

            optimizer.step()

            stats = log_stat(args, stats, iteration, loss, loss_tuple)

            # Make a model tuple
            if args.model_type == 'MWS':
                model = generative_model, guide, memory
            else:
                model = generative_model, guide

            # Save Checkpoint
            if iteration % args.save_interval == 0 or iteration == \
                                                            args.num_iterations:
                save(args, iteration, model, optimizer, stats)
            
            # End training based on `iteration`
            iteration += 1
            if iteration == args.num_iterations:
                break
        epoch += 1
        writer.flush()

        # Test every epoch
        # if val_loader:
        #     test_model(model, stats, val_loader, args, epoch=epoch, writer=writer)
    writer.close()
    
    save(args, iteration, model, optimizer, stats)
    return model

def test_model(model, stats, test_loader, args, save_imgs_dir=None, epoch=None, 
                                                                writer=None):
    test.marginal_likelihoods(model, stats, test_loader, args, 
                                            save_imgs_dir, epoch, writer, k=1,
                                            dataset_name=args.dataset)
    
def get_model_named_params(args, guide, generative_model):
    '''Return the trainable parameters of the models
    '''
    if args.model_type == 'Sequential' and args.prior_dist == 'Sequential':
            named_params = itertools.chain(
                    guide.named_parameters(), 
                    guide.internal_decoder.no_img_dist_named_params(),
                    generative_model.img_dist_named_params())
    elif args.model_type in ['AIR']:
        if not args.execution_guided and args.prior_dist == 'Sequential':
            named_params = itertools.chain(
                            guide.non_decoder_named_params(),
                            generative_model.no_img_dist_named_params()
                                        ) 
        elif args.execution_guided or args.prior_dist == 'Sequential':
            named_params = itertools.chain(
                                # guide.non_decoder_named_params(),
                                # guide.non_decoder_named_params(),
                                # generative_model.decoder_named_params()
                                        )
        elif args.prior_dist == 'Independent':
            named_params = itertools.chain(guide.named_parameters(),
                                    generative_model.named_parameters()) 
        elif not args.execution_guided:
            named_params = itertools.chain(
                                guide.non_decoder_named_params(),
                                generative_model.decoder_named_params())
        else:
            named_params = itertools.chain(
                                    guide.named_parameters(),
                                    generative_model.named_parameters())
    elif args.model_type in ['VAE']:
        named_params = itertools.chain(
                                guide.named_parameters(),
                                generative_model.named_parameters())
    else:
        named_params = guide.named_parameters()
    return named_params

MWSLoss = namedtuple('MWSLoss', ["neg_elbo", "theta_loss", "phi_loss", 
                                 "prior_loss","accuracy", "novel_proportion", 
                                 "new_map", "overall_loss"])
def get_loss_tuple(args, generative_model, guide, iteration, imgs, writer, 
                   obs_id):
    if args.model_type == 'Base':
        base.schedule_model_parameters(generative_model, guide, 
                                    iteration, args.loss, args.device)
        loss_tuple = losses.get_loss_base(
                                generative_model, guide, imgs, 
                                loss=args.loss,)
    elif args.model_type == 'Sequential':
        loss_tuple = losses.get_loss_sequential(
                                generative_model=generative_model, 
                                guide=guide,
                                imgs=imgs, 
                                loss_type=args.loss, 
                                k=1,
                                iteration=iteration,
                                writer=writer,
                                beta=float(args.beta),
                                args=args)
    elif args.model_type == 'AIR':
        air.schedule_model_parameters(generative_model, guide,
                                      iteration, args)
        loss_tuple = losses.get_loss_air(
                                generative_model=generative_model, 
                                guide=guide,
                                imgs=imgs, 
                                loss_type=args.loss, 
                                iteration=iteration,
                                writer=writer,
                                beta=float(args.beta),
                                )
    elif args.model_type == 'VAE':
        loss_tuple = losses.get_loss_vae(
                                generative_model=generative_model,
                                guide=guide,
                                imgs=imgs,
                                iteration=iteration,
                                writer=writer)
    elif args.model_type == 'MWS':
        loss_tuple = get_mws_loss_tuple(generative_model,
                                                guide,
                                                memory,
                                                imgs.squeeze(1).round(),
                                                obs_id,
                                                args.num_particles,)
    else:
        raise NotImplementedError
    return loss_tuple

def get_mws_loss_tuple(generative_model,
                        guide,
                        memory,
                        imgs,
                        obs_id,
                        num_particles,):
    (loss,
    theta_loss,
    phi_loss,
    prior_loss,
    accuracy,
    novel_proportion,
    new_map) = get_mws_loss(
                            generative_model,
                            guide,
                            memory,
                            imgs.squeeze(1).round(),
                            obs_id,
                            args.num_particles,
                            )
    return MWSLoss(neg_elbo=loss, theta_loss=theta_loss, phi_loss=phi_loss, 
                    prior_loss=prior_loss, accuracy=accuracy, 
                    novel_proportion=novel_proportion, 
                    new_map=new_map, overall_loss=loss)

def log_stat(args, stats, iteration, loss, loss_tuple):
    '''Log to stats and generate output for display
    '''
    # Record stats
    if args.model_type != 'MWS':
        stats.trn_losses.append(loss.item())
    else:
        stats.theta_losses.append(loss_tuple.theta_loss)
        stats.phi_losses.append(loss_tuple.phi_loss)
        stats.prior_losses.append(loss_tuple.prior_loss)
        if accuracy is not None:
            stats.accuracies.append(loss_tuple.accuracy)
        if novel_proportion is not None:
            stats.novel_proportions.append(loss_tuple.novel_proportion)
        if new_map is not None:
            stats.new_maps.append(loss_tuple.new_map)
    
    # Log
    if iteration % args.log_interval == 0:
        if args.model_type == 'MWS':
            util.logging.info(
                "it. {}/{} | prior loss = {:.2f} | theta loss = {:.2f} | "
                "phi loss = {:.2f} | accuracy = {}% | novel = {}% | new map = {}% "
                "| last log_p = {} | last kl = {} | GPU memory = {:.2f} MB".format(
                    iteration,
                    args.num_iterations,
                    loss_tuple.prior_loss,
                    loss_tuple.theta_loss,
                    loss_tuple.phi_loss,
                    loss_tuple.accuracy * 100 if loss_tuple.accuracy is not None else None,
                    loss_tuple.novel_proportion * 100 if loss_tuple.novel_proportion is not None else None,
                    loss_tuple.new_map * 100 if loss_tuple.new_map is not None else None,
                    "N/A" if len(stats.log_ps) == 0 else stats.log_ps[-1],
                    "N/A" if len(stats.kls) == 0 else stats.kls[-1],
                    (
                        torch.cuda.max_memory_allocated(device=args.device) / 1e6
                        if args.device.type == "cuda"
                        else 0
                    ),
                )
            )
        else:
            util.logging.info(f"Iteration {iteration} | Loss = {stats.trn_losses[-1]:.3f}")
    return stats

def save(args, iteration, model, optimizer, stats):
    # save iteration.pt
    if args.save_history_ckpt:
        util.save_checkpoint(
            util.get_checkpoint_path(args, 
            checkpoint_iteration=iteration),
            model,
            optimizer,
            stats,
            run_args=args,
        )
    # save latest.pt
    util.save_checkpoint(
        util.get_checkpoint_path(args),
        model,
        optimizer,
        stats,
        run_args=args,
    )