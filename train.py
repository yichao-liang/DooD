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

# @profile
def train(model, 
          optimizer, 
          scheduler, 
          stats, 
          data_loader, 
          args, 
          writer, 
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
        memory = None
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
    fix_img, fix_tar = next(iter(val_loader))
    if args.model_type == 'MWS':
        obs_id = imgs.type(torch.int64)            
        imgs = mws_transform(target)
        fix_img = mws_transform(fix_tar)
    # if not args.bern_img_dist:
    #     imgs = util.blur(imgs)
    #     fix_img = util.blur(fix_img)
    imgs, fix_img = imgs.to(args.device), fix_img.to(args.device)

    args.num_iterations = 1e6
    while iteration < args.num_iterations:
        # freeze pres mlp
        # if iteration == 0:
        #     for p in guide.pres_mlp.parameters(): 
        #         p.requires_grad=False
        # Log training reconstruction in Tensorboard
        ite_since_last_plot = 0
        with torch.no_grad():
            plot.plot_reconstructions(imgs=imgs, guide=guide, 
                                      generative_model=generative_model, 
                                      args=args, writer=writer, epoch=epoch,
                                      writer_tag='Train', 
                                      dataset_name=args.dataset, 
                                      fix_img=fix_img,
                                      has_fixed_img=True)
            # test.stroke_mll_plot(model, val_loader, args, writer, epoch)

        # train_loader = itertools.chain(train_loader, val_loader)
        # torch.autograd.set_detect_anomaly(True)
        for imgs, target in train_loader:
            # Special data pre-processing for MWS
            obs_id = None
            if args.model_type == 'MWS':
                obs_id = imgs.type(torch.int64)            
                imgs = mws_transform(target)

            # prepare the data
            if iteration < 0 and not args.bern_img_dist: #np.inf:
                imgs = util.blur(imgs)
            imgs = imgs.to(args.device)
            # fit on only 1 batch

            optimizer.zero_grad()
            loss_tuple = get_loss_tuple(args, generative_model, guide, 
                                    iteration, imgs, writer, obs_id, memory)
            loss = loss_tuple.overall_loss.mean()
            loss.backward()

            # Constrain the baseline gradients
            # torch.nn.utils.clip_grad_norm_(guide.parameters(), max_norm=1e5)
                
            # Log loss, gradients and some parameters
            for n, l in zip(loss_tuple._fields, loss_tuple):
                writer.add_scalar("Train curves/"+n, l.detach().mean(), 
                                                                    iteration)
            writer.add_scalar("Train curves/lr", 
                              optimizer.state_dict()['param_groups'][0]['lr'], 
                              iteration)
            if args.anneal_non_pr_net_lr:
                writer.add_scalar("Train curves/lr.pres", 
                              optimizer.state_dict()['param_groups'][1]['lr'], 
                              iteration)

            # Check for nans gradients, parameters
            # nan_grad = False
            # named_params = get_model_named_params(args, guide, generative_model)
            # for name, p in named_params:
            #     try:
            #         if torch.isnan(p).any() or torch.isnan(p.grad).any():
            #             print(f"{name}.grad has {p.grad.isnan().sum()}"
            #                     f"/{np.prod(p.shape)} nan parameters")
            #             nan_grad = True
            #         # if args.log_grad:
            #         #     writer.add_scalar(f"Grad_norm/{name}", 
            #         #                 p.grad.norm(2), iteration)
            #     except Exception as e:
            #         print(e)
            #         breakpoint()
            # if nan_grad:
            #     breakpoint()

                    ## if (name == 'style_mlp.seq.linear_modules.2.weight' and
                    ##     (parameter.grad.norm(2) > 6e4)):
                    ##     print(f'{name} has grad_norm = {parameter.grad.norm(2)}')
                    ##     util.debug_gradient(name, parameter, imgs, 
                    ##                         guide, generative_model, optimizer,
                    ##                         iteration, writer, args)


            optimizer.step()
            if args.anneal_lr:
                scheduler.step(loss_tuple.log_posterior.detach().mean())

            stats = log_stat(args, stats, iteration, loss, loss_tuple)

            # Make a model tuple
            if args.model_type == 'MWS':
                model = generative_model, guide, memory
            else:
                model = generative_model, guide

            # Save Checkpoint
            # if iteration % 100 == 0 or iteration == \
            if iteration % args.save_interval == 0 or iteration == \
                                                            args.num_iterations:
                save(args, iteration, model, optimizer, scheduler, stats)
            
            # End training based on `iteration`
            iteration += 1
            if iteration == args.num_iterations:
                break
            # plot every 1k ite if an epoch takes too long
            ite_since_last_plot += 1
            if ite_since_last_plot > 1000:
                ite_since_last_plot = 0
                with torch.no_grad():
                    plot.plot_reconstructions(imgs=imgs, guide=guide, 
                                      generative_model=generative_model, 
                                      args=args, writer=writer, epoch=epoch,
                                      writer_tag='Train', 
                                      dataset_name=args.dataset, 
                                      fix_img=fix_img,
                                      has_fixed_img=True)

        epoch += 1
        writer.flush()

        # Test every epoch
        # if epoch % 5 == 0:
        #     if val_loader:
        #         test_model(model, stats, val_loader, args, epoch=epoch, writer=writer)
        # else:
        stats.tst_losses.append([])
        
        # count epochs when test is not used

    writer.close()
    
    save(args, iteration, model, optimizer, scheduler, stats)
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
        if not args.use_canvas and args.prior_dist == 'Sequential':
            named_params = itertools.chain(
                            guide.non_decoder_named_params(),
                            generative_model.no_img_dist_named_params()
                                        ) 
        elif args.use_canvas or args.prior_dist == 'Sequential':
            named_params = itertools.chain(
                                # guide.non_decoder_named_params(),
                                # guide.non_decoder_named_params(),
                                # generative_model.decoder_named_params()
                                        )
        elif args.prior_dist == 'Independent':
            named_params = itertools.chain(guide.named_parameters(),
                                    generative_model.named_parameters()) 
        elif not args.use_canvas:
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
                                 "prior_loss", "novel_proportion", 
                                 "new_map", "overall_loss"])
def get_loss_tuple(args, generative_model, guide, iteration, imgs, writer, 
                   obs_id, memory=None):
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
                                k=1,
                                iteration=iteration,
                                writer=writer,
                                beta=float(args.beta),
                                args=args)
    elif args.model_type == 'AIR':
        loss_tuple = losses.get_loss_air(
                                generative_model=generative_model, 
                                guide=guide,
                                imgs=imgs, 
                                k=1,
                                iteration=iteration,
                                writer=writer,
                                beta=float(args.beta),
                                args=args,
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
                            num_particles,
                            )
    return MWSLoss(neg_elbo=loss, theta_loss=torch.tensor(theta_loss), 
                   phi_loss=torch.tensor(phi_loss), 
                    prior_loss=torch.tensor(prior_loss), 
                    # accuracy=accuracy, 
                    novel_proportion=torch.tensor(novel_proportion), 
                    new_map=torch.tensor(new_map), 
                    overall_loss=loss)

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
        # if accuracy is not None:
        #     stats.accuracies.append(loss_tuple.accuracy)
        # if novel_proportion is not None:
        #     stats.novel_proportions.append(loss_tuple.novel_proportion)
        # if new_map is not None:
        #     stats.new_maps.append(loss_tuple.new_map)
    
    # Log
    if iteration % args.log_interval == 0:
        if args.model_type == 'MWS':
            util.logging.info(
                "it. {}/{} | prior loss = {:.2f} | theta loss = {:.2f} | "
                "phi loss = {:.2f} | novel = {}% | new map = {}% "
                "| last log_p = {} | last kl = {} | GPU memory = {:.2f} MB".format(
                    iteration,
                    args.num_iterations,
                    loss_tuple.prior_loss,
                    loss_tuple.theta_loss,
                    loss_tuple.phi_loss,
                    # loss_tuple.accuracy * 100 if loss_tuple.accuracy is not None else None,
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

def save(args, iteration, model, optimizer, scheduler, stats):
    # save iteration.pt
    if iteration % 5000 == 0:
    # if iteration % 100 == 0:
        if args.save_history_ckpt:
            util.save_checkpoint(
                util.get_checkpoint_path(args, 
                checkpoint_iteration=iteration),
                model,
                optimizer,
                scheduler,
                stats,
                run_args=args,
            )
    # save latest.pt
    util.save_checkpoint(
        util.get_checkpoint_path(args),
        model,
        optimizer,
        scheduler,
        stats,
        run_args=args,
    )