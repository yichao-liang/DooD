import itertools
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

import util
import plot
import losses
import test
from models import base, sequential 

def train(model, optimizer, stats, data_loader, args, writer):

    checkpoint_path = util.get_checkpoint_path(args)
    num_iterations_so_far = len(stats.trn_losses)
    num_epochs_so_far = len(stats.tst_losses)
    iteration = num_iterations_so_far
    epoch = num_epochs_so_far

    generative_model, guide = model
    guide.train()
    generative_model.train()
    train_loader, test_loader = data_loader
    data_size = len(train_loader.dataset)
    
    # For ploting first
    imgs, _ = next(iter(train_loader))
    imgs = util.transform(imgs.to(args.device))
    while iteration < args.num_iterations:

        # Log training reconstruction in Tensorboard
        with torch.no_grad():
            plot.plot_reconstructions(imgs=imgs, 
                                      guide=guide, 
                                      generative_model=generative_model, 
                                      args=args, 
                                      writer=writer, 
                                      epoch=epoch,
                                      is_train=True)

        for imgs, _ in train_loader:
            if iteration < np.inf:
                imgs = util.transform(imgs.to(args.device))
            else:
                imgs = imgs.to(args.device)
            # fit on only 1 batch
            # imgs = one_batch.to(args.device)

            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
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
                                            writer=writer)
                elif args.model_type == 'AIR':
                    loss_tuple = losses.get_loss_air(
                                            generative_model=generative_model, 
                                            guide=guide,
                                            imgs=imgs, 
                                            loss_type=args.loss, 
                                            iteration=iteration,
                                            writer=writer)
                elif args.model_type == 'VAE':
                    loss_tuple = losses.get_loss_vae(
                                            generative_model=generative_model,
                                            guide=guide,
                                            imgs=imgs,
                                            iteration=iteration,
                                            writer=writer,
                    )
                else:
                    raise NotImplementedError

                loss = loss_tuple.overall_loss.mean()
                loss.backward()

            # Constrain the baseline gradients
            # for "independent" prior
            # torch.nn.utils.clip_grad_norm_(guide.parameters(), max_norm=1e4)
            # for "sequential" prior
            # torch.nn.utils.clip_grad_norm_(guide.parameters(), max_norm=1e5)
            # torch.nn.utils.clip_grad_norm_(guide.get_baseline_params(), 
            #                                max_norm=1e6)
                
            # Log loss, gradients and some parameters
            for n, l in zip(loss_tuple._fields, loss_tuple):
                writer.add_scalar("Train curves/"+n, l.detach().mean(), iteration)
            if args.model_type == 'Sequential' and args.prior_dist == 'Sequential':
                named_params = itertools.chain(
                            guide.named_parameters(), 
                            guide.internal_decoder.no_img_dist_named_params(),
                            generative_model.img_dist_named_params())
            else:
                named_params = guide.named_parameters()
            
            writer.add_scalars("Gradient Norm", {f"Grad/{n}":
                                        p.grad.norm(2) for n, p in 
                                        named_params}, 
                                        iteration)
            if args.model_type in ['Sequential', 'Base']:
                if generative_model.input_dependent_param:
                    # writer.add_histogram("Parameters/gen.sigma",
                    #                         generative_model.sigma, iteration)
                    # writer.add_histogram("Parameters/tanh.norm.slope",
                    #             generative_model.tanh_norm_slope, iteration)
                    # if args.model_type == 'sequential':
                    #     writer.add_histogram("Parameters/tanh.norm.slope.per_stroke",
                    #     generative_model.tanh_norm_slope_stroke, iteration)
                    pass
                else:
                    # writer.add_histogram("Parameters/gen.sigma", 
                    #                 generative_model.get_sigma(),iteration)
                    # writer.add_histogram("Parameters/tanh.norm.slope",
                    #             generative_model.get_tanh_slope(), iteration)
                    # writer.add_histogram("Parameters/imgs_dist.std",
                    #             generative_model.get_imgs_dist_std(), iteration)
                    # if args.model_type == 'Sequential':
                    #     writer.add_scalar("Parameters/tanh.norm.slope.per_stroke",
                    #     generative_model.get_tanh_slope_strk().mean(), iteration)
                    pass
            elif args.model_type in ['VAE', 'AIR']:
                writer.add_scalars("Gradient Norm", {f"Grad/{n}":
                            p.grad.norm(2) for n, p in 
                            itertools.chain(generative_model.named_parameters()
                            )}, 
                            iteration)
            # Check for nans
            if args.model_type in ["VAE", "AIR", "Base"]:
                params = itertools.chain(
                                        generative_model.named_parameters(), 
                                        guide.named_parameters())
            elif args.model_type in ['Sequential']:
                if args.execution_guided:
                    params = itertools.chain(
                            guide.named_parameters(),
                            guide.internal_decoder.no_img_dist_named_params(),
                            generative_model.img_dist_named_params()
                                        )
                else:
                    params = itertools.chain(
                            guide.named_parameters(),
                            generative_model.learnable_named_parameters()
                                        )
            for name, parameter in params:
                # print(f"{name} has norm: {parameter.norm(1)}")
                # print(f"{name}.grad has norm: {parameter.grad.norm(2)}")
                # mod the grad
                try:
                    if torch.isnan(parameter).any() or torch.isnan(parameter.grad).any():
                        print(f"{name}.grad has {parameter.grad.isnan().sum()}/{np.prod(parameter.shape)} nan parameters")
                        breakpoint()
                except:
                    breakpoint()

            optimizer.step()

            # Record stats
            stats.trn_losses.append(loss.item())
            
            # Log
            if iteration % args.log_interval == 0:
                util.logging.info(f"Iteration {iteration} | Loss = {stats.trn_losses[-1]:.3f}")

            # Make a model tuple
            model = generative_model, guide

            # Save Checkpoint
            if iteration % args.save_interval == 0 or iteration == \
                                                            args.num_iterations:
                # save iteration.pt
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
            
            # End training based on `iteration`
            iteration += 1
            if iteration == args.num_iterations:
                break
        epoch += 1
        writer.flush()

        # Test every epoch
        if test_loader:
            test_model(model, stats, test_loader, args, epoch=epoch, writer=writer)
    writer.close()
    return model

def test_model(model, stats, test_loader, args, save_imgs_dir=None, epoch=None, 
                                                                writer=None):
    with torch.no_grad():
        test.marginal_likelihoods(model, stats, test_loader, args, 
                                            save_imgs_dir, epoch, writer, k=1)