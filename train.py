import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

import util
import plot
import losses
from models import base, sequential 

def train(model, optimizer, stats, data_loader, args):
    # Write will output to ./log
    writer = SummaryWriter(log_dir="./log", comment="L1 loss")

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
            # fit on only 1 batch
            if iteration < 10000:
                imgs = util.transform(imgs.to(args.device))
            else:
                imgs = imgs.to(args.device)
            # imgs = one_batch.to(args.device)
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                if args.model_type == 'base':
                    base.schedule_model_parameters(generative_model, guide, 
                                                iteration, args.loss, args.device)
                    loss_tuple = losses.get_loss_base(
                                            generative_model, guide, imgs, 
                                            loss=args.loss,)
                elif args.model_type == 'sequential':
                    loss_tuple = losses.get_loss_sequential(
                                                    generative_model, guide,
                                                    imgs, args.loss)
                else:
                    raise NotImplementedError

                loss = loss_tuple.overall_loss.mean()
                loss.backward()

            # Log loss, gradients and some parameters
            for n, l in zip(loss_tuple._fields, loss_tuple):
                writer.add_scalar("Train curves/"+n, l.detach().mean(), iteration)
            # writer.add_scalars("Train curves", {n:l.detach().mean() for n, l in 
            #                     zip(loss_tuple._fields, loss_tuple)}, iteration)                                
            writer.add_scalars("Gradient Norm", {f"Grad/{n}":
                                        p.grad.norm(2) for n, p in 
                                        guide.named_parameters()}, iteration)
            if generative_model.input_dependent_param:
                writer.add_scalar("Parameters/gen.sigma",
                                        generative_model.sigma.mean(),iteration)
                # writer.add_scalar("Parameters/tanh.norm.slope",
                #              generative_model.tanh_norm_slope.mean(), iteration)
                # if args.model_type == 'sequential':
                #     writer.add_scalar("Parameters/tanh.norm.slope.per_stroke",
                #       generative_model.tanh_norm_slope_stroke.mean(), iteration)
            else:
                writer.add_scalar("Parameters/gen.sigma", 
                                 generative_model.get_sigma().mean() ,iteration)
                writer.add_scalar("Parameters/tanh.norm.slope",
                            generative_model.get_tanh_slope().mean(), iteration)
                writer.add_scalar("Parameters/imgs_dist.std",
                            generative_model.get_imgs_dist_std().mean(), iteration)
                if args.model_type == 'sequential':
                    writer.add_scalar("Parameters/tanh.norm.slope.per_stroke",
                       generative_model.get_tanh_slope_strk().mean(), iteration)
            # Check for nans
            for name, parameter in guide.named_parameters():
                # print(f"{name} has norm: {parameter.norm(1)}")
                # print(f"{name}.grad has norm: {parameter.grad.norm(2)}")
                # mod the grad
                if torch.isnan(parameter).any() or torch.isnan(parameter.grad).any():
                    print(f"{name}.grad has {parameter.grad.isnan().sum()}/{np.prod(parameter.shape)} nan parameters")
                    breakpoint()
                if torch.isnan(parameter).any(): 
                    breakpoint()
                    raise RuntimeError(f"nan in guide parameter {name}: {parameter}")

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
                util.save_checkpoint(
                    util.get_checkpoint_path(args, 
                    checkpoint_iteration=iteration),
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
            test(model, stats, test_loader, args, epoch=epoch, writer=writer)
    writer.close()

def test(model, stats, test_loader, args, save_imgs_dir=None, epoch=None, 
                                                                writer=None):
    '''
    Args:
        test_loader (DataLoader): testset dataloader
    '''
    if args.model_type == 'sequential':
        cum_losses = [0]*8
    elif args.model_type == 'base':
        cum_losses = [0]*3

    generative_model, guide = model
    generative_model.eval(); guide.eval()

    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(args.device)

            if args.model_type == 'base':
                loss_tuple = losses.get_loss_base(
                                        generative_model, guide, imgs, 
                                        loss=args.loss,)
            elif args.model_type == 'sequential':
                loss_tuple = losses.get_loss_sequential(
                                                generative_model, guide,
                                                imgs, args.loss)
            for i in range(len(loss_tuple)):
                cum_losses[i] += loss_tuple[i].sum()      
        
        # Logging
        data_size = len(test_loader.dataset)
        for i in range(len(cum_losses)):
            cum_losses[i] /= data_size

        if args.model_type == 'sequential':
            loss_tuple = losses.SequentialLoss(*cum_losses)
        elif args.model_type == 'base':
            loss_tuple = losses.BaseLoss(*cum_losses)

        stats.tst_losses.append(loss_tuple.overall_loss)
        
        for n, l in zip(loss_tuple._fields, loss_tuple):
            writer.add_scalar("Test curves/"+n, l, epoch)
        # writer.add_scalars("Test curves", {n:l for n, l in 
        #                         zip(loss_tuple._fields, loss_tuple)}, epoch)   
        util.logging.info(f"Epoch {epoch} Test loss | Loss = {stats.tst_losses[-1]:.3f}")

        plot.plot_reconstructions(imgs=imgs, 
                                      guide=guide, 
                                      generative_model=generative_model, 
                                      args=args, 
                                      writer=writer, 
                                      epoch=epoch,
                                      is_train=False)

        writer.flush()