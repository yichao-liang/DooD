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
    
    # one_batch, _ = next(iter(train_loader))
    while iteration < args.num_iterations:
        for imgs, _ in train_loader:
            # fit on only 1 batch
            imgs = imgs.to(args.device)
            # imgs = one_batch.to(args.device)

            optimizer.zero_grad()
            if args.model_type == 'base':
                base.schedule_model_parameters(generative_model, guide, 
                                            iteration, args.loss, args.device)
                loss, neg_gen_prob, inf_prob = losses.get_loss_base(
                                        generative_model, guide, imgs, 
                                        loss=args.loss,)
            elif args.model_type == 'sequential':
                loss, neg_gen_prob, inf_prob = losses.get_loss_sequential(
                                                generative_model, guide,
                                                imgs, args.loss)
            else:
                raise NotImplementedError

            loss = loss.mean()
            neg_gen_prob = neg_gen_prob.mean()
            inf_prob = inf_prob.mean()
            writer.add_scalars("Train curves", {'Train/-ELBO': loss,
                                'Train/inference_log_prob': inf_prob, 
                                'Train/generative_negative_log_prob': 
                                                            neg_gen_prob},
                                iteration)
                                
            loss.backward()

            # Check for nans
            writer.add_scalars("Gradient Norm", {f"Grad/{n}":
                                        p.grad.norm(2) for n, p in 
                                        guide.named_parameters()}, iteration)
            writer.add_scalar("Parameters/gen.sigma",
                            util.constrain_parameter(generative_model.sigma, 
                                                min=.01, max=.05),iteration)
            writer.add_scalar("Parameters/tanh.norm.slope",
                util.constrain_parameter(generative_model.tanh_norm_slope, 
                                                min=.1,max=.7),iteration)
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

        # Log training reconstruction in Tensorboard
        with torch.no_grad():
            plot.plot_reconstructions(imgs=imgs, 
                                      guide=guide, 
                                      generative_model=generative_model, 
                                      args=args, 
                                      writer=writer, 
                                      epoch=epoch,
                                      is_train=True)

        # Test every epoch
        test(model, stats, test_loader, args, epoch=epoch, writer=writer)

def test(model, stats, test_loader, args, save_imgs_dir=None, epoch=None, 
                                                                writer=None):
    '''
    Args:
        test_loader (DataLoader): testset dataloader
    '''
    generative_model, guide = model
    generative_model.eval(); guide.eval()

    with torch.no_grad():
        loss, neg_gen_prob, inf_prob = 0, 0, 0
        for imgs, _ in test_loader:
            imgs = imgs.to(args.device)

            if args.model_type == 'base':
                loss, neg_gen_prob, inf_prob = losses.get_loss_base(
                                        generative_model, guide, imgs, 
                                        loss=args.loss,)
            elif args.model_type == 'sequential':
                loss_tp, neg_gen_prob_tp, inf_prob_tp = losses.get_loss_sequential(
                                                generative_model, guide,
                                                imgs, args.loss)
            loss += loss_tp.sum()
            neg_gen_prob += neg_gen_prob_tp.sum()
            inf_prob += inf_prob_tp.sum()
            
        plot.plot_reconstructions(imgs=imgs, 
                                      guide=guide, 
                                      generative_model=generative_model, 
                                      args=args, 
                                      writer=writer, 
                                      epoch=epoch,
                                      is_train=False)
        
        # Logging
        data_size = len(test_loader.dataset)
        loss /= data_size
        neg_gen_prob /= data_size
        inf_prob /= data_size
        stats.tst_losses.append(loss)
        writer.add_scalars("Test curves", {'Test/-ELBO': loss,
                                'Test/inference_log_prob': inf_prob, 
                                'Test/generative_negative_log_prob': 
                                                                neg_gen_prob,
                                }, epoch)
        util.logging.info(f"Epoch {epoch} Test loss | Loss = {stats.tst_losses[-1]:.3f}")
