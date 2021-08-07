import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

import util
import losses

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

            # if args.loss == 'elbo':
            loss, neg_gen_prob, inf_prob = losses.get_elbo_loss(
                                        generative_model, guide, imgs, 
                                        loss=args.loss)

            loss = loss.mean()
            neg_gen_prob = neg_gen_prob.mean()
            inf_prob = inf_prob.mean()
            writer.add_scalars("Train curves", {'Train/-ELBO': loss,
                                'Train/inference_log_prob': inf_prob, 
                                'Train/generative_negative_log_prob': 
                                                            neg_gen_prob},
                                iteration)
            # else:
                # raise NotImplementedError()

            loss.backward()

            # Check for nans
            writer.add_scalars("Gradient Norm", {f"Grad/{n}":
                                        p.grad.norm(2) for n, p in 
                                        guide.named_parameters()}, iteration)
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

        # Log training reconstruction
        with torch.no_grad():
            n = min(imgs.shape[0], 16)
            latent = guide.rsample(imgs)
            recon_img = generative_model.img_dist_b(latent).mean
        fillers = torch.zeros(16-n, 1, 28, 28).to(args.device)
        comparision = torch.cat([imgs[:8], recon_img[:8], imgs[8:n], fillers, recon_img[8:n], fillers])
        img_grad = make_grid(comparision, nrow=8)
        # draw control points
        if args.inference_dist == 'Dirichlet':
            # reshape for Dir
            # breakpoint()
            latent = latent.chunk(2, -1)[0].view(*latent.shape[:-3], 
                            args.strokes_per_img, args.points_per_stroke, -1)
        writer.add_image("Train/Reconstruction", img_grad, epoch)
        util.add_control_points_plot(generative_model, latent, writer, tag="Train/Control Points", epoch=epoch)
        
        # test every epoch
        save_imgs_dir = util.get_save_test_img_dir(args, epoch)
        test(model, stats, test_loader, args, save_imgs_dir, epoch=epoch, writer=writer)

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

            loss_tp, neg_gen_prob_tp, inf_prob_tp = losses.get_elbo_loss(
                                                generative_model, guide, imgs,
                                                loss=args.loss)
            loss += loss_tp.sum()
            neg_gen_prob += neg_gen_prob_tp.sum()
            inf_prob += inf_prob_tp.sum()

        if save_imgs_dir is not None:
            n = min(imgs.shape[0], 16)
            latent = guide.rsample(imgs)
            recon_img = generative_model.img_dist_b(latent).mean
            fillers = torch.zeros(16-n, 1, 28, 28).to(args.device)
            comparision = torch.cat([imgs[:8], recon_img[:8], imgs[8:n], fillers, recon_img[8:n], fillers])
            save_image(comparision.cpu(), save_imgs_dir, nrow=8)
        
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
        img_grad = make_grid(comparision, nrow=8)
        writer.add_image("Test/Reconstruction", img_grad, epoch)
        util.logging.info(f"Epoch {epoch} Test loss | Loss = {stats.tst_losses[-1]:.3f}")
        # Log the control points plot
        if args.inference_dist == 'Dirichlet':
            # reshape for Dir
            latent = latent.chunk(2, -1)[0].view(*latent.shape[:-3], 
                            args.strokes_per_img, args.points_per_stroke, -1)
        util.add_control_points_plot(generative_model, latent, writer, tag="Test/Control Points", epoch=epoch)
