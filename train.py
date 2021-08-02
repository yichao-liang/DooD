import torch
from torchvision.utils import save_image

import util
import losses

def train(model, optimizer, stats, data_loader, args):
    checkpoint_path = util.get_checkpoint_path(args)
    num_iterations_so_far = len(stats.trn_losses)
    iteration = num_iterations_so_far

    generative_model, guide = model
    train_loader, test_loader = data_loader

    while iteration < args.num_iterations:
        for imgs, _ in train_loader:
            if args.dataset == "mnist":
                imgs = imgs.to(args.device)

            optimizer.zero_grad()

            if args.loss == 'elbo':
                loss = losses.get_elbo_loss(generative_model, guide, imgs).mean()
            else:
                raise NotImplementedError()

            loss.backward()

            # Check for nans
            for name, parameter in guide.named_parameters():
                if torch.isnan(parameter).any() or torch.isnan(parameter.grad).any():
                    breakpoint()
                    raise RuntimeError(f"nan in guide parameter {name}: {parameter}")

            optimizer.step()

            # Record stats
            stats.trn_losses.append(loss.item())
            
            # Log
            if iteration % args.log_interval == 0:
                util.logging.info(f"Iteration {iteration} | Loss = {stats.trn_losses[-1]:.0f}")

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
        
        # test every epoch
        save_imgs_dir = util.get_save_test_img_dir(args, iteration)
        test(model, stats, test_loader, args, save_imgs_dir)

def test(model, stats, test_loader, args, save_imgs_dir=None):
    '''
    Args:
        test_loader (DataLoader): testset dataloader
    '''
    generative_model, guide = model
    generative_model.eval(); guide.eval()

    with torch.no_grad():
        loss = 0
        for imgs, _ in test_loader:
            imgs = imgs.to(args.device)

            loss += losses.get_elbo_loss(generative_model, guide, imgs).sum()
        
        if save_imgs_dir is not None:
            n = min(imgs.shape[0], 8)
            latent = guide.rsample(imgs)
            recon_img = generative_model.img_dist_b(latent).mean
            comparision = torch.cat([imgs[:n],
                                    recon_img.view(-1, 1, 28, 28)[:n]])
            save_image(comparision.cpu(), save_imgs_dir, nrow=n)
            
        stats.tst_losses.append(loss/ len(test_loader.dataset))
        util.logging.info(f"Test loss | Loss = {stats.tst_losses[-1]:.0f}")
