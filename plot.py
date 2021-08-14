import argparse
from einops import rearrange
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

import util

def plot_reconstructions(imgs:torch.Tensor, 
                         guide:torch.nn.Module, 
                         generative_model:torch.nn.Module, 
                         args:argparse.ArgumentParse, 
                         writer:SummaryWriter, 
                         epoch:int, 
                         is_train:bool=True):
    '''Plot 
    1) reconstructions in the format: target -> reconstruction
    2) control points
    Args:
        tag:str: in the format of, e.g. Train or Test
    '''
    bs = imgs.shape[0]
    n = min(bs, 16)
    if args.model_type == 'base':
        if args.inference_net_architecture == "STN":
            latent, stn_out = guide.rsample(imgs, stn_out=True)
        else:
            latent = guide.rsample(imgs)
        generative_model.stn_transform = guide.stn_transform
        recon_img = generative_model.img_dist_b(latent).mean
    elif args.model_type == 'sequential':
        latent = guide(imgs).z_smpl
        # invert z_where for adding bounding boxes
        n_strks = latent.z_where.shape[1]
        z_where_inv = util.invert_z_where(
            latent.z_where.view(bs*n_strks, 3) # 3 for z_where type
        ).view(bs, n_strks, 3)
        recon_img = generative_model.img_dist(latent).mean[:n].expand(n,3,28,28)
        imgs_w_box = util.batch_add_bounding_boxes(
                                imgs=imgs, 
                                z_wheres=z_where_inv, 
                                n_obj=2*torch.ones(imgs.shape[0]).cuda()
                        ).to(args.device)
        stn_out = util.spatial_transform(imgs, 
            util.get_affine_matrix_from_param(latent.z_where[:,0], 
                                z_where_type='3'))[:n].expand(n,3,28,28)
        print(f"epoch {epoch}, z_pres", latent.z_pres[:n].squeeze().cpu())
    fillers = torch.zeros(16-n, 3, 28, 28).to(args.device)

    if args.model_type == 'base':
        if args.inference_net_architecture == 'STN':
            comparision = torch.cat([imgs[:8], stn_out[:8], recon_img[:8] ,
                                    imgs[8:n], fillers, stn_out[8:n], fillers, 
                                    recon_img[8:n], fillers])        
        else:
            comparision = torch.cat([imgs[:8], recon_img[:8], imgs[8:n], 
                                        fillers, recon_img[8:n], fillers])
    elif args.model_type == 'sequential':
            comparision = torch.cat([imgs_w_box[:8], stn_out[:8], recon_img[:8] ,
                                    imgs_w_box[8:n], fillers, stn_out[8:n], fillers, 
                                    recon_img[8:n], fillers]) 

    # Save image in a dir
    suffix = 'trn' if is_train else 'tst'
    save_imgs_dir = util.get_save_test_img_dir(args, epoch, suffix='tst')
    save_image(comparision.cpu(), save_imgs_dir, nrow=8)

    # Log image in Tensorboard
    img_grid = make_grid(comparision, nrow=8)
    # draw control points
    if args.inference_dist == 'Dirichlet':
        # reshape for Dir
        # breakpoint()
        latent = latent.chunk(2, -1)[0].view(*latent.shape[:-3], 
                        args.strokes_per_img, args.points_per_stroke, -1)
    writer.add_image("Train/Reconstruction", img_grid, epoch)

    if args.model_type == "sequential":
            # todo 2: add transform to z_what for plotting
            latent = latent.z_what

    tag = 'Train/Reconstruction' if is_train else 'Test/Reconstruction'
    add_control_points_plot(gen=generative_model, 
                            latents=latent, 
                            writer=writer, 
                            epoch=epoch,
                            is_train=is_train, 
                            )

def add_control_points_plot(gen, latents, writer, tag=None, epoch=None,
                            is_train:bool=True):
    num_shown = 8
    num_steps_per_strk = 500
    num_strks_per_img, num_pts_per_strk = latents.shape[1], latents.shape[2]
    total_pts_num = num_strks_per_img * num_pts_per_strk

    steps = torch.linspace(0, 1, num_steps_per_strk).cuda()
    latents = latents[:num_shown]
    curves = gen.bezier.sample_curve(latents[:num_shown], steps)

    fig, ax= plt.subplots(2,4,figsize=(10, 4))

    # [num_shown, num_strks, num_pts, 2]-> [num_shown, total_pts_num, 2]
    latents = latents.view([num_shown, total_pts_num, 2]).cpu()
    # [num_shown, num_strks, 2, num_steps]
    curves = curves.cpu().flip(1)
    curves = rearrange(curves, 'b k xy p -> b (k p) xy')
    # transpose(2, 3).reshape(num_shown, num_strks_per_img*
    #                                                 num_steps_per_strk, 2).cpu()
    for i in range(num_shown):
        ax[i//4][i%4].set_aspect('equal')
        ax[i//4][i%4].axis('equal')
        ax[i//4][i%4].invert_yaxis()
        im = ax[i//4][i%4].scatter(x=latents[i][:,0],
                                   y=latents[i][:,1],
                                    marker='x',
                                    s=30,c=np.arange(total_pts_num),
                                            cmap='rainbow')
        ax[i//4][i%4].scatter(x=curves[i][:,0], y=curves[i][:,1],
                                                s=0.1,
                                                c=-np.arange(num_strks_per_img*
                                                    num_steps_per_strk),
                                                cmap='rainbow')       
    fig.subplots_adjust(right=0.94)
    cbar_ax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
    fig.colorbar(im, cax=cbar_ax)

    canvas = FigureCanvas(fig)
    canvas.draw()       # draw the canvas, cache the renderer

    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    width, height = fig.get_size_inches() * fig.get_dpi() 
    image = torch.tensor(image.reshape(int(height), int(width), 3)).float()
    image = rearrange(image, 'h w c -> c h w')

    tag = "Train/Control Points" if is_train else "Test/Control Points"
    writer.add_image(tag, image, epoch)
    plt.close('all')