import argparse
from einops import rearrange
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import offsetbox
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
import seaborn as sns
import torch
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from skimage.draw import line_aa, line
from kornia.geometry.transform import invert_affine_transform

import util
from splinesketch.code.bezier import Bezier


# [bs, 3 channels, pixel]
pres_clr = torch.tensor((0,.5,.5), dtype=torch.float).view(1, 3, 1) # gold
no_pres_clr = torch.tensor((.5,0,0.5), dtype=torch.float).view(1, 3, 1) # orange red
nrow = 16 # number of imgs/row

resize_res = 64
display_transform = transforms.Compose([
                        transforms.Resize([resize_res, resize_res]),
                        ])
def plot_stroke_mll_swarm_plot(dataframe, args, writer, epoch):
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    sns.swarmplot(x='Num_strokes', y='ELBO', hue='Label', dodge=True, size=1, 
                    data=dataframe,ax=ax).set_title(
                    'Number of strokes vs. ELBO on images of 1s and 7s')
    plt.tight_layout()
    writer.add_figure("Stroke count plot", fig, epoch)
    save_imgs_dir = util.get_save_count_swarm_img_dir(args, epoch, suffix='')
    plt.savefig(save_imgs_dir)

def plot_stroke_mll_swarm_plot(dataframe, args, writer, epoch):
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    sns.swarmplot(x='Num_strokes', y='ELBO', hue='Label', dodge=True, size=1, 
                    data=dataframe,ax=ax).set_title(
                    'Number of strokes vs. ELBO on images of 1s and 3s')
    plt.tight_layout()
    writer.add_figure("Stroke count plot", fig, epoch)
    save_imgs_dir = util.get_save_count_swarm_img_dir(args, epoch, suffix='')
    plt.savefig(save_imgs_dir)

def plot_reconstructions(imgs:torch.Tensor, 
                         guide:torch.nn.Module, 
                         generative_model:torch.nn.Module, 
                         args:argparse.ArgumentParser, 
                         writer:SummaryWriter, 
                         iteration:int=None,
                         epoch:int=None, 
                         writer_tag:str='Train',
                         dataset_name:str=None,
                         max_display=16):
    '''Plot 
    1) reconstructions in the format: target -> reconstruction
    2) control points
    Args:
        tag:str: in the format of, e.g. Train or Test
    '''
    bs = imgs.shape[0]
    n_strks = args.strokes_per_img

    n = min(bs, max_display)
    imgs = imgs[:n]
    res = imgs.shape[-1]

    if args.model_type == 'Base':
        if args.inference_net_architecture == "STN":
            latent, stn_out = guide.rsample(imgs, stn_out=True)
        else:
            latent = guide.rsample(imgs)
        generative_model.stn_transform = guide.stn_transform
        recon_img = generative_model.img_dist_b(latent).mean
        pts = latent

        if args.inference_net_architecture == 'STN':
            comparision = torch.cat([imgs[:8], stn_out[:8], recon_img[:8] ,
                                    imgs[8:n], fillers, stn_out[8:n], fillers, 
                                    recon_img[8:n], fillers])        
        else:
            comparision = torch.cat([imgs[:8], recon_img[:8], imgs[8:n], 
                                        fillers, recon_img[8:n], fillers])
                            

    elif args.model_type == 'Sequential' or args.model_type == 'AIR':

        # Get the info for make plots
        guide_out = guide(imgs, num_particles=1)
        latent = guide_out.z_smpl
        z_where_dim = latent.z_where.shape[-1]

        # Invert z_where for adding bounding boxes
        ptcs, bs, n_strks = latent.z_where.shape[:3]
        z_where_inv = util.invert_z_where(
            latent.z_where.view(n*n_strks, z_where_dim) # 3 for z_where type
        ).view(n, n_strks, z_where_dim)
        z_where_mtrx = util.get_affine_matrix_from_param(
                        latent.z_where.view(n*n_strks, -1), 
                        z_where_type=args.z_where_type)
                        
        # Reconstruction images; z_what: [bs, n_strks, n_pts, 2]
        if args.model_type == 'Sequential' and \
           generative_model.input_dependent_param:
            generative_model.sigma = guide_out.decoder_param.sigma[0]
            generative_model.single_strk_tanh_slope = \
                                    guide_out.decoder_param.slope[0]
            generative_model.add_strk_tanh_slope = \
                                    guide_out.decoder_param.slope[1][:, :, -1]
        recon_glimpse = generative_model.renders_glimpses(latent.z_what)
        recon_glimpse = recon_glimpse.squeeze(0)
        if args.execution_guided:
            if guide.intr_ll is not None:
                recon_img = guide_out.canvas[0, :, -1].expand(n,3,res,res)
            else:
                recon_img = guide_out.canvas[0].expand(n,3,res,res)
        else:
            recon_img = generative_model.renders_imgs(latent)[0].expand(n,3,res,res)

        print(f"epoch {epoch}")
        print(np.array(guide_out.z_pms.z_pres.detach().cpu()).round(3))
        # if args.z_where_type == '4_rotate':
        #     print(np.array(latent.z_where[:n,:,3].detach().cpu()).round(2))
        # Add bounding boxes based on z_where
        imgs_w_box = batch_add_bounding_boxes_skimage(
                                imgs=display_transform(imgs).cpu(), 
                                z_where_mtrx=z_where_mtrx.cpu().clone(), 
                                n_objs=n_strks*torch.ones(imgs.shape[0]),
                                n_strks=n_strks,
                        ).to(args.device)
        # imgs_w_box = batch_add_bounding_boxes(
        #                         imgs=transform(imgs), 
        #                         z_wheres=z_where_inv, 
        #                         n_obj=n_strks*torch.ones(imgs.shape[0]).cuda(),
        #                         z_where_type=args.z_where_type,
        #                 ).to(args.device)

        # Get the transformed imgs for each z_where

        stn_out = util.spatial_transform(imgs.repeat_interleave(repeats=n_strks,
                              dim=0),z_where_mtrx).view(n, n_strks, 1, res, res)

        # Take only n image for display, and make it has RGB channels
        # Then view it in shape [n * n_strkes, 3, res, res]
        stn_out = stn_out.expand(n,n_strks,3,res,res).view(n*n_strks,3,res,res)

        # Add color to indicate whether its kept
        # z_pres: originally in [n, n_strkes] -> [n * n_strkes]
        z_pres = latent.z_pres.view(n * n_strks).bool()
        stn_out = color_img_edge(imgs=stn_out, z_pres=z_pres, color=pres_clr)
        stn_out = color_img_edge(imgs=stn_out, z_pres=~z_pres, color=no_pres_clr)

        # stn_out: [n, 3, n_strks * res, res]
        stn_out = display_transform(stn_out)
        stn_out = stn_out.view([n,n_strks,3,resize_res,resize_res])
        stn_out = stn_out.transpose(1,2).reshape([n,3,n_strks*resize_res,
                                                                    resize_res])

        # resize recon_glimpse: 
        # [n, n_strks, 1, res, res] -> [n, 3, n_strks * resize_res, resize_res]
        recon_glimpse =  recon_glimpse.expand([n, n_strks, 3, res, res])
        recon_glimpse = display_transform(recon_glimpse.view([n*n_strks, 3, res, res]))
        recon_glimpse = recon_glimpse.view([n*n_strks,3,resize_res,resize_res])
        recon_glimpse = recon_glimpse.view([n, n_strks, 3, resize_res, 
                        resize_res]).transpose(1,2)
        recon_glimpse = recon_glimpse.reshape([n,3,n_strks*resize_res,
                                                                    resize_res])

        recon_img = display_transform(recon_img)
        # pre canvas for debugging execution guided
        if args.execution_guided:
            if args.exec_guid_type == 'canvas':
                if guide.intr_ll is not None:
                    cum_canvas = display_transform(guide_out.canvas[0, :, -1
                                                    ].expand(n, 3, res,res))
                else:
                    cum_canvas = display_transform(guide_out.canvas[0].expand(
                                                            n, 3, res, res))
            elif args.exec_guid_type == 'residual':
                # just reuse the name cum_canvas; this is actually just the
                # final residual
                cum_canvas = display_transform(guide_out.residual[0].expand(n, 3, res, res))
            else: raise NotImplementedError
        else:
            cum_canvas = torch.zeros_like(imgs_w_box)
        # comparision: [n, 3, res * (1_for_target + n_strks_stn + 1_for_recon), res]
        comparision = torch.cat([imgs_w_box, 
                                 recon_img, 
                                 stn_out,
                                 recon_glimpse,
                                 cum_canvas], dim=2)
        # comparision = torch.cat([imgs_w_box, recon_img, stn_out], dim=2)
        assert comparision.shape == torch.Size([n, 3, 
                                    resize_res * (1+n_strks*2+1+1), resize_res])

    elif args.model_type == 'VAE':
        latent = guide(imgs)
        recon_img = generative_model.img_dist(latent.z_smpl).mean[0]
        comparision = torch.cat([display_transform(imgs), 
                                 display_transform(recon_img)], dim=2)
    elif args.model_type == 'MWS':
        latent_dist = guide.get_latent_dist(imgs.squeeze(1).round())
        latent = guide.sample_from_latent_dist(latent_dist, 1)
        num_particles, batch_size, num_arcs, _ = latent.shape
        recon_dist = generative_model.get_obs_dist(latent[0])

        # Get sample
        mixture_ids = torch.randint(recon_dist.num_thetas, torch.Size())
        probs = recon_dist.cond_transformed[mixture_ids.view(-1)].view(
            *torch.Size(), batch_size, args.img_res, args.img_res
        )
        recon_img = torch.distributions.Bernoulli(probs=probs)._param
        
        imgs.unsqueeze_(1)
        recon_img.unsqueeze_(1)
        comparision = torch.cat([display_transform(imgs), 
                                 display_transform(recon_img)], dim=2)

    # Save image in a dir
    # suffix = 'trn' if is_train else 'tst'
    if writer_tag[-1] == '/': writer_tag = writer_tag[:-1]
    writer_tag = writer_tag + '_' + dataset_name
    save_imgs_dir = util.get_save_test_img_dir(args, epoch, suffix=writer_tag)
    try:
        save_image(comparision.cpu(), save_imgs_dir, nrow=nrow)
    except Exception as e:
        print(e)
        breakpoint()

    # Log image in Tensorboard
    img_grid = make_grid(comparision, nrow=nrow)
    tag = writer_tag
    if dataset_name is not None:
        tag = f'{dataset_name}/Reconstruction/{writer_tag}/'
    else:
        tag = f'Reconstruction/{writer_tag}'
    writer.add_image(tag, img_grid, epoch)

    # Draw control points
    if args.inference_dist == 'Dirichlet':
        # reshape for Dir
        latent = latent.chunk(2, -1)[0].view(*latent.shape[:-3], 
                        args.strokes_per_img, args.points_per_stroke, -1)

    if args.model_type == "Sequential":
            # todo 2: add transform to z_what for plotting
        # Get affine matrix: [bs * n_strk, 2, 3]
        pts_per_strk = latent.z_what.shape[3]
        z_what = latent.z_what.view(n * n_strks, pts_per_strk, 2)
        z_where = latent.z_where.view(n * n_strks, -1)
        pts = transformed_z_what = util.transform_z_what(
                                    z_what=z_what, 
                                    z_where=z_where,
                                    z_where_type=args.z_where_type,
                                    res=res).view(n, n_strks, pts_per_strk, 2)
    # removed to test spline z_what with neural decoder
    # if args.model_type in ['Sequential', 'Base']:
    #     add_control_points_plot(gen=generative_model, 
    #                             latents=pts, 
    #                             writer=writer, 
    #                             epoch=epoch,
    #                             writer_tag=writer_tag, 
    #                             dataset_name=dataset_name,
    #                             )
def color_img_edge(imgs, z_pres, color):
    '''
    Args:
        imgs [bs, 3, res, res]
        z_pres [bs,]
        color [1, 3, 1]
    return:
        colored_imgs [bs, 3, res, res]
    '''
    res = imgs.shape[-1]
    color = color.to(imgs.device)
    new_imgs = imgs.clone()
    new_imgs[:, :, 0:res, 0][z_pres] = color
    new_imgs[:, :, 0:res, -1][z_pres] = color
    new_imgs[:, :, 0, 0:res][z_pres] = color
    new_imgs[:, :, -1, 0:res][z_pres] = color
    return new_imgs

def add_control_points_plot(gen, latents, writer, epoch=None,
                            writer_tag:bool=True, dataset_name=None):
    num_shown = 16
    n = min(num_shown, latents.shape[0])
    num_steps_per_strk = 500
    num_strks_per_img, num_pts_per_strk = latents.shape[1], latents.shape[2]
    total_pts_num = num_strks_per_img * num_pts_per_strk

    steps = torch.linspace(0, 1, num_steps_per_strk).cuda()
    latents = latents[:num_shown]
    curves = gen.decoder.sample_curve(latents[:num_shown], steps)

    fig, ax= plt.subplots(2,8,figsize=(18, 4))

    # [num_shown, num_strks, num_pts, 2]-> [num_shown, total_pts_num, 2]
    latents = latents.reshape([n, total_pts_num, 2]).cpu()
    # [num_shown, num_strks, 2, num_steps]
    curves = curves.cpu().flip(1)
    curves = rearrange(curves, 'b k xy p -> b (k p) xy')
    # transpose(2, 3).reshape(num_shown, num_strks_per_img*
    #                                                 num_steps_per_strk, 2).cpu()
    for i in range(n):
        ax[i//8][i%8].set_aspect('equal')
        ax[i//8][i%8].axis('equal')
        ax[i//8][i%8].invert_yaxis()
        im = ax[i//8][i%8].scatter(x=latents[i][:,0],
                                   y=latents[i][:,1],
                                    marker='x',
                                    s=30,c=np.arange(total_pts_num),
                                            cmap='rainbow')
        ax[i//8][i%8].scatter(x=curves[i][:,0], y=curves[i][:,1],
                                                s=0.1,
                                                c=-np.arange(num_strks_per_img*
                                                    num_steps_per_strk),
                                                cmap='rainbow')       
        line = mlines.Line2D([0, 0], [1, 0], color='black', alpha=.5)
        ax[i//8][i%8].add_line(line)
        line = mlines.Line2D([0, 1], [1, 1], color='black', alpha=.5)
        ax[i//8][i%8].add_line(line)
        line = mlines.Line2D([0, 1], [0, 0], color='black', alpha=.5)
        ax[i//8][i%8].add_line(line)
        line = mlines.Line2D([1, 1], [0, 1], color='black', alpha=.5)
        ax[i//8][i%8].add_line(line)
    fig.subplots_adjust(right=0.94)
    cbar_ax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
    fig.colorbar(im, cax=cbar_ax)

    # Add as figure
    tag = writer_tag
    if dataset_name is not None:
        tag = f'{dataset_name}/Control Points/{tag}'
    else:
        tag = f'Constrol Points/{writer_tag}'
    writer.add_figure(tag, fig, epoch)
    # Adding as image
    # canvas = FigureCanvas(fig)
    # canvas.draw()       # draw the canvas, cache the renderer

    # image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    # width, height = fig.get_size_inches() * fig.get_dpi() 
    # image = torch.tensor(image.reshape(int(height), int(width), 3)).float()
    # image = rearrange(image, 'h w c -> c h w')

    # writer.add_image(tag, image, epoch)
    plt.close('all')

def plot_stroke_tsne(ckpt_path:str, title:str, save_dir:str='plots/', 
                                                z_what_to_keep=5000):
    '''
    Args:
        ckpt_path: path to checkpoint
        title: title for the plot and also saving
        save_dir: dir to save the figure to
        z_what_to_keep: we stop after generating more than this number. If none
            use all the dataset
    '''
    from sklearn.manifold import TSNE

    # Load checkpoint ----------------------------------------------------------
    util.logging.info("Loading the checkpoint...")
    device = util.get_device()
    model, _, _, data_loader, run_args = util.load_checkpoint(ckpt_path, device)
    _, guide = model
    data_loader, _ = data_loader

    # Get z_what and renders them to curves ------------------------------------
    util.logging.info("Generating z_what and curves...")
    all_latents = []
    all_curves = []
    num_steps_per_strk = 500
    bezier = Bezier(res=28, steps=num_steps_per_strk, method='bounded')
    bs = data_loader.batch_size
    n_strks = guide.max_strks
    pts_per_strk = guide.pts_per_strk

    with torch.no_grad():
        keep_num = 0
        for imgs, _ in data_loader:
            # get latents
            imgs = imgs.to(device)
            guide_out = guide(imgs)
            bs = imgs.shape[0]
            z_pres, z_what, z_where = guide_out.z_smpl
            
            # Create curves
            steps = torch.linspace(0, 1, bezier.steps).to(z_what.device)
            curves = rearrange(bezier.sample_curve(z_what,steps), 'b strk xy pts -> (b strk) pts xy')
            
            # Keep z_what with z_pres==1
            z_pres = z_pres.flatten().bool()
            z_what = z_what.view(bs*n_strks, pts_per_strk* 2)[z_pres]
            curves = curves[z_pres]

            # Store them
            all_latents.append(z_what.cpu().numpy())
            all_curves.append(curves.cpu().numpy())
            keep_num += z_pres.sum().item()
            if z_what_to_keep:
                if keep_num > z_what_to_keep: break

    # Concat
    all_z_whats = np.concatenate(all_latents)
    all_curves = np.concatenate(all_curves)
    assert keep_num == all_z_whats.shape[0] == all_curves.shape[0]
    print("z_what shape:", all_z_whats.shape, "curves shape:", all_curves.shape)

    # Generate visualization plot ----------------------------------------------
    # t-sne z_whats
    util.logging.info("Generating t-sne embeddings...")
    all_z_whats_embedded = TSNE(n_components=2).fit_transform(all_z_whats)

    # plot the curves
    util.logging.info("Generating curve images...")
    images = []
    for curve in all_curves:
        fig, ax = plt.subplots(1,1,figsize=(.5,.5))
        i = 19
        ax.scatter(curve[:,0],curve[:,1], s=0.1,)
        ax.set_aspect('equal')
        ax.axis('equal')
        ax.invert_yaxis()
        plt.xticks([]), plt.yticks([])

        # draw as image
        canvas = FigureCanvas(fig)
        canvas.draw()       # draw the canvas, cache the renderer

        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        width, height = fig.get_size_inches() * fig.get_dpi() 
        image = torch.tensor(image.reshape(int(height), int(width), 3)).float()
        
        images.append(image)
        plt.close()
    
    images_arr = np.stack(images)
    images_arr = images_arr.astype('int')

    util.logging.info("Plotting visualization...")
    plot_embeddings(all_z_whats_embedded, imgs=images_arr, title=title, 
                                                        save_dir=save_dir)

def plot_embeddings(X, imgs, title=None, save_dir=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    _, axs = plt.subplots(1,2,figsize=(20,10))
    
    shown_images = np.array([[1. ,1.]])
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        imagebox =  offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(imgs[i]),
                X[i], frameon=False)
        axs[0].add_artist(imagebox)
    axs[0].scatter(X[:,0],X[:,1])
    axs[0].axes.xaxis.set_visible(False)
    axs[0].axes.yaxis.set_visible(False)
    plt.xticks([]), plt.yticks([])
    axs[1].scatter(X[:,0],X[:,1])
    
    plt.tight_layout()
        
    if title:
        plt.suptitle(title, y=1.01,fontsize=16)
        plt.savefig(save_dir + title + ".pdf")

# Plotting
def batch_add_bounding_boxes_skimage(imgs, z_where_mtrx, n_objs, n_strks):
    '''
    Args:
        imgs: [bs, 1 (c), res, res]
        z_where_mtrx: [bs*n_strks, 2, 3]
        n_obj: [bs]
        color: [some, 3]
        n_strks: int
    '''
    colors = torch.tensor([[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.],
                           [1., 1., 0.],
                           [0., 1., 1.]]).view(5,3,1)
    bs, res = imgs.shape[0], imgs.shape[-1]
    imgs = imgs.expand(bs, 3, res, res)
    new_img = imgs.clone()
    # Get end point of glimpse
    ends = torch.tensor([[.99,.99,.99,-.99], 
                         [.99,.99,-.99,.99], 
                         [-.99,.99,-.99,-.99], 
                         [.99,-.99,-.99,-.99]], dtype=torch.float, 
                                                device=imgs.device).view(8, 2)
    homo_coord = torch.cat([ends, torch.ones_like(ends)[:,0:1]], dim=1)
    # get 1 for each image, step
    # breakpoint()
    homo_coord = homo_coord.expand(bs*n_strks, 8, 3)
    homo_coord = (z_where_mtrx @ homo_coord.transpose(1,2)).transpose(1,2)
    homo_coord = homo_coord.reshape(bs, n_strks, 4, 4)
    # change from image canvas coord to array coord
    homo_coord = ((homo_coord + torch.ones_like(homo_coord))/2) * res
    for i, (img, bbox_coords, n_obj) in enumerate(zip(imgs, homo_coord, n_objs)):
        new_img[i] = add_bboxes_sk(img, bbox_coords, int(n_obj.item()), colors)
    return new_img

def add_bboxes_sk(img, bbox_coords, n_obj, colors):
    '''
    Args:
        img: [1, res, res]
        bbox_coords: [2, 4, 4]
        n_obj: int
        colors: [n, 3]
    return:
        img with boundings box add to it
    '''
    # breakpoint()
    for i in range(int(n_obj)):
        img = add_bbox_sk(img, bbox_coords[i], colors[i])
    return img

def add_bbox_sk(img, bbox_coords, color):
    '''
    Args:
        img: [1, res, res]
        bbox_coords: [4, 4]
        color: [3]
    Return:
        img with bounding box
    '''
    res = img.shape[-1] - 1
    new_img = img.clone()
    
    for i in range(4):
        rr, cc = line(*(bbox_coords[i,:].cpu().numpy()).astype(int))
        keep_idx = ((rr >= 0) & (rr <= res)) & ((cc >= 0) & (cc <=res))
        rr, cc = rr[keep_idx], cc[keep_idx]
        # breakpoint()
        new_img[:, cc, rr] = color
    return new_img

def batch_add_bounding_boxes(imgs, z_wheres, n_obj, color=None, n_img=None, 
                                z_where_type=None):
    """
    :param imgs: 4d tensor of numpy array, channel dim either 1 or 3
    :param z_wheres: tensor or numpy of shape (n_imgs, max_n_objects, 3)
    :param n_obj:
    :param color:
    :param n_img:
    :return:
    """
    if color is None:
        color = np.array([[1., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., 1.]])

    # Check arguments
    assert len(imgs.shape) == 4
    assert imgs.shape[1] in [1, 3]
    assert len(z_wheres.shape) == 3
    assert z_wheres.shape[0] == imgs.shape[0]
    if z_wheres.shape[2] == 4:
        batch_add_bounding_boxes_skimage(imgs, z_wheres, n_obj, color,
                                                z_where_type=z_where_type)
    assert z_wheres.shape[2] == 3

    target_shape = list(imgs.shape)
    target_shape[1] = 3

    if n_img is None:
        n_img = len(imgs)

    out = torch.stack([
        add_bounding_boxes(imgs[j], z_wheres[j], color, n_obj[j])
        for j in range(n_img)
    ])

    out_shape = tuple(out.shape)
    target_shape = tuple(target_shape)
    assert out_shape == target_shape, "{}, {}".format(out_shape, target_shape)
    return out


def add_bounding_boxes(img, z_wheres, color, n_obj):
    """
    Adds bounding boxes to the n_obj objects in img, according to z_wheres.
    The output is never on cuda.
    :param img: image in 3d or 4d shape, either Tensor or numpy. If 4d, the
                first dimension must be 1. The channel dimension must be
                either 1 or 3.
    :param z_wheres: tensor or numpy of shape (1, max_n_objects, 3) or
                (max_n_objects, 3)
    :param color: color of all bounding boxes (RGB)
    :param n_obj: number of objects in the scene. This controls the number of
                bounding boxes to be drawn, and cannot be greater than the
                max number of objects supported by z_where (dim=1). Has to be
                a scalar or a single-element Tensor/array.
    :return: image with required bounding boxes, with same type and dimension
                as the original image input, except 3 color channels.
    """

    try:
        n_obj = n_obj.item()
    except AttributeError:
        pass
    n_obj = int(round(n_obj))
    assert n_obj <= z_wheres.shape[1]

    try:
        img = img.cpu()
    except AttributeError:
        pass

    if len(img.shape) == 3:
        color_dim = 0
    else:
        color_dim = 1

    if len(z_wheres.shape) == 3:
        assert z_wheres.shape[0] == 1
        z_wheres = z_wheres[0]

    target_shape = list(img.shape)
    target_shape[color_dim] = 3

    for i in range(n_obj):
        img = add_bounding_box(img, z_wheres[i:i+1], color[i])
    if img.shape[color_dim] == 1:  # this might happen if n_obj==0
        reps = [3, 1, 1]
        if color_dim == 1:
            reps = [1] + reps
        reps = tuple(reps)
        if isinstance(img, torch.Tensor):
            img = img.repeat(*reps)
        else:
            img = np.tile(img, reps)

    target_shape = tuple(target_shape)
    img_shape = tuple(img.shape)
    assert img_shape == target_shape, "{}, {}".format(img_shape, target_shape)
    return img


def add_bounding_box(img, z_where, color):
    """
    Adds a bounding box to img with parameters z_where and the given color.
    Makes a copy of the input image, which is left unaltered. The output is
    never on cuda.
    :param img: image in 3d or 4d shape, either Tensor or numpy. If 4d, the
                first dimension must be 1. The channel dimension must be
                either 1 or 3.
    :param z_where: tensor or numpy with 3 elements, and shape (1, ..., 1, 3)
    :param color:
    :return: image with required bounding box in the specified color, with same
                type and dimension as the original image input, except 3 color
                channels.
    """
    def _bounding_box(z_where, x_size, rounded=True, margin=1):
        z_where = z_where.cpu().numpy().flatten()
        assert z_where.shape[0] == z_where.size == 3
        s, x, y = tuple(z_where)
        w = x_size / s
        h = x_size / s
        xtrans = -x / s * x_size / 2
        ytrans = -y / s * x_size / 2
        x1 = (x_size - w) / 2 + xtrans - margin
        y1 = (x_size - h) / 2 + ytrans - margin
        x2 = x1 + w + 2 * margin
        y2 = y1 + h + 2 * margin
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        coords = (x1, x2, y1, y2)
        if rounded:
            coords = (int(round(t)) for t in coords)
        return coords

    target_shape = list(img.shape)
    collapse_first = False
    torch_tensor = isinstance(img, torch.Tensor)
    img = img.cpu().numpy().copy()
    if len(img.shape) == 3:
        collapse_first = True
        img = np.expand_dims(img, 0)
        target_shape[0] = 3
    else:
        target_shape[1] = 3
    assert len(img.shape) == 4 and img.shape[0] == 1
    if img.shape[1] == 1:
        img = np.tile(img, (1, 3, 1, 1))
    assert img.shape[1] == 3
    color = color[:, None]

    x1, x2, y1, y2 = _bounding_box(z_where, img.shape[2])
    x_max = y_max = img.shape[2] - 1
    if 0 <= y1 <= y_max:
        img[0, :, y1, max(x1, 0):min(x2, x_max)] = color
    if 0 <= y2 - 1 <= y_max:
        img[0, :, y2 - 1, max(x1, 0):min(x2, x_max)] = color
    if 0 <= x1 <= x_max:
        img[0, :, max(y1, 0):min(y2, y_max), x1] = color
    if 0 <= x2 - 1 <= x_max:
        img[0, :, max(y1, 0):min(y2, y_max), x2 - 1] = color

    if collapse_first:
        img = img[0]
    if torch_tensor:
        img = torch.from_numpy(img)

    target_shape = tuple(target_shape)
    img_shape = tuple(img.shape)
    assert img_shape == target_shape, "{}, {}".format(img_shape, target_shape)
    return img