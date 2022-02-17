import itertools
import random
import collections
import os
import sys
import subprocess
import getpass
import logging
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image
from einops import rearrange
from kornia.geometry.transform import invert_affine_transform, get_affine_matrix2d

from models import base, ssp, air, vae#, mws
# from data.omniglot_dataset.omniglot_dataset import TrainingDataset
from data import synthetic, multimnist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(pathname)s:%(lineno)d | %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

ZWhereParam = collections.namedtuple("ZWhereParam", "loc std dim")

def heat_weight(init_val, final_val, cur_ite, heat_step, init_ite=0):
    '''
    heat_step::int: ite where the heat stop
    init_ite::int: ite where the heat start
    '''
    ite = max((cur_ite - init_ite), 0)
    val = final_val + (init_val - final_val) * (1 - ite/(heat_step-init_ite))
    heat_weight = min(final_val, val)
    return heat_weight

def anneal_weight(init_val, final_val, cur_ite, anneal_step, init_ite=0):
    '''
    anneal_step::int: ite where the anneal stop
    init_ite::int: ite where the anneal start
    '''
    ite = max((cur_ite - init_ite), 0)
    val = final_val + (init_val - final_val) * (1 - ite/anneal_step)
    anneal_weight = max(final_val, val)
    return anneal_weight

def sampling_gauss_noise(base_tensor, var):
    out = base_tensor + torch.empty_like(base_tensor).normal_(
                                                        mean=0, 
                                                        std=torch.sqrt(var))
    return out

def character_conditioned_sampling(n_samples, guide_out, decoder):
    '''Take the guide_out and decoder, return the samples with some variations
    Args:
        n_samples::int
        guide_out::GuideReturn
        decoder::GenerativeModel
        var::float: Variance value for the Gaussian noise
    Return:
        sampled_img [n_samples, bs, 1, res, res]
    '''
    out = guide_out
    _, bs, n_strks = out.z_smpl.z_pres.shape
    var = torch.tensor(1e-3)
    render_var = torch.tensor(1e-6)

    canvas = torch.zeros([n_samples, bs, 1, decoder.res, decoder.res], 
                                    device=next(decoder.parameters()).device)
    for t in range(n_strks):
        # z_pres_p [n_smpls, bs, 1]
        # don't sample z_pres
        z_pres_p = out.z_smpl.z_pres[:, :, t: t+1].expand(n_samples, bs, 1)

        # z_where_loc [n_smpls, bs, z_where_dim]
        z_where_loc = out.z_smpl.z_where[:, :, t, :].expand(n_samples, bs, 
                                                            decoder.z_where_dim)
        z_where_loc = sampling_gauss_noise(z_where_loc, var)
        # todo: make sure the params are still legal

        # sigma, strk_slope, add_slope
        sigma = out.decoder_param.sigma[:, :, t].expand(n_samples, bs)
        sigma = sampling_gauss_noise(sigma, render_var) # todo: smaller noise
        strk_slope = out.decoder_param.slope[0][:, :, t].expand(n_samples, bs)
        strk_slope = sampling_gauss_noise(strk_slope, render_var)

        decoder.sigma = sigma
        decoder.tanh_norm_slope_stroke = strk_slope

        # z_what_loc [n_smpls, bs, pts, 2]
        z_what_loc = out.z_smpl.z_what[:, :, t, :, :].expand(n_samples, bs,
                                                        decoder.pts_per_strk, 2)
        z_what_loc = sampling_gauss_noise(z_what_loc, var)
        
        # render step
        # canvas_step [n_smples, bs, 1, res, res]
        canvas_step = decoder.renders_imgs((z_pres_p,
                                            z_what_loc.unsqueeze(2),
                                            z_where_loc.unsqueeze(2)))
        # update the canvas
        img_shape = canvas_step.shape[1:]
        canvas_step = canvas_step.view(n_samples, bs, *img_shape)
        canvas = canvas + canvas_step
        add_slope = out.decoder_param.slope[1][:, :, t].expand(n_samples, bs)
        add_slope = sampling_gauss_noise(add_slope, render_var)
        canvas = normalize_pixel_values(canvas, method='tanh', 
                                                     slope=add_slope)
    
    # return canvas
    return canvas


def geom_weights_from_z_pres(z_pres, p):
    '''
    Args:
        z_pres [bs, n_steps] (Assuming the ptcs dim has been reshaped to bs)
    '''
    bs, max_steps = z_pres.shape
    num_steps = z_pres.sum(1)
    # weights = torch.distributions.Geometric(p).log_prob(
    #                                     torch.arange(max_steps).to(p.device))
    # weights = torch.exp(weights)
    weights = ((1-p) ** (torch.arange(max_steps)).to(p.device)) * p
    weights = weights.expand(z_pres.shape).clone()

    for i, n_stps in enumerate(num_steps):
        if n_stps > 0:
            cnt = n_stps.int()-1
            weights[i, cnt] = 1 - weights[i][:cnt].sum()
    # idx = (num_steps != 0)
    # torch.gather(weights[idx], -1, num_steps[idx].type(torch.int64).unsqueeze(1))
    # torch.index_select(weights[idx], 1, num_steps[idx].type(torch.int)-1)
    # weights[idx, num_steps[idx].type(torch.int)-1] = weights[num_steps-1:].sum(1)
    weights = weights * z_pres
    return weights

def debug_gradient(name, param, imgs, guide, generative_model, optimizer, 
                                                    iteration, writer, args):
    import plot, losses
    # print the gradient
    logging.info(f"{name} has grad norm: {param.grad.norm(2)}")

    # plot reconstruction
    with torch.no_grad():
        plot.plot_reconstructions(imgs=imgs, 
                                guide=guide, 
                                generative_model=generative_model, 
                                args=args, 
                                writer=writer, 
                                iteration=iteration,
                                writer_tag='Debug',
                                max_display=64
                            )
    
    # go through single images:
    for idx, img in enumerate(imgs):
        img = img.unsqueeze(0)
        optimizer.zero_grad()
        loss_tuple = losses.get_loss_sequential(
                                            generative_model=generative_model, 
                                            guide=guide,
                                            imgs=img, 
                                            loss_type=args.loss, 
                                            k=1,
                                            iteration=iteration,
                                            writer=writer,
                                            writer_tag=f'Debug-img{idx}/')
        loss = loss_tuple.overall_loss.mean()
        loss.backward()
        for n, p in guide.named_parameters():
            # check
            if (n == name and p.grad.norm(2) > 1e4):
                print(f'img {idx}: {n} has grad {p.grad.norm(2)}')
                for param_n, param in guide.named_parameters():
                    writer.add_scalar(f"Grad_norm_debug/img{idx}/{param_n}", 
                                                param.grad.norm(2), iteration)
                for loss_n, l in zip(loss_tuple._fields, loss_tuple):
                    writer.add_scalar(f"Loss_debug/img{idx}/{loss_n}/", 
                                                l.detach().mean(), iteration)

                with torch.no_grad():
                    plot.plot_reconstructions(imgs=img, 
                                guide=guide, 
                                generative_model=generative_model, 
                                args=args, 
                                writer=writer, 
                                writer_tag=f'Debug-img{idx}/',
                                max_display=64,
                            )
                breakpoint()
        va = 1+1

def transform_z_what(z_what, z_where, z_where_type, res):
    '''Apply z_where_mtrx to z_what such that it has a similar result as
    applying the inverse z_where_mtrx to the z_what-rendering.
    Args:
        z_what [bs, n_pts, 2]
        z_where_mtrx [bs, z_where_dim]
        res (int): resolution for target image; used in affine_grid
    Return:
        transformed_z_what [bs, n_pts, 2]
    '''
    bs, n_pts, _ = z_what.shape
    # Get the center of the inverse affine grid
    z_where_mtrx = get_affine_matrix_from_param(thetas=z_where, 
                                                z_where_type=z_where_type, 
                                                center=None)
    inv_z_where_mtrx = invert_affine_transform(z_where_mtrx)
    ag = F.affine_grid(inv_z_where_mtrx, [bs, 1, res, res], align_corners=True)

    l_coord, r_coord = int(np.ceil((res/2) - 1)),  int(np.floor(res/2))
    center = ((ag[:, l_coord, l_coord].cuda() + 
               ag[:, r_coord, r_coord].cuda()) / 2) + 0.5
    
    # Get the adjusted z_where_mtrx, apply to z_what
    adj_z_where_mtrx = get_affine_matrix_from_param(thetas=z_where, 
                                                z_where_type=z_where_type, 
                                                center=center)
    homo_coord = torch.cat([z_what, torch.ones(bs, n_pts, 1).to(z_what.device)], 
                                                                        dim=2)
    transformed_z_what = ((adj_z_where_mtrx @ homo_coord.transpose(1,2)
                                                            ).transpose(1,2))
    return transformed_z_what


def incremental_average(m_prev, a, n):
    '''
    Args:
        m_pres: mean of n-1 elements
        a: new elements
        n (int): n-th element
    '''
    return m_prev + ((a - m_prev)/n)

def init_z_where(z_where_type):
    '''
    '3': (scale, shift x, y)
    '4_no_rotate': (scale x, y, shift x, y)
    '4_rotate': (scale, shift x, y, rotate)
    '5': (scale x, y, shift x, y, rotate
    '''
    init_z_where_params = {'3': ZWhereParam(torch.tensor([.8,0,0]), 
                                                torch.tensor([.2,.2,.2]), 3), # AIR?
                                                # torch.tensor([.2,1,1]), 3), # spline?
                                                # torch.ones(3)/5, 3),
                           '4_no_rotate': ZWhereParam(torch.tensor([.3,1,0,0]),
                                                torch.ones(4)/5, 4),
                           '4_rotate': ZWhereParam(
                                                torch.tensor([.8,0,0,0]),
                                                torch.tensor([0.2,1,1,1]), 4),
                                                # torch.ones(4)/5, 4),
                           '5': ZWhereParam(torch.tensor([1,1,0,0,0]),
                                                torch.ones(5)/5, 5),
                        }
    assert z_where_type in init_z_where_params
    return init_z_where_params.get(z_where_type)
    
def safe_div(dividend, divisor):
    '''divident / divisor, only do it for divisor larger than 1e-4
    '''
    # idx = divisor = 0
    idx = divisor >= 1e-4
    idx = idx.squeeze()
    # divisor[idx] = divisor[idx].clamp(min=1e-6)
    dividend[idx] =  dividend[idx] / divisor[idx] 
    return dividend

def sigmoid(x, min=0., max=1., shift=0., slope=1.):
    return (
        (max-min)/(1+torch.exp(-slope * (-shift + x))
        ) + min)

def constrain_parameter(param, min=5e-2, max=1e-2):
    return torch.sigmoid(param) * (max-min) + min

def normalize_pixel_values(img, method='tanh', slope=0.6):
    '''non inplace operation
    Args:
        img:
            for 'tanh' 1: [bs, 1, res, res] or 2: [bs, n_strk, 1, res, res]
            for 'maxnorm': [bs, 1, res, res]
        slope int or Tensor: if 1: [bs]; if 2: [bs, n_strk]
    '''
    if method == 'tanh':
        try:
            if type(slope) == float:
                img_ = torch.tanh(img/slope)
            elif len(slope.shape) > 0 and slope.shape[0] == img.shape[0]:
                assert (len(img.shape) - len(slope.shape) == 3) 
                slope = slope.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # img_ = torch.tanh(img/slope)
                # if execution guided
                img_ = torch.tanh(img/(slope.clone()))
                return img_
            else: 
                breakpoint()
        except:
            breakpoint()
    elif method == 'maxnorm':
        batch_dim = img.shape[0]
        max_per_recon = img.detach().clone().reshape(batch_dim, -1).max(1)[0]
        max_per_recon = max_per_recon.reshape(batch_dim, 1, 1, 1) #/ maxnorm_max
        img = safe_div(img, max_per_recon)
        return img
    else:
        raise NotImplementedError
    return img_


def get_baseline_save_dir():
    return "save/baseline"


def get_baseline_posterior_path():
    return f"{get_baseline_save_dir()}/posterior.pt"


def get_path_base_from_args(args):
    if args.save_model_name is not None:
        return f"{args.save_model_name}"
    else:
        return f"{args.model_type}"


def get_save_job_name_from_args(args):
    return get_path_base_from_args(args)


def get_save_dir_from_path_base(path_base):
    return f"save/{path_base}"


def get_save_dir(args):
    return get_save_dir_from_path_base(get_path_base_from_args(args))

def get_save_test_img_dir(args, iteration, suffix='tst'):
    Path(f"{get_save_dir(args)}/images").mkdir(parents=True, exist_ok=True)
    return f"{get_save_dir(args)}/images/reconstruction_ep{iteration}_{suffix}.pdf"

def get_save_count_swarm_img_dir(args, iteration, suffix='tst'):
    Path(f"{get_save_dir(args)}/images").mkdir(parents=True, exist_ok=True)
    return f"{get_save_dir(args)}/images/count_swarm_ep{iteration}_{suffix}.pdf"
def get_checkpoint_path(args, checkpoint_iteration=-1):
    '''e.g. get_path_base_from_args: "base"
    '''
    return get_checkpoint_path_from_path_base(get_path_base_from_args(args), 
                                              checkpoint_iteration)


def get_checkpoint_path_from_path_base(path_base, checkpoint_iteration=-1):
    checkpoints_dir = f"{get_save_dir_from_path_base(path_base)}/checkpoints"
    if checkpoint_iteration == -1:
        return f"{checkpoints_dir}/latest.pt"
    else:
        return f"{checkpoints_dir}/{checkpoint_iteration}.pt"


def get_checkpoint_paths(checkpoint_iteration=-1):
    save_dir = "./save/"
    for path_base in sorted(os.listdir(save_dir)):
        yield get_checkpoint_path_from_path_base(path_base, checkpoint_iteration)

transform = transforms.Compose([
                       transforms.GaussianBlur(kernel_size=3)
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])

def init(run_args, device):  
    # Data
    # breakpoint()
    if run_args.dataset == 'omniglot':
        data_loader = omniglot_dataset.init_training_data_loader(
                                                run_args.data_dir, 
                                                device=device,
                                                batch_size=run_args.batch_size,
                                                shuffle=False,  
                                                mode="original",
                                                one_substroke='angle',
                                                use_interpolate=20)
    elif run_args.dataset == 'MNIST' or run_args.dataset == 'mnist':
        # Training and Testing dataset
        res = run_args.img_res

        trn_dataset = datasets.MNIST(root='./data', train=True, download=True,
                transform=transforms.Compose([
                transforms.RandomRotation(30, fill=(0,)),
                transforms.Resize([res,res], antialias=True),
                transforms.ToTensor(),
                #transforms.Normalize((0.1307,), (0.3081,))
            ]))
        val_dataset = datasets.MNIST(root='./data', train=False, download=False,
                transform=transforms.Compose([
                transforms.Resize([res,res], antialias=True),
                transforms.ToTensor(),
                #transforms.Normalize((0.1307,), (0.3081,))
            ]))

        # Get index of trn, val set
        # num_train = len(trn_dataset)
        # valid_size = 0.2
        # indices = list(range(num_train))
        # np.random.shuffle(indices)
        # split = int(np.floor(valid_size * num_train))
        # trn_idx, val_idx = indices[split:], indices[:split]

        # trn_sampler = SubsetRandomSampler(trn_idx)
        # val_sampler = SubsetRandomSampler(val_idx)

        # To only use a subset
        # idx = torch.logical_or(trn_dataset.targets == 1, 
        #                        trn_dataset.targets == 2)
        # idx = torch.logical_or(trn_dataset.targets == 0, trn_dataset.targets == 8)
        # idx = trn_dataset.targets == 1
        # trn_dataset.targets = trn_dataset.targets[idx]
        # trn_dataset.data= trn_dataset.data[idx]

        # idx = torch.logical_or(val_dataset.targets == 1, 
        #                        val_dataset.targets == 2)
        # # idx = val_dataset.targets == 1
        # val_dataset.targets = val_dataset.targets[idx]
        # val_dataset.data= val_dataset.data[idx]

        train_loader = DataLoader(trn_dataset,
                                batch_size=run_args.batch_size, 
                                num_workers=2,
                                shuffle=True,
                                # sampler=trn_sampler
                                )

        valid_loader = DataLoader(val_dataset,
                                batch_size=run_args.batch_size,
                                num_workers=2,
                                shuffle=True,
                                # sampler=val_sampler
                                )

        # Test dataset
        # tst_dataset = datasets.MNIST(root='./data', train=False,
        # tst_dataset = datasets.EMNIST(root='./data', train=False, split='balanced',
        #                 transform=transforms.Compose([
        #                     # transforms.RandomRotation(30, fill=(0,)),
        #                     transforms.Resize([res,res], antialias=True),
        #                     transforms.ToTensor(),
        #                     #transforms.Normalize((0.1307,), (0.3081,))
        #                 ]),
        #                 download=True)
        # test_loader = DataLoader(tst_dataset,
        #         batch_size=run_args.batch_size, shuffle=True, num_workers=4
        # )
        if run_args.model_type == 'MWS':
            from models.mws import handwritten_characters as mws
            train_loader = mws.data.get_data_loader(trn_dataset.data/255,
                                                    run_args.batch_size,
                                                    run_args.device, ids=True)
            valid_loader = mws.data.get_data_loader(val_dataset.data/255,
                                                    run_args.batch_size, 
                                                    run_args.device, 
                                                    id_offset=len(trn_dataset),
                                                    ids=True)
        data_loader = train_loader, valid_loader
    elif run_args.dataset == 'generative_model':
        # train and test dataloader
        data_loader = synthetic.get_data_loader(generative_model, 
                                                batch_size=run_args.batch_size,
                                                device=run_args.device)
    elif run_args.dataset == 'multimnist':
        keep_classes = ['11', '77', '17', '71',]# '171']
        data_loader = (multimnist.get_dataloader(keep_classes=keep_classes,
                                                device=run_args.device,
                                                batch_size=run_args.batch_size,
                                                shuffle=True,
                                                transform=transforms.Compose([
                    transforms.GaussianBlur(kernel_size=3),
                    transforms.Resize([28, 28])
                   ])), None)
    else:
        raise NotImplementedError
      
    if run_args.model_type == 'Base':
        # Generative model
        generative_model = base.GenerativeModel(
                                ctrl_pts_per_strk=run_args.points_per_stroke,
                                prior_dist=run_args.prior_dist,
                                likelihood_dist=run_args.likelihood_dist,
                                strks_per_img=run_args.strokes_per_img,
                                res=run_args.img_res,
                                ).to(device)

        # Guide
        guide = base.Guide(ctrl_pts_per_strk=run_args.points_per_stroke,
                                dist=run_args.inference_dist,
                                net_type=run_args.inference_net_architecture,
                                strks_per_img=run_args.strokes_per_img,
                                img_dim=[1, run_args.img_res, run_args.img_res],
                                ).to(device)

    elif run_args.model_type == 'Sequential':
        # assert ((run_args.execution_guided == (not run_args.no_sgl_strk_tanh)) or
        #         not run_args.execution_guided),\
            # "execution_guided should be used in accordance with strk_tanh norm"
        generative_model = ssp.GenerativeModel(
                    max_strks=run_args.strokes_per_img,
                    pts_per_strk=run_args.points_per_stroke,
                    z_where_type=run_args.z_where_type,
                    res=run_args.img_res,
                    execution_guided=run_args.execution_guided,
                    transform_z_what=run_args.transform_z_what,
                    input_dependent_param=run_args.input_dependent_render_param,
                    prior_dist=run_args.prior_dist,
                    num_mlp_layers=run_args.num_mlp_layers,
                    maxnorm=not run_args.no_maxnorm,
                    sgl_strk_tanh=not run_args.no_sgl_strk_tanh,
                    add_strk_tanh=not run_args.no_add_strk_tanh,
                    constrain_param=not run_args.constrain_sample,
                    spline_decoder=not run_args.no_spline_renderer,
                    render_method=run_args.render_method,
                    intermediate_likelihood=run_args.intermediate_likelihood,
                    # comment out for eval old models
                    # dependent_prior=run_args.dependent_prior
                                    ).to(device)
        guide = ssp.Guide(
                max_strks=run_args.strokes_per_img,
                pts_per_strk=run_args.points_per_stroke,
                z_where_type=run_args.z_where_type,
                img_dim=[1, run_args.img_res, run_args.img_res],
                execution_guided=run_args.execution_guided,
                transform_z_what=run_args.transform_z_what,
                input_dependent_param=run_args.input_dependent_render_param,
                exec_guid_type=run_args.exec_guid_type,
                prior_dist=run_args.prior_dist,
                target_in_pos=run_args.target_in_pos,
                feature_extractor_sharing=run_args.feature_extractor_sharing,
                num_mlp_layers=run_args.num_mlp_layers,
                num_bl_layers=run_args.num_baseline_layers,
                bl_mlp_hid_dim=run_args.bl_mlp_hid_dim,
                bl_rnn_hid_dim=run_args.bl_rnn_hid_dim,
                maxnorm=not run_args.no_maxnorm,
                sgl_strk_tanh=not run_args.no_sgl_strk_tanh,
                add_strk_tanh=not run_args.no_add_strk_tanh,
                z_what_in_pos=run_args.z_what_in_pos,
                constrain_param=not run_args.constrain_sample,
                render_method=run_args.render_method,
                intermediate_likelihood=run_args.intermediate_likelihood,
                # comment out for eval old models
                dependent_prior=run_args.dependent_prior,
                residual_pixel_count=run_args.residual_pixel_count,
                spline_decoder=not run_args.no_spline_renderer,
                sep_where_pres_mlp=run_args.sep_where_pres_mlp,
                render_at_the_end=run_args.render_at_the_end,
                simple_pres=run_args.simple_pres,
                simple_arch=run_args.simple_arch,
                residual_no_target=run_args.residual_no_target,
                                ).to(device)
    elif run_args.model_type == 'AIR':
        run_args.z_where_type = '3'
        generative_model = air.GenerativeModel(
                                max_strks=run_args.strokes_per_img,
                                res=run_args.img_res,
                                z_where_type=run_args.z_where_type,
                                execution_guided=run_args.execution_guided,
                                z_what_dim=run_args.z_dim,
                                prior_dist=run_args.prior_dist,
                        ).to(device)
        guide = air.Guide(
                        max_strks=run_args.strokes_per_img,
                        img_dim=[1, run_args.img_res, run_args.img_res],
                        z_where_type=run_args.z_where_type,
                        execution_guided=run_args.execution_guided,
                        exec_guid_type=run_args.exec_guid_type,
                        feature_extractor_sharing=\
                                            run_args.feature_extractor_sharing,
                        z_what_dim=run_args.z_dim,
                        z_what_in_pos=run_args.z_what_in_pos,
                        prior_dist=run_args.prior_dist,
                        target_in_pos=run_args.target_in_pos,
                        sep_where_pres_mlp=run_args.sep_where_pres_mlp,
                        ).to(device)
    elif run_args.model_type == 'VAE':
        generative_model = vae.GenerativeModel(
                                res=run_args.img_res,
                                z_dim=run_args.z_dim,
                                ).to(device)
        guide = vae.Guide(
                        img_dim=[1, run_args.img_res, run_args.img_res],
                        z_dim=run_args.z_dim,
                        ).to(device)
    elif run_args.model_type == 'MWS':
        from models.mws import handwritten_characters as mws
        mws_args = mws.run.get_args_parser().parse_args([
                                            '--img_res', str(run_args.img_res)
        ])
        res = run_args.img_res
        mws_args.num_rows, mws_args.num_cols = res, res
        # mws_args.num_train_data = len(trn_dataset)
        mws_args.num_train_data = len(trn_dataset)
        run_args.num_particles = mws_args.num_particles
        generative_model, guide, optimizer, memory, stats = \
                                        mws.util.init(mws_args, device)

    else:
        raise NotImplementedError
    # Model tuple
    if run_args.model_type == 'MWS':
        model = (generative_model, guide, memory) 
    else:
        model = (generative_model, guide)

    # Optimizer
    optimizer = init_optimizer(run_args, model)

    # Stats
    if run_args.model_type != 'MWS':
        stats = Stats([], [], [], [])


    return model, optimizer, stats, data_loader

def init_optimizer(run_args, model):
    # Model tuple
    if run_args.model_type == 'MWS':
        (generative_model, guide, _) = model
    else:
        (generative_model, guide) = model

    # parameters = guide.parameters()
    if run_args.model_type == 'AIR':
        air_parameters = itertools.chain(guide.air_params(), 
                                         generative_model.parameters()
                                         )
        optimizer = torch.optim.Adam([
            {
                'params': air_parameters, 
                'lr': run_args.lr,
                'weight_decay': run_args.weight_decay
            },
            {
                'params': guide.baseline_params(),
                'lr': run_args.bl_lr,
                'weight_decay': run_args.weight_decay,
            }
        ])
    elif run_args.model_type == 'Sequential':
        train_style_mlp = True
        if run_args.execution_guided or run_args.prior_dist == 'Sequential':
            if train_style_mlp:
                air_parameters = itertools.chain(
                                             guide.air_params(), 
                                             guide.internal_decoder.parameters(),
                                             generative_model.parameters())
            else:
                air_parameters = itertools.chain(
                                             guide.no_style_mlp_air_parameters(), 
                                             guide.internal_decoder.parameters(),
                                             generative_model.parameters())
        else:
            if train_style_mlp:
                air_parameters = itertools.chain(guide.air_params(), 
                                                generative_model.parameters())
            else:
                air_parameters = itertools.chain(
                                            guide.no_style_mlp_air_parameters(),
                                            generative_model.parameters())
        optimizer = torch.optim.Adam([
            {
                'params': air_parameters, 
                'lr': run_args.lr,
                'weight_decay': run_args.weight_decay
            },
            {
                'params': guide.baseline_params(),
                'lr': run_args.bl_lr,
                'weight_decay': run_args.weight_decay,
            }
        ]) 
    else:
        parameters = itertools.chain(guide.parameters(), 
                                     generative_model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=run_args.lr)
        
    return optimizer

def save_checkpoint(path, model, optimizer, stats, run_args=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if run_args.model_type == 'MWS':
        generative_model, guide, memory = model
        torch.save(
            {
                "generative_model_state_dict": generative_model.state_dict(),
                "guide_state_dict": guide.state_dict(),
                "memory": memory,
                "optimizer_state_dict": optimizer.state_dict(),
                "stats": stats,
                "run_args": run_args,
            },
            path,
        )
    else:
        generative_model, guide = model
        torch.save(
            {
                "generative_model_state_dict": generative_model.state_dict(),
                "guide_state_dict": guide.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stats": stats,
                "run_args": run_args,
            },
            path,
        )
    logging.info(f"Saved checkpoint to {path}")


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as e:
        print(e)
        print(f"failed loading {path}")
    run_args = checkpoint["run_args"]
    model, optimizer, stats, data_loader = init(run_args, device)

    if run_args.model_type == 'MWS':
        generative_model, guide, memory = model

        generative_model.load_state_dict(checkpoint["generative_model_state_dict"])
        guide.load_state_dict(checkpoint["guide_state_dict"])
        memory = checkpoint["memory"]

        model = generative_model, guide, memory
    else:
        generative_model, guide = model

        generative_model.load_state_dict(checkpoint["generative_model_state_dict"])
        guide.load_state_dict(checkpoint["guide_state_dict"])

        model = (generative_model, guide)

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    stats = checkpoint["stats"]
    return model, optimizer, stats, data_loader, run_args

ClfStats = collections.namedtuple("ClfStats", ['trn_accuracy', 'tst_accuracy'])
Stats = collections.namedtuple("Stats", ["trn_losses", "trn_elbos", 
                                            "tst_losses", "tst_elbos"])


def save_baseline_posterior(
    path,
    color_variabilitiess,
    global_color_probss,
    color_probs_log_prob,
    color_probs_grid,
    marbless,
    run_args=None,
):
    """Save approximation of p(ɑ, β, θ | y).

    Args:
        color_variabilitiess: [num_iterations - burn_in]
        global_color_probss: [num_iterations - burn_in, num_colors]
        color_probs_log_prob: [num_bags, num_grid_points]
        color_probs_grid: [num_grid_steps, 2]
        marbless: [num_bags, num_colors]
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "color_variabilitiess": color_variabilitiess,
            "global_color_probss": global_color_probss,
            "color_probs_log_prob": color_probs_log_prob,
            "color_probs_grid": color_probs_grid,
            "marbless": marbless,
            "run_args": run_args,
        },
        path,
    )
    logging.info(f"Saved baseline run to {path}")


def load_baseline_posterior(path, device):
    checkpoint = torch.load(path, map_location=device)

    color_variabilitiess = checkpoint["color_variabilitiess"]
    global_color_probss = checkpoint["global_color_probss"]
    color_probs_log_prob = checkpoint["color_probs_log_prob"]
    color_probs_grid = checkpoint["color_probs_grid"]
    marbless = checkpoint["marbless"]
    run_args = checkpoint["run_args"]

    return (
        color_variabilitiess,
        global_color_probss,
        color_probs_log_prob,
        color_probs_grid,
        marbless,
        run_args,
    )


def set_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_fig(fig, path, dpi=100, tight_layout_kwargs={}):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(**tight_layout_kwargs)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    logging.info("Saved to {}".format(path))
    plt.close(fig)


class MultilayerPerceptron(nn.Module):
    def __init__(self, dims, non_linearity):
        """
        Args:
            dims: list of ints
            non_linearity: differentiable function
        Returns: nn.Module which represents an MLP with architecture
            x -> Linear(dims[0], dims[1]) -> non_linearity ->
            ...
            Linear(dims[-3], dims[-2]) -> non_linearity ->
            Linear(dims[-2], dims[-1]) -> y
        """

        super(MultilayerPerceptron, self).__init__()
        self.dims = dims
        self.non_linearity = non_linearity
        self.linear_modules = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.linear_modules.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        temp = x
        for linear_module in self.linear_modules[:-1]:
            temp = self.non_linearity(linear_module(temp))
        return self.linear_modules[-1](temp)


def init_mlp(in_dim, out_dim, hidden_dim, num_layers, non_linearity=None,):
    """Initializes a MultilayerPerceptron.
    Args:
        in_dim: int
        out_dim: int
        hidden_dim: int
        num_layers: int
        non_linearity: differentiable function (tanh by default)
    Returns: a MultilayerPerceptron with the architecture
        x -> Linear(in_dim, hidden_dim) -> non_linearity ->
        ...
        Linear(hidden_dim, hidden_dim) -> non_linearity ->
        Linear(hidden_dim, out_dim) -> y
        where num_layers = 0 corresponds to
        x -> Linear(in_dim, out_dim) -> y
    """
    if non_linearity is None:
        # non_linearity = nn.ReLU()
        non_linearity = nn.Tanh()
    dims = [in_dim] + [hidden_dim for _ in range(num_layers)] + [out_dim]

    return MultilayerPerceptron(dims, non_linearity)

class ConvolutionNetwork(nn.Module):
    def __init__(self, n_in_channels=1, n_mid_channels=32, n_out_channels=64,
                                                                    mlp=None):
        super().__init__()
        self.conv1 = nn.Conv2d(n_in_channels, n_mid_channels, 3, 1)
        self.conv2 = nn.Conv2d(n_mid_channels, n_out_channels, 3, 1)
        self.mlp = mlp
        self.dropout = nn.Dropout(0.50)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        if self.mlp:
            x = self.mlp(x)
        return x

def init_cnn(mlp_out_dim, 
             cnn_out_dim,
             num_mlp_layers, 
             n_in_channels=1, 
             n_mid_channels=32, 
             n_out_channels=64, 
             non_linearity=None, 
             mlp_hidden_dim=256):

    """Initializes a convnet, assuming input:
         - has 1 channel and; 
         - it's a squared image.
    """
    conv_output_size = cnn_out_dim #4608 #1600 #9216
    mlp = init_mlp(in_dim=conv_output_size, out_dim=mlp_out_dim, 
                                            hidden_dim=mlp_hidden_dim, 
                                            num_layers=num_mlp_layers)
    conv_net = ConvolutionNetwork(mlp=mlp, n_in_channels=n_in_channels,
                                            n_mid_channels=n_mid_channels, 
                                            n_out_channels=n_out_channels)

    return conv_net

class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization1 = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
            )
        # Regressor for the 3x2 affine matrix
        # todo: contraint the angle to be within [-pi, pi]
        self.localization2 = nn.Sequential(
                nn.Linear(10 * 3 * 3, 32),
                nn.ReLU(True),
                nn.Linear(32, 4) # translation x, y; scale; angle
            )
        # Initialize the weight/bias with identity transformation
        self.localization2[2].weight.data.zero_()
        self.localization2[2].bias = torch.nn.Parameter(torch.tensor([0,0,1,0], 
                                                            dtype=torch.float))
    def get_transform_param(self, x):
        '''
        Args:
            x [n_batch, n_channel, h, w]: input map
        Return:
            thetas [n_batch, 4 (translation x, y; scale; angle)]: contraint 
                affine matrix
        '''
        # step1: a transformation conditional on the input
        xs = self.localization1(x)
        xs = xs.view(-1, 10 * 3 * 3)
        thetas = self.localization2(xs)
        thetas = thetas.view(-1, 4)
        thetas = SpatialTransformerNetwork.get_affine_matrix_from_param(thetas)
        return thetas
    
    def forward(self, x, output_theta=False):
        thetas = self.get_transform_param(x)
        x_out = spatial_transform(x, thetas)
        if output_theta:
            return x_out, thetas
        else:
            x_out

def spatial_transform(x, theta):
    '''
    Args:
        x: unstransformed image from the dataset
        theta: [2, 3] affine transformation matrix. Output of a spatial
            transfomer's localization output.
    Return:
        transformed image
    '''
    # step2: the transformation parameters are used to create a sampling grid
    grid = F.affine_grid(theta, x.shape, align_corners=True)
    # step3: take the image, sampling grid to the sampler, producing output
    x = F.grid_sample(x, grid, align_corners=True,)
    return x

def inverse_spatial_transformation(x, theta):
    '''
    Args:
        x: transformed images
        theta: [2, 3] affine transformation parameters output by the 
            localization network to "zoom in"/focus on a part of the image.
    Return:
        x: images put back to its original background
    '''
    r_theta = invert_affine_transform(theta)
    x_out = spatial_transform(x, r_theta)
    return x_out

def invert_z_where(z_where):
    z_where_inv = torch.zeros_like(z_where)
    scale = z_where[:, 0:1]   # (batch, 1)
    z_where_inv[:, 1:3] = -z_where[:, 1:3] / scale   # (batch, 2)
    z_where_inv[:, 0:1] = 1 / scale    # (batch, 1)
    return z_where_inv

def get_affine_matrix_from_param(thetas, z_where_type, center=None):
    '''Get a batch of 2x3 affine matrix from transformation parameters 
    thetas of shape [batch_size, 4 (shift x, y; scale; angle)]. <-deprecated format
    # todo make capatible with base model stn
    # todo constraint the scale, shift parameters
    Args:
        thetas [bs, z_where_dim]
        z_where_type::str:
            '3': (scale, shift x, y)
            '4_rotate': (scale, shift x, y, rotate) or 
            '4_no_rotate': (scale x, y, shift x, y)
            '5': (scale x, y, shift x, y, rotate)
    '''
    if center is None:
        center = torch.zeros_like(thetas[:, :2])
    if z_where_type=='4_rotate':
        bs = thetas.shape[0]
        # new
        # rotate
        angle = thetas[:, 3]
        angle_cos, angle_sin = (torch.cos(torch.deg2rad(angle)),\
                                torch.sin(torch.deg2rad(angle)))
        affine_matrix = torch.stack([angle_cos, -angle_sin, torch.empty_like(angle), 
                                     angle_sin, angle_cos, torch.empty_like(angle)], 
                                     dim=0).T.view(-1,2,3)
        # scale
        affine_matrix *= thetas[:, 0].unsqueeze(-1).unsqueeze(-1)  
        # translate
        affine_matrix[:, :, 2] = thetas[:, 1:3] 
        return affine_matrix

        # old
        # scale = thetas[:, 0:1].expand(-1, 2)
        # translations = thetas[:, 1:3]
        # angle = thetas[:, 3]
    elif z_where_type == '4_no_rotate':
        scale = thetas[:, 0:2]
        translations = thetas[:, 2:4]
        angle = torch.zeros_like(thetas[:, 0])
    elif z_where_type == '3':
        bs = thetas.shape[0]
        affine_matrix = torch.eye(2,3).unsqueeze(0).expand(bs,2,3).to(thetas.device)
        # scale
        affine_matrix = affine_matrix * thetas[:, 0].unsqueeze(-1).unsqueeze(-1)  
        # translate
        affine_matrix[:, :, 2] = thetas[:, 1:3] 
        # scale = thetas[:, 0:1].expand(-1, 2)
        # translations = thetas[:, 1:3]
        # angle = torch.zeros_like(thetas[:, 0]) 
        return affine_matrix
    elif z_where_type == '5':
        scale = thetas[:, 0:2]
        translations = thetas[:, 2:4]
        angle = thetas[:, 3]
    else:
        raise NotImplementedError
    affine_matrix = get_affine_matrix2d(translations=translations,
                                        center=center,
                                        scale=scale,
                                        angle=angle)[:, :2]
    return affine_matrix    



def init_stn(in_dim=None, out_dim=None, num_mlp_layers=None, 
                        non_linearity=None, end_cnn=False):
    stn = SpatialTransformerNetwork()
    if end_cnn:
        cnn = init_cnn(in_dim=in_dim, out_dim=out_dim, 
                                num_mlp_layers=num_mlp_layers,
                        ) 
        return stn, cnn
    else:
        return stn

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device


def dirichlet_raw_params_transform(raw_concentration):
    return raw_concentration.exp()


def gamma_raw_params_transform(raw_concentration, raw_rate):
    return raw_concentration.exp(), raw_rate.exp()


def max_normalize(imgs, max_per_img):
    """Only normalize the images where the max is greater than 0.
    """
    non_zero_max = max_per_img[max_per_img != 0].reshape(-1, 1, 1, 1)
    imgs[(max_per_img != 0).squeeze()] = imgs[(max_per_img != 0).squeeze()] / non_zero_max
    return imgs

def normal_raw_params_transform(raw_loc, raw_scale):
    return raw_loc, raw_scale.exp()


def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                                 exp(values[i_1, ..., i_N])
            log( ------------------------------------------------------------ )
                    sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
    # log_numerator = values
    return values - log_denominator


def exponentiate_and_normalize(values, dim=0):
    """Exponentiates and normalizes a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                            exp(values[i_1, ..., i_N])
            ------------------------------------------------------------
             sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    return torch.exp(lognormexp(values, dim=dim))


def cancel_all_my_non_bash_jobs():
    logging.info("Cancelling all non-bash jobs.")
    jobs_status = (
        subprocess.check_output(f"squeue -u {getpass.getuser()}", shell=True)
        .decode()
        .split("\n")[1:-1]
    )
    non_bash_job_ids = []
    for job_status in jobs_status:
        if not ("bash" in job_status.split() or "zsh" in job_status.split()):
            non_bash_job_ids.append(job_status.split()[0])
    if len(non_bash_job_ids) > 0:
        cmd = "scancel {}".format(" ".join(non_bash_job_ids))
        logging.info(cmd)
        logging.info(subprocess.check_output(cmd, shell=True).decode())
    else:
        logging.info("No non-bash jobs to cancel.")


def step_lstm(lstm, input_, h_0_c_0=None):
    """LSTMCell-like API for LSTM.

    Args:
        lstm: nn.LSTM
        input_: [batch_size, input_size]
        h_0_c_0: None or
            h_0: [num_layers, batch_size, hidden_size]
            c_0: [num_layers, batch_size, hidden_size]

    Returns:
        output: [batch_size, hidden_size]
        h_1_c_1:
            h_1: [num_layers, batch_size, hidden_size]
            c_1: [num_layers, batch_size, hidden_size]
    """
    output, h_1_c_1 = lstm(input_[None], h_0_c_0)
    return output[0], h_1_c_1
