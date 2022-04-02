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
from torchvision.transforms.functional import adjust_sharpness
from torchvision.utils import save_image
from einops import rearrange
from kornia.geometry.transform import invert_affine_transform, get_affine_matrix2d

from models import base, ssp, air, vae#, mws
# from data.omniglot_dataset.omniglot_dataset import TrainingDataset
from data import synthetic, multimnist, cluster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(pathname)s:%(lineno)d | %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

ZWhereParam = collections.namedtuple("ZWhereParam", "loc std dim")

def init_dataloader(res, dataset, batch_size=64, rot=True, shuffle=True):
    # Dataloader
    # Train dataset
    if dataset in [
        'Omniglot',
        'Quickdraw',
        ]: 
        rot = False
    if rot:
        trn_transform_lst = [
                            transforms.Resize([120, 120], antialias=True),
                            transforms.RandomRotation(30, fill=(0,)),
                            transforms.Resize([res, res], antialias=True),
                            transforms.ToTensor(),
                            ]
    else:
        trn_transform_lst = [
                            # transforms.Resize([120, 120], antialias=True),
                            transforms.Resize([res, res], antialias=True),
                            transforms.ToTensor(),
                            ]
    tst_transform_lst = [
                            transforms.Resize([res,res], antialias=True),
                            transforms.ToTensor(),
                        ]
    if dataset == "EMNIST": # trn 697,932 tst 116,323
        trn_dataset = datasets.EMNIST(root='./data', train=True, 
                            split='balanced',
                            transform=transforms.Compose(trn_transform_lst), 
                            download=True)
        tst_dataset = datasets.EMNIST(root='./data', train=False, 
                            split='balanced',
                            transform=transforms.Compose(tst_transform_lst), 
                            download=True)
    elif dataset == "MNIST": # trn 60k; tst 10k
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
        # trn_dataset.targets = trn_dataset.targets[idx]
        # trn_dataset.data= trn_dataset.data[idx]
        trn_dataset = datasets.MNIST(root='./data', train=True,
                            transform=transforms.Compose(trn_transform_lst), 
                            download=True)
        tst_dataset = datasets.MNIST(root='./data', train=False,
                            transform=transforms.Compose(tst_transform_lst), 
                            download=True)    
    elif dataset == 'KMNIST': # trn 60k; tst 10k
        trn_dataset = datasets.KMNIST(root='./data', train=True,
                            transform=transforms.Compose(trn_transform_lst), 
                            download=True)

        tst_dataset = datasets.KMNIST(root='./data', train=False,
                            transform=transforms.Compose(tst_transform_lst), 
                            download=True)
    elif dataset == 'QMNIST': # trn, tst 60k
        trn_dataset = datasets.QMNIST(root='./data', train=True, compat=True,
                            transform=transforms.Compose(trn_transform_lst), 
                            download=True)
        tst_dataset = datasets.QMNIST(root='./data', train=False, compat=True,
                            transform=transforms.Compose(tst_transform_lst), 
                            download=True)
    elif dataset == 'Quickdraw': # all: trn 1.38m; tst 345k; 100 cat: .4m, .1m
        from data.QuickDraw_pytorch.DataUtils.load_data import QD_Dataset
        trn_dataset = QD_Dataset(mtype="train", 
                        transform=transforms.Compose([
                        transforms.ToPILImage(mode=None)] + trn_transform_lst +[
                        transforms.GaussianBlur(kernel_size=3),
                        transforms.Lambda(lambda x: adjust_sharpness(x, 5))
                        ]),
                            root="./data/QuickDraw_pytorch/Dataset")
        tst_dataset = QD_Dataset(mtype="test", 
                        transform=transforms.Compose([
                        transforms.ToPILImage(mode=None)] + tst_transform_lst +[
                        transforms.GaussianBlur(kernel_size=3),
                        transforms.Lambda(lambda x: adjust_sharpness(x, 5))
                        ]),
                            root="./data/QuickDraw_pytorch/Dataset")
    elif dataset == 'Omniglot': # trn 19,280; tst 13,180
        trn_dataset = datasets.Omniglot(root='./data', background=True,
                        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: 1-x),
                        transforms.ToPILImage(mode=None)]+ trn_transform_lst), 
                            download=True)
        tst_dataset = datasets.Omniglot(root='./data', background=False,
                        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: 1-x),
                        transforms.ToPILImage(mode=None)]+ tst_transform_lst), 
                            download=True)        
    else: raise NotImplementedError

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(6)

    train_loader = DataLoader(trn_dataset, batch_size=batch_size, 
                                shuffle=True if shuffle else False, 
                                num_workers=4,
                                # worker_init_fn=seed_worker, generator=g,
                                )
    test_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=4,
                                worker_init_fn=seed_worker, generator=g,)
    
    return train_loader, test_loader, trn_dataset, tst_dataset

def init_classification_nets(guide, args, dataset, batch_size, trned_ite):
    # Dataset
    res = guide.res if args.save_model_name == 'MWS' else guide.img_dim[-1]
    # source dataset
    train_loader, test_loader, _,_ = init_dataloader(res, dataset, 
                                                batch_size=batch_size)

    # only_z_what = True if args.model_type == 'Sequential' else False
    only_z_what = False
    # latent variable dataset
    train_loader, test_loader = cluster.get_lv_data_loader(
                                                args.save_model_name,
                                                guide, 
                                                (train_loader, test_loader),
                                                dataset_name=dataset,
                                                args=args,
                                                only_z_what=only_z_what,
                                                trned_ite=trned_ite)
    # Model
    if args.save_model_name == 'VAE':
        classifier_in_dim = guide.z_dim
    elif args.save_model_name == 'MWS':
        classifier_in_dim = guide.num_arcs * 2
    elif args.model_type in ['AIR', 'Sequential']:
        if only_z_what:
            classifier_in_dim = guide.max_strks * guide.z_what_dim
        else:
            classifier_in_dim = guide.max_strks * (
                                    guide.z_where_dim + guide.z_what_dim)
    classifier_out_dim = test_loader.dataset.num_classes

    lv_classifier = init_mlp(in_dim=classifier_in_dim, 
                             out_dim=classifier_out_dim,
                             hidden_dim=256,
                             num_layers=0,).to(args.device)
    # Optimizer
    optimizer = torch.optim.Adam([
        {
            'params': lv_classifier.parameters(), 
            'lr': 1e-5
        }
    ])
    
    # Stats
    stats = ClfStats([], [])

    return (lv_classifier, (train_loader, test_loader), 
            optimizer, stats)


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

def transform_z_what(z_what, z_where, z_where_type):
    '''Apply z_where_mtrx to z_what such that it has a similar result as
    applying the inverse z_where_mtrx to the z_what-rendering.
    Args:
        z_what [bs, n_strk, n_pts, 2]
        z_where [bs, n_strk, z_where_dim]
        z_where_type::str
        res (int): resolution for target image; used in affine_grid
    Return:
        transformed_z_what [bs, n_strk, n_pts, 2]
    '''
    bs, n_strk, n_pts, _ = z_what.shape
    z_where_mtrx = get_affine_matrix_from_param(
                                    thetas=z_where.view(bs*n_strk, -1), 
                                    z_where_type=z_where_type
                                    ).view(bs, n_strk, 2, 3)
    z_what = (z_what - .5) * 2

    homo_coord = torch.cat([z_what, torch.ones_like(z_what)[..., :1]], dim=3)
    transformed_z_what = ((z_where_mtrx @ homo_coord.transpose(2,3)
                                                    ).transpose(2,3))
    transformed_z_what = transformed_z_what * .5 +.5

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
    '3': (shift x, y, scale, )
    # '4_no_rotate': (scale x, y, shift x, y)
    '4_rotate': (shift x, y, scale, rotate)
    '5': (shift x, y, scale x, y, rotate)
    '''
    init_z_where_params = {'3': ZWhereParam(torch.tensor([0,0,.8]), 
                                                torch.tensor([.2,.2,.2]), 3), # AIR?
                                                # torch.tensor([.2,1,1]), 3), # spline?
                                                # torch.ones(3)/5, 3),
                        #    '4_no_rotate': ZWhereParam(torch.tensor([.3,1,0,0]),
                        #                         torch.ones(4)/5, 4),
                           '4_rotate': ZWhereParam(
                                                torch.tensor([0,0,.8,0]),
                                                torch.tensor([.2,.2,.2,.4]), 4),
                                                # torch.ones(4)/5, 4),
                           '5': ZWhereParam(torch.tensor([0,0,.8,.8,0]),
                                            torch.tensor([.2,.2,.2,.2,.4]), 5),
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
    # return f"save/{path_base}"
    # to avoid the disk storage issues
    return f"/om/user/ycliang/save/{path_base}"


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

blur = transforms.Compose([
                       transforms.GaussianBlur(kernel_size=3)
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])

def init(run_args, device):  
    # Data
    # res = run_args.img_res
    train_loader, test_loader, trn_dataset, tst_dataset = init_dataloader(
                                                        run_args.img_res, 
                                                        run_args.dataset,
                                                        run_args.batch_size)
    data_loader = train_loader, test_loader

    if run_args.model_type == 'MWS':
        from models.mws import handwritten_characters as mws
        train_loader = mws.data.get_data_loader(trn_dataset.data/255,
                                                run_args.batch_size,
                                                run_args.device, ids=True)
        test_loader = mws.data.get_data_loader(tst_dataset.data/255,
                                                run_args.batch_size, 
                                                run_args.device, 
                                                id_offset=len(trn_dataset),
                                                ids=True)
    data_loader = train_loader, test_loader
    # elif run_args.dataset == 'generative_model':
    #     # train and test dataloader
    #     data_loader = synthetic.get_data_loader(generative_model, 
    #                                             batch_size=run_args.batch_size,
    #                                             device=run_args.device)
    # elif run_args.dataset == 'multimnist':
    #     keep_classes = ['11', '77', '17', '71',]# '171']
    #     data_loader = (multimnist.get_dataloader(keep_classes=keep_classes,
    #                                             device=run_args.device,
    #                                             batch_size=run_args.batch_size,
    #                                             shuffle=True,
    #                                             transform=transforms.Compose([
    #                 transforms.GaussianBlur(kernel_size=3),
    #                 transforms.Resize([28, 28])
    #                ])), None)
    # else:
    #     raise NotImplementedError
      
    if run_args.model_type == 'Base':
        # Generative model
        # removing stand alone gen
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
        # assert ((run_args.use_canvas == (not run_args.no_sgl_strk_tanh)) or
        #         not run_args.use_canvas),\
            # "use_canvas should be used in accordance with strk_tanh norm"
        hid_dim = 256
        generative_model = ssp.GenerativeModel(
                    max_strks=int(run_args.strokes_per_img),
                    pts_per_strk=run_args.points_per_stroke,
                    z_where_type=run_args.z_where_type,
                    res=run_args.img_res,
                    use_canvas=run_args.use_canvas,
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
                    sep_where_pres_net=run_args.sep_where_pres_net,
                    no_rnn=run_args.no_rnn,
                    no_pres_rnn=run_args.no_pres_rnn,
                    # comment out for eval old models
                    dependent_prior=run_args.dependent_prior,
                    prior_dependency=run_args.prior_dependency,
                    hidden_dim=hid_dim,
                    bern_img_dist=run_args.bern_img_dist,
                    n_comp=int(run_args.num_mixtures),
                    correlated_latent=run_args.correlated_latent,
                    use_bezier_rnn=run_args.use_bezier_rnn,
                    condition_by_img=run_args.condition_by_img,
                                    ).to(device)
        guide = ssp.Guide(
                max_strks=int(run_args.strokes_per_img),
                pts_per_strk=run_args.points_per_stroke,
                z_where_type=run_args.z_where_type,
                img_dim=[1, run_args.img_res, run_args.img_res],
                use_canvas=run_args.use_canvas,
                transform_z_what=run_args.transform_z_what,
                input_dependent_param=run_args.input_dependent_render_param,
                use_residual=run_args.use_residual,
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
                dependent_prior=run_args.dependent_prior,
                prior_dependency=run_args.prior_dependency,
                residual_pixel_count=run_args.residual_pixel_count,
                spline_decoder=not run_args.no_spline_renderer,
                sep_where_pres_net=run_args.sep_where_pres_net,
                render_at_the_end=run_args.render_at_the_end,
                simple_pres=run_args.simple_pres,
                residual_no_target=run_args.residual_no_target,
                canvas_only_to_zwhere=run_args.canvas_only_to_zwhere,
                detach_canvas_so_far=run_args.detach_canvas_so_far,
                detach_canvas_embed=run_args.detach_canvas_embed,
                detach_rsd=not run_args.no_detach_rsd,
                detach_rsd_embed=run_args.detach_rsd_embed,
                no_rnn=run_args.no_rnn,
                no_pres_rnn=run_args.no_pres_rnn,
                no_post_rnn=run_args.no_post_rnn,
                # comment out for eval old models
                no_what_post_rnn=run_args.no_what_post_rnn,
                no_pres_post_rnn=run_args.no_pres_post_rnn,
                only_rsd_ratio_pres=run_args.only_rsd_ratio_pres,
                hidden_dim=hid_dim,
                bern_img_dist=run_args.bern_img_dist,
                dataset=run_args.dataset,
                linear_sum=run_args.linear_sum,
                n_comp=int(run_args.num_mixtures),
                correlated_latent=run_args.correlated_latent,
                use_bezier_rnn=run_args.use_bezier_rnn,
                condition_by_img=run_args.condition_by_img,
                                ).to(device)
    elif run_args.model_type == 'AIR':
        run_args.z_where_type = '3'
        generative_model = air.GenerativeModel(
                                max_strks=run_args.strokes_per_img,
                                res=run_args.img_res,
                                z_where_type=run_args.z_where_type,
                                use_canvas=run_args.use_canvas,
                                z_what_dim=run_args.z_dim,
                                prior_dist=run_args.prior_dist,
                        ).to(device)
        guide = air.Guide(
                        max_strks=run_args.strokes_per_img,
                        img_dim=[1, run_args.img_res, run_args.img_res],
                        z_where_type=run_args.z_where_type,
                        use_canvas=run_args.use_canvas,
                        use_residual=run_args.use_residual,
                        feature_extractor_sharing=\
                                            run_args.feature_extractor_sharing,
                        z_what_dim=run_args.z_dim,
                        z_what_in_pos=run_args.z_what_in_pos,
                        prior_dist=run_args.prior_dist,
                        target_in_pos=run_args.target_in_pos,
                        sep_where_pres_net=run_args.sep_where_pres_net,
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
    optimizer, scheduler = init_optimizer(run_args, model)

    # Stats
    if run_args.model_type != 'MWS':
        stats = Stats([], [], [], [])


    return model, optimizer, scheduler, stats, data_loader

def init_optimizer(run_args, model):
    # Model tuple
    scheduler = None
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
        if run_args.use_canvas or run_args.prior_dist == 'Sequential':
            if run_args.anneal_non_pr_net_lr:
                assert run_args.sep_where_pres_net != run_args.simple_pres,\
                        "sep lrs needs sep nets"
                pr_net_param = guide.pr_net_param()
                none_pr_air_param = itertools.chain(
                                            guide.non_pr_net_air_param(),
                                            guide.internal_decoder.parameters(),
                                            generative_model.parameters())
            air_parameters = itertools.chain(
                                            guide.air_params(), 
                                            guide.internal_decoder.parameters(),
                                            generative_model.parameters())
        else:
            air_parameters = itertools.chain(guide.air_params(), 
                                                generative_model.parameters())
        if run_args.anneal_non_pr_net_lr:
            optimizer = torch.optim.Adam([
            {'params': none_pr_air_param, 
             'lr': run_args.lr,
             'weight_decay': run_args.weight_decay},
            {'params': pr_net_param, 
             'lr': run_args.lr,
             'weight_decay': run_args.weight_decay},
            {'params': guide.baseline_params(),
             'lr': run_args.bl_lr,
             'weight_decay': run_args.weight_decay,}
            ])
            patience, threshold = 5000, 30
            if run_args.dataset in ['EMNIST', 
                                    'Omniglot', 
                                    'KMNIST', 
                                    'Quickdraw']:
                patience, threshold = 0, 100
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='max',factor=0.1, patience=patience, 
                        threshold=threshold, threshold_mode='abs', min_lr=[
                            1e-4, 
                            1e-3, 
                            1e-3
                            ])
            # lambda1 = lambda epoch: 9*torch.heaviside(torch.tensor(epoch,dtype=int), torch.tensor(-1,dtype=int)) + 1
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)
        else:
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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max',factor=0.1, patience=5000, threshold=30, 
            min_lr=[1e-4, 1e-3], threshold_mode='abs'
            )
    else:
        parameters = itertools.chain(guide.parameters(), 
                                     generative_model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=run_args.lr)
    return optimizer, scheduler

def save_checkpoint(path, model, optimizer, schedular, stats, run_args=None):
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
        "schedular_state_dict": schedular.state_dict() if schedular != None\
                                                       else None,
        "stats": stats,
        "run_args": run_args,
        },
            path,
        )
    logging.info(f"Saved checkpoint to {path}")


def load_checkpoint(path, device, finetune_dataset_name=None):
    scheduler = None
    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as e:
        print(e)
        print(f"failed loading {path}")
    run_args = checkpoint["run_args"]
    if finetune_dataset_name != None:
        run_args.dataset = finetune_dataset_name
    model, optimizer, scheduler, stats, data_loader = init(run_args, device)

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

        # # reset pres, where mlp for omniglot
        # guide.where_mlp.seq.linear_modules[-1].weight.data.zero_()
        # guide.where_mlp.seq.linear_modules[-1].bias = torch.nn.Parameter(
        #     torch.tensor([2,2,0,0,0, -4,-4,-4,-4,-4], dtype=torch.float,
        #                  device=device), 
        #     requires_grad=True) 
        # guide.pres_mlp.seq.linear_modules[-1].weight.data.zero_()
        # guide.pres_mlp.seq.linear_modules[-1].bias = torch.nn.Parameter(
        #     torch.tensor([5], dtype=torch.float, 
        #                  device=device), 
        #     requires_grad=True) # when scaling up the reparam likelihood, it's better to use this
        #     # [6], dtype=torch.float)) # works for stable models
        # from models.ssp_mlp import PresWherePriorMLP
        # guide.internal_decoder.gen_pr_wr_mlp = PresWherePriorMLP(
        #                                         in_dim=266,
        #                                         z_where_type='5',
        #                                         z_where_dim=5,
        #                                         hidden_dim=256,
        #                                         num_layers=2,
        #                                         constrain_param=True).to(device)
        model = (generative_model, guide)

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    stats = checkpoint["stats"]
    return model, optimizer, scheduler, stats, data_loader, run_args

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
        # self.conv1 = nn.Conv2d(n_in_channels, n_mid_channels, 9, 1)
        # self.conv2 = nn.Conv2d(n_mid_channels, n_out_channels, 7, 1)
        self.mlp = mlp
        self.dropout = nn.Dropout(0.50)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2) # addition
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        # print("CNN out dim", x.shape)
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

# def invert_z_where(z_where):
#     z_where_inv = torch.zeros_like(z_where)
#     scale = z_where[:, 0:1]   # (batch, 1)
#     z_where_inv[:, 1:3] = -z_where[:, 1:3] / scale   # (batch, 2)
#     z_where_inv[:, 0:1] = 1 / scale    # (batch, 1)
#     return z_where_inv

def get_affine_matrix_from_param(thetas, z_where_type, center=None):
    '''Get a batch of 2x3 affine matrix from transformation parameters 
    thetas of shape [batch_size, 4 (shift x, y; scale; angle)]. <-deprecated format
    # todo make capatible with base model stn
    # todo constraint the scale, shift parameters
    Args:
        thetas [bs, z_where_dim]
        z_where_type::str:
            '3': (scale, shift x, y)
            '4_rotate': (shift x, y, scale, rotate) or 
            '4_no_rotate': (scale x, y, shift x, y)
            '5': (scale x, y, shift x, y, rotate)
    '''
    if center is None:
        center = torch.zeros_like(thetas[:, :2])
    if z_where_type=='4_rotate':
        bs = thetas.shape[0]
        # new
        # 1 rotate
        angle = thetas[:, 3]
        angle_cos, angle_sin = (torch.cos(angle),\
                                torch.sin(angle))
        affine_matrix = torch.stack([angle_cos, -angle_sin, 
                                    torch.ones_like(angle), 
                                     angle_sin, angle_cos, 
                                     torch.ones_like(angle)], 
                                     dim=1)
        affine_matrix = affine_matrix.view(-1,2,3)
        # 2 scale
        affine_matrix = affine_matrix * thetas[:, 2].unsqueeze(-1).unsqueeze(-1)  
        # 3 translate
        affine_matrix[:, :, 2] = thetas[:, 0:2] 
        return affine_matrix

        # old
        # scale = thetas[:, 0:1].expand(-1, 2)
        # translations = thetas[:, 1:3]
        # angle = thetas[:, 3]
    elif z_where_type == '5':
        bs = thetas.shape[0]
        # new
        # 1 rotate
        angle = thetas[:, 4]
        angle_cos, angle_sin = (torch.cos(angle),\
                                torch.sin(angle))
        affine_matrix = torch.stack([angle_cos, -angle_sin, torch.empty_like(angle), 
                                     angle_sin, angle_cos, torch.empty_like(angle)], 
                                     dim=0).T.view(-1,2,3)
        # 2 scale
        affine_matrix[:, :, 0] *= thetas[:, 2].unsqueeze(-1)  
        affine_matrix[:, :, 1] *= thetas[:, 3].unsqueeze(-1)  
        # 3 translate
        affine_matrix[:, :, 2] = thetas[:, 0:2] 
        return affine_matrix
        
        #old
        scale = thetas[:, 0:2]
        translations = thetas[:, 2:4]
        angle = thetas[:, 3]
    elif z_where_type == '7':
        # shear
        sx = thetas[:, 5]
        sy = thetas[:, 6]
        
        # rotate
        angle = thetas[:, 4]
        angle_cos, angle_sin = (torch.cos(torch.deg2rad(angle)),\
                                torch.sin(torch.deg2rad(angle)))
        affine_matrix = torch.stack([
                                    angle_cos * (1 + sx * sy) + angle_sin * sx, 
                                    -angle_sin * (1 + sx * sy) + sx * angle_cos, 
                                    torch.empty_like(angle), 
                                    angle_cos * sy + angle_sin, 
                                    -angle_sin * sy + angle_cos, 
                                    torch.empty_like(angle)
                                ], dim=0).T.view(-1,2,3)
        # 2 scale
        affine_matrix[:, :, 0] *= thetas[:, 2].unsqueeze(-1)  
        affine_matrix[:, :, 1] *= thetas[:, 3].unsqueeze(-1)  
        # 3 translate
        affine_matrix[:, :, 2] = thetas[:, 0:2] 
        return affine_matrix
 
    elif z_where_type == '4_no_rotate':
        scale = thetas[:, 0:2]
        translations = thetas[:, 2:4]
        angle = torch.zeros_like(thetas[:, 0])
    elif z_where_type == '3':
        bs = thetas.shape[0]
        affine_matrix = torch.eye(2,3).unsqueeze(0).expand(bs,2,3).to(thetas.device)
        # scale
        affine_matrix = affine_matrix * thetas[:, 2].unsqueeze(-1).unsqueeze(-1)  
        # translate
        affine_matrix[:, :, 2] = thetas[:, 0:2] 
        # scale = thetas[:, 0:1].expand(-1, 2)
        # translations = thetas[:, 1:3]
        # angle = torch.zeros_like(thetas[:, 0]) 
        return affine_matrix
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
