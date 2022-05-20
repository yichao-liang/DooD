"""test.py: Load the models and test it throught classification and marginal 
likelihood
"""
import os
from pathlib import Path
from sched import scheduler
import numpy as np
import itertools
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pylab as plt

import util, plot, losses, train
from plot import display_transform
from plot import save_img_debug as sid
from models.mws.handwritten_characters.train import get_log_p_and_kl
from models.template import ZSample, ZLogProb
from models.ssp import SampleCurveDist, AffineSampleCurveDist, \
            SampleCurveDistWithAffine

mws_transform = transforms.Resize([50, 50], antialias=True)
gns_transform = transforms.Resize([105,105], antialias=True)
NROW = 16

def marginal_likelihoods(model, stats, test_loader, args, 
                        save_imgs_dir=None,
                        epoch=None, writer=None, k=100, 
                        train_loader=None, optimizer=None, scheduler=None,
                        dataset_name=None, model_tag=None, finetune_ite=0,
                        only_marginal_likelihood_evaluation=True,
                        only_reconstruction=False,
                        dataset_derived_std=False,
                        log_to_file=False,
                        recons_per_img=1,
                        test_run=False,
                        save_as_individual_img=False):
    '''Compute the marginal likelihood through IWAE's k samples and log the 
    reconstruction through `writer` and `save_imgs_dir`.
        If `train_loader` and `optimizer` is not None, do some finetuning
    Args:
        test_loader (DataLoader): testset dataloader
        dataset_name: used in logging; in normal training this is None.
        only_marginal_likelihood_evaluation::bool: if True not recon ploting
        only_reconstruction::bool: if >0, recon such number of batches of img
    '''
    ite_so_far = len(stats.trn_losses)
    # Fining tunning
    if finetune_ite > 0:
        assert train_loader and optimizer, "finetuning requires data and optim"
        args.num_iterations = ite_so_far + finetune_ite 
        model = train.train(model, optimizer, scheduler, stats, 
                            (train_loader, None), args, 
                            writer=writer, dataset_name=dataset_name)

    if args.model_type in ['Sequential', 'AIR']:
        cum_losses = [0]*8
    elif args.model_type in ['Base', 'VAE']:
        cum_losses = [0]*5
    elif args.model_type in ['MWS']:
        cum_losses = [0]*3

    if args.model_type == 'MWS':
        generative_model, guide, memory = model
    else:
        generative_model, guide = model

    # todo: make it condition on trn set
    # if dataset_derived_std:
    #     # use dataset derived std for evaluating marginal likelihood
    #     ds = train_loader.dataset
    #     ds = torch.stack([img for img, _ in ds], dim=0)
    #     ds_var = torch.var(ds,dim=0)
    #     imgs_dist_scale = torch.nn.Parameter(
    #                                             torch.sqrt(ds_var/2).cuda()
    #                                         )
    # if args.model_type == 'AIR' or args.no_spline_renderer:
    #     guide.internal_decoder.imgs_dist_std = torch.nn.Parameter(
    #                     torch.sqrt((guide.internal_decoder.get_imgs_dist_std()**2)/2)
    #                 )

    generative_model.eval(); guide.eval()

    val_ite = 0
    with torch.no_grad():
        if only_reconstruction:
            args.log_param = True
        else:
            args.log_param = False
        for imgs, target in test_loader:
            # for i, im in enumerate(imgs):
            #     im = 1 - gns_transform(im)
            #     save_image(im, f'/om2/user/ycliang/im{i}.png')
            # breakpoint()
            # sid(imgs[-8], 'target')
            if args.model_type == 'MWS':
                    # for mll evaluation
                    imgs = mws_transform(imgs).squeeze(1)
                    obs_id = target

            imgs = imgs.to(args.device)

            if args.model_type == 'Base':
                loss_tuple = losses.get_loss_base(
                                                generative_model, 
                                                guide, 
                                                imgs, 
                                                )
            elif args.model_type == 'Sequential':
                loss_tuple = losses.get_loss_sequential(
                                                generative_model, guide, imgs, 
                                                k=k, iteration=ite_so_far,
                                                args=args,
                                                writer_tag=f'{dataset_name}/',
                                                train=False,
                                                writer=writer,
                                                )
            elif args.model_type == 'AIR':
                loss_tuple = losses.get_loss_air(
                                                # generative_model, 
                                                guide,
                                                imgs, 
                                                k=k,
                                                args=args,
                                                writer_tag=f'{dataset_name}/',
                                                )
            elif args.model_type == 'VAE':
                loss_tuple = losses.get_loss_vae(
                                                generative_model,
                                                guide,
                                                imgs, 
                                                k)
            elif args.model_type == 'MWS':
                loss_tuple = get_log_p_and_kl(
                                                generative_model,
                                                guide,
                                                imgs.round(),
                                                obs_id,
                                                k,)
            else:
                raise NotImplementedError
                
            for i in range(len(loss_tuple)):
                cum_losses[i] += loss_tuple[i].sum()      
        
            val_ite += 1
            if val_ite % 25 == 0:
                util.logging.info(f"Validation Iteration {val_ite}") 

            if only_reconstruction:
                break
            
            if val_ite == 2 and test_run:
                break
        # Logging
        data_size = len(test_loader.dataset)
        for i in range(len(cum_losses)):
            cum_losses[i] /= data_size

        if args.model_type in ['Sequential', 'AIR']:
            loss_tuple = losses.SequentialLoss(*cum_losses)
        elif args.model_type in ['Base', 'VAE']:
            loss_tuple = losses.BaseLoss(*cum_losses)

        if stats is not None:
            if args.model_type != 'MWS':
                stats.tst_losses.append(loss_tuple.overall_loss)
        
        # write to tensorboard
        if not only_reconstruction:
            if args.model_type == 'MWS':
                names = ['neg_elbo', 'kl', 'neg_log_likelihood']
                for n, l in zip(names, loss_tuple):
                    writer.add_scalar(f"{dataset_name}/Test curves/"+n, 
                                        l.detach().mean(), 
                                        epoch)
            else:
                for n, l in zip(loss_tuple._fields, loss_tuple):
                    # writer.add_scalar(f"{test_loader.dataset.__name__}/Test curves/"+n, 
                    if dataset_name is not None:
                        writer.add_scalar(f"{dataset_name}/Test curves/"+n, 
                                        l, ite_so_far)
                    else:
                        writer.add_scalar("Test curves/"+n, l, epoch)
        # writer.add_scalars("Test curves", {n:l for n, l in 
        #                         zip(loss_tuple._fields, loss_tuple)}, epoch)   
        if args.model_type == 'MWS':
            util.logging.info(f"Epoch {epoch} Test loss | Loss = {loss_tuple[0].detach().mean():.3f}")
        else:
            util.logging.info(f"Epoch {epoch} Test loss | Loss = {loss_tuple.overall_loss:.3f}")

        if not only_marginal_likelihood_evaluation:
            plot.plot_reconstructions(
                    imgs=imgs, 
                    guide=guide, 
                    # generative_model=generative_model, 
                    args=args, 
                    writer=writer, 
                    epoch=ite_so_far,
                    writer_tag=f'Test_{args.save_model_name}',
                    dataset_name=dataset_name,
                    recons_per_img=recons_per_img,
                    has_fixed_img=False,
                    target=target,
                    dataset=test_loader.dataset,
                    invert_color=True,
                    save_as_individual_img=save_as_individual_img,
                    multi_sample=only_reconstruction,
                )

        if log_to_file:
            log_file_name = "eval_mll.csv"
            if os.path.exists(log_file_name):
                print("File exist.") 
                eval_df = pd.read_csv(log_file_name, header=0, index_col=0)
                print(eval_df)
            else:
                eval_df = pd.DataFrame()
            if args.model_type != 'MWS':
                mll = loss_tuple.neg_elbo.detach().cpu().numpy()
            else:
                mll = loss_tuple[0].mean().detach().cpu().numpy()
            eval_df = eval_df.append({
                                "model_name": args.save_model_name,
                                "seed": args.seed,
                                "source_dataset": args.dataset,
                                "source_ite": ite_so_far,
                                "target_dataset": dataset_name,
                                "num_samples": int(k),
                                "marginal_likelihood": mll,
                            }, ignore_index=True)
            eval_df.to_csv(log_file_name)
        writer.flush()
        
        if args.model_type == 'MWS':
            model = generative_model, guide, memory
        else:
            model = generative_model, guide
        return model

def stroke_mll_plot(model, val_loader, args, writer, epoch):
    '''make the stroke vs mll plot for images of 1s and 7s and save it both 
    in a file and log it on tensorboard.
    '''
    n_strks, mlls, labels = [], [], []
    gen, guide = model

    for imgs, labs in val_loader:
        imgs = imgs.to(args.device)
        guide_out = guide(imgs, 1)
        latents, log_post, _, mask_prev, canvas, z_prior = (
            guide_out.z_smpl, guide_out.z_lprb, guide_out.baseline_value, 
            guide_out.mask_prev, guide_out.canvas, guide_out.z_prior)
        z_pres, _, _ = latents
        # num_strks
        num_strks = z_pres.squeeze(0).sum(1)
        # elbo
        log_post_z = torch.cat(
                        [prob.sum(-1, keepdim=True) for prob in log_post], 
                        dim=-1).sum(-1)
        log_prior, log_likelihood = gen.log_prob(
                                            latents=latents, 
                                            imgs=imgs,
                                            z_pres_mask=mask_prev,
                                            canvas=canvas,
                                            z_prior=z_prior)
        log_prior_z = torch.cat(
                        [prob.sum(-1, keepdim=True) for prob in 
                            log_prior], dim=-1).sum(-1)
        generative_joint_log_prob = (log_likelihood + log_prior_z)
        elbo = - log_post_z + generative_joint_log_prob
        # add to list
        n_strks.extend(num_strks.detach().int().cpu().tolist())
        mlls.extend(elbo.squeeze(0).detach().round().cpu().tolist())
        labels.extend(labs.int().tolist())
    # calculate the accuracy of stroke usage assuming
    # 7s need 2 strokes, 1s need 1.
    id7 = torch.tensor(labels) == 7
    strks7 = torch.tensor(n_strks)[id7]
    correct7 = (strks7 == 2).sum()
    num7s = len(strks7)
    id1 = torch.tensor(labels) == 1
    strks1 = torch.tensor(n_strks)[id1]
    correct1 = (strks1 == 1).sum()
    num1s = len(strks1)

    acc1 = correct1 / num1s
    acc7 = correct7 / num7s
    overall_acc = (correct7 + correct1) / (num7s + num1s) 
    writer.add_scalar("Train curves/stroke accuray 1", acc1, epoch)
    writer.add_scalar("Train curves/stroke accuray 7", acc7, epoch)
    writer.add_scalar("Train curves/stroke accuray 1,7", overall_acc, epoch)
    # plot
    plot_data = pd.DataFrame({'Num_strokes': n_strks,
                             'ELBO': mlls,
                             'Label': labels})
    plot.plot_stroke_mll_swarm_plot(plot_data, args, writer, epoch)
    
def classification_evaluation(guide, args, writer, dataset, stats, 
                                                            model_tag=None,
                                                            dataset_name=None,
                                                            batch_size=64,
                                                            log_to_file=True,
                                                        clf_trn_interations=10):
    '''Train two classification models:
    one using the `guide` output; 
    the other using the raw image output
    and log both's accuracy throught `writer`
    '''
    ite_so_far = len(stats.trn_losses)
    guide_classifier, dataloaders, optimizer, clf_stats = \
                            util.init_classification_nets(guide, args, dataset, 
                                                            batch_size, 
                                                        trned_ite=ite_so_far)
    train_loader, test_loader = dataloaders
    log_interval = 100
    num_iterations = clf_trn_interations
    iteration = 0 # iterations-so-far
    epoch = 0
    best_accuray, best_epoch = 0, 0
    while iteration < num_iterations:
        # Train
        for zs, target in train_loader:

            zs, target = zs.to(args.device), target.to(args.device)
            bs = zs.shape[0]
            # Forward
            guide_clf_pred = guide_classifier(zs)

            assert guide_clf_pred.shape[:1] == target.shape
            guide_clf_loss = F.cross_entropy(guide_clf_pred, target)
            
            # Backward, optimize
            guide_clf_loss.backward()
            # for n, p in guide_classifier.named_parameters(): 
            #     assert (p.grad.isnan().any() == False)
            optimizer.step()

            # Log
            # Get the accuracy
            pred_label = guide_clf_pred.argmax(dim=1)
            clf_accuracy = (pred_label == target).sum() / bs

            if iteration % log_interval == 0:
                util.logging.info('Train iteration: {}/{}\t clf_accuracy: {:.6f}'.format(
                    iteration, num_iterations, clf_accuracy))
            writer.add_scalar(f"{dataset_name}/Train/Classification_accuracy", 
                                                      clf_accuracy, iteration)
            clf_stats.trn_accuracy.append(clf_accuracy.item())

            iteration += 1

            if iteration > num_iterations:
                break

        epoch += 1 

        # Test
        with torch.no_grad():
            guide_clf_acc_sum = 0
            for zs, target in test_loader:
                # if args.save_model_name == 'MWS':
                #     imgs = imgs.squeeze(1)
                #     obs_id = target
                bs = zs.shape[0]
                zs, target = zs.to(args.device), target.to(args.device)

                # Forward
                guide_clf_pred = guide_classifier(zs)

                try:
                    guide_clf_loss = F.cross_entropy(guide_clf_pred, target)
                except:
                    breakpoint()

                # get the accuracy
                pred_label = guide_clf_pred.argmax(dim=1)
                correct_count = (pred_label == target).sum()
                guide_clf_acc_sum += correct_count

            n = len(test_loader.dataset)
            guide_clf_loss = guide_clf_acc_sum / n
            if guide_clf_loss > best_accuray:
                best_accuray, best_epoch = guide_clf_loss, epoch

            # Log
            util.logging.info(f'Test epoch {epoch}: ' +
                              f'Clf_accuracy: {guide_clf_loss:.4f}, ' +
                              f'Best so far: {best_accuray:.4f} ' +
                              f'@ epoch {best_epoch}\n')
            writer.add_scalar(f"{dataset}/Test/Classification_accuracy", 
                                                    guide_clf_loss, epoch)
            clf_stats.tst_accuracy.append(guide_clf_loss.item())
            
    if log_to_file:
        log_file_name = "eval_clf.csv"
        if os.path.exists(log_file_name):
            print("File exist.") 
            eval_df = pd.read_csv(log_file_name, header=0, index_col=0)
            print(eval_df)
        else:
            eval_df = pd.DataFrame()

        eval_df = eval_df.append({
                            "model_name": args.save_model_name,
                            "seed": args.seed,
                            "source_dataset": args.dataset,
                            "source_ite": ite_so_far,
                            "train_ite": num_iterations,
                            "target_dataset": dataset_name,
                            "target_dataset": dataset_name,
                            "best_accuracy": best_accuray.detach().cpu().numpy(),
                            "best_epoch": best_epoch
                        }, ignore_index=True)
        eval_df.to_csv(log_file_name)


def unconditioned_generation(model, args, writer, in_img, stats, 
                             max_strks=None):
    '''draw samples conditioned on nothing from the generative model
    Args:
        n::int: number of samples to draw
        in_img [bs, 1, res, res]: input_image; use these for the sigma, 
            normalization slope prediction.
    '''
    # Display the in_img
    bs = in_img.shape[0]
    ite_so_far = len(stats.trn_losses)
    if max_strks == None:
        imgs_disp = display_transform(in_img)
        in_img_grid = make_grid(imgs_disp, nrow=NROW)
        in_img_grid = 1 - in_img_grid
        writer.add_image(f'Unconditioned Generation/in_img', in_img_grid,
                        ite_so_far)

    (gen, guide), dec_param = init_generator(args, model, in_img)

    gen_imgs = draw_samples(gen, guide, bs, dec_param, max_strks)
    with torch.no_grad():
        # if gen_imgs == None:
        #     gen_imgs = gen.renders_imgs(gen_out.z_smpl)
        if max_strks == None:
            gen_imgs_l = display_transform(gen_imgs.squeeze(0))
            gen_img_grid = make_grid(gen_imgs_l, nrow=NROW)
            gen_img_grid = 1 - gen_img_grid
            writer.add_image(f'Unconditioned Generation/out_img', gen_img_grid, 
                            ite_so_far)
            save_imgs_dir = util.get_save_test_img_dir(args, ite_so_far, 
                                               prefix='uncon_generation',
                                               suffix=f'out')
            save_image(gen_img_grid, save_imgs_dir, nrow=NROW)
            # cummulative rendering
            plot.plot_cum_recon(args=args, imgs=None, gen=gen, 
                            latent=gen_out.z_smpl, 
                            z_where_type=gen.z_where_type, writer=writer,
                            dataset_name=None, 
                            epoch=ite_so_far, tag2='out_cum', 
                            tag1='Unconditioned Generation')
        return gen_imgs

def one_shot_classification(model, args, writer, its_so_far, 
                            do_fine_tune=False):
    '''Perform one shot clf as in Omniglot challenge
    '''
    from train import save
    from classify import parse, fine_tune, score, optimize_and_score
    if args.model_type == 'AIR':
        n_parse_per_ite, run_per_eps = 3, 3
    else:
        n_parse_per_ite, run_per_eps = 3, 4
        
    
    # gen, guide = model

    # Init plot of parses
    # tst_loader = util.init_ontshot_clf_data_loader(res=args.img_res)
    # latents, dec_params, gen_probs = parse(args, model, tst_loader, writer, 
    #                                 n_parse=n_parse, tag="Before Finetune Parse")
    # Finetune the model
    if do_fine_tune:
        tst_loader = util.init_ontshot_clf_data_loader(res=args.img_res, 
                                                        shuffle=True)
        model = fine_tune(args, model, tst_loader, writer)

    # Parse again
    tst_loader = util.init_ontshot_clf_data_loader(res=args.img_res, 
                                                   shuffle=False)
    # latents, dec_params, gen_probs = parse(args, model, tst_loader, writer, 
    #                                 n_parse=n_parse, tag="After Finetune Parse")
    # Compute the score
    accuracies = optimize_and_score(
                                    # latents, dec_params, gen_probs,
                                    args, model, tst_loader, writer, 
                                    n_parse_per_ite=n_parse_per_ite, 
                                    run_per_eps=run_per_eps,
                                    two_way_clf=False,
                                    optimize=True, 
                                    tag="After Finutune")
        # get pred
        # pred = score.argmax(score, dim=1)
        # accuracy = compute_accuracy(pred, label)
        # accuracys.append(accuracy)

    av_acc = np.mean(accuracies)
    print("#### One shot classification accuracy per run:", accuracies)
    print("#### One shot classification accuracy average:", av_acc)
    writer.add_scalar("One shot classification/accuracy", av_acc, its_so_far)
    writer.close()

def LoadImgAsPoints(I):
	# Load image file and return coordinates of 'inked' pixels in the binary image
	# 
	# Output:
	#  D : [n x 2] rows are coordinates
    I.squeeze_()
    # I = I.flatten()
    I = np.array(I,dtype=bool)
    # I = np.logical_not(I)
    (row,col) = I.nonzero()
    D = np.array([row,col])
    D = np.transpose(D)
    D = D.astype(float)
    n = D.shape[0]
    mean = np.mean(D,axis=0)
    for i in range(n):
        D[i,:] = D[i,:] - mean
    return D       

def ModHausdorffDistance(itemA,itemB):
	# Modified Hausdorff Distance
	#
	# Input
	#  itemA : [n x 2] coordinates of "inked" pixels
	#  itemB : [m x 2] coordinates of "inked" pixels
	#
	#  M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
	#  International Conference on Pattern Recognition, pp. 566-568.
	#
	D = cdist(itemA,itemB)
	mindist_A = D.min(axis=1)
	mindist_B = D.min(axis=0)
	mean_A = np.mean(mindist_A)
	mean_B = np.mean(mindist_B)
	return max(mean_A,mean_B)
        
def num_strokes_plot(model, stats, args, writer, max_strks=10):
    '''Plot the distribution of number of stroke use in each dataset
    '''
    ds_counts = {} # {ds_name: {0: x0, 1: x1,..., n: xn}}
    ite_so_far = len(stats.trn_losses)
    gen, guide = model
    guide.max_strks = max_strks
    test_run = True
    datasets = [
                "MNIST",
                "Omniglot", 
                "Quickdraw",
                "KMNIST", 
                "EMNIST", 
                ]
    for ds in ['KMNIST', 'EMNIST']: datasets.remove(ds) # simplify dsets

    # loop through all dataset
    for ds_name in datasets:
        _, test_loader, _, _ = util.init_dataloader(args.img_res, ds_name, 
                                                    batch_size=args.batch_size)
        all_counts = [] # stroke counts for each data point

        for i, (imgs, _) in enumerate(tqdm(test_loader)):
            imgs = imgs.to(args.device)
            out = guide(imgs, num_particles=1,)
            z_pres = out.z_smpl.z_pres
            counts = z_pres.sum(-1).squeeze(0).detach().cpu().int()
            all_counts.append(counts)

            # make a cum plot
            if i == 0:
                if args.model_type == 'Sequential':
                    gen.sigma = out.decoder_param.sigma[0]
                    gen.sgl_strk_tanh_slope = \
                                        out.decoder_param.slope[0]
                    if guide.linear_sum:
                        gen.add_strk_tanh_slope =\
                                            out.decoder_param.slope[1][:, :, 0]
                    else: 
                        gen.add_strk_tanh_slope =\
                                        out.decoder_param.slope[1][:, :, -1]
                plot.plot_cum_recon(args, imgs, gen, out.z_smpl, 
                               args.z_where_type, writer, dataset_name=None, 
                               epoch=ite_so_far, tag1='#Stroke Histgram',
                               tag2=ds_name)
            if test_run:
                if i == 10: break
            # i += 1

        # count frequency and store to dict
        all_counts = torch.cat(all_counts, dim=0)
        # all_counts = torch.bincount(all_counts, minlength=max_strks+1)
        ds_counts[ds_name] = all_counts
    
    # plot
    df = pd.concat([pd.DataFrame(
                        {'dataset': name, 'number of strokes': n_strks}
                                    ) for name, n_strks in ds_counts.items()
                    ], ignore_index=True)   

    ax = sns.histplot(df, x='number of strokes', hue='dataset', stat='percent', 
                    #   kde=True, 
                      discrete=True)
    # Set plot font size
    plt.legend(title='Dataset', labels=["MNIST","Omniglot","Quickdraw"])
    plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='16') # for legend title
    ax.set_xlabel("Number of strokes",fontsize=18)
    ax.set_ylabel("Percentage",fontsize=18)
    ax.tick_params(labelsize=16)

    ax.set_xlim(0,max_strks+1)
    ax.set_ylim(0,34)
    fig = ax.get_figure()
    fig.set_tight_layout(tight=True)
    fig.savefig(f"plots/z_pr_generalize_{args.save_model_name}.pdf") 
    writer.add_figure('#Stroke Histgram', fig, ite_so_far)

def auto_complete(model, args, writer, partial_img, stats):
    '''Perform auto-completion conditioned on "partial-completed" images
    '''
    # Setup
    par_img_set = display_transform(partial_img)
    partial_img = torch.repeat_interleave(partial_img,8,dim=0)
    bs = partial_img.shape[0]
    ite_so_far = len(stats.trn_losses)
    imgs_disp = display_transform(partial_img)
    in_img_grid = 1-make_grid(imgs_disp, nrow=NROW)
    writer.add_image(f'Partial completion/in_img', in_img_grid,
                     ite_so_far)

    _, guide = model
    gen = guide.internal_decoder
    gen.img_feature_extractor = guide.img_feature_extractor
    if guide.sep_where_pres_net:
        if guide.no_pres_rnn:
            gen.wr_rnn = guide.wr_rnn
        else:    
            gen.wr_rnn = guide.wr_rnn
            gen.pr_rnn = guide.pr_rnn
    else:
        gen.pr_wr_rnn = guide.pr_wr_rnn
    gen.wt_rnn = guide.wt_rnn

    # Parse the partial image to get hidden states
    # single parse
    # out = guide(partial_img)
    # hs, z_smpl, dec_param = out.hidden_states, out.z_smpl, out.decoder_param
    # last_hs, last_z_sample = util.get_last_vars(hs, z_smpl)
    # recs = out.canvas

    # top parse
    z_smpl, hs, dec_param, recs, _, _ = util.get_top_latents_hiddens(guide, gen, 
                                                    partial_img, ptcs=1)
    last_hs, last_z_sample = util.get_last_vars(hs, z_smpl)

    from models.ssp import DecoderParam
    new_dec_param = DecoderParam(
        sigma=dec_param.sigma[0:1,:,0:1].median().repeat(1,bs,guide.max_strks),
        slope=[
            dec_param.slope[0][0:1,:,0:1].median().repeat(1,bs,guide.max_strks),
            dec_param.slope[1][0:1,:,0:1].repeat(1,1,guide.max_strks)
            ])


    # Resume completing the samples
    # partial_img = (dec_param.slope[1][0:1,:,0].squeeze(0)[:,None,None,None] * 
    partial_img = (dec_param.slope[1][0:1,:,0].squeeze(0)[:,None,None,None] * 
                    # torch.atanh(partial_img))
                    torch.atanh(recs.squeeze(0)))
                    
    gen_out = gen.sample(bs=[bs], 
                            init_canvas=partial_img,
                            init_h=last_hs,
                            init_z=last_z_sample,
                            decoder_param=new_dec_param,
                            linear_sum=guide.linear_sum,
                        )
    
    # Log the results
    # - check the posterior quality
    recs = display_transform(recs.squeeze(0))
    rec_img_grid = 1-make_grid(recs, nrow=NROW)
    writer.add_image(f'Partial completion/partial image recons', rec_img_grid, 
                    ite_so_far)
    # - log the completion results
    gen_imgs = gen_out.canvas
    gen_imgs_l = display_transform(gen_imgs.squeeze(0))
    gen_img_grid = 1-make_grid(gen_imgs_l, nrow=NROW)
    writer.add_image(f'Partial completion/out_img', gen_img_grid, 
                    ite_so_far)
    # save
    gen_imgs_wrap = gen_imgs_l.view(4,8,1,64,64).transpose(1,2).reshape(4,1,8*64,64)
    cat = 1 - torch.cat([par_img_set, gen_imgs_wrap], dim=2)
    for i in range(4):
        save_imgs_dir = f'/om/user/ycliang/{args.dataset}-par_com-{(7+int(args.seed))*4 + i}.pdf'
        # util.get_save_test_img_dir(args, ite_so_far, 
        #                             prefix=f'partial_completion/{args.dataset}_partial_completion',
        #                             suffix=f'out{int(args.seed)*4 + i}')
        save_image(cat[i], save_imgs_dir)
  
    # cat_img = torch.cat([imgs_disp, gen_imgs_l],dim=-1)
    # # cat_img_grid = make_grid(cat_img, nrow=2)

    # save_imgs_dir = util.get_save_test_img_dir(args, ite_so_far, 
    #                                            prefix='partial_completion',
    #                                            suffix=f'out')
    # cat_img = 1 - make_grid(cat_img, nrow=1)
    # save_image(cat_img, save_imgs_dir)
    # cummulative rendering
    gen.sigma = new_dec_param.sigma
    gen.sgl_strk_tanh_slope = new_dec_param.slope[0]
    if guide.linear_sum:
        gen.add_strk_tanh_slope = new_dec_param.slope[1][:, :, 0]
    else:
        gen.add_strk_tanh_slope = new_dec_param.slope[1][:, :, -1]
    # plot.plot_cum_recon(args=args, imgs=None, gen=gen, 
    #                         latent=gen_out.z_smpl, 
    #                         z_where_type=gen.z_where_type, writer=writer,
    #                         dataset_name=None, 
    #                         epoch=ite_so_far, tag2='out_cum', 
    #                         tag1='Partial completion')

def draw_samples(generative_model, guide, bs, dec_param, max_strks=None):
    with torch.no_grad():
        # sample
        # generative_model.transform_z_what = True
        gen_out = generative_model.sample(bs=[bs], 
                                          decoder_param=dec_param,
                                          linear_sum=guide.linear_sum,
                                          max_strks=max_strks,
                                          )

        
        # display the results
        return gen_out.canvas

def init_generator(args, model, in_img):
    bs = in_img.shape[0]

    if args.model_type == 'MWS':
        generative_model, guide, memory = model
    else:
        _, guide = model
        generative_model = guide.internal_decoder

    with torch.no_grad():
        # pass on the parameters
        decoder_param = guide(in_img).decoder_param
        generative_model.img_feature_extractor = guide.img_feature_extractor
        if guide.sep_where_pres_net:
            if guide.no_pres_rnn:
                generative_model.wr_rnn = guide.wr_rnn
            else:    
                generative_model.wr_rnn = guide.wr_rnn
                generative_model.pr_rnn = guide.pr_rnn
        else:
            generative_model.pr_wr_rnn = guide.pr_wr_rnn
        generative_model.wt_rnn = guide.wt_rnn

        from models.ssp import DecoderParam
        # breakpoint()
        decoder_param = DecoderParam(
            sigma=decoder_param.sigma[:,:,0:1].median().repeat(1,bs,
                                                        guide.max_strks),
            slope=[
                decoder_param.slope[0][:,:,0:1].median().repeat(1,bs,
                                                            guide.max_strks),
                decoder_param.slope[1][:,:,0:1].median().repeat(1,bs,
                                                            guide.max_strks)
            ]
        )
    model = generative_model, guide 

    gen.sigma = decoder_param.sigma
    gen.sgl_strk_tanh_slope = decoder_param.slope[0]
    if guide.linear_sum:
        gen.add_strk_tanh_slope = decoder_param.slope[1][:, :, 0]
    else:
        gen.add_strk_tanh_slope = decoder_param.slope[1][:, :, -1]

    return model, decoder_param

def compute_fid_score(model, args, data_loader, writer):
    from ignite.metrics import FID
    from ignite.engine import Engine
    # Init FID evaluator
    def eval_step(engine, batch):
        return batch
    default_evaluator = Engine(eval_step)
    metric = FID()
    metric.attach(default_evaluator, "fid")
    res = 128
    eval_trans = transforms.Compose([transforms.Resize([res, res])])

    # Obtain some samples
    # Draw samples from model
    samples_to_use = 1500
    in_img, _ = next(iter(data_loader))
    in_img = in_img.to(args.device)
    bs = in_img.shape[0]
    (gen, guide), dec_param = init_generator(args, model, in_img)
    gen.eval(), guide.eval()

    print("===> Generating samples from model")
    y_preds, count = [], 0
    while count < samples_to_use:
        gen_imgs = draw_samples(gen, guide, bs, dec_param)
        y_preds.append(gen_imgs)
        count += bs
    y_preds = torch.cat(y_preds, dim=0)
    y_preds = y_preds[:samples_to_use]
    # trans to [bs,3,res,res]
    y_preds = eval_trans(y_preds).repeat(1,3,1,1).cpu() 
    print("===> Done\n")

    # Draw samples from dataset
    print("===> Taking dataset samples")
    y_trues, count = [], 0
    while count < samples_to_use:
        for imgs,_ in data_loader:
            y_trues.append(imgs)
            count += bs
            if count >= samples_to_use:
                break
    y_trues = torch.cat(y_trues, dim=0)
    y_trues = y_trues[:samples_to_use]
    # trans to [bs,3,res,res]
    y_trues = eval_trans(y_trues).repeat(1,3,1,1).cpu()
    print("===> Done\n")

    # Compute score
    state = default_evaluator.run([[y_preds, y_trues]])
    fid = state.metrics["fid"]
    writer.add_scalar("FID/", fid)
    print("## FID score:", fid)

def get_args_parser():
    parser = argparse.ArgumentParser(formatter_class=
                                        argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt_path", 
        default=None,
        type=str,
        help="Path to checkpoint for evaluation")
    
    parser.add_argument("--save_model_name", default=None, type=str, 
                        help='name for ckpt dir')
    parser.add_argument("--tb_name", default=None, type=str, 
                        help='name for tensorboard log')
    parser.add_argument("--clf_trn_interations", default=800000, type=int, help=" ")
    # parser.add_argument("--clf_trn_interations", default=20, type=int, help=" ")
    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # Choose the tasks to test on
    test_run = False
    # broad generalization
    mll_eval = True
    recon_eval = False
    save_as_individual_img = True
    num_strokes_eval = False # plot a distribution of number of strokes / dataset
    clf_eval = False
    # deep generalization
    uncon_sample = False
    auto_complete_demo = False
    char_con_sample = False
    one_shot_clf_eval = False
    plot_tsne = False
    compute_fid = False

    

    if test_run:
        marginal_likelihood_test_datasets = ["MNIST"]
    else:
        marginal_likelihood_test_datasets = [
                     "Omniglot", 
                     "Quickdraw",
                     "KMNIST", 
                     "EMNIST", 
                     "MNIST",
                     ]

    if test_run:
        clf_test_datasets = ["MNIST"]
    else:
        clf_test_datasets = [
                     "KMNIST", 
                     "EMNIST", 
                     "MNIST",
                     "Quickdraw",
                    ]
    
    # Init the models
    device = util.get_device()
    args.device = device
    if args.ckpt_path != None:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = util.get_checkpoint_path_from_path_base(args.save_model_name
                                                            )
    model, optimizer, scheduler, stats, _, trn_args = util.load_checkpoint(
                                                        path=ckpt_path,
                                                        device=device,
                                                        init_data_loader=False)
    if trn_args.model_type == 'MWS':
        gen, guide, memory = model
    else:
        gen, guide = model
    res = gen.res
    
    if args.tb_name == None:
        tb_dir = f"/om/user/ycliang/log/hyper/{args.save_model_name}"
    else:
        tb_dir = f"/om/user/ycliang/log/hyper/{args.tb_name}"
    writer = SummaryWriter(log_dir=tb_dir)

    if plot_tsne:
        with torch.no_grad():
            coloring = 'kmeans'
            # coloring = 'dbscan'
            n_clusters = [8, 10, 15, 5]
            for n in n_clusters:
                plot.plot_stroke_tsne(trn_args,
                                ckpt_path=ckpt_path,
                                title=f'tsne',
                                save_dir=f'plots/tsne_{args.save_model_name}'+
                                        f'_{coloring}{n}',
                                z_what_to_keep=10000,
                                clustering=coloring,
                                n_clusters=n)
    if num_strokes_eval:
        with torch.no_grad():
            num_strokes_plot(model=model, stats=stats, args=trn_args, 
                         writer=writer)

    for dataset in marginal_likelihood_test_datasets:
        if recon_eval:
            train_loader, test_loader, _, _ = util.init_dataloader(res, dataset, 
                                                            batch_size=64, 
                                                            rot=False)
            print(f"===> Begin Reconstruction testing on {dataset}")
            trn_args.save_model_name = args.save_model_name
            marginal_likelihoods(model=model, stats=stats, 
                                test_loader=test_loader, 
                                args=trn_args, save_imgs_dir=None, epoch=None, 
                                writer=writer, k=1,
                                train_loader=None, optimizer=None,
                                dataset_name=dataset, 
                                only_marginal_likelihood_evaluation=False,
                                only_reconstruction=True,
                                save_as_individual_img=save_as_individual_img)
            print(f"===> Done Reconstruction on {dataset}\n")


        # Evaluate marginal likelihood
        if mll_eval:
            train_loader, test_loader, _,_ = util.init_dataloader(res, dataset, 
                                                            batch_size=2)
            print(f"===> Begin Marginal Likelihood evaluation on {dataset}")
            model = marginal_likelihoods(model=model, stats=stats, 
                            test_loader=test_loader, 
                            args=trn_args, save_imgs_dir=None, epoch=None, 
                            writer=writer, 
                            # k=1, # used for debug why the signs are different
                            k=200, # use for current results
                            train_loader=train_loader, 
                            optimizer=None, scheduler=scheduler,
                            # train_loader=train_loader, optimizer=optimizer,
                            dataset_name=dataset,
                            finetune_ite=0,
                            only_marginal_likelihood_evaluation=True,
                            only_reconstruction=False,
                            dataset_derived_std=False,
                            log_to_file=True,
                            test_run=test_run)
            print(f"===> Done elbo_evalution on {dataset}\n")
    if char_con_sample:
        train_loader, test_loader, _, _ = util.init_dataloader(res, 
                                                        trn_args.dataset, 
                                                        batch_size=32, 
                                                        rot=False)
        print(f'===> Generating multiple parse for a data')
        trn_args.save_model_name = args.save_model_name
        marginal_likelihoods(model=model, stats=stats, 
                            test_loader=test_loader, 
                            args=trn_args, save_imgs_dir=None, epoch=None, 
                            writer=writer, k=1,
                            train_loader=None, optimizer=None,
                            dataset_name=dataset, 
                            only_marginal_likelihood_evaluation=False,
                            only_reconstruction=True,
                            recons_per_img=10)
        print(f'===> Done generating multiple parse for a data')
        

    # Evaluate classification
    if clf_eval:
        for dataset in clf_test_datasets:
            train_loader, test_loader, _,_ = util.init_dataloader(res, dataset, 
                                                             batch_size=64)
            print(f"===> Begin Classification evaluation on {dataset}")
            classification_evaluation(guide, trn_args, writer, dataset, stats=stats, 
                                        dataset_name=dataset, batch_size=64,
                                        clf_trn_interations=10 if test_run\
                                            else args.clf_trn_interations)    
            print(f"===> Done clf_evaluation on {dataset}\n")
                
            # Character-conditioned generation
            # Alphabet-conditioned generation
            # print(f"===> Done testing on {dataset} dataset \n\n")

    # Unconditioned generation
    if uncon_sample or auto_complete_demo or compute_fid:
        num_to_sample = 64
        trn_loader, _, _, _ = util.init_dataloader(res, trn_args.dataset, 
                                                    batch_size=num_to_sample,
                                                    rot=True)
        in_img, _ = next(iter(trn_loader))
        in_img = in_img.to(device)
        # in_img, n = [], 0
        # for imgs, _ in train_loader:
        #     imgs = imgs.to(device)
        #     in_img.append(imgs)
        #     n += imgs.shape[0]
        #     if n >= num_to_sample:
        #         in_img = torch.cat(in_img, dim=0)
        #         break

        if compute_fid:
            print(f"===> Begin FID eval on with {args.save_model_name}")
            compute_fid_score(model, trn_args, trn_loader, writer)
            
            print(f"===> Done FID eval on with {args.save_model_name}\n")
        if uncon_sample:
            print(f"===> Begin Unconditioned generation on with {args.save_model_name}")
            unconditioned_generation(model=model, 
                                args=trn_args,
                                writer=writer, 
                                in_img=in_img[:num_to_sample],
                                stats=stats,
                                max_strks=None
                                )
            print(f"===> Done Unconditioned generation on with {args.save_model_name}\n")
        if auto_complete_demo:
            print(f"===> Begin autocompletion demo with {args.save_model_name}")
            partial_img = None
            if partial_img == None:
                # Sample some synthetic partial img
                partial_img = unconditioned_generation(model=model, 
                                args=trn_args,
                                writer=writer, 
                                in_img=in_img[:num_to_sample],
                                stats=stats,
                                max_strks=1,
                                )
            partial_img = partial_img[0:4]
            # partial_img = partial_img[0:16].repeat(4,1,1,1)
            auto_complete(model=model, 
                            args=trn_args, 
                            writer=writer, 
                            partial_img=partial_img,
                            stats=stats)
            print(f"===> Done autocompletion demo with {args.save_model_name}")

    if one_shot_clf_eval:
        one_shot_classification(model, trn_args, writer, len(stats.trn_losses))
    
    # print(f"===> Begin Character-conditioned generation on with {args.save_model_name}")
    # character_conditioned_generation(model=model,
    #                                  args=trn_args,
    #                                  writer=writer,
    #                                  imgs=in_img[:8])
    # print(f"===> Done Character-conditioned generation on with {args.save_model_name}\n")
    writer.flush()
    writer.close()