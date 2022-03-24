"""test.py: Load the models and test it throught classification and marginal 
likelihood
"""
import os
from sched import scheduler
import numpy as np
import itertools
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm


import util, plot, losses, train
from plot import display_transform
from models.mws.handwritten_characters.train import get_log_p_and_kl

mws_transform = transforms.Resize([50, 50], antialias=True)
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
                        recons_per_img=1):
    '''Compute the marginal likelihood through IWAE's k samples and log the 
    reconstruction through `writer` and `save_imgs_dir`.
        If `train_loader` and `optimizer` is not None, do some finetuning
    Args:
        test_loader (DataLoader): testset dataloader
        dataset_name: used in logging; in normal training this is None.
        only_marginal_likelihood_evaluation::bool: if True not recon ploting
        only_reconstruction::bool: if True only see one batch of data
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
    if dataset_derived_std:
        # use dataset derived std for evaluating marginal likelihood
        ds = train_loader.dataset
        ds = torch.stack([img for img, _ in ds], dim=0)
        ds_var = torch.var(ds,dim=0)
        imgs_dist_scale = torch.nn.Parameter(torch.sqrt(ds_var/2).cuda())
        generative_model.imgs_dist_std = imgs_dist_scale

    generative_model.eval(); guide.eval()

    val_ite = 0
    with torch.no_grad():
        for imgs, target in test_loader:
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
                                                args=args)
            elif args.model_type == 'AIR':
                loss_tuple = losses.get_loss_air(
                                                generative_model, 
                                                guide,
                                                imgs, 
                                                k=k)
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
                                        l, epoch)
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
                                    generative_model=generative_model, 
                                    args=args, 
                                    writer=writer, 
                                    epoch=ite_so_far,
                                    writer_tag=f'Test_{args.save_model_name}',
                                    dataset_name=dataset_name,
                                    recons_per_img=recons_per_img,
                                    has_fixed_img=False,
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
                                                         batch_size)
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
            writer.add_scalar(f"{dataset_name}/Train/clf_accuracy", 
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

                # Get the z through guide
                # if args.save_model_name == 'MWS':
                #     latent_dist = guide.get_latent_dist(imgs.round())
                #     zs = guide.sample_from_latent_dist(latent_dist, 1)
                #     zs = zs.view(bs, -1).type(torch.float)
                # else:
                #     zs = guide(imgs).z_smpl
                #     if args.save_model_name != 'VAE':
                #         z_pres, z_what, z_where = zs
                #         max_strks = guide.max_strks
                #         # method 1:
                #         # z_what = (z_what.view(bs, max_strks, -1) * 
                #         #           z_pres.view(bs, max_strks, -1)).view(bs, -1)
                #         # z_where = (z_where.view(bs, max_strks, -1) * 
                #         #            z_pres.view(bs, max_strks, -1)).view(bs, -1)

                #         # method 2:
                #         z_what = (z_what.view(bs, max_strks, -1) * 
                #                   z_pres.view(bs, max_strks, -1)).sum(1).view(bs, -1)
                #         z_where = (z_where.view(bs, max_strks, -1) * 
                #                    z_pres.view(bs, max_strks, -1)).sum(1).view(bs, -1)
                #         zs = torch.cat([z_what, z_where], dim=-1)
                #     else:
                #         zs = zs.squeeze(0)

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
                            "target_dataset": dataset_name,
                            "target_dataset": dataset_name,
                            "best_accuracy": best_accuray.detach().cpu().numpy(),
                            "best_epoch": best_epoch
                        }, ignore_index=True)
        eval_df.to_csv(log_file_name)


def unconditioned_generation(model, args, writer, in_img):
    '''draw samples conditioned on nothing from the generative model
    Args:
        n::int: number of samples to draw
        in_img [bs, 1, res, res]: input_image; use these for the sigma, 
            normalization slope prediction.
    '''
    # Display the in_img
    imgs_disp = display_transform(in_img)
    in_img_grid = make_grid(imgs_disp, nrow=NROW)
    writer.add_image(f'Unconditioned Generation/in_img', in_img_grid)

    if args.model_type == 'MWS':
        generative_model, guide, memory = model
    else:
        _, guide = model
        generative_model = guide.internal_decoder
    
    # todo: check for AIR, Full, Full-seq_prir
    with torch.no_grad():
        # pass on the parameters
        decoder_param = guide(in_img).decoder_param
        generative_model.img_feature_extractor = guide.img_feature_extractor
        if guide.sep_where_pres_net:
            generative_model.wr_rnn = guide.wr_rnn
            generative_model.pr_rnn = guide.pr_rnn
        else:
            generative_model.pr_wr_rnn = guide.pr_wr_rnn
        generative_model.wt_rnn = guide.wt_rnn
        # generative_model.renderer_param_mlp = guide.renderer_param_mlp
        # generative_model.target_in_pos = guide.target_in_pos
        max_step = guide.max_strks

        from models.ssp import DecoderParam
        decoder_param = DecoderParam(
            sigma=decoder_param.sigma[:,:,0:1].repeat(1,1,max_step),
            slope=[
                decoder_param.slope[0][:,:,0:1].repeat(1,1,max_step),
                decoder_param.slope[1]
            ]
        )
        # sample
        gen_out = generative_model.sample(bs=[in_img.shape[0]], 
                                          decoder_param=decoder_param,
                                          char_cond_gen=False,
                                          linear_sum=guide.linear_sum)

        gen.sigma = decoder_param.sigma
        gen.sgl_strk_tanh_slope = decoder_param.slope[0]
        if guide.linear_sum:
            gen.add_strk_tanh_slope = decoder_param.slope[1][:, :, 0]
        else:
            gen.add_strk_tanh_slope = decoder_param.slope[1][:, :, -1]
        
        # display the results
        gen_imgs = gen_out.canvas
        if gen_imgs == None:
            gen_imgs = gen.renders_imgs(gen_out.z_smpl)
        gen_imgs = display_transform(gen_imgs.squeeze(0))
        gen_img_grid = make_grid(gen_imgs, nrow=NROW)
        writer.add_image(f'Unconditioned Generation/out_img', gen_img_grid)
        # cummulative rendering
        plot.plot_cum_recon(imgs=None, gen=gen, latent=gen_out.z_smpl, 
                            z_where_type=gen.z_where_type, writer=writer,
                            dataset_name=None, 
                            epoch=None, tag2='out_cum', 
                            tag1='Unconditioned Generation')

def character_conditioned_generation(model, args, writer, imgs):
    '''Get the first (or more) hidden_states and pass it to the generative model
    to get samples.
    Args:
        imgs [bs, 1, res, res]: character imgs to be conditioned on
    '''
    if args.model_type == 'MWS':
        generative_model, guide, memory = model
    else:
        generative_model, guide = model

    # todo: check for AIR, FULL, Full-seq_prir
    n_samples = 1
    bs = imgs.shape[0]
    # res = generative_model.res
    # out = guide(imgs)

    # # [n_samples, bs, 1, res, res]
    # gen_imgs = util.character_conditioned_sampling(n_samples, 
    #                                                 out, 
    #                                                 generative_model)
    out = guide(imgs)

    # Load the generative states
    generative_model.img_feature_extractor = guide.img_feature_extractor
    generative_model.style_rnn = guide.style_rnn
    generative_model.z_what_rnn = guide.z_what_rnn
    generative_model.renderer_param_mlp = guide.renderer_param_mlp
    generative_model.target_in_pos = guide.target_in_pos

    gen_out = generative_model.sample(bs=[imgs.shape[0]], 
                                      in_img=imgs,
                                    #   hs=out.hidden_states,
                                      z_pms=out.z_pms,
                                      decoder_param=out.decoder_param,
                                    )
    
    gen_imgs = gen_out.canvas
    gen_imgs = display_transform(gen_imgs).unsqueeze(0)

    disp_imgs = torch.cat([display_transform(imgs).unsqueeze(0), gen_imgs],
                                                                        dim=0)                                       
    disp_imgs = disp_imgs.transpose(0, 1).reshape(2*bs, 1, 64, 64)
    gen_img_grid = make_grid(disp_imgs, nrow=n_samples+1)
    writer.add_image(f'Character-conditioned Generation', gen_img_grid) 
    return
    

def get_args_parser():
    parser = argparse.ArgumentParser(formatter_class=
                                        argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt_path", 
        default=None,
        type=str,
        help="Path to checkpoint for evaluation")
    
    parser.add_argument("--save_model_name", default=None, type=str, 
                        help='name for ckpt dir')
    parser.add_argument("--clf_trn_interations", default=400000, type=int, help=" ")
    # parser.add_argument("--clf_trn_interations", default=20, type=int, help=" ")
    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # Choose the dataset to test on
    mll_eval = False
    recon_eval = False
    clf_eval = False
    uncon_eval = True
    marginal_likelihood_test_datasets = [
                     "Omniglot", 
                     "Quickdraw",
                     "MNIST",
                     "KMNIST", 
                     "EMNIST", 
                     ]

    clf_test_datasets = [
                     "EMNIST", 
                     "MNIST",
                     "KMNIST", 
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
                                                            device=device)
    if trn_args.model_type == 'MWS':
        gen, guide, memory = model
    else:
        gen, guide = model
    res = gen.res
    writer = SummaryWriter(log_dir=f"/om/user/ycliang/log/debug1/{args.save_model_name}",)

    for dataset in marginal_likelihood_test_datasets:

        # Evaluate marginal likelihood
        if mll_eval:
            train_loader, test_loader, _,_ = util.init_dataloader(res, dataset, 
                                                            batch_size=2)
        if mll_eval:
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
                            dataset_derived_std=True,
                            log_to_file=True)
            print(f"===> Done elbo_evalution on {dataset}\n")
        
        if recon_eval:
            train_loader, test_loader, _, _ = util.init_dataloader(res, dataset, 
                                                            batch_size=64)
            # print(f"===> Begin Reconstruction testing on {dataset}")
            # trn_args.save_model_name = args.save_model_name
            # marginal_likelihoods(model=model, stats=stats, 
            #                     test_loader=test_loader, 
            #                     args=trn_args, save_imgs_dir=None, epoch=None, 
            #                     writer=writer, k=1,
            #                     train_loader=None, optimizer=None,
            #                     dataset_name=dataset, 
            #                     only_marginal_likelihood_evaluation=False,
            #                     only_reconstruction=True)
            # print(f"===> Done Reconstruction on {dataset}\n")

            print(f'===> Generating multiple parse for a data')
            trn_args.save_model_name = args.save_model_name
            marginal_likelihoods(model=model, stats=stats, test_loader=train_loader, 
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
                                clf_trn_interations=args.clf_trn_interations)    
            print(f"===> Done clf_evaluation on {dataset}\n")
                
            # Character-conditioned generation
            # Alphabet-conditioned generation
            # print(f"===> Done testing on {dataset} dataset \n\n")

    # Unconditioned generation
    if uncon_eval:
        num_to_sample = 64
        trn_loader, _, _, _ = util.init_dataloader(res, trn_args.dataset, 
                                                    batch_size=num_to_sample,
                                                    rot=False)
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

        print(f"===> Begin Unconditioned generation on with {args.save_model_name}")
        unconditioned_generation(model=model, 
                                args=trn_args,
                                writer=writer, 
                                in_img=in_img[:num_to_sample])
        print(f"===> Done Unconditioned generation on with {args.save_model_name}\n")

    # print(f"===> Begin Character-conditioned generation on with {args.save_model_name}")
    # character_conditioned_generation(model=model,
    #                                  args=trn_args,
    #                                  writer=writer,
    #                                  imgs=in_img[:8])
    # print(f"===> Done Character-conditioned generation on with {args.save_model_name}\n")
    writer.flush()
    writer.close()