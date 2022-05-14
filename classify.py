import math

import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torch.distributions import Independent, Normal

# import plot
import util
from plot import display_transform

def parse(args, model, data_loader, writer, n_parse=1, tag=None):
    '''Parse the images with the model and log the results.
    Return:
        parse
    '''
    gen, guide = model
    latents = []
    dec_params = []
    # split = ['Support', 'Query'] 
    with torch.no_grad():
        for i, (sup_img, qry_img, _) in enumerate(data_loader):
            # Parse
            sup_img.squeeze_(0), qry_img.squeeze_(0)
            sup_img = sup_img.to(args.device)
            qry_img = qry_img.to(args.device) 
            imgs = torch.cat([sup_img, qry_img], dim=0)
            out = guide(imgs, n_parse)
            recons = out.canvas[0]

            latents.append(out.z_smpl)
            dec_params.append(out.decoder_param)

            # Plot
            imgs = display_transform(imgs.cpu())
            recons = display_transform(recons.cpu())
            comparison = torch.cat([imgs, recons], dim=2)
            img_grid = make_grid(comparison, nrow=40)
            writer.add_image('One shot classification/'+tag,
                             img_grid,
                             i)
    return latents, dec_params
                

def fine_tune(args, model):
    # Train
    # check if it has been finetuned
    # pretrained base model ckpt path
    base_checkpoint_path = util.get_checkpoint_path(args)
    # ckpt path for the finetune model
    args.save_model_name = args.save_model_name + '_OneShotClf'
    clf_checkpoint_path = util.get_checkpoint_path(args)

    continue_training = False
    if Path(clf_checkpoint_path).exists() and continue_training:
        model, optimizer, scheduler, stats, _, trn_args = util.load_checkpoint(
                                                path=clf_checkpoint_path,
                                                device=device,
                                                init_data_loader=False)
    else:
        _, optimizer, scheduler, stats, _, trn_args = util.load_checkpoint(
                                                    path=base_checkpoint_path,
                                                    device=device,
                                                    init_data_loader=False)
        stats.tst_elbos.clear()

    # Fine-tuning
    finetune_ite = 1e4
    ft_ite_so_far = 0 if not continue_training else len(stats.tst_elbos)
    while ft_ite_so_far < finetune_ite: 
        for sup_img, qry_img, _ in tst_loader:
            sup_img.squeeze_(0), qry_img.squeeze_(0)
            sup_img = sup_img.to(args.device)
            qry_img = qry_img.to(args.device) 
            imgs = torch.cat([sup_img, qry_img], dim=0)

            # optimzier
            optimizer.zero_grad()
            loss_tuple = losses.get_loss_sequential(
                                generative_model=gen, 
                                guide=guide,
                                imgs=imgs, 
                                k=1,
                                iteration=ft_ite_so_far,
                                writer=writer,
                                beta=float(args.beta),
                                args=args)
            loss = loss_tuple.overall_loss.mean()
            loss.backward()
            optimizer.step()
            # log
            for n, l in zip(loss_tuple._fields, loss_tuple):
                writer.add_scalar("OneShot clf finetuning/"+n, 
                                    l.detach().mean(), ft_ite_so_far)
            stats.tst_elbos.append(loss)

            if ft_ite_so_far % 50 == 0:
                util.logging.info(f"Iteration {ft_ite_so_far} |"+
                                  f"Loss = {stats.tst_elbos[-1]:.3f}")
            # save
            if ft_ite_so_far % args.save_interval == 0 or ft_ite_so_far == \
                                                            finetune_ite:
                save(args, ft_ite_so_far, model, optimizer, scheduler, stats) 
            ft_ite_so_far += 1

    save(args, ft_ite_so_far, model, optimizer, scheduler, stats) 

def optimize_and_score(args, model, latentss, dec_paramss, 
                                    tst_loader, 
                                    writer, 
                                    n_parse=1):
    # For each episode
    for run, (sup_img, qry_img, label) in tqdm(enumerate(tst_loader)):
        latents, dec_params = latentss[run], dec_paramss[run]
        # Create a token model for each n_parse X query image
        cm = ClassifyModel(latents, dec_params, model)
        # Optimizer
        optim = torch.optim.Adam([{
                                    'param':cm.parameters(),
                                    'lr': 1e-4}]
                                )
        # Optimize
        cm.train()
        for tqdm(range(1000)):
            cm.zero_grad()
            score = -cm(qry_img)

            score.mean().bacward()
            optim.step()

        # Score
        cm.eval()
        score = cm(qry_img)
    # Get accuracies
    pass

class ClassifyModel(nn.Modules):
    def __init__(self, latents, dec_param, model, qs, ss, ps, 
                 direction="q|s"):
        '''
        Args:
            latents: used as the conditioning variable of the token model
                        in the order of [support; query]
                    z_pres [ptcs, bs, strks]
                    z_what [ptcs, bs, strks, pts, 2]
                    z_where [ptcs, bs, strks, 3]
            qs: number of querys to simaltaneously optimize
            ss: number of supports
            ps: number of particles per query-support
            direction: deciding which way the score is computed
        '''
        z_pr, z_wt, z_wr = latents
        # z_pres [1, ss, ps, strks]
        # z_what [1, ss, ps, strks, pts, 2]
        # z_where [1, ss, ps, strks, z_where_dim]
        z_pr, z_wt, z_wr = (z_pr.transpose(0,1)[None,], 
                            z_wt.transpose(0,1)[None,], 
                            z_wr.transpose(0,1)[None,])
        if direction == 'q|s':
            # z_pres [qs, ss, ps, strks]
            # z_what [qs, ss, ps, strks, pts, 2]
            # z_where [qs, ss, ps, strks, z_where_dim]
            self.cond_pr, 
            self.cond_wt, 
            self.cond_wr = (z_pr[:, :20].repeat(qs, 1, 1, 1), 
                            z_wt[:, :20].repeat(qs, 1, 1, 1, 1, 1), 
                            z_wr[:, :20].repeat(qs, 1, 1, 1, 1))
            
        elif direction == 's|q':
            self.cond_pr, 
            self.cond_wt, 
            self.cond_wr = (z_pr[:, 20:].repeat(qs, 1, 1, 1), 
                            z_wt[:, 20:].repeat(qs, 1, 1, 1, 1, 1), 
                            z_wr[:, 20:].repeat(qs, 1, 1, 1, 1))
            
        else:
            raise NotImplementedError

        self.optm_pr, self.optm_wt, self.optm_wr = (self.cond_pr, 
                                                    self.cond_wt, 
                                                    self.cond_wr)
        self.cond_wt_std = torch.zeros_like(self.cond_wt) + 0.2
        optm_wt = nn.Parameters(self.optm_wt)
        self.affines_param = nn.Parameters(torch.zeros([qs, ss, ps, 7]))

    def get_affines(self):
        shift = util.constrain_param(self.affines_param[...,:2], -.2, .2)
        scale = util.constrain_param(self.affines_param[...,2:4], .7, .8)
        rot = util.constrain_param(self.affines_param[...,4:5], -.2*math.pi, 
                                                                .2*math.pi)
        shear = util.constrain_param(self.affines_param[...,5:7], -.2*math.pi, 
                                                                  .2*math.pi)
        return util.constrain_param(torch.cat(shift,scale,rot,shear), dim=3)
    
    def get_cond_token_dist(self):
        affine_matrices = util.get_affine_from_param(
                                    self.get_affines(self.affine_params)
                                    type='7'
                                )
        transform_cond = util.transform_z_what(self.cond_wt, affine_matrices)
        token_dist = Independent(
                                Normal(transform_cond, self.cond_wt_std),
                                reinterpreted_batch_ndims=2
                            )
        return token_dist

    def forward(self, qry_img):
        '''
            score [qs, ss, ps]
        '''
        # Compute the token model log_prob
        strks_per_samp = self.cond_pr.sum(-1)
        token_score = self.get_cond_token_dist.log_prob(self.optm_wt)
        # out: [qs, ss, ps], in: [qs, ss, ps, n_strokes]
        token_score = token_score[self.cond_pr].sum(-1) / strks_per_samp

        # Compute the image likelihood
        img_dist = self.gen.renders_image(
                                        self.optm_wr,
                                        self.optm_wt,
                                        self.optm_pr,
                                          )
        image_score = img_dist(qry_img)

        # Return the scores for each optimize on each conditioning variable
        return token_score + image_score

def score():
    with torch.no_grad():
        accuracys = []
        for run, (sup_img, qry_img, label) in tqdm(enumerate(tst_loader)):
            sup_img.squeeze_(0), qry_img.squeeze_(0), label.squeeze_(0)
            sup_img = sup_img.to(args.device)
            qry_img = qry_img.to(args.device) 
            label = label.to(args.device)

            # our model:
            # get choose top k latents from ptcs
            sup_latents = util.get_top_k_latents(guide, gen, sup_img, 
                                                  ptcs=3, k=1)
            sup_out = guide(sup_img)
            sup_latents, mask_prev, canvas, z_prior = (
                                                sup_out.z_smpl, sup_out.mask_prev,
                                                sup_out.canvas,sup_out.z_prior)

            # get support sample curve dist 
            sample_res = 200 # works well wo affine
            # norm_std = 0.05
            norm_std = 0.05
            sup_crv = gen.get_sample_curve(sup_latents, uni_out_dim=True, 
                                            sample_res=sample_res)
            sup_crv_dist = SampleCurveDistWithAffine(sup_crv, norm_std)
            
            # get query sample curves
            qry_out = guide(qry_img)
            qry_latents = qry_out.z_smpl
            qry_crv = gen.get_sample_curve(qry_latents, uni_out_dim=True, 
                                            sample_res=sample_res)

            # for each of n_class, repeat for n_test time, n_dim
            # latents = ZSample(latents[0].repeat(20,1,1).transpose(0,1),
            #                   latents[1].repeat(20,1,1,1,1).transpose(0,1),
            #                   latents[2].repeat(20,1,1,1).transpose(0,1))
            # mask_prev = mask_prev.repeat(20,1,1).transpose(0,1)
            # canvas = canvas.repeat(20,1,1,1,1).transpose(0,1)
            # z_prior = ZLogProb(z_prior[0].repeat(20,1,1).transpose(0,1),
            #                     z_prior[1].repeat(20,1,1).transpose(0,1),
            #                     z_prior[2].repeat(20,1,1).transpose(0,1))
            # tar_img = tar_img.repeat(20,1,1,1,1)
                            
            batch_mode = False # faster if memory is not a issue
            if batch_mode:
                score_mtrx = sup_crv_dist.log_prob(qry_crv[:, :13])
                raise NotImplementedError
                preds = score_mtrx.argmax(dim=1)+1
                trues = label[:,1]
                n_correct = (preds==trues).sum().item()
            else:
                score_lst, preds, trues = [], [], []
                n_correct = 0
                for i, qry in tqdm(enumerate(qry_img)):

                    # Our model
                    # image_dist likelihood: for each src_img, score each tar_img
                    # qry = qry.qry_img[i:i+1].repeat(20,1,1,1)
                    # _, log_lld = guide.internal_decoder.log_prob(latents=latents,
                    #                     imgs=qry,
                    #                     z_pres_mask=mask_prev,
                    #                     # canvas=canvas,
                    #                     canvas=sup_img.unsqueeze(0),
                    #                     z_prior=z_prior)
                    # scores = log_lld # + log_lld_rv

                    # sample curve likelihood
                    scores = sup_crv_dist.log_prob(qry_crv[:,i:i+1])

                    # pred
                    pred_class = scores.argmax()+1
                    score_lst.append(scores)

                    # Lake baseline: hausdorff distance
                    # scores = []
                    # f_cost = ModHausdorffDistance
                    # f_load = LoadImgAsPoints
                    # qry_load = f_load(qry.detach().cpu())
                    # for sup in sup_img:
                    #     scores.append(f_cost(f_load(sup.detach().cpu()), qry_load))
                    # scores = torch.tensor(scores)
                    # score_lst.append(scores)
                    # pred_class = scores.argmin()+1

                    # compute the accuracy
                    lbl_class = label[label[:,0]==i+1][0][1]
                    preds.append(pred_class)
                    trues.append(lbl_class)
                    # get the label of the current test img
                    if pred_class.item() == lbl_class.item():
                        n_correct += 1
                score_mtrx = torch.concat(score_lst)

            # plot and log 
            score_mtrx = score_mtrx.detach().cpu().numpy()
            score_fig = plot.plot_clf_score_heatmap(score_mtrx, preds, trues)
            writer.add_figure("One shot classification/score", score_fig, run+1)
            writer.add_image("One shot classification/query img", make_grid(
                                        display_transform(qry_img), nrow=10), run+1)
            writer.add_image("One shot classification/support img", make_grid(
                                        display_transform(sup_img), nrow=10), run+1)
            
            accuracys.append(n_correct/20)
            print(f"##### Run {run} accuracy", n_correct/20)
