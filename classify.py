import math
from pathlib import Path

from numpy import prod
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torch.distributions import Independent, Normal
from scipy.special import logsumexp

# import plot
import util, losses, plot
from train import save
from models.template import ZSample
from models.ssp import DecoderParam

display_trans = transforms.Compose([
                        transforms.Resize([30, 30]),
                        ])

# def parse(args, model, data_loader, writer, n_parse=1, tag=None):
def parse(args, imgs, model, writer=None, eps=None, n_parse=1, tag=None, 
          crit='joint', device=None):
    '''Parse the images with the model and log the results.
    Return:
        List of latents, where each is:
            z_pres [ptcs, bs, strks]
            z_what [ptcs, bs, strks, pts, 2]
            z_where [ptcs, bs, strks, 3]
        List of dec_param, where each is:
            sigma, strk_slope [ptcs, bs, strks]
            add_slope [ptcs, bs]
    '''
    if device == None:
        device = 'cpu'
    # Each n_parse is picked from k parses
    k_to_choose_from = 20
    gen, guide = model

    if args.model_type == 'Sequential':
        latents, dec_params, joint_probs, recons, elbos = [], [], [], [], []
        for _ in range(n_parse):
            zs, _, dec_param, recon, joint_prob, elbo = util.get_top_latents_hiddens(
                                        guide, gen, imgs, k_to_choose_from, crit
                                            )
            
            latents.append(zs)
            dec_params.append(dec_param)
            joint_probs.append(joint_prob)
            elbos.append(elbo)
            recons.append(recon.squeeze(0))
        
        re_latents = ZSample(
                            torch.cat([l.z_pres.to(device) for l in latents], dim=0),
                            torch.cat([l.z_what.to(device) for l in latents], dim=0),
                            torch.cat([l.z_where.to(device) for l in latents], dim=0),
                        )
        re_dec_params = DecoderParam(
                sigma=torch.cat([d.sigma.to(device) for d in dec_params], dim=0),
                slope=(
                        torch.cat([d.slope[0].to(device) for d in dec_params], dim=0),
                        torch.cat([d.slope[1].to(device) for d in dec_params], dim=0),
                    )
                )
        re_joint_prob = torch.stack(joint_probs).to(device)
        re_elbos = torch.stack(elbos,dim=0).to(device)
    else: # DAIR case
        recons = None
        re_dec_params = None
        from models.air import ZLogProb
        out = guide(imgs, n_parse)
        generative_model = guide.internal_decoder
        # wheater to mask current value based on prev.z_pres; 
        # more doc in model
        latents, log_post, mask_prev, canvas, z_prior = (
            out.z_smpl, out.z_lprb, 
            out.mask_prev, out.canvas, out.z_prior)
        re_latents = latents
        
        # multi by beta
        log_post_ = {}
        for i, z in enumerate(log_post._fields):
            log_post_[z] = log_post[i]
        log_post = ZLogProb(**log_post_)
        
        # Posterior log probability: [batch_size, max_strks] (before summing)
        log_post_z = torch.cat(
                            [prob.sum(-1, keepdim=True) for prob in log_post], 
                            dim=-1).sum(-1)

        # Prior and Likelihood probability
        # Prior: [batch_size, max_strks]
        # Likelihood: [batch_size]
        log_prior, log_likelihood = generative_model.log_prob(
                                                    latents=latents, 
                                                    imgs=imgs,
                                                    z_pres_mask=mask_prev,
                                                    canvas=canvas,
                                                    z_prior=z_prior,
                                                    )
        log_prior_z = torch.cat(
                            [prob.sum(-1, keepdim=True) for prob in 
                                log_prior], dim=-1).sum(-1)
        re_joint_prob = (log_likelihood + log_prior_z).cpu()

        # Compute ELBO: [bs, ]
        re_elbos = (- log_post_z.cpu() + re_joint_prob)
        recons = generative_model.renders_imgs(latents)
    # Plot
    if writer != None and recons != None:
        imgs = display_trans(imgs.cpu())
        recons = display_trans(recons[0].cpu())
        comparison = torch.cat([imgs, recons], dim=2)
        img_grid = make_grid(comparison, nrow=20)
        # reconss.append(img_grid)
        writer.add_image('One shot classification/'+tag, img_grid, eps)

    return re_latents, re_dec_params, re_joint_prob, re_elbos
                

def fine_tune(args, model, tst_loader, writer, continue_training=True):
    print("==> Being finetuning")
    device = args.device
    gen, guide = model
    # Train
    # check if it has been finetuned
    # pretrained base model ckpt path
    base_checkpoint_path = util.get_checkpoint_path(args)
    # ckpt path for the finetune model
    args.save_model_name = args.save_model_name + '_OneShotClf'
    clf_checkpoint_path = util.get_checkpoint_path(args)

    continue_training = continue_training
    if Path(clf_checkpoint_path).exists() and continue_training:
        model, optimizer, scheduler, stats, _, trn_args = util.load_checkpoint(
                                                path=clf_checkpoint_path,
                                                device=device,
                                                init_data_loader=False)
        if len(stats.tst_elbos) < 50:
            stats.tst_elbos.clear()
    else:
        _, optimizer, scheduler, stats, _, trn_args = util.load_checkpoint(
                                                    path=base_checkpoint_path,
                                                    device=device,
                                                    init_data_loader=False)
        stats.tst_elbos.clear()

    # Fine-tuning
    finetune_ite = 1e5
    finetune_ite = 8e4
    ft_ite_so_far = 0 if not continue_training else len(stats.tst_elbos)
    print(f"==> Finetuned iteration: {ft_ite_so_far}")
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
            if ft_ite_so_far % 1000 == 0 or ft_ite_so_far == finetune_ite:
                with torch.no_grad():
                    plot.plot_reconstructions(imgs=imgs, guide=guide, 
                                      args=args, writer=writer, 
                                      epoch=ft_ite_so_far,
                                      writer_tag='Train', 
                                      dataset_name='OneShot clf finetuning')

                save(args, ft_ite_so_far, model, optimizer, scheduler, stats, 
                     save_ite=4000) 
            ft_ite_so_far += 1
        writer.flush()

    save(args, ft_ite_so_far, model, optimizer, scheduler, stats, save_ite=4000) 
    print("==> Using Fintuned model")
    return model

def averaged_score(log_probs, log_score):
    '''
    log_probs: [ss, particles] joint prob of prior * likelihood
    log_score: [qs, ss, ps]
    '''
    # weights: [1, ss, ps]
    weights = util.normlized_exp(log_probs)[None, ...]
    # score [qs, ss]
    ave_log_score = logsumexp(log_score.detach().numpy(), axis=-1, 
                            b=weights.detach().numpy())
    return torch.tensor(ave_log_score)
    
def get_accuracy(score, label):
    '''
    score [qs, ss]
    label: right from dataloader
    '''
    preds = score.argmax(dim=1) + 1
    trues = label[0, :, 1]
    n_correct = (preds==trues).sum().item()
    accuracy = n_correct/20
    return accuracy

def optimize_and_score(args, model, tst_loader, writer, 
                       n_parse_per_ite=1, run_per_eps=1, two_way_clf=True,
                       tag=None, optimize=False):
    accuracies = []
    if two_way_clf:
        directions = ['q|s', 's|q']
    else:
        directions = ['q|s']
    gen, guide = model

    # For each episode
    for eps, (sup_img, qry_img, label) in tqdm(enumerate(tst_loader)):
        sup_img.squeeze_(0), qry_img.squeeze_(0)
        sup_img = sup_img.to(args.device)
        qry_img = qry_img.to(args.device) 

        # label = label.to(args.device)
        imgs = torch.cat([sup_img, qry_img], dim=0)
        
        # # Parse
        # with torch.no_grad():
        #     gen, guide = model
        #     model = gen.eval(), guide.eval()
        #     latents, dec_params, gen_log_probs = parse(
        #                 imgs, model, writer, eps=eps, n_parse=n_parse_per_ite, 
        #                 tag=tag)
        #     gen_log_probs = gen_log_probs.T # [bs, particals]
        if optimize:
            scores_way = []
            for direction in directions:
                gen_log_probs_run = [] # store the unnorm weights
                log_score_run = [] # store the scores
                elbos_run = []
                for run in range(run_per_eps):
                    # Parse; this is slightly imefficient, but will do 4 now
                    with torch.no_grad():
                        gen, guide = model
                        model = gen.eval(), guide.eval()
                        latents, dec_params, gen_log_probs, elbos = parse(
                                    args,
                                    imgs, model, writer, eps=eps, 
                                    n_parse=n_parse_per_ite, 
                                    tag=tag)
                        elbos = elbos.T
                        gen_log_probs = gen_log_probs.T # [bs, particals]

                    # Create a token model for each n_parse X query image
                    # currently 1 way clf; fit support z to query x
                    cm = ClassifyModel(latents, dec_params, model, 
                                        ps=n_parse_per_ite, 
                                        direction=direction,
                                        model_type=args.model_type,
                                        writer=writer)
                    # Optimizer
                    optim = torch.optim.Adam([{
                                                'params':cm.optim_parameters(),
                                                'lr': 1e-2}]
                                            )
                    # Optimize
                    cm.train()
                    writer.add_image(f'One shot classification/Run{eps} Query',
                            make_grid(display_trans(qry_img), nrow=1))
                    writer.add_image(f'One shot classification/Run{eps} Support',
                            make_grid(display_trans(sup_img), nrow=20))
                    if eps == 0:
                        render_untrans = True
                    else:
                        render_untrans = False

                    for ite in tqdm(range(100)):
                        optim.zero_grad()
                        if direction == 'q|s':
                            in_img = qry_img
                        if direction == 's|q':
                            in_img = sup_img
                        log_score, recons, qrys, untrans_optim_z_plot = cm(in_img, 
                                                        render_untrans=render_untrans,
                                                        ite=ite, writer=writer)

                        m_score = -log_score.mean() # Try to maximize the score
                        m_score.backward()
                        optim.step()
                        
                        # Log
                        writer.add_scalar(f'One shot classification/Run{eps} scores',
                                            m_score, ite)
                        if ite % 100 == 0:
                            disp = make_grid(display_trans(recons[:, 0]), nrow=20)
                            writer.add_image(f'One shot classification/eps{eps} recons0', 
                                            disp, ite)
                            disp = make_grid(display_trans(recons[:, 1]), nrow=20)
                            writer.add_image(f'One shot classification/eps{eps} recons1', 
                                            disp, ite)
                            disp = make_grid(display_trans(qrys[:, 0]), nrow=20)
                            writer.add_image(f'One shot classification/eps{eps} qry_mtrx0', 
                                            disp, ite)
                            disp = make_grid(display_trans(qrys[:, 1]), nrow=20)
                            writer.add_image(f'One shot classification/eps{eps} qry_mtrx1', 
                                            disp, ite)
                            if render_untrans:
                                disp = make_grid(display_trans(untrans_optim_z_plot[:, 0]), nrow=20)
                                writer.add_image(f'One shot classification/eps{eps} untrans_optim_z0', 
                                            disp, ite)
                                disp = make_grid(display_trans(untrans_optim_z_plot[:, 1]), nrow=20)
                                writer.add_image(f'One shot classification/eps{eps} untrans_optim_z1', 
                                            disp, ite)

                    # Score
                    # Compute the weighted one-way score
                    cm.eval()
                    # log_score [qs, ss, ps]
                    # if direction == 'q|s':
                    log_score, _, _, _ = cm(in_img) 
                    # if direction == 's|q':
                    #     log_score, _, _, _ = cm(sup_img) 
                    log_score = log_score.cpu()

                    if direction == 'q|s':
                        gen_log_probs_ = gen_log_probs[:20]
                        elbos_ = elbos[:20]
                    elif direction == 's|q':
                        gen_log_probs_ = gen_log_probs[20:]
                    # weights = util.normlized_exp(gen_log_probs_)[None, ...]
                    # logsumexp prefers larger values compared to direct average
                    # one_score = logsumexp(log_score.detach().numpy(), axis=-1, 
                    #                     b=weights.detach().numpy())
                    
                    gen_log_probs_run.append(gen_log_probs_.detach())
                    elbos_run.append(elbos_.detach())
                    log_score_run.append(log_score.detach())
                    
                    # ===> Intermediate feedback; not used in 2nd way
                    gen_log_probs = torch.cat(gen_log_probs_run, dim=-1)
                    log_score = torch.cat(log_score_run, dim=-1)
                    # score [qs, ss]
                    inter_ave_score = averaged_score(gen_log_probs, log_score)

                    if direction == 'q|s':
                        # Compute a quick one-way score
                        one_w_accuracy = get_accuracy(inter_ave_score, label)
                        print(f"##### eps {eps} run {run}/{run_per_eps-1} accuracy (1-way):", 
                              one_w_accuracy)
                    # ===> Intermediate feedback End

                # Aggregate scores from all runs
                if direction == 'q|s':
                    sup_gen_log_probs = torch.cat(gen_log_probs_run, dim=-1)
                    sup_elbos = torch.cat(elbos_run, dim=-1)
                gen_log_probs = torch.cat(gen_log_probs_run, dim=-1)
                log_score = torch.cat(log_score_run, dim=-1)
                # score [qs, ss]
                one_way_score = averaged_score(gen_log_probs, log_score)

                # End of one-way scoring
                scores_way.append(one_way_score)

                # if two_way_clf and direction == 'q|s':
                #     # Compute a quick one-way score
                #     one_w_preds = scores_way[0].argmax(dim=1) + 1
                #     trues = label[0, :, 1]
                #     one_w_n_correct = (one_w_preds==trues).sum().item()
                #     one_w_accuracy = one_w_n_correct/20
                #     print(f"##### eps {eps} accuracy (1-way):", one_w_accuracy)

            # ===> End two-way scoring

            # Compute final scores
            if two_way_clf:
                # mll: [1, ss]
                # mll = (torch.logsumexp(gen_log_probs[:20], dim=0, keepdim=True) 
                #         - torch.log(torch.tensor(n_parse_per_ite)))
                # [1, ss]
                # breakpoint()
                # img_lld = torch.logsumexp(sup_gen_log_probs, dim=1)[None,]
                ptcs = sup_elbos.shape[-1]
                img_lld = (torch.logsumexp(sup_elbos, dim=1)[None,] -
                            torch.log(torch.tensor(ptcs)))
                # scores[0]: [qs, ss]
                scores = scores_way[0] + scores_way[1].T 
                final_scores = scores - img_lld
            else:
                final_scores = scores_way[0]
                
            # Combining scores from multi runs to get prediction
            preds = final_scores.argmax(dim=1) + 1
            trues = label[0, :, 1]
            n_correct = (preds==trues).sum().item()
            # n_correct = torch.nan_to_num(n_correct)
            accuracy = n_correct/20

            # Log and display score
            if two_way_clf:
                print(f"##### eps {eps} accuracy (2-way):", accuracy)
            else:
                print(f"##### eps {eps} accuracy:", accuracy)
            if args.model_type == 'Sequential':
                final_scores = final_scores.detach().cpu().numpy()
                score_fig = plot.plot_clf_score_heatmap(final_scores, preds, trues)
                writer.add_figure("One shot classification/eps{eps} score", score_fig)
            accuracies.append(accuracy)
    return accuracies

class ClassifyModel(nn.Module):
    def __init__(self, latents, dec_param, model, qs=20, ss=20, ps=1, 
                 direction="q|s", model_type='Sequential', writer=None):
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
        super().__init__()
        self.model_type = model_type
        if model_type == 'Sequential':
            self.gen, _ = model
            z_pr, z_wt, z_wr = latents
            ps, ss, strks, pts, _ = z_wt.shape
            ss = int(ss/2)
            self.z_wr_dim = z_wr.shape[-1]
            # transform z_wt by z_wr
            z_wt = util.transform_z_what(z_wt.view(ps*ss*2, strks, pts, 2),
                                        z_wr.view(ps*ss*2, strks, self.z_wr_dim),
                                        model[0].z_where_type
                                        ).view(ps, ss*2, strks, pts, 2)
            # z_pres [1, ss, ps, strks]
            # z_what [1, ss, ps, strks, pts, 2]
            # z_where [1, 1, 1, 1, z_where_dim]
            z_pr, z_wt, z_wr = (z_pr.transpose(0,1)[None,].cuda(), 
                                z_wt.transpose(0,1)[None,].cuda(), 
                                torch.tensor([0,0,1,0],device=z_wt.device
                                            )[None,None,None,None,:].cuda(),
                                )

            if direction == 'q|s':
                # z_pres [qs, ss, ps, strks]
                # z_what [qs, ss, ps, strks, pts, 2]
                # z_where [qs, ss, ps, strks, z_where_dim]
                self.cond_pr, self.cond_wt, self.cond_wr = (
                                z_pr[:, :20].clone().repeat(qs, 1, 1, 1), 
                                z_wt[:, :20].clone().repeat(qs, 1, 1, 1, 1, 1), 
                                z_wr.clone().repeat(qs, ss, ps, strks, 1)
                                    )
                optm_sigma, optm_strk_tanh, optm_add_tanh = (
                    dec_param[0][None, :, :20].clone().transpose(1,2).repeat(qs, 1, 1, 1),
                    dec_param[1][0][None, :, :20].clone().transpose(1,2).repeat(qs, 1, 1, 1),
                    dec_param[1][1][None, :, :20, 0].clone().transpose(1,2).repeat(qs, 1, 1),
                )
                
            elif direction == 's|q':
                self.cond_pr, self.cond_wt, self.cond_wr = (
                                z_pr[:, 20:].clone().repeat(qs, 1, 1, 1), 
                                z_wt[:, 20:].clone().repeat(qs, 1, 1, 1, 1, 1), 
                                z_wr.clone().repeat(qs, ss, ps, strks, 1)
                                    )
                optm_sigma, optm_strk_tanh, optm_add_tanh = (
                    dec_param[0][None, :, 20:].clone().transpose(1,2).repeat(qs, 1, 1, 1),
                    dec_param[1][0][None, :, 20:].clone().transpose(1,2).repeat(qs, 1, 1, 1),
                    dec_param[1][1][None, :, 20:, 0].clone().transpose(1,2).repeat(qs, 1, 1),
                )
                
            else:
                raise NotImplementedError

            self.optm_pr, self.optm_wt, self.optm_wr = (
                                                self.cond_pr.detach().clone(), 
                                                self.cond_wt.detach().clone(), 
                                                self.cond_wr.detach().clone())
            self.cond_wt_std = torch.zeros_like(self.cond_wt) + 0.001
            # self.cond_wt_std = torch.zeros_like(self.cond_wt) + 0.0001
            self.optm_wt = torch.nn.Parameter(self.optm_wt, requires_grad=True)
            # Global affine transformations
            self.affines_param_ = torch.nn.Parameter(torch.zeros([qs, ss, ps, 7], 
                                            device=z_wt.device), requires_grad=True)
            self.gen.sigma = optm_sigma.view(qs*ss, ps, strks).cuda()
            self.gen.sgl_strk_tanh_slope = optm_strk_tanh.view(qs*ss, ps, strks).cuda()
            self.gen.add_strk_tanh_slope = optm_add_tanh.view(qs*ss, ps).cuda()
        elif model_type == 'AIR':
            _, guide = model
            self.gen = guide.internal_decoder
            z_pr, z_wt, z_wr = latents
            ps, ss, strks, pts = z_wt.shape
            ss = int(ss/2)
            self.z_wr_dim = z_wr.shape[-1]
            # transform z_wt by z_wr
            # z_wt = util.transform_z_what(z_wt.view(ps*ss*2, strks, pts),
            #                             z_wr.view(ps*ss*2, strks, self.z_wr_dim),
            #                             model[0].z_where_type
            #                             ).view(ps, ss*2, strks, pts)
            # z_pres [1, ss, ps, strks]
            # z_what [1, ss, ps, strks, pts, 2]
            # z_where [1, 1, 1, 1, z_where_dim]
            z_pr, z_wt, z_wr = (z_pr.transpose(0,1)[None,].cuda(), 
                                z_wt.transpose(0,1)[None,].cuda(), 
                                z_wr.transpose(0,1)[None,].cuda(), 
                                )

            # z_pres [qs, ss, ps, strks]
            # z_what [qs, ss, ps, strks, pts, 2]
            # z_where [qs, ss, ps, strks, z_where_dim]
            self.cond_pr, self.cond_wt, self.cond_wr = (
                            z_pr[:, :20].clone().repeat(qs, 1, 1, 1), 
                            z_wt[:, :20].clone().repeat(qs, 1, 1, 1, 1), 
                            z_wr[:, :20].clone().repeat(qs, 1, 1, 1, 1), 
                            )
            # breakpoint()
            # writer.add_image(f'One shot classification/debug',make_grid(
            #     self.gen.renders_imgs((z_pr[0], z_wt[0], z_wr[0]))[:,0], nrow=20))
            # writer.add_image(f'One shot classification/debug',make_grid(
            #     self.gen.renders_imgs((self.cond_pr[0], self.cond_wt[0], self.cond_wr[0]))[:,0], nrow=20))           
                

            self.optm_pr, self.optm_wt, self.optm_wr = (
                                                self.cond_pr.detach().clone(), 
                                                self.cond_wt.detach().clone(), 
                                                self.cond_wr.detach().clone())
            self.cond_wt_std = torch.zeros_like(self.cond_wt) + 0.001
            # self.cond_wt_std = torch.zeros_like(self.cond_wt) + 0.0001
            self.optm_wt = torch.nn.Parameter(self.optm_wt, requires_grad=True)
            # Global affine transformations
            self.affines_param_ = torch.nn.Parameter(torch.zeros([qs, ss, ps, 7], 
                                        device=z_wt.device), requires_grad=True)

    def get_affines(self):
        shift = util.constrain_parameter(self.affines_param_[...,:2], -.2, .2)
        scale = util.constrain_parameter(self.affines_param_[...,2:4], .6, 1.4)
        rot = util.constrain_parameter(self.affines_param_[...,4:5], 
                                                        -.25*math.pi, .25*math.pi)
        shear = util.constrain_parameter(self.affines_param_[...,5:7], 
                                                        -.3*math.pi, .3*math.pi)
        return torch.cat([shift,scale,rot,shear], dim=3)
        # shift = util.constrain_parameter(self.affines_param_[...,:2], -.1, .1)
        # scale = util.constrain_parameter(self.affines_param_[...,2:4], .8, 1.2)
        # rot = util.constrain_parameter(self.affines_param_[...,4:5], -.1*math.pi, .1*math.pi)
        # shear = util.constrain_parameter(self.affines_param_[...,5:7], -.1*math.pi, .1*math.pi)
        # return torch.cat([shift,scale,rot,shear], dim=3)
    
    def get_cond_token_dist(self):
        if self.model_type == 'Sequential':
            qs, ss, ps, strks, pts, _ = self.cond_wt.shape
            # affine_params = self.get_affines()[..., None, :].repeat(1,1,1,strks,1)
            
            # self.trans_cond = util.transform_z_what(
            #                         self.cond_wt.view(qs*ss*ps, strks, pts, 2),
            #                         affine_params.view(qs*ss*ps, strks, 7),
            #                         '7',                                    
            #                     ).view(qs, ss, ps, strks, pts, 2)
            token_dist = Independent(
                                Normal(self.cond_wt, self.cond_wt_std),
                                reinterpreted_batch_ndims=2
                            )
        elif self.model_type == 'AIR':
            qs, ss, ps, strks, pts = self.cond_wt.shape
            # affine_params = self.get_affines()[..., None, :].repeat(1,1,1,strks,1)
            
            # self.trans_cond = util.transform_z_what(
            #                         self.cond_wt.view(qs*ss*ps, strks, pts, 2),
            #                         affine_params.view(qs*ss*ps, strks, 7),
            #                         '7',                                    
            #                     ).view(qs, ss, ps, strks, pts, 2)
            token_dist = Independent(
                                Normal(self.cond_wt, self.cond_wt_std),
                                reinterpreted_batch_ndims=1)
        return token_dist

    def forward(self, qry_img, render_untrans=False, ite=None, writer=None):
        '''
            score [qs, ss, ps]
        '''
        if self.model_type == 'Sequential':
            # Init
            qs, ss, ps, strks, pts, _ = self.optm_wt.shape
            wr_dim = self.optm_wr.shape[-1]

            # Compute the token model log_prob; averaged by the number of steps
            strks_per_samp = self.cond_pr.sum(-1) # [qs, ss, ps]
            token_dist= self.get_cond_token_dist()
            token_score = token_dist.log_prob(self.optm_wt)
            # out: [qs, ss, ps], in: [qs, ss, ps, n_strokes]
            token_score = (token_score * self.cond_pr).sum(-1) / strks_per_samp
            
            # Compute the image likelihood
            untrans_optim_z_plot = None
            if render_untrans and ite % 100 == 0:
                # For plotting
                with torch.no_grad():
                    untrans_optim_z_plot = self.gen.renders_imgs((
                                        self.optm_pr.view(qs*ss,ps,strks),
                                        self.optm_wt.view(qs*ss,ps,strks,pts,2),
                                        self.optm_wr.view(qs*ss,ps,strks,wr_dim),
                                        )).cpu()

            # trans z'
            # qs, ss, ps, strks, pts, _ = self.cond_wt.shape
            affine_params = self.get_affines()[..., None, :].repeat(1,1,1,strks,1)
            self.trans_optm_wt = util.transform_z_what(
                                    self.optm_wt.view(qs*ss*ps, strks, pts, 2),
                                    affine_params.view(qs*ss*ps, strks, 7),
                                    '7',                                    
                                ).view(qs, ss, ps, strks, pts, 2).clone()
            recons = self.gen.renders_imgs((
                                    self.optm_pr.view(qs*ss,ps,strks),
                                    self.trans_optm_wt.view(qs*ss,ps,strks,pts,2),
                                    self.optm_wr.view(qs*ss,ps,strks,wr_dim),
                                        ))
            img_shp = qry_img.shape[1:]
            # qry_img = qry_img.repeat(qs*ps,1,1,1).view(qs,ss,ps,*img_shp
            #                         ).transpose(0,1).reshape(qs*ss,ps,*img_shp)
            qry_img = qry_img[:,None,None,...].repeat(1,ss,ps,1,1,1
                                                    ).view(qs*ss,ps,*img_shp)
            img_dist = self.gen.img_dist(canvas=recons)
            recons = recons.cpu()
            image_score = img_dist.log_prob(qry_img).view(qs,ss,ps)

            # Return the scores for each optimize on each conditioning variable
            # [qs, ss, ps]
            log_score = token_score + image_score
            # print(f"token_score: {token_score.mean().item():.2f} " + 
            #       f"image_score: {image_score.mean().item():.2f} " +
            #       f"sum score: {log_score.mean().item():.2f}")
        elif self.model_type == 'AIR':
            # Init
            qs, ss, ps, strks, pts = self.optm_wt.shape
            img_shp = qry_img.shape[1:]
            wr_dim = self.optm_wr.shape[-1]


            self.optm_wt = torch.nn.Parameter(
                torch.nan_to_num(self.optm_wt), 
                requires_grad=True)
            # Compute the token model log_prob; averaged by the number of steps
            strks_per_samp = self.cond_pr.sum(-1) # [qs, ss, ps]
            token_dist= self.get_cond_token_dist()
            token_score = token_dist.log_prob(self.optm_wt)
            # out: [qs, ss, ps], in: [qs, ss, ps, n_strokes]
            token_score = (token_score * self.cond_pr).sum(-1) / strks_per_samp
            
            # Compute the image likelihood
            untrans_optim_z_plot = None

            if render_untrans and ite % 100 == 0:
                # For plotting
                with torch.no_grad():
                    untrans_optim_z_plot = self.gen.renders_imgs((
                                    self.optm_pr.view(qs*ss,ps,strks),
                                    self.optm_wt.view(qs*ss,ps,strks,pts),
                                    self.optm_wr.view(qs*ss,ps,strks,wr_dim)
                                    )).cpu()

            # trans z'
            # qs, ss, ps, strks, pts, _ = self.cond_wt.shape
            affine_params = self.get_affines()
            recons = self.gen.renders_imgs((
                                    self.optm_pr.view(qs*ss,ps,strks),
                                    self.optm_wt.view(qs*ss,ps,strks,pts),
                                    self.optm_wr.view(qs*ss,ps,strks,wr_dim)))
            # trans_recon
            af_mtrx = util.get_affine_matrix_from_param(
                                        affine_params.view(qs*ss*ps,-1),
                                        '4_rotate')
            trans_recons = util.inverse_spatial_transformation(
                                        recons.view(qs*ss*ps, *img_shp), af_mtrx)
            trans_recons = trans_recons.view(qs*ss,ps,*img_shp)
            img_shp = qry_img.shape[1:]
            # qry_img = qry_img.repeat(qs*ps,1,1,1).view(qs,ss,ps,*img_shp
            #                         ).transpose(0,1).reshape(qs*ss,ps,*img_shp)
            qry_img = qry_img[:,None,None,...].repeat(1,ss,ps,1,1,1
                                                    ).view(qs*ss,ps,*img_shp)
            recons = torch.clip(recons, min=0.,max=1.)
            qry_img = torch.clip(qry_img, min=0.,max=1.)
            img_dist = self.gen.img_dist(canvas=recons)
            recons = recons.cpu()
            image_score = img_dist.log_prob(qry_img).view(qs,ss,ps)

            # Return the scores for each optimize on each conditioning variable
            # [qs, ss, ps]
            log_score = token_score + image_score
            # print(f"token_score: {token_score.mean().item():.2f} " + 
            #       f"image_score: {image_score.mean().item():.2f} " +
            #       f"sum score: {log_score.mean().item():.2f}")
        return log_score, recons, qry_img, untrans_optim_z_plot
    
    def optim_named_parameters(self):
        for n, p in self.named_parameters():
            if n.split(".")[0] != 'gen':
                yield n,p
    def optim_parameters(self):
        for _, p in self.optim_named_parameters():
            yield p

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
