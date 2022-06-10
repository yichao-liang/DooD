
strokes_per_img = '6'

'''
Our model = AIR 
        + sequential prior (with 
                    a) explicit conditioning or either wr|wt or wt|wr;
                    b) GMM prior)
        + spline decoder
        + intermediate rendering (canvas, residual, residual pixel ratio)
        (
            + separated z_{where, pre} and z_what
            + 4-dim z_where vs 3 in AIR
            + input target at MLP, as opposed to RNN, for more sensible 
                generative model
            + separated z_pres, z_where nets
            + no RNN for z_pres 
            + more constrained latent distribution
            + learnable observation std
            + capability with multi-sample learning/evaluation
            + ability to transform z_what by z_where then render vs having to
                render out z_what and transform with z_where through inverse STN
        )
'''
full_config = [
                'sequential_prior',  
                'spline_decoder',  
                'canvas',  
                'seperated_z',
            ]

models_2_cmd = {
    'VAE': [ 
        '--model_type', 'VAE',
        '--lr', '1e-4',
    ],
    'AIR': [
        '--model_type', 'AIR',
        '--prior_dist', 'Independent',
        '--lr', '1e-4', 
        '--bl_lr', '1e-3',
        '--z_where_type', '3',
        '--strokes_per_img', strokes_per_img,
        '--z_what_in_pos', 'z_where_rnn',
        '--target_in_pos', 'RNN',
        '--save_history_ckpt',
    ],
    'AIR4_Lapl': [
        '--model_type', 'AIR',
        '--prior_dist', 'Independent',
        '--lr', '1e-4', 
        '--bl_lr', '1e-3',
        '--z_where_type', '4_rotate',
        '--strokes_per_img', strokes_per_img,
        '--z_what_in_pos', 'z_where_rnn',
        '--target_in_pos', 'RNN',
        '--save_history_ckpt',
        '--log_param',
    ],
    'AIR4_Gaus': [
        '--model_type', 'AIR',
        '--prior_dist', 'Independent',
        '--lr', '1e-4', 
        '--bl_lr', '1e-3',
        '--z_where_type', '4_rotate',
        '--strokes_per_img', strokes_per_img,
        '--z_what_in_pos', 'z_where_rnn',
        '--target_in_pos', 'RNN',
        '--save_history_ckpt',
        '--z_dim', '10',
        "--likelihood_dist", 'Normal',
        '--log_param',
    ],
    'AIR4_50': [
        '--model_type', 'AIR',
        '--prior_dist', 'Independent',
        '--lr', '1e-4', 
        '--bl_lr', '1e-3',
        '--z_where_type', '4_rotate',
        '--strokes_per_img', strokes_per_img,
        '--z_what_in_pos', 'z_where_rnn',
        '--target_in_pos', 'RNN',
        '--save_history_ckpt',
        '--z_dim', '50',
    ],
    'AIR4_50G': [
        '--model_type', 'AIR',
        '--prior_dist', 'Independent',
        '--lr', '1e-4', 
        '--bl_lr', '1e-3',
        '--z_where_type', '4_rotate',
        '--strokes_per_img', strokes_per_img,
        '--z_what_in_pos', 'z_where_rnn',
        '--target_in_pos', 'RNN',
        '--save_history_ckpt',
        '--z_dim', '50',
        "--likelihood_dist", 'Normal',
    ],
    'AIR4_50_mlp': [
        '--model_type', 'AIR',
        '--prior_dist', 'Independent',
        '--lr', '1e-4', 
        '--bl_lr', '1e-3',
        '--z_where_type', '4_rotate',
        '--strokes_per_img', strokes_per_img,
        '--z_what_in_pos', 'z_where_rnn',
        '--target_in_pos', 'RNN',
        '--save_history_ckpt',
        '--z_dim', '50',
        '--feature_extractor_type', 'MLP',
    ],
    'DAIR_Lapl': [
        '--model_type', 'AIR',
        '--prior_dist', 'Independent',
        '--lr', '1e-4', 
        '--bl_lr', '1e-3',
        '--z_where_type', '4_rotate',
        '--strokes_per_img', strokes_per_img,
        '--z_what_in_pos', 'z_where_rnn',
        '--target_in_pos', 'RNN',
        '--use_residual',
        # '--use_canvas',
        '--residual_no_target',
        '--save_history_ckpt',
        # '--batch-size', '128',
        '--log_param',
    ],
    'DAIR_Gaus': [
        '--model_type', 'AIR',
        '--prior_dist', 'Independent',
        '--lr', '1e-4', 
        '--bl_lr', '1e-3',
        '--z_where_type', '4_rotate',
        '--strokes_per_img', strokes_per_img,
        '--z_what_in_pos', 'z_where_rnn',
        '--target_in_pos', 'RNN',
        '--use_residual',
        # '--use_canvas',
        '--residual_no_target',
        '--save_history_ckpt',
        '--log_param',
        "--likelihood_dist", 'Normal',
    ],
    'DAIR50': [
        '--model_type', 'AIR',
        '--prior_dist', 'Independent',
        '--lr', '1e-4', 
        '--bl_lr', '1e-3',
        '--z_where_type', '4_rotate',
        '--strokes_per_img', strokes_per_img,
        '--z_what_in_pos', 'z_where_rnn',
        '--target_in_pos', 'RNN',
        '--use_residual',
        # '--use_canvas',
        '--residual_no_target',
        '--save_history_ckpt',
        '--z_dim', 50,
    ],
    'MWS': [
        '--model_type', 'MWS',
    ],
    'AIR+seq_prir': [
        '--model_type', 'AIR',
        '--prior_dist', 'Sequential',
        '--lr', '1e-4', 
        '--bl_lr', '1e-3',
        '--z_where_type', '3',
        '--strokes_per_img', strokes_per_img,
        '--z_what_in_pos', 'z_where_rnn',
        '--target_in_pos', 'RNN'
    ]
}

def args_from_kw_list(kw_config):
    '''Get the arguments for running the full model/ablations from keywords
    '''
    args = []
    args.extend(['--model_type', 'Sequential'])
    # --- Prior distribution:
    args.append('--prior_dist')
    if 'sequential_prior' in kw_config:
        args.extend(['Sequential',
                     '--dependent_prior'])
    else:
        args.extend(['Independent'])
    # --- Spline decoder: 
    if 'spline_decoder' in kw_config:
        pass
    else:
        args.extend([
            '--no_spline_renderer',
            '--no_maxnorm',
            '--no_sgl_strk_tanh',
            ])
    # --- Intermediate rendering:
    if 'canvas' in kw_config:
        args.extend([
            '--use_canvas', '--detach_canvas_so_far',
            '--use_residual', '--residual_pixel_count', '--detach_rsd_embed',
            ])
    # --- z_what in position:
    args.append('--z_what_in_pos')
    if 'seperated_z' in kw_config:
        args.append('z_what_rnn')
    else:
        args.append('z_where_rnn')
    # --- Extra terms:
    args.extend(['--anneal_lr', '--anneal_non_pr_net_lr',
                 '--no_pres_rnn',
                 '--update_reinforce_loss',
                 '--sep_where_pres_net',
                 '--save_history_ckpt',
                 ])
    return args

def ablation_args():
    all_args = {}
    for ablation in full_config:
        exp_config = full_config.copy()
        exp_config.remove(ablation)
        args = args_from_kw_list(exp_config)
        # args.append("--no_maxnorm")
        all_args[f"Full-{ablation}"] = args

    return all_args


# Record all experiment configs
# full_model_args = args_from_kw_list(full_config)
# i.e. ['--prior_dist', 'Sequential', 
    #   '--model_type', 'Sequential', 
    #   '--z_what_in_pos', 'z_what_rnn',
    #   '--use_canvas']
basic_full_model = ['--prior_dist', 'Sequential', 
                    '--model_type', 'Sequential', 
                    '--use_canvas', 
                    '--z_what_in_pos', 'z_what_rnn']
full_no_canvas = basic_full_model.copy()
full_no_canvas.remove('--use_canvas')

exp_dict = {
    # MNIT: no image
    'Full-spDec-sq40MCorPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            # '--condition_by_img', #Imc
            '--transform_z_what', #tranWhat
            '--residual_no_target_pres',
         ],

    # Mar 30
    # 20MT
    'Full-spDec-sq1MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '1', #20M
            '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            '--transform_z_what', #tranWhat
            '--residual_no_target_pres',
         ],
    # 1MNCT
    'Full-spDec-sq1MImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '1', #20M
            # '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            '--transform_z_what', #tranWhat
            '--residual_no_target_pres',
         ],
    # MNCT
    'Full-spDec-sq40MImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            # '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            '--transform_z_what', #tranWhat
            '--residual_no_target_pres',
         ],
 
    # MTnT
    'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-noTgt-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            '--transform_z_what', #tranWhat
         ],
    # MTT
    'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnnTg-normRfLoss-anNonPrLr-lapl-tranWhat-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            '--transform_z_what', #tranWhat
            # '--residual_no_target_pres',
         ],
    # MTL
    'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEmL-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            '--transform_z_what', #tranWhat
            '--residual_no_target_pres',
            '--img_feat_dim', '512',
         ],
    # M
    'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            # '--transform_z_what', #tranWhat
            '--residual_no_target_pres',         
            '--img_feat_dim', '256',
            ],
    # Ma1: ablation 1: full - sequential prior
    'Full-spDec-indPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk': basic_full_model +\
        [
            '--prior_dist', 'Independent',
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            # '--transform_z_what', #tranWhat
            '--residual_no_target_pres',         
            '--img_feat_dim', '256',
            ],
    # Ma2: ablation 2: full - spline decoder
    # with nn_dec, maxnorm and sgl_strk norm are disabled for it to work
    'Full-nnDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            # '--transform_z_what', #tranWhat
            '--residual_no_target_pres',         
            '--img_feat_dim', '256',
            '--no_spline_renderer',
            # '--no_maxnorm', # seem necessary for no spline_dec model 
            # '--no_sgl_strk_tanh', # seem necessary for no spline_dec model
            # '--no_add_strk_tanh',
            ], 
    # Ma3: ablation 3: full - intermediate rendering
    'Full-spDec-sq40MCorPrior-dp-tr-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk': full_no_canvas +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            # '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            # '--use_residual', # detachRsdNotRsdEm
            # '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            # '--condition_by_img', #Imc
            # '--transform_z_what', #tranWhat
            # '--residual_no_target_pres',         
            '--img_feat_dim', '256',
            ],
    # MS
    'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEmNoShrg-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            # '--transform_z_what', #tranWhat
            '--residual_no_target_pres',         
            '--img_feat_dim', '256',
            '--no_feature_extractor_sharing',
            ],
    # MT
    'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            '--transform_z_what', #tranWhat
            '--residual_no_target_pres',
            '--img_feat_dim', '256',
            # '--batch-size', '128',
         ],
    # MTa1: ablation 1: full - sequential prior
    'Full-spDec-indPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk': basic_full_model +\
        [
            '--prior_dist', 'Independent',
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            '--transform_z_what', #tranWhat
            '--residual_no_target_pres',         
            '--img_feat_dim', '256',
            ],
    # MTa3: ablation 3: full - intermediate rendering
    'Full-spDec-sq40MCorPrior-dp-tr-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk': full_no_canvas +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            # '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            # '--use_residual', # detachRsdNotRsdEm
            # '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            # '--condition_by_img', #Imc
            '--transform_z_what', #tranWhat
            # '--residual_no_target_pres',         
            '--img_feat_dim', '256',
            ],
    # MTmlp
    'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEmMlp-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            # '--condition_by_img', #Imc
            '--transform_z_what', #tranWhat
            '--residual_no_target_pres',
            '--img_feat_dim', '256',
            # '--batch-size', '128',
            '--feature_extractor_type', 'MLP',
         ],
    # MTS
    'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEmNoShrg-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            '--transform_z_what', #tranWhat
            '--residual_no_target_pres',
            '--no_feature_extractor_sharing',
         ],
    # MT5
    'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPr5WrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            '--transform_z_what', #tranWhat
            '--residual_no_target_pres',
            '--z_where_type', '5',
         ],
    # 20MT
    'Full-spDec-sq20MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '20', #20M
            '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            '--transform_z_what', #tranWhat
            '--residual_no_target_pres',
         ],
    # MnT
    'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-noTgt-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'MNIST',
            '--linear_sum',
            '--num_mixtures', '40', #20M
            '--correlated_latent', #Cor
            '--condition_by_img', #Imc
            # '--transform_z_what', #tranWhat
            '--residual_no_target_pres',         
            ],
    # Mar 28
    'Full-spDec-sq4MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-om': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'Omniglot',
            '--linear_sum',
            '--num_mixtures', '4',
            '--correlated_latent',
            '--condition_by_img',
         ],

    # Mar 27
    # om: 20 mixture, bezier curve rnn
    'Full-spDec-sq50MCorImcPrior-bzRnn-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-om': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'Omniglot',
            '--linear_sum',
            '--num_mixtures', '50',
            '--correlated_latent',
            '--condition_by_img',
            '--use_bezier_rnn',
         ],
    'Full-spDec-sq50MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-om': basic_full_model +\
    [
        '--anneal_lr', # anNonPrLr
        '--anneal_non_pr_net_lr', # anNonPrLr
        '--log_param',
        '--detach_canvas_so_far', # detachRsdNotRsdEm
        # '--no_pres_rnn',
        '--no_pres_post_rnn', # noPrPosRnn

        '--use_residual', # detachRsdNotRsdEm
        '--residual_pixel_count', # detachRsdNotRsdEm
        # '--detach_rsd_embed', #detachRsdNotRsdEm
        '--update_reinforce_loss', # normRfLoss
        '--sep_where_pres_net', # sepPrWrNet
        '--dependent_prior', # dp
        '--prior_dependency', 'wt|wr', # t|r
        # '--save_history_ckpt',
        '--strokes_per_img', '6', # 6strk
        # '--residual_no_target', # noTarget
        # '--no_what_post_rnn', # noWtPrPosRnn
        '--dataset', 'Omniglot',
        '--linear_sum',
        '--num_mixtures', '50',
        '--correlated_latent',
        '--condition_by_img',
        ],
    # om: 20 mixture
    'Full-spDec-sq20MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk-om': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'Omniglot',
            '--linear_sum',
            '--num_mixtures', '20',
            '--correlated_latent',
            '--condition_by_img',
            '--transform_z_what',
         ],
    # Mar 26
    # omBrT
    'Full-spDec-sq20MCorImcPrior-bzRnn-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk-om': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'Omniglot',
            '--linear_sum',
            '--num_mixtures', '20',
            '--correlated_latent',
            '--condition_by_img',
            '--use_bezier_rnn',
            '--transform_z_what',
         ],
    # mnImT
    'Full-spDec-sq20MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-45strk-mn': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '4', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--linear_sum',
            '--num_mixtures', '20',
            '--correlated_latent',
            '--condition_by_img',
            '--transform_z_what',
         ],
    # Mar 25
    # om: 20 mixture, bezier curve rnn
    'Full-spDec-sq20MCorImcPrior-bzRnn-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-om': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'Omniglot',
            '--linear_sum',
            '--num_mixtures', '20',
            '--correlated_latent',
            '--condition_by_img',
            '--use_bezier_rnn',
         ],
    # Mar 24
    # om: 20 mixture
    'Full-spDec-sq20MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-om': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'Omniglot',
            '--linear_sum',
            '--num_mixtures', '20',
            '--correlated_latent',
            '--condition_by_img',
         ],
    # Mar 22
    # image cond 10m
    'Full-spDec-sq20MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-45strk-mn': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '4', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--linear_sum',
            '--num_mixtures', '20',
            '--correlated_latent',
            '--condition_by_img',
         ],
    # image conditioned corr
    'Full-spDec-sq4MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-45strk-mn': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '4', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--linear_sum',
            '--num_mixtures', '4',
            '--correlated_latent',
            '--condition_by_img',
         ],
    # spline+img rnn
    'Full-spDec-sq20MCorImcPrior-bzRnn-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-45strk-mn': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '4', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--linear_sum',
            '--num_mixtures', '20',
            '--correlated_latent',
            '--use_bezier_rnn',
            '--condition_by_img',
         ],
    # spline rnn
    'Full-spDec-sqMCorPrior-bzRnn-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-45strk-mn': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '4', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--linear_sum',
            '--num_mixtures', '4',
            '--correlated_latent',
            '--use_bezier_rnn',
         ],
    # cor
    'Full-spDec-sqMCorPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-45strk-mn': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '4', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--linear_sum',
            '--num_mixtures', '4',
            '--correlated_latent',
         ],
    # Mar 20
    # gmm mnist
    'Full-spDec-sqMPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-45strk-mn': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '4', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--linear_sum',
            '--num_mixtures', '4',
         ],
    'Full-spDec-sqMPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-mn': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--linear_sum',
            '--num_mixtures', '4',
         ],
    'Full-spDec-sqMPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-om': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'Omniglot',
            '--linear_sum',
            '--num_mixtures', '4',
         ],
    'Full-spDec-sqMPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-km': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'KMNIST',
            '--linear_sum',
            '--num_mixtures', '4',
         ],
    'Full-spDec-sqMPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-em': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'EMNIST',
            '--linear_sum',
            '--num_mixtures', '4',
         ],
    'Full-spDec-sqMPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-qd': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--dataset', 'Quickdraw',
            '--linear_sum',
            '--num_mixtures', '4',
         ],
    # Mar 19
    'Full-spDec-sqPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-em': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            # "--z_where_type", '5', #5wr
            '--dataset', 'EMNIST',
            # '--lr', '1e-4',
            '--linear_sum',
         ],
    'Full-spDec-sqPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-km': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            # "--z_where_type", '5', #5wr
            '--dataset', 'KMNIST',
            # '--lr', '1e-4',
            '--linear_sum',
         ],
    'Full-spDec-sqPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-qd': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            # "--z_where_type", '5', #5wr
            '--dataset', 'Quickdraw',
            # '--lr', '1e-4',
            '--linear_sum',
         ],
    # Mar 16
    # om_tr
    'Full-spDec-sqPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-om': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            # "--z_where_type", '5', #5wr
            '--dataset', 'Omniglot',
            # '--lr', '1e-4',
            '--linear_sum',
         ],
    # om5
    'Full-spDec-sqPrior-dp-5wr-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-omni': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            "--z_where_type", '5', #5wr
            '--dataset', 'Omniglot',
            # '--lr', '1e-4',
         ],
    # omv2
    'Full-spDec-sqPrior-dp-tr-detachRsdNotRsdEm-noTgt-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-omni': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            # "--z_where_type", '5', #5wr
            '--dataset', 'Omniglot',
            # '--lr', '1e-4',
         ],
    # omv3
    'Full-spDec-sqPrior-dp-tr-detachRsdNotRsdEm-noTgt-sepPrWrNet-noWtPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-omni': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            '--residual_no_target', # noTarget
            '--no_what_post_rnn', # noWtPrPosRnn
            # "--z_where_type", '5', #5wr
            '--dataset', 'Omniglot',
            # '--lr', '1e-4',
         ],
    'Full-spDec-sqPrior-dp-rt-detachRsdNotRsdEm-noTgt-sepPrWrNet-noWtPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '4', # 6strk
            '--residual_no_target', # noTarget
            '--no_what_post_rnn', # noWtPrPosRnn
            '--linear_sum',
            # "--z_where_type", '5', #5wr
            # '--dataset', 'Omniglot',
            # '--lr', '1e-4',
         ],
    # Mar 9
    # v3.1: based on v0.1, uses continousBernoulli img dist; β9
    'Full-spDec-sqPrior-dp-rt-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-bern-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wr|wt', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--bern_img_dist', # bern
         ],
    # Mar 6
    # v2.1: train directly on omniglot
    'Full-spDec-sqPrior-dp-rt-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-em': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wr|wt', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            # "--z_where_type", '5', #5wr
            '--dataset', 'EMNIST',
            # '--lr', '1e-4',
         ],
    'Full-spDec-sqPrior-dp-rt-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-km': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wr|wt', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            # "--z_where_type", '5', #5wr
            '--dataset', 'KMNIST',
            # '--lr', '1e-4',
         ],
    'Full-spDec-sqPrior-dp-rt-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-qd': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wr|wt', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            # "--z_where_type", '5', #5wr
            '--dataset', 'Quickdraw',
            # '--lr', '1e-4',
         ],
    'Full-spDec-sqPrior-dp-rt-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-omni': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wr|wt', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            # "--z_where_type", '5', #5wr
            '--dataset', 'Omniglot',
            # '--lr', '1e-4',
         ],
    # 2.1.2
    'Full-spDec-sqPrior-dp-rt-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-omni-consPr': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wr|wt', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            # "--z_where_type", '5', #5wr
            '--dataset', 'Omniglot',
            '--constrain_z_pres_param',
            # '--lr', '1e-4',
         ],
    # v2.2
    'Full-spDec-sqPrior-dp-5wr-detachRsdNotRsdEm-noTarget-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-6strk-omni': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            # '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            "--z_where_type", '5', #5wr
            '--dataset', 'Omniglot',
         ],
    # v2.3
    'Full-spDec-sqPrior-dp-5wr-detachRsdNotRsdEm-noTarget-sepPrWrNet-noWtPrPosRnn-normRfLoss-anNonPrLr-6strk-omni': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            # '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            '--residual_no_target', # noTarget
            '--no_what_post_rnn', # noWtPrPosRnn
            "--z_where_type", '5', #5wr
            '--dataset', 'Omniglot',
         ],
    # Mar 5
    # v1.1 # use 5dim z_where
    'Full-spDec-sqPrior-dp-5wr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            # '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wr|wt', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            "--z_where_type", '5', #5wr
         ],
    # v1.1 44strk
    'Full-spDec-sqPrior-dp-5wr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-44strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wr|wt', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '4', # 4strk
            "--points-per-stroke", '4',
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            "--z_where_type", '5', #5wr
         ],
    # v1.2
    'Full-spDec-sqPrior-dp-5wr-detachRsdNotRsdEm-noTarget-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            # '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            "--z_where_type", '5', #5wr
         ],
    # v1.3
    'Full-spDec-sqPrior-dp-5wr-detachRsdNotRsdEm-noTarget-sepPrWrNet-noWtPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            # '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            '--residual_no_target', # noTarget
            '--no_what_post_rnn', # noWtPrPosRnn
            "--z_where_type", '5', #5wr
         ],
    # current best model v0.1; β3
    'Full-spDec-sqPrior-dp-rt-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
         ],
    # reverse dependency
    'Full-spDec-sqPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-mn': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            # '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
            '--linear_sum',
         ],
    # v0.2; β2
    'Full-spDec-sqPrior-dp-tr-detachRsdNotRsdEm-noTarget-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            # '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            '--residual_no_target', # noTarget
            # '--no_what_post_rnn', # noWtPrPosRnn
         ],
    # v0.3; β2
    'Full-spDec-sqPrior-dp-tr-detachRsdNotRsdEm-noTarget-sepPrWrNet-noWtPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk': basic_full_model +\
        [
            '--anneal_lr', # anNonPrLr
            '--anneal_non_pr_net_lr', # anNonPrLr
            # '--log_param',
            '--detach_canvas_so_far', # detachRsdNotRsdEm
            # '--no_pres_rnn',
            '--no_pres_post_rnn', # noPrPosRnn

            '--use_residual', # detachRsdNotRsdEm
            '--residual_pixel_count', # detachRsdNotRsdEm
            # '--detach_rsd_embed', #detachRsdNotRsdEm
            '--update_reinforce_loss', # normRfLoss
            '--sep_where_pres_net', # sepPrWrNet
            '--dependent_prior', # dp
            '--prior_dependency', 'wt|wr', # t|r
            # '--save_history_ckpt',
            '--strokes_per_img', '6', # 6strk
            '--residual_no_target', # noTarget
            '--no_what_post_rnn', # noWtPrPosRnn
         ],
    # Mar 4
    'Full-spDec-sqPrior-dp-useDetachRsd-sepPrWrNet-noPosPrRnn-normRfLoss-anNonPrLr-6strk': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            # '--no_pres_rnn',
            '--no_pres_post_rnn',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
            '--dependent_prior',
            # '--save_history_ckpt',
            '--strokes_per_img', '6',
            "--points-per-stroke", '5',
         ],
    # prev best model
    'Full-spDec-sqPrior-dp-useDetachRsd-sepPrWrNet-noPosPrRnn-normRfLoss-anNonPrLr': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            # '--no_pres_rnn',
            '--no_pres_post_rnn',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
            '--dependent_prior',
            # '--save_history_ckpt',
         ],
    'Full-spDec-sqPrior-useDetachRsdNotRsdEm-noTarget-sepPrWrNet-noWtPosPrRnn-normRfLoss-anNonPrLr-6strk': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            '--no_pres_rnn',

            '--use_residual',
            '--residual_pixel_count',
            # '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
            '--residual_no_target',
            '--no_what_post_rnn',
            # '--save_history_ckpt',
            '--strokes_per_img', '6',
         ],
    # β1 is already use var number of strokes; β4 collapse; β3 works
    'Full-spDec-sqPrior-useDetachRsdNotRsdEm-noTarget-sepPrWrNet-noWtPosPrRnn-normRfLoss-anNonPrLr': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            '--no_pres_rnn',

            '--use_residual',
            '--residual_pixel_count',
            # '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
            '--residual_no_target',
            '--no_what_post_rnn',
            # '--save_history_ckpt',
         ],
    # β1 is already use var number of strokes 
    'Full-spDec-sqPrior-useDetachRsdNotRsdEm-noTarget-sepPrWrNet-noPrRnn-normRfLoss-anNonPrLr': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            '--no_pres_rnn',

            '--use_residual',
            '--residual_pixel_count',
            # '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
            '--residual_no_target',
            # '--save_history_ckpt',
         ],
    'Full-spDec-sqPrior-useDetachRsd-noTarget-sepPrWrNet-noPrRnn-normRfLoss-anNonPrLr': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            '--no_pres_rnn',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
            '--residual_no_target',
            # '--save_history_ckpt',
         ],
    # Feb 27
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-normRfLoss-anNonPrLr-intrll': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
            '--intermediate_likelihood', 'Mean',
         ],
    # Feb 26
    'Full-spDec-sqPrior-detachCanvRsd-sepPrWrNet-noRnn-normRfLoss-anLr': basic_full_model +\
        [
            '--anneal_lr',
            # '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            '--no_rnn',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            # '--no_detach_rsd',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
         ],
    'Full-spDec-sqPrior-detachCanvRsd-sepPrWrNet-noRnn-normRfLoss': basic_full_model +\
        [
            # '--anneal_lr',
            # '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            '--no_rnn',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            # '--no_detach_rsd',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
         ],
    'Full-spDec-sqPrior-unDetachCanvRsd-sepPrWrNet-noRnn-normRfLoss': basic_full_model +\
        [
            # '--anneal_lr',
            # '--anneal_non_pr_net_lr',
            # '--log_param',
            # '--detach_canvas_so_far',
            '--no_rnn',

            '--use_residual',
            '--residual_pixel_count',
            # '--detach_rsd_embed',
            '--no_detach_rsd',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
         ],
    # Feb 25
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-noPrRnn-normRfLoss': basic_full_model +\
        [
            # '--anneal_lr',
            # '--anneal_non_pr_net_lr',
            '--log_param',
            '--detach_canvas_so_far',
            '--no_pres_rnn',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            # '--update_reinforce_ll',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
         ],
    # The 3 models with no_post_rnn, with +detachCanv, + anLr progressivly.
    'Full-spDec-sqPrior-no_post_rnn-detachCanvas-normRfLoss-anLr': basic_full_model +\
        [
            '--anneal_lr',
            '--no_post_rnn',
            '--log_param',
            '--detach_canvas_so_far',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
        ],
    'Full-spDec-sqPrior-no_post_rnn-detachCanvas-normRfLoss': basic_full_model +\
        [
            '--no_post_rnn',
            '--log_param',
            '--detach_canvas_so_far',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
        ],
    'Full-spDec-sqPrior-no_post_rnn-normRfLoss': basic_full_model +\
        [
            '--no_post_rnn',
            '--log_param',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
        ],
    # current not working very well
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-normGlobalRfLoss-anNonPrLr': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            '--log_param',
            '--detach_canvas_so_far',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            # '--update_reinforce_ll',
            '--update_reinforce_loss',
            '--global_reinforce_signal',
            '--sep_where_pres_net',
         ],
    # Feb 24
    # this with β4 is able to learn variables strokes ~3/5 seeds
    # the prev best model: v0
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-normRfLoss-anNonPrLr': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            # '--no_pres_rnn',

            # '--only_rsd_ratio_pres',
            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
         ],
    # v2
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-noPrRnn-normRfLoss-anNonPrLr': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            '--no_pres_rnn',

            # '--only_rsd_ratio_pres',
            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
            '--save_history_ckpt',
         ],
    # v2.1 dependent prior
    'Full-spDec-sqPrior-dp-useDetachRsd-sepPrWrNet-noPrRnn-normRfLoss-anNonPrLr': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            '--no_pres_rnn',

            # '--only_rsd_ratio_pres',
            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
            '--dependent_prior',
            '--save_history_ckpt',
         ],
    # v2.2 dependent prior but the other way around
    'Full-spDec-sqPrior-dp-t|r-useDetachRsd-sepPrWrNet-noPrRnn-normRfLoss-anNonPrLr': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            '--no_pres_rnn',

            # '--only_rsd_ratio_pres',
            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
            '--dependent_prior',
            '--prior_dependency', 'wt|wr',
            # '--save_history_ckpt',
         ],
    
    # v1
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-onlyRsdRatPr-'+\
        'normRfLoss-anNonPrLr': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            # '--no_pres_rnn',

            '--only_rsd_ratio_pres',
            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
         ],
    # v3
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-noPrRnn-'+\
        'onlyRsdRatPr-normRfLoss-anNonPrLr': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            '--no_pres_rnn',

            '--only_rsd_ratio_pres',
            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
         ],
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-noPrRnn-'+\
        'normRfLoss-anNonPrLr-omni': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            # '--log_param',
            '--detach_canvas_so_far',
            '--no_pres_rnn',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
            '--dataset', 'Omniglot',
         ],
    # Q: what's the diff between simple_pres + sep_wr_pre vs just simple_pres?
    # A: not much but should use just simple_pres as it would require the z_pres
    # prior would need the hidden states.
    'Full-spDec-sqPrior-useDetachRsd-simPres-normRfLoss-anNonPrLr': basic_full_model +\
        [
            '--anneal_lr',
            '--anneal_non_pr_net_lr',
            '--log_param',
            '--detach_canvas_so_far',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            # '--update_reinforce_ll',
            '--update_reinforce_loss',
            # '--sep_where_pres_net',
            '--simple_pres',
         ],
    'Full-spDec-sqPrior-useDetachRsd-simPres-normRfLoss-anLr': basic_full_model +\
        [
            '--anneal_lr',
            # '--anneal_non_pr_net_lr',
            '--log_param',
            '--detach_canvas_so_far',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            # '--update_reinforce_ll',
            '--update_reinforce_loss',
            # '--sep_where_pres_net',
            '--simple_pres',
         ],
    # Feb 23 
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-normRfLoss-anLr': basic_full_model +\
        [
            '--anneal_lr',
            '--log_param',
            '--detach_canvas_so_far',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            # '--update_reinforce_ll',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
         ],
    # 500k
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-anLr': basic_full_model +\
        [
            '--anneal_lr',
            '--log_param',
            '--detach_canvas_so_far',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_ll',
            '--sep_where_pres_net',
         ],
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-normRfLoss': basic_full_model +\
        [
            # '--anneal_lr',
            '--log_param',
            '--detach_canvas_so_far',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_loss',
            '--sep_where_pres_net',
         ],
    'Full-spDec-sqPrior-useDetachRsd-normRfLoss': basic_full_model +\
        [
            # '--anneal_lr',
            '--log_param',
            '--detach_canvas_so_far',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_loss',
         ],
    # the below used update_reinforce_ll = True in loss
    # Feb 21: shown yesterday that Full-spDec-fxPrior-useCanvas-anLr-β3
    # is already able to learn a variable number of strokes.
    # Now exp fxPrior model that uses 4dim z_where and residual
    # works 400k+
    'Full-spDec-sqPrior-useDetachRsd-anLr': basic_full_model +\
        [
            '--anneal_lr',
            '--log_param',
            '--detach_canvas_so_far',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--update_reinforce_ll',
         ],
    # works 250k+
    'Full-spDec-sqPrior-useDetachCanvRsd-anLr': basic_full_model +\
        [
            '--anneal_lr',
            '--log_param',
            '--detach_canvas_so_far',

            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--detach_canvas_embed',
            '--update_reinforce_ll',
         ],
    # works 250k+
    'Full-spDec-sqPrior-useCanv-anLr': basic_full_model +\
        [
            '--anneal_lr',
            '--log_param',
            '--detach_canvas_so_far',
            '--update_reinforce_ll',
         ],
    # works 300k+
    'Full-spDec-sqPrior-noEG-anLr': full_no_canvas +\
        [
            '--anneal_lr',
            '--log_param',
            '--update_reinforce_ll',
         ],
    # works 300k+
    'Full-spDec-fxPrior-useDetachCanvRsd-anLr-wr4': basic_full_model +\
        [
            '--prior_dist', 'Independent',
            '--anneal_lr',
            '--detach_canvas_so_far',
            '--log_param',
            '--use_residual',
            '--residual_pixel_count',
            '--detach_rsd_embed',
            '--detach_canvas_embed',
            # '--save_history_ckpt',
        ],
    # works 300k+
    'Full-spDec-fxPrior-useRsd-anLr-wr4-simplePres': basic_full_model +\
        [
            '--prior_dist', 'Independent',
            '--anneal_lr',
            '--detach_canvas_so_far',
            '--log_param',
            '--use_residual',
            '--residual_pixel_count',
            '--simple_pres',
            # '--save_history_ckpt',
        ],
    # works 300k+
    'Full-spDec-fxPrior-useRsd-anLr-wr4': basic_full_model +\
        [
            '--prior_dist', 'Independent',
            '--anneal_lr',
            '--detach_canvas_so_far',
            '--log_param',
            '--use_residual',
            '--residual_pixel_count',
            # '--save_history_ckpt',
        ],
    'Full-spDec-fxPrior-useCanvas-anLr-wr4': basic_full_model +\
        [
            '--prior_dist', 'Independent',
            '--anneal_lr',
            '--detach_canvas_so_far',
            '--log_param',
            # '--save_history_ckpt',
        ],
    'Full-spDec-fxPrior-useCanvas-anLr-wr3': basic_full_model +\
        [
            '--prior_dist', 'Independent',
            '--anneal_lr',
            '--z_where_type', '3',
            '--detach_canvas_so_far',
            '--log_param',
            # '--save_history_ckpt',
        ],
    # Feb 17
    'Full-fixed_prir-useRsd-anLr': basic_full_model +\
        [
            '--anneal_lr',
            '--prior', "Independent",
            '--use_residual',
        ],
    'Full-spDec-noPost_rnn': basic_full_model +\
        [
            '--no_post_rnn',
            '--log_param',
            '--use_residual',
        ],
    'Full-spDec-simple_pres': basic_full_model +\
        [
            '--simple_pres',
            '--log_param',
            '--use_residual',
        ],
    'Full': basic_full_model,
    # model that doesn't collapse, a.k.a. baseline:
    'Full-neuralDec-fixed_prir-noEG-sepPrWr': full_no_canvas +\
        [
            '--no_spline_renderer',
            '--prior_dist', 'Independent',
            '--no_maxnorm',
            '--no_sgl_strk_tanh',
            '--sep_where_pres_net',
            '--log_param',
        ],
    'Full-spDec-fxPrior-useCanvas-anLr-sepPrWr': basic_full_model +\
        [
            '--prior_dist', 'Independent',
            '--z_where_type', '3',
            '--anneal_lr',
            '--detach_canvas_so_far',
            '--log_param',
            # '--constrain_z_pres_param',
            '--sep_where_pres_net',
            '--save_history_ckpt',
        ],
    'Full-spDec-fxPrior-useCanvas-anLr': basic_full_model +\
        [
            '--prior_dist', 'Independent',
            '--z_where_type', '3',
            '--anneal_lr',
            '--detach_canvas_so_far',
            '--log_param',
            # '--save_history_ckpt',
        ],
    # exp with constraining z_pres_param, as normally it doesn't produce 0s once
    # converged
    'Full-spDec-fxPrior-useCanvas-anLr-consPres': basic_full_model +\
        [
            '--prior_dist', 'Independent',
            '--z_where_type', '3',
            '--anneal_lr',
            '--detach_canvas_so_far',
            '--log_param',
            '--constrain_z_pres_param',
            '--save_history_ckpt',
        ],
    # this shouldn't collapse from conclusion from investigation into collapse
    'Full-spDec-fxPrior-noEG-anLr-sepPrWr': full_no_canvas +\
        [
            '--prior_dist', 'Independent',
            '--anneal_lr',
            '--sep_where_pres_net',
            '--log_param',
        ],
    # we can potentially also use: 1) a detach canvas before passing to both rnn
    # and 2)return no canvas
    'Full-spDec-fxPrior-useDetachCanvas-anLr': basic_full_model +\
        [
            '--prior_dist', 'Independent',
            '--z_where_type', '3',
            '--anneal_lr',
            '--detach_canvas_so_far',
            '--log_param',
        ],
    # use canvas by passing it only to zwhere but not zwhat, see above for 
    # another way
    'Full-spDec-fxPrior-useUndetachetachCanvas-anLr': basic_full_model +\
        [
            '--prior_dist', 'Independent',
            '--z_where_type', '3',
            '--anneal_lr',
            '--canvas_only_to_zwhere',
            '--log_param',
        ],
    # 2 minimal models that collapse from early exp:
    'Full-neuralDec-fxPrior-useUndetachCanvas-anLr': basic_full_model +\
        [
            '--no_spline_renderer',
            '--prior_dist', 'Independent',
            '--z_where_type', '3',
            '--no_maxnorm',
            '--no_sgl_strk_tanh',
            '--anneal_lr',
            '--log_param',
            '--update_reinforce_ll',
        ],  
    'Full-spDec-fxPrior-noEG-anLr': full_no_canvas +\
        [
            '--prior_dist', 'Independent',
            '--z_where_type', '3',
            '--anneal_lr',
            '--log_param',
        ],
    # model that doesn't collapse, a.k.a. baseline:
    'Full-neuralDec-fxPrior-noEG': full_no_canvas +\
        [
            '--no_spline_renderer',
            '--prior_dist', 'Independent',
            '--no_maxnorm',
            '--no_sgl_strk_tanh',
        ],
    'AIR-seq': 
        [
            '--model_type', 'Sequential',
            '--prior_dist', 'Independent',
            '--no_spline_renderer',
            '--lr', '1e-4', 
            '--bl_lr', '1e-4',
            '--z_where_type', '3',
            '--strokes_per_img', strokes_per_img,
            '--z_what_in_pos', 'z_where_rnn',
            '--target_in_pos', 'RNN',
            '--save_history_ckpt',
        ],
    **models_2_cmd
}

full_and_ablation_dict = {
    'Full': args_from_kw_list(full_config)+\
        [
            # '--beta', '4'
         ],
    **ablation_args()
}
exp_dict.update(full_and_ablation_dict)

standard = "Full-spDec-sqPrior-dp-useDetachRsd-sepPrWrNet-noPrRnn-normRfLoss-"+\
            "anNonPrLr"
assert set(exp_dict['Full']) == set(exp_dict[standard]), "Full != Standard"