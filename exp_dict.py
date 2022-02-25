
strokes_per_img = '4'

'''
Our model = AIR + sequential prior
                + canvas-so-far
                + seperated z_{where, pre} and z_what (4-dim z_where)
                + spline renderer
                (+ input target at mlp, i.e., target_in_pos = 'RNN', for 
                    writing-completion)
'''
full_config = [
                'sequential_prior',  
                'spline_latent',  
                'canvas',  
                'seperated_z',
            ]

models_2_cmd = {
    'VAE': [ 
        '--model-type', 'VAE',
        '--lr', '1e-3',
    ],
    'AIR': [
        '--model-type', 'AIR',
        '--prior_dist', 'Independent',
        '--lr', '1e-4', 
        '--bl_lr', '1e-3',
        '--z_where_type', '3',
        '--strokes_per_img', strokes_per_img,
        '--z_what_in_pos', 'z_where_rnn',
        '--target_in_pos', 'RNN'
    ],
    'AIR4': [
        '--model-type', 'AIR',
        '--prior_dist', 'Independent',
        '--lr', '1e-4', 
        '--bl_lr', '1e-3',
        '--z_where_type', '4_rotate',
        '--strokes_per_img', strokes_per_img,
        '--z_what_in_pos', 'z_where_rnn',
        '--target_in_pos', 'RNN'
    ],
    'MWS': [
        '--model-type', 'MWS',
    ],
    'AIR+seq_prir': [
        '--model-type', 'AIR',
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
    # ---
    args.append('--prior_dist')
    if 'sequential_prior' in kw_config:
        args.extend(['Sequential'])
    else:
        args.extend(['Independent'])
    # ---    
    args.append('--model-type')
    if 'spline_latent' in kw_config:
        args.extend(['Sequential'])
    else:
        args.extend(['AIR'])
        args.extend(['--lr', '1e-4'])
    # ---
    if 'canvas' in kw_config:
        args.extend(['--use_canvas'])
    # ---
    args.append('--z_what_in_pos')
    if 'seperated_z' in kw_config:
        args.append('z_what_rnn')
    else:
        args.append('z_where_rnn')
    return args

def ablation_args():
    all_args = {}
    for ablation in full_config:
        # if ablation == 'spline_latent':
            # skipping Full-spline_latent b/c the model didn't work
            # continue
        exp_config = full_config.copy()
        exp_config.remove(ablation)
        args = args_from_kw_list(exp_config)
        if ablation == 'canvas':
            args.append("--no_strk_tanh")
        # args.append("--no_maxnorm")
        all_args[f"Full-{ablation}"] = args

    return all_args

# common options
                    # '--no_spline_renderer',
                    # '--prior', "Independent",
                    # '--simple_pres',
                    # '--z_what_in_pos', 'z_where_rnn',
                    # '--target_in_pos', 'RNN',
                    # '--z_where_type', '3',
                    # '--no_baseline',
                    # '--lr', '1e-3', 
                    # '--sep_where_pres_mlp',
                    # '--render_at_the_end',
                    # '--beta', f'{run_args.beta}',
                    # "--increase_beta",
                    # '--final_beta', f'{run_args.final_beta}',
                    # '--use_residual',
                    # '--residual_pixel_count',
                    # '--dependent_prior',
                    # '--no_maxnorm',
                    # '--no_sgl_strk_tanh',
                    # '--no_add_strk_tanh',
                    # "--anneal_lr",
                    # '--continue_training',
                    # "--log_grad",
                    # "--log_param",
                    # '--save_history_ckpt',

# Record all experiment configs
full_model_args = args_from_kw_list(full_config)
# i.e. ['--prior_dist', 'Sequential', 
    #   '--model-type', 'Sequential', 
    #   '--z_what_in_pos', 'z_what_rnn',
    #   '--use_canvas']
full_no_canvas = args_from_kw_list(full_config)
full_no_canvas.remove('--use_canvas')

exp_dict = {
    # Feb 25
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-normGlobalRfLoss-anNonPrLr': full_model_args +\
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
    # Q: what's the diff between simple_pres + sep_wr_pre vs just simple_pres?
    # A: not much but should use just simple_pres as it would require the z_pres
    # prior would need the hidden states.
    'Full-spDec-sqPrior-useDetachRsd-simPres-normRfLoss-anNonPrLr': full_model_args +\
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
    'Full-spDec-sqPrior-useDetachRsd-simPres-normRfLoss-anLr': full_model_args +\
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
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-normRfLoss-anNonPrLr': full_model_args +\
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
            '--sep_where_pres_net',
         ],
    # Feb 23
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-normRfLoss-anLr': full_model_args +\
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
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-anLr': full_model_args +\
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
    'Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-normRfLoss': full_model_args +\
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
    'Full-spDec-sqPrior-useDetachRsd-normRfLoss': full_model_args +\
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
    # works 250k+
    'Full-spDec-sqPrior-useDetachRsd-anLr': full_model_args +\
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
    'Full-spDec-sqPrior-useDetachCanvRsd-anLr': full_model_args +\
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
    'Full-spDec-sqPrior-useCanv-anLr': full_model_args +\
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
    'Full-spDec-fxPrior-useDetachCanvRsd-anLr-wr4': full_model_args +\
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
    'Full-spDec-fxPrior-useRsd-anLr-wr4-simplePres': full_model_args +\
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
    'Full-spDec-fxPrior-useRsd-anLr-wr4': full_model_args +\
        [
            '--prior_dist', 'Independent',
            '--anneal_lr',
            '--detach_canvas_so_far',
            '--log_param',
            '--use_residual',
            '--residual_pixel_count',
            # '--save_history_ckpt',
        ],
    'Full-spDec-fxPrior-useCanvas-anLr-wr4': full_model_args +\
        [
            '--prior_dist', 'Independent',
            '--anneal_lr',
            '--detach_canvas_so_far',
            '--log_param',
            # '--save_history_ckpt',
        ],
    'Full-spDec-fxPrior-useCanvas-anLr-wr3': full_model_args +\
        [
            '--prior_dist', 'Independent',
            '--anneal_lr',
            '--z_where_type', '3',
            '--detach_canvas_so_far',
            '--log_param',
            # '--save_history_ckpt',
        ],
    # Feb 17
    'Full-fixed_prir-useRsd-anLr': full_model_args +\
        [
            '--anneal_lr',
            '--prior', "Independent",
            '--use_residual',
        ],
    'Full-spDec-simple_arch': full_model_args +\
        [
            '--simple_arch',
            '--log_param',
            '--use_residual',
        ],
    'Full-spDec-simple_pres': full_model_args +\
        [
            '--simple_pres',
            '--log_param',
            '--use_residual',
        ],
    'Full': full_model_args,
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
    'Full-spDec-fxPrior-useCanvas-anLr-sepPrWr': full_model_args +\
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
    'Full-spDec-fxPrior-useCanvas-anLr': full_model_args +\
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
    'Full-spDec-fxPrior-useCanvas-anLr-consPres': full_model_args +\
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
    'Full-spDec-fxPrior-useDetachCanvas-anLr': full_model_args +\
        [
            '--prior_dist', 'Independent',
            '--z_where_type', '3',
            '--anneal_lr',
            '--detach_canvas_so_far',
            '--log_param',
        ],
    # use canvas by passing it only to zwhere but not zwhat, see above for 
    # another way
    'Full-spDec-fxPrior-useUndetachetachCanvas-anLr': full_model_args +\
        [
            '--prior_dist', 'Independent',
            '--z_where_type', '3',
            '--anneal_lr',
            '--canvas_only_to_zwhere',
            '--log_param',
        ],
    # 2 minimal models that collapse from early exp:
    'Full-neuralDec-fxPrior-useUndetachCanvas-anLr': full_model_args +\
        [
            '--no_spline_renderer',
            '--prior_dist', 'Independent',
            '--z_where_type', '3',
            '--no_maxnorm',
            '--no_sgl_strk_tanh',
            '--anneal_lr',
            '--log_param',
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
}
