
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
    # this shouldn't collapse from conclusion from investigation into collapse
    'Full-spDec-fxPrior-noEG-anLr-sepPrWr': full_no_canvas +\
        [
            '--prior_dist', 'Independent',
            '--anneal_lr',
            '--sep_where_pres_net',
            '--log_param',
        ],
    # we can potentially also use it detach canvas before passing to both rnn
    # and return no canvas
    'Full-spDec-fxPrior-useCanvas-anLr': full_model_args +\
        [
            '--prior_dist', 'Independent',
            '--z_where_type', '3',
            '--anneal_lr',
            '--detach_canvas_so_far',
            '--log_param',
        ],
    # use canvas by passing it only to zwhere but not zwhat, see above for 
    # another way
    'Full-spDec-fxPrior-useCanvas-anLr': full_model_args +\
        [
            '--prior_dist', 'Independent',
            '--z_where_type', '3',
            '--anneal_lr',
            '--canvas_only_to_zwhere',
            '--log_param',
        ],
    # 2 minimal models that collapse from early exp:
    'Full-neuralDec-fxPrior-useCanvas-anLr': full_model_args +\
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
