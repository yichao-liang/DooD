
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
        args.extend(['--execution_guided'])
        args.extend(['--exec_guid_type', 'canvas'])
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
                    # '--exec_guid_type', 'residual',
                    # '--residual_pixel_count',
                    # '--dependent_prior',
                    # '--no_maxnorm',
                    # '--no_sgl_strk_tanh',
                    # '--no-add_strk_tanh',
                    # "--anneal_lr",
                    # '--continue_training',
                    # "--log_grad",
                    # "--log_param",
                    # '--save_history_ckpt',

# Record all experiment configs
exp_dict = {
    # 'Full': args_from_kw_list(full_config),
    'Full-simple_arch': args_from_kw_list(full_config) +\
        [
            '--simple_arch',
            '--log_param',
            '--exec_guid_type', 'residual',
        ]
}