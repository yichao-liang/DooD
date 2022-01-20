'''
Train all the selected models
Test them on the test suite
Output the results in a pdf
'''
from itertools import combinations
import subprocess
import numpy as np

import util

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
                'canvas',  
                'seperated_z',
                'spline_latent',  
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
        if ablation == 'spline_latent':
            # skipping Full-spline_latent b/c the model didn't work
            continue
        exp_config = full_config.copy()
        exp_config.remove(ablation)
        args = args_from_kw_list(exp_config)
        if ablation == 'canvas':
            args.append("--no_strk_tanh")
        args.append("--no_maxnorm")
        all_args[f"Full-{ablation}"] = args

    return all_args
    
if __name__ == '__main__':
    all_exp_args = {}

    # VAE and AIR 
    # all_exp_args['VAE'] = models_2_cmd['VAE']
    all_exp_args['AIR10'] = models_2_cmd['AIR']

    # Full model
    all_exp_args['Full'] = args_from_kw_list(full_config) + [
                                                        '--constrain_sample']

    # Full - spline
    all_exp_args['Full-spline_decoder'] = [
                                    '--model-type', 'Sequential',
                                    '--prior_dist', 'Sequential',
                                    '--strokes_per_img', '4',
                                    '--lr', '1e-4', '--bl_lr', '1e-3',
                                    '--z_where_type', '3',
                                    '--execution_guided',
                                    '--z_what_in_pos', 'z_what_rnn',
                                    '--target_in_pos', 'MLP',
                                    '--no_spline_renderer']

    # (Full minus 3 features except spline_latent)
    all_exp_args['AIR+spline_latent'] = (args_from_kw_list(['spline_latent']) + 
                                            ['--no_maxnorm', 
                                             '--no_strk_tanh',
                                             '--target_in_pos', 'RNN',
                                             '--z_where_type', '3',
                                             '--constrain_sample'])

    # Full model ablation (Full minus 1 feature)
    all_ablations = ablation_args()
    all_exp_args.update(all_ablations)
    all_exp_args['Full-sequential_prior'] = all_exp_args['Full-sequential_prior'
                                            ] + ['--constrain_sample']

    # MWS
    all_exp_args['MWS'] = models_2_cmd['MWS']

    for n, args in all_exp_args.items():
        print(f"==> Begin training the '{n}' model")
        args.extend(['--save_model_name', n])
        subprocess.run(['python', 'run.py'] + args)# + ['--continue_training'])
        print(f"==> Done training {n}\n")

        # print(f"==> Begin evaluating the '{n}' model")
        # ckpt_path = util.get_checkpoint_path_from_path_base(n, -1)
        # subprocess.run(['python', 'test.py', '--save_model_name', n])
        # print(f"==> Done evaluating the '{n}' model\n\n")