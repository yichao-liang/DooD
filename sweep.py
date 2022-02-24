'''
Train all the selected models
Test them on the test suite
Output the results in a pdf
'''
import subprocess
import argparse

import util
import exp_dict as ed

def get_args_parser():
    parser = argparse.ArgumentParser(formatter_class=
                                        argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=0)
    parser.add_argument("--beta", default=1, help="beta term as in beta-VAE")
    parser.add_argument("--final_bern", default=.5, type=float, 
                        help="Minimal value for the z_pres Bern param")
    parser.add_argument("--final_beta", default=1, type=float, 
                        help="Minimal value for the beta")
    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    run_args = parser.parse_args()

    all_exp_args = {}

    # VAE and AIR 
    # all_exp_args['VAE'] = ed.models_2_cmd['VAE']
    # all_exp_args['AIR10'] = ed.models_2_cmd['AIR']

    # # Full model
    # all_exp_args['Full'] = ed.exp_dict['Full']

    # # # Full model ablation (Full minus 1 feature)
    # all_ablations = ed.ablation_args()
    # all_exp_args.update(all_ablations)

    # # MWS
    # all_exp_args['MWS'] = models_2_cmd['MWS']

    # Not in final list: AIR+seq_prir
    # all_exp_args['AIR+seq_prir'] = models_2_cmd['AIR+seq_prir']

    exp_name = "Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-normRfLoss-anLr"
    all_exp_args[exp_name] = ed.exp_dict[exp_name]
    
    for n, args in all_exp_args.items():
        print(f"==> Begin training the '{n}' model")
        # args.remove('--use_canvas')
        model_name = n + f'-β{run_args.beta}-{run_args.seed}'
        args.extend(['--save_model_name', model_name,
                     '--tb_dir', f'./log/full_bu/{model_name}',
                     '--beta', f'{run_args.beta}',

                    '--seed', f'{run_args.seed}',
                    # '--continue_training',
                    ])
        subprocess.run(['python', 'run.py'] + args)# + ['--continue_training'])
        print(f"==> Done training {n}\n")
        # print(f"==> Begin evaluating the '{n}' model")
        # ckpt_path = util.get_checkpoint_path_from_path_base(n, -1)
        # subprocess.run(['python', 'test.py', '--save_model_name', n])
        # print(f"==> Done evaluating the '{n}' model\n\n")