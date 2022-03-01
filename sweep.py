'''
Train all the selected models
Test them on the test suite
Output the results in a pdf
'''
import subprocess
import argparse

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
    # Full model
    # Full model ablation (Full minus 1 feature)
    # MWS

    # exp_name = "Full-spDec-sqPrior-unDetachCanvRsd-sepPrWrNet-noRnn-normRfLoss"
    # exp_name = "Full-spDec-sqPrior-detachCanvRsd-sepPrWrNet-noRnn-normRfLoss"
    # exp_name = "Full-spDec-sqPrior-detachCanvRsd-sepPrWrNet-noRnn-normRfLoss-anLr"
    # exp_name = "Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-normRfLoss-anNonPrLr"
    exp_name = "Full-spDec-sqPrior-useDetachRsd-sepPrWrNet-noPrRnn-normRfLoss-anNonPrLr"
    # exp_name = 'MWS'
    all_exp_args[exp_name] = ed.exp_dict[exp_name]
    
    train = True
    evalulate = False

    for n, args in all_exp_args.items():
        model_name = n + f'-β{run_args.beta}-{run_args.seed}'
        if train:
            print(f"==> Begin training the '{model_name}' model")
            args.extend(['--save_model_name', model_name,
                        '--tb_dir', f'./log/full/{model_name}',
                        #  '--tb_dir', f'./log/full-beta/{model_name}',
                        '--beta', f'{run_args.beta}',

                        '--seed', f'{run_args.seed}',
                        # '--continue_training',
                        ])
            subprocess.run(['python', 'run.py'] + args)# + ['--continue_training'])
            print(f"==> Done training {n}\n")

        if evalulate:
            print(f"==> Begin evaluating the '{n}' model")

            # ckpt_path = util.get_checkpoint_path_from_path_base(model_name, -1)
            subprocess.run(['python', 'test.py', 
                            # '--ckpt_path', ckpt_path,
                            # for old models
                            '--save_model_name', n])
                            # for new models
                            # '--save_model_name', model_name])
            print(f"==> Done evaluating the '{n}' model\n\n")
