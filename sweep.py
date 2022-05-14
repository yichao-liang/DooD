'''
Train all the selected models
Test them on the test suite
Output the results in a pdf
'''
import subprocess
import argparse

import exp_dict as ed
import util

def get_args_parser():
    parser = argparse.ArgumentParser(formatter_class=
                                        argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=0)
    parser.add_argument("--beta", default=1, help="beta term as in beta-VAE")
    parser.add_argument("--final_bern", default=.5, type=float, 
                        help="Minimal value for the z_pres Bern param")
    parser.add_argument("--final_beta", default=1, type=float, 
                        help="Minimal value for the beta")
    parser.add_argument("-ct", action='store_true',
                        help='continue training')
    parser.add_argument("-it", default=-1,
                        type=int, help='''iteration to run test on, default -1
                        means latest''')
    parser.add_argument("-m", default='mn',
                        type=str, help='model code')
    parser.add_argument("-ds", default='mn',
                        type=str, help='dataset name')
    parser.add_argument("-trn", "--train_not_test", action='store_true',
                        help='if specified, just train or else just eval')
    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    run_args = parser.parse_args()

    all_exp_args = {}

    ablation_exp_name = [
            'Full-sequential_prior',
            'Full-spline_decoder',
            'Full-canvas',
            'Full-seperated_z',
        ]

    ds_dict = {
        'mn': 'MNIST',
        'om': 'Omniglot',
        'em': 'EMNIST',
        'km': 'KMNIST',
        'qm': 'QMNIST',
        'qd': 'Quickdraw',
        'sy': 'Synthetic',
    }
    ds_name = ds_dict[run_args.ds]

    code_dict = {
        'AIR3': 'AIR',
        'AIR': 'AIR4',
        'AIR_g': 'AIR4_Gaus',
        'AIR_l': 'AIR4_Lapl',
        'DAIR_l': 'DAIR_Lapl',
        'DAIR_g': 'DAIR_Gaus',
        'M': 'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk',
        'Ma1': 'Full-spDec-indPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk',
        'Ma2': 'Full-nnDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk',
        'Ma3': 'Full-spDec-sq40MCorPrior-dp-tr-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk',
        'MS': 'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEmNoShrg-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk',
        'MnT': 'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-noTgt-65strk',
        # '1MT': 'Full-spDec-sq1MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk',
        # '20MT': 'Full-spDec-sq20MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk',
        # 'MNCT': 'Full-spDec-sq40MImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk',
        # 'MDT': 'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk',
        # '1MNCT': 'Full-spDec-sq1MImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk',
        'MT': 'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk',
        # 'MTL': 'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEmL-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk',
        # 'MTL2': 'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEmL-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk',
        # 'MTS': 'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEmNoShrg-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk',
        # 'MT5': 'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPr5WrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk',
        'MTnT': 'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-noTgt-65strk',
        # 'MTnT5': 'Full-spDec-sq40MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPr5WrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-noTgt-65strk',
    }
    exp_name = code_dict[run_args.m]
    all_exp_args[exp_name] = ed.exp_dict[exp_name]
    
    train_not_test = run_args.train_not_test
    if train_not_test:
        train, evaluate = True, False
    else:
        train, evaluate = False, True

    for n, args in all_exp_args.items():
        model_name = n + f'-{run_args.ds}-β{run_args.beta}-{run_args.seed}'
        tb_name = f'{run_args.m}-{run_args.ds}-β{run_args.beta}-{run_args.seed}'
        if train:
            print(f"==> Begin training the '{model_name}' model")
            args.extend(['--save_model_name', model_name,
                        '--tb_dir', f'/om/user/ycliang/log/hyper/{tb_name}',
                        # '--tb_dir', f'/om/user/ycliang/log/full-{run_args.m}/{model_name}',
                        #  '--tb_dir', f'./log/full-beta/{model_name}',
                        '--beta', f'{run_args.beta}',
                        '--dataset', ds_name,

                        '--seed', f'{run_args.seed}',
                        ])
            if run_args.ct:
                args.append('--continue_training')
            subprocess.run(['python', 'run.py'] + args)# + ['--continue_training'])
            print(f"==> Done training {n}\n")

        if evaluate:
            print(f"==> Begin evaluating the '{model_name}' model")

            ckpt_path = util.get_checkpoint_path_from_path_base(model_name, 
                                                                run_args.it)
            subprocess.run(['python', 'test.py', 
                            '--ckpt_path', ckpt_path,
                            # for old models
                            # '--save_model_name', n])
                            # for new models
                            '--tb_name', tb_name,
                            '--save_model_name', model_name])
            print(f"==> Done evaluating the '{n}' model\n\n")
