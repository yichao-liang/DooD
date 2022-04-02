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
    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    run_args = parser.parse_args()

    all_exp_args = {}

    # VAE and AIR 
    # Full model
    # Full model ablation (Full minus 1 feature)
    # MWS

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
    }
    ds_name = ds_dict[run_args.ds]
    # ---
    # v3.1 bernoulli: β8, 9 works well
    # exp_name = 'Full-spDec-sqPrior-dp-rt-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-bern-65strk'
    # v0.1: β3 works
    # exp_name = 'Full-spDec-sqPrior-dp-rt-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk'
    # v0.2: β2-4 works 
    # exp_name = 'Full-spDec-sqPrior-dp-detachRsdNotRsdEm-noTarget-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-6strk'
    # v0.3: β2 a bit more strokes then needed, 4 collapse
    # exp_name = 'Full-spDec-sqPrior-dp-detachRsdNotRsdEm-noTarget-sepPrWrNet-noWtPrPosRnn-normRfLoss-anNonPrLr-6strk'

    # v1.1 β3 works better than 4; 4 stops using strokes
    # exp_name = 'Full-spDec-sqPrior-dp-5wr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk'
    # v1.2 β2-4 all works
    # exp_name = 'Full-spDec-sqPrior-dp-5wr-detachRsdNotRsdEm-noTarget-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk'
    # v1.2 β2 works
    # exp_name = 'Full-spDec-sqPrior-dp-5wr-detachRsdNotRsdEm-noTarget-sepPrWrNet-noWtPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk'
    # exp_name = 'Full-neuralDec-fxPrior-useUndetachCanvas-anLr'
    
    # v2.1
    # exp_name = 'Full-spDec-sqPrior-dp-rt-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-omni'
    # exp_name = 'Full-spDec-sqPrior-dp-rt-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-km'
    # exp_name = 'Full-spDec-sqPrior-dp-rt-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-em'
    # exp_name = 'Full-spDec-sqPrior-dp-rt-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk-qd'
    code_dict = {
        'M': 'Full-spDec-sq20MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-65strk',
        # 'MT': 'Full-spDec-sq20MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk',
        'MT': 'Full-spDec-sq50MCorImcPrior-dp-tr-detachRsdNotRsdEm-sepPrWrNet-noPrPosRnn-normRfLoss-anNonPrLr-lapl-tranWhat-65strk',
        'AIR': 'AIR',
    }
    exp_name = code_dict[run_args.m]
    # exp_name = 'Full' 
    # β4->4/4 collapse; β1->4/4 steps; β3->3collapse 1full; β2->1var; 3full
    # exp_name = ablation_exp_name[0] 
    # also means no strk_tanh and add_tanh; β10 
    # exp_name = ablation_exp_name[1]
    # extramely hard to get to learn variable steps; maybe β4
    # exp_name = ablation_exp_name[2]
    # β4: 1/4 var, 3/4 full
    # exp_name = ablation_exp_name[3]
    # exp_name = 'MWS'
    # exp_name = 'AIR'
    all_exp_args[exp_name] = ed.exp_dict[exp_name]
    
    train_not_test = False
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
                        '--tb_dir', f'/om/user/ycliang/log/debug1/{tb_name}',
                        # '--tb_dir', f'/om/user/ycliang/log/full-{run_args.m}/{model_name}',
                        #  '--tb_dir', f'./log/full-beta/{model_name}',
                        '--beta', f'{run_args.beta}',
                        '--dataset', ds_name,

                        '--seed', f'{run_args.seed}',
                        # '--dataset', 'Omniglot',
                        # '--dataset', 'KMNIST',
                        # '--dataset', 'Quickdraw',
                        # '--dataset', 'EMNIST',
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
