from pprint import pprint

import util
import train
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from models import base, ssp, air, vae# , mws

def main(args):
    # Cuda
    device = util.get_device()
    args.device = device

    # seed
    util.set_seed(args.seed)

    # Write will output to ./log
    # When doing sweep evaluation
    log_dir = args.tb_dir
    # log_dir = f"./log/debug_full{args.seed}/{args.save_model_name}"
    writer = SummaryWriter(log_dir=log_dir)

    # When doing hyperop
    # writer = SummaryWriter(log_dir=f"./log/hyperop/" + 
    #                             f"name={args.save_model_name}-" + 
    #                             f"intr_ll={args.intermediate_likelihood}-" + 
    #                             f"constrain_sample={args.constrain_sample}-" +
    #                             f"z_where={args.z_where_type}-" + 
    #                             f"bl_lr={args.bl_lr}-" +
    #                             f"n_lyr={args.num_baseline_layers}-" +
    #                             f"mlp_h_dim={args.bl_mlp_hid_dim}-" + 
    #                             f"rnn_h_dim={args.bl_rnn_hid_dim}-" + 
    #                             f"maxnorm={str(not args.no_maxnorm)}-" + 
    #                             f"strk_tanh={str(not args.no_strk_tanh)}-" + 
    #                             f"render={args.render_method}")

    # initialize models, optimizer, stats, data
    checkpoint_path = util.get_checkpoint_path(args)
    num_iterations = args.num_iterations
    if not (args.continue_training and Path(checkpoint_path).exists()):
        util.logging.info("Training from scratch")
        model, optimizer, scheduler, stats, data_loader = util.init(args, 
                                                                      device)
    else:
        model, optimizer, scheduler, stats, data_loader, args =\
                                util.load_checkpoint(checkpoint_path, device)
    args.num_iterations = num_iterations

    # train
    if args.model_type == 'MWS':
        import models.mws.handwritten_characters as mws
        generative_model, guide, memory = model
        mws_args = mws.run.get_args_parser().parse_args([])

        if not Path(checkpoint_path).is_file():
            mws.train.train_sleep(
                                generative_model,
                                guide,
                                min(500, 
                                  mws_args.batch_size * mws_args.num_particles),
                                mws_args.pretrain_iterations,
                                mws_args.log_interval
                                )

    train.train(model, optimizer, scheduler, stats, data_loader, args, writer,
                dataset_name=args.dataset)

def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=
                                        argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=10)

    # Model
    parser.add_argument("--save_model_name", 
        default="full-test", 
        type=str, help='name for ckpt dir')
    parser.add_argument(
        "--model-type",
        default="Sequential", # Spline latent representation
        # default="AIR",        # Distribution latent representation    
        # default="VAE", 
        # default="Base", 
        choices=['Base', 'Sequential', 'AIR', 'VAE', 'MWS'],
        type=str,
        help='''Sequential has spline latent representation; AIR has distributed 
        representation'''
    )
    parser.add_argument(
        '--render_method', type=str,
        default='base',
        choices=['bounded', 'base'],
        help='method for Bezier renderer',
    )
    parser.add_argument(
        "--prior_dist",
        # default='Dirichlet',
        # default='Normal',
        # default='Independent',
        default='Sequential',
        type=str,
        choices=['Dirichlet', 'Normal', 'Independent', 'Sequential'],
        help='''
        For the `base` model, it can choose between "Dirichlet", "Normal" prior;
        For the `sequential` model, it always uses a Normal distribution for 
        each step, but it can choose between "independent" or "sequential"
        between steps.
        '''
    )
    parser.add_argument(
        "--target_in_pos",
        # default='RNN',
        default='MLP',
        type=str,
        choices=['RNN', 'MLP'],
        help='''Only used when `prior_dist` = Independent and `execution_guided`
        = True. This controls where the target image is input to the guide net; 
        either at the RNN or the MLP. For `prior_dist` = Squential, this is 
        always set to MLP, as the RNN is also used for the generation task where 
        the target image doesn't exist.
        ''') 
    parser.add_argument(
        "--likelihood_dist",
        # default='Normal',
        default='Laplace',
        type=str,
        choices=['Normal', 'Laplace'],
        help=""
    )
    parser.add_argument(
        "--inference-dist",
        # default='Dirichlet',
        default='Normal',
        type=str,
        choices=['Dirichlet', 'Normal'],
        help=""
    )
    parser.add_argument(
        "--inference_net_architecture",
        default='CNN',
        type=str,
        choices=['STN','CNN','MLP'],
        help=""
    )
    parser.add_argument(
        "--strokes_per_img",
        default=4,
        type=int,
        help="Maximum number of strokes per image"
    )
    parser.add_argument("--points-per-stroke", default=5, type=int,
        help="Number of control points per stroke curve."
    )
    parser.add_argument(
        "--z_where_type",
        default='4_rotate',
        # default='3',
        type=str, choices=['3', '4_rotate', '4_no_rotate', '5'],
        help='''
        "3": (scale, shift x, y)
        "4_rotate": (scale, shift x, y, rotate)
        "4_no_rotate": (scale x, y, shift x, y)
        "5": (scale x, y, shift x, y, rotate)'''
    ) 
    parser.add_argument(
        "--input_dependent_render_param",
        # default=False,
        default=True, type=bool, help="",
    )
    # parser.add_argument(
    #     '-eg', "--execution_guided",
    #     action='store_true', 
    #     # default=True,
    #     help="if not declared, False"
    # )
    # parser.add_argument(
    #     '--exec_guid_type',
    #     # default='canvas_so_far',
    #     default='canvas', choices=['residual', 'canvas', 'target+residual'], 
    #     type=str,
    #     help="""Only useful is --execution_guided = True. 
    #     Residual: only uses the difference between the target and canvas-so-far;
    #     Canvas_so_far: stack the target image on canvas-so-far"""
    # )
    parser.add_argument(
        '--use_canvas',
        action='store_true',
        help="equivalent to specifying execution_guided as before"
    )
    parser.add_argument(
        '--use_residual',
        action='store_true',
        help="equivalent to specifying exec_guid_type == residual as before"
    )
    parser.add_argument(
        "--transform_z_what",
        default=False,
        # default=True,
        type=bool,
        help=" "
    )
    parser.add_argument(
        "--feature_extractor_sharing",
        # default=False,
        default=True,
        type=bool,
        help='''Sharing the feature extractor for the target, canvas and glimpse
        The advantage of sharing is more training data but it may also result in
        is being less stable as the input can be quite differert.'''
    )
    parser.add_argument("--z_what_in_pos", 
        default='z_what_rnn',
        # default='z_where_rnn',
        type=str,
        help='Whether to input prev.z_what to z_where_rnn or z_what_rnn')
    parser.add_argument("--z_dim", default=10, type=int, 
        help="Only for VAE/AIR")
    parser.add_argument(
        '--num_mlp_layers', default=2, type=int,
        help="num mlp layers for style_mlp and z_what_mlp, and their prior nets"
    )
    parser.add_argument(
        '-cs', '--constrain_sample',
        action='store_true',
        help='''if not specified then False; then the distribution parameters 
        will be constrained'''
    )
    parser.add_argument(
        '--no_spline_renderer',
        action='store_true',
        help='if not specified then False'
    )
    parser.add_argument(
        '--intermediate_likelihood',
        type=str,
        default=None,
        choices=[None, 'Mean', 'Geom'],
        help='''Which intermediate likelihood to use for computing the final
        likelihood'''
    )
    parser.add_argument(
        '--dependent_prior',
        action='store_true',
        help='if specified then True',
    )
    parser.add_argument(
        '--residual_pixel_count',
        action='store_true',
        help='if specified then True',
    )
    parser.add_argument(
        '--sep_where_pres_net',
        action='store_true',
        help='if specified then True',
    )
    parser.add_argument(
        '--render_at_the_end',
        action='store_true',
        help='''This is useful when canvas is used. If Ture, a recon is computed 
        at the end is used for the likelihood loss, instead of the canvas'''
    )
    parser.add_argument('--no_maxnorm', action='store_true', 
                                     help='if specified then True.')
    parser.add_argument('--no_sgl_strk_tanh', action='store_true', 
                                     help='if specified then True.')
    parser.add_argument('--no_add_strk_tanh', action='store_true', 
                                     help='if specified then True.')
    parser.add_argument('--simple_pres', action='store_true',
                        help='''if specified, use residual pixel as z_pres param
                        specifically, z_pres_prob = residual ** r where r is
                        a positive integer''')
    parser.add_argument('--residual_no_target', action='store_true',
                        help='''if true then only residual is passed, not target
                        and residual''')
    parser.add_argument('--canvas_only_to_zwhere', action='store_true',
                        help='''only pass canvas to z_where (z_pres) rnn, not
                        z_what rnn. Because for the model with spline decoder,
                        fixed prior pass it to both makes the model unable to 
                        start doing good quality reconstructions''')
    parser.add_argument('--detach_canvas_so_far', action='store_true',
                        help='''detach canvas-so-far from the computation graph 
                        before passing into the rnn at each step, in the end, 
                        a None canvas is returned and the latent variables are 
                        used to render out recons. This simplifies the 
                        gradient graph.
                        Not using this is DISENCOURAGED.''')
    parser.add_argument('--detach_canvas_embed', action='store_true',
                        help='''detach the embedding of canvas, if it's used. 
                        The feature extractor cnn will not be updated through 
                        the gradient from canvas''')
    parser.add_argument('--detach_rsd_embed', action='store_true',
                        help='''detach the embedding of residual and 
                        residual ratio, if they are used. 
                        If False, residual will have a seperate cnn, thereby 
                        avoid interfereing with the learning for the other image
                        cnn;
                        If True, residual and img will share cnn, but there
                        won't be gradient from residual_embed to cnn.''')
    parser.add_argument('--no_detach_rsd', action='store_true')
    parser.add_argument('--constrain_z_pres_param', action='store_true',
                        help='''constrain the z_pres parameters according to the
                        schedule in loss.py''')
    # parser.add_argument('--update_reinforce_ll', action='store_true',
    #                      help='whether to modify the reinforce likelihood')
    parser.add_argument('--update_reinforce_loss', action='store_true',
                        help='''whether to center and normalize the reinforce loss
                        as in NVIL paper''')
    parser.add_argument('--update_reinforce_ll', action='store_true',
                        help='''doing this can reduce initial fluctuation, and
                        this wihout updating the reinforce_loss have been shown
                        to allow running to 300k iteration with lr scheduler.
                        ''')
    parser.add_argument('--anneal_non_pr_net_lr', action='store_true',
                        help='''only useable when sep_wr_pr_net is used, this
                        allows different learning rate scheduling on pr net and 
                        the rest''')
    parser.add_argument('--global_reinforce_signal', action='store_true',
                        help='''This concerns, for REINFORCE, at time t, whether
                        to include the KL loss from z^<t.''')
    parser.add_argument('--no_post_rnn', action='store_true',
                        help='''if specified, only use residual for passing
                        sequential information for the guide, not RNNs. The RNNs
                        are still used in sequential prior setting for the prior
                        MLPs. The target image is also detached before pres MLP
                        to avoid influences of REINFORCE to the CNN.''')
    parser.add_argument('--no_rnn', action='store_true', 
                        help='''This is used with sequential prior. 
                        For prior, we have:
                            a MLP for z_where and z_what respectively that 
                            inputs the canvas so far; outputs the prior 
                            distribution parameters a vector of learnable 
                            parameter for z_pres that’s instance independent.
                        For posterior, we have:
                            an MLP for z_pres, z_where z_what respectively that 
                            inputs residual and (optionally) the target; outputs 
                            the posterior parameters.
                            Shared between them, we have a CNN for extracting 
                            the image embeddings.''')
    parser.add_argument('--no_pres_rnn', action='store_true',
                        help='''remove pres rnn, keep the z_pres MLP for 
                        posterior param prediction, and use a num_strks-dim
                        learnable vector for the z_pres prior prob. The target
                        image is also detached before pres MLP to avoid 
                        influences of REINFORCE to the CNN.''')
    # parser.add_argument('--half_1s')

    # Baseline network
    parser.add_argument('--num_baseline_layers', default=3, type=int, help='')
    parser.add_argument('--bl_mlp_hid_dim', default=256, type=int, help='')
    parser.add_argument('--bl_rnn_hid_dim', default=256, type=int, help='')
    # parser.add_argument('--maxnorm', default=True, type=bool,
    #                                  help='if not specified then True.')

    # Optimization
    parser.add_argument("--continue_training", action="store_true", help=" ")
    parser.add_argument("--num-iterations", default=500000, type=int, help=" ")
    parser.add_argument("--bl_lr", default=1e-3, type=float, help='''
    1e-3 worked for Sequential though has collapse for vrnn; 
    1e-3 is suggested for AIR''')
    parser.add_argument("--lr", default=1e-3, type=float, help='''
    1e-3 worked for VAE, Base, Sequential
    1e-4 is suggested for AIR
    ''')
    parser.add_argument("--weight_decay", default=0.0, type=float, help="")
    parser.add_argument("--log-interval", default=50, type=int, help=" ")
    parser.add_argument("--save-interval", default=1000, type=int, help=" ")
    # Loss
    parser.add_argument("--loss", default="elbo", choices=['elbo','l1','nll'], 
        type=str, help=" ",
    )

    # Dataset
    parser.add_argument("--dataset",
                    # default="multimnist",
                    default='MNIST',
                    # default="generative_model",
                    choices=['MNIST', 'Omniglot', 'generative_model', 
                    'multimnist'],
                    type=str, help=" ")
    parser.add_argument("--data-dir", 
                        default="./omniglot_dataset/omniglot/",
                        type=str, help=" ")
    # 32 worked for all losses, but for elbo sometimes it miss the "-" in "7"s
    # 64 works well for elbos and most others loss (in "1, 7" dataset).
    # 128, the model stops learning anything quite often (in "1, 7").
    parser.add_argument("--batch-size", default=64, type=int, help=" ")
    parser.add_argument("--img_res", default=50, type=int, help=" ")
    parser.add_argument("--beta", default=1, type=float, 
                        help="beta term as in beta-VAE")
    parser.add_argument("--final_bern", default=.5, type=float, 
                        help="Minimal value for the z_pres Bern param")
    parser.add_argument("--anneal_lr", action='store_true',
        help='if not specified then False')
    parser.add_argument("--increase_beta", action='store_true',
        help='if not specified then False')
    parser.add_argument("--final_beta", default=1, type=float, 
                        help="Minimal value for the beta")
    parser.add_argument("--log_grad", action='store_true',
                        help="store gradient values in tensorboard")
    parser.add_argument("--log_param", action='store_true',
                    help="store distribution, rendering params in tensorboard")
    parser.add_argument("--save_history_ckpt", action='store_true',
                help='''store distribution, save not only latest checkput but
                also the all the past ones''')
    parser.add_argument("--tb_dir", default='./log/model',
                        type=str, 
                    help="tensorboard log dir, recommand: ./{dir}/{model_name}")


    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    pprint(vars(args))
    main(args)