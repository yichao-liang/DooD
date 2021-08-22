import util
import train
from pathlib import Path

def main(args):
    # Cuda
    device = util.get_device()
    args.device = device

    # seed
    util.set_seed(args.seed)

    # initialize models, optimizer, stats, data
    checkpoint_path = util.get_checkpoint_path(args)
    if not (args.continue_training and Path(checkpoint_path).exists()):
        util.logging.info("Training from scratch")
        model, optimizer, stats, data_loader = util.init(args, device)
    else:
        model, optimizer, stats, data_loader, _ = util.load_checkpoint(
                                                    checkpoint_path, device)

    # train
    train.train(model, optimizer, stats, data_loader, args)

def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=
                                        argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=1)

    # Model
    parser.add_argument(
        "--model-type",
        default="sequential", 
        # default="base", 
        choices=['base', 'sequential'],
        type=str,
        help=" "
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
        default=2,
        type=int,
        help="Maximum number of strokes per image"
    )
    parser.add_argument(
        "--points-per-stroke",
        default=5,
        type=int,
        help="Number of control points per stroke curve."
    )
    parser.add_argument(
        "--z_where_type",
        default='4_rotate',
        # default='3',
        type=str,
        choices=['3', '4_rotate', '4_no_rotate', '5'],
        help='''
        "3": (scale, shift x, y)
        "4_rotate": (scale, shift x, y, rotate)
        "4_no_rotate": (scale x, y, shift x, y)
        "5": (scale x, y, shift x, y, rotate)'''
    ) 
    parser.add_argument(
        "--input_dependent_render_param",
        # default=False,
        default=True,
        type=bool,
        help="",
    )
    parser.add_argument(
        "--execution_guided",
        default=False,
        # default=True,
        type=bool,
        help=" "
    )
    parser.add_argument(
        '--exec_guid_type',
        # default='canvas_so_far',
        default='canvas',
        choices=['residual', 'canvas', 'target+residual'],
        type=str,
        help="""Only useful is --execution_guided = True. 
        Residual: only uses the difference between the target and canvas-so-far;
        Canvas_so_far: stack the target image on canvas-so-far"""
    )
    parser.add_argument(
        "--transform_z_what",
        default=False,
        # default=True,
        type=bool,
        help=" "
    )

    # Optimization
    parser.add_argument("--continue-training", action="store_true", help=" ")
    parser.add_argument("--num-iterations", default=1000000, type=int, help=" ")
    parser.add_argument("--lr", default=1e-3, type=float, help=" ")
    parser.add_argument("--log-interval", default=50, type=int, help=" ")
    parser.add_argument("--save-interval", default=1000, type=int, help=" ")
    # Loss
    parser.add_argument(
        "--loss",
        default="elbo",
        choices=['elbo','l1','nll'],
        type=str,
        help=" ",
    )

    # Dataset
    parser.add_argument("--dataset",
                    # default="multimnist",
                    default='mnist',
                    # default="generative_model",
                    choices=['mnist', 'omniglot', 'generative_model', 
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

    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)