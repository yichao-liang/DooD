import util
import train
from pathlib import Path

def main(args):
    # cuda
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

    # model
    parser.add_argument(
        "--model-type",
        default="base", # base is a 2-level (obs, hidden) model
        type=str,
        help=" "
    )
    parser.add_argument(
        "--prior_dist",
        default='Dirichlet',
        # default='Normal',
        type=str,
        choices=['Dirichlet', 'Normal'],
        help=""
    )
    parser.add_argument(
        "--inference-dist",
        default='Dirichlet',
        # default='Normal',
        type=str,
        choices=['Dirichlet', 'Normal'],
        help=""
    )

    # optimization
    parser.add_argument("--continue-training", action="store_true", help=" ")
    parser.add_argument("--num-iterations", default=100000, type=int, help=" ")
    parser.add_argument("--lr", default=1e-3, type=float, help=" ")
    parser.add_argument("--log-interval", default=50, type=int, help=" ")
    parser.add_argument("--save-interval", default=1000, type=int, help=" ")
    # parser.add_argument("--checkpoint-interval", default=1000, type=int, help=" ")

    # Loss
    parser.add_argument(
        "--loss",
        default="elbo",
        type=str,
        help=" ",
    )

    # data
    parser.add_argument("--dataset",
                    default="mnist",
                    choices=['mnist', 'omniglot'],
                    type=str, help=" ")
    parser.add_argument("--data-dir", 
                        default="./omniglot_dataset/omniglot/",
                        type=str, help=" ")
    parser.add_argument("--batch-size", default=64, type=int, help=" ")

    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)