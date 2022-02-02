from pprint import pprint
from collections import namedtuple
import itertools

import util_lite as util
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.lite import LightningLite
from models import base, sequential, air, vae# , mws

class Lite(LightningLite):
    def run(self, args):
        # Cuda
        device = util.get_device()
        args.device = device

        # seed
        util.set_seed(args.seed)

        # Write will output to ./log
        # When doing sweep evaluation
        writer = SummaryWriter(log_dir=f"./log/debug/{args.save_model_name}")

        # When doing hyperop
        # writer = SummaryWriter(log_dir=f"./log/AIR_hyperop/" + 
        # writer = SummaryWriter(log_dir=f"./log/test/" + 
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
        # writer = SummaryWriter(log_dir=f"./log/AIR_hyperop_debug/name={args.save_model_name}-constrain_sample={args.constrain_sample}-z_where={args.z_where_type}-bl_lr={args.bl_lr}-n_lyr={args.num_baseline_layers}-mlp_h_dim={args.bl_mlp_hid_dim}-rnn_h_dim={args.bl_rnn_hid_dim}-maxnorm={str(not args.no_maxnorm)}-strk_tanh={str(not args.no_strk_tanh)}-render={args.render_method}")

        # initialize models, optimizer, stats, data
        checkpoint_path = util.get_checkpoint_path(args)
        num_iterations = args.num_iterations
        if not (args.continue_training and Path(checkpoint_path).exists()):
            util.logging.info("Training from scratch")
            model, optimizer, stats, data_loader = util.init(args, device)
        else:
            model, optimizer, stats, data_loader, args = util.load_checkpoint(
                                                        checkpoint_path, device)
        args.num_iterations = num_iterations

        if args.model_type == 'MWS':
            # todo
            (generative_model, guide, memory) = model
            generative_model = setup(generative_model)
            guide, optimizer = setup(guide, optimizer)
            memory = setup(memory)
        else:
            model, optimizer = self.setup(model, optimizer)

        train_loader, test_loader = data_loader
        train_loader = self.setup_dataloaders(train_loader)  # Scale your dataloaders

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

        self.train(model, optimizer, stats, train_loader, test_loader, args, 
                    writer)
    
    def train(self, model, optimizer, stats, train_loader, test_loader, args, 
                writer, dataset_name=None):
        
        checkpoint_path = util.get_checkpoint_path(args)
        if args.model_type == 'MWS':
            num_iterations_so_far = len(stats.theta_losses)
            num_epochs_so_far = 0
        else:
            num_iterations_so_far = len(stats.trn_losses)
            num_epochs_so_far = len(stats.tst_losses)
        iteration = num_iterations_so_far
        epoch = num_epochs_so_far

        if args.model_type != 'MWS':
            breakpoint()
            generative_model, guide = model.gen, model.guide
        else:
            mws_transform = transforms.Resize([args.img_res, args.img_res], 
                                                antialias=True)
            generative_model, guide, memory = model
            if memory is not None:
                util.logging.info(
                f"Size of MWS's memory of shape {memory.shape}: "
                f"{memory.element_size() * memory.nelement() / 1e6:.2f} MB"
                )
        guide.train()
        generative_model.train()
        
        data_size = len(train_loader.dataset)
        
        # For ploting first
        imgs, target = next(iter(train_loader))
        if args.model_type == 'MWS':
            obs_id = imgs.type(torch.int64)            
            imgs = mws_transform(target)
        imgs = util.transform(imgs)
        if dataset_name is not None and dataset_name == "Omniglot":
            imgs = 1 - imgs

        while iteration < args.num_iterations:

            # Log training reconstruction in Tensorboard
            with torch.no_grad():
                plot.plot_reconstructions(imgs=imgs, 
                                        guide=guide, 
                                        generative_model=generative_model, 
                                        args=args, 
                                        writer=writer, 
                                        epoch=epoch,
                                        writer_tag='Train',
                                        dataset_name=args.dataset)

            for imgs, target in train_loader:
                # Special data processing
                if args.model_type == 'MWS':
                    obs_id = imgs.type(torch.int64)            
                    imgs = mws_transform(target)

                if dataset_name is not None and dataset_name == "Omniglot":
                    imgs = 1 - imgs

                # prepare the data
                if iteration < np.inf:
                    imgs = util.transform(imgs)
                else:
                    imgs = imgs
                # fit on only 1 batch
                # imgs = one_batch.to(args.device)

                optimizer.zero_grad()
                loss_tuple = get_loss_tuple(args, generative_model, guide, 
                                            iteration, imgs, writer, obs_id)
                loss = loss_tuple.overall_loss.mean()
                self.backward(loss)

                # Constrain the baseline gradients
                # torch.nn.utils.clip_grad_norm_(guide.parameters(), max_norm=1e5)
                    
                # Log loss, gradients and some parameters
                if args.model_type == 'MWS':
                    for n, l in named_loss_list:
                        if l is not None:
                            writer.add_scalar("Train curves/"+n, l, iteration)
                else:
                    for n, l in zip(loss_tuple._fields, loss_tuple):
                        writer.add_scalar("Train curves/"+n, l.detach().mean(), 
                                                                        iteration)

                # Check for nans gradients, parameters
                named_params = get_model_named_params(args, guide, generative_model)
                for name, parameter in named_params:
                    try:
                        writer.add_scalar(f"Grad_norm/{name}", parameter.grad.norm(
                                                                2), iteration)
                        # check for abnormal gradients
                        # if (name == 'style_mlp.seq.linear_modules.2.weight' and
                        #     (parameter.grad.norm(2) > 6e4)):
                        #     print(f'{name} has grad_norm = {parameter.grad.norm(2)}')
                        #     util.debug_gradient(name, parameter, imgs, 
                        #                         guide, generative_model, optimizer,
                        #                         iteration, writer, args)
                        if torch.isnan(parameter).any() or \
                            torch.isnan(parameter.grad).any():
                            print(f"{name}.grad has {parameter.grad.isnan().sum()}"
                                f"/{np.prod(parameter.shape)} nan parameters")
                            breakpoint()
                    except Exception as e:
                        print(e)
                        breakpoint()

                optimizer.step()

                stats = log_stat(args, iteration, loss, loss_tuple)

                # Make a model tuple
                if args.model_type == 'MWS':
                    model = generative_model, guide, memory
                else:
                    model = generative_model, guide

                # Save Checkpoint
                if iteration % args.save_interval == 0 or iteration == \
                                                                args.num_iterations:
                    pass
                    # save(args, iteration, model, optimizer, stats)

                # End training based on `iteration`
                iteration += 1
                print(iteration)
                if iteration == args.num_iterations:
                    break
            epoch += 1
            writer.flush()

            # Test every epoch
            if test_loader:
                test_model(model, stats, test_loader, args, epoch=epoch, writer=writer)
        writer.close()
        # save(args, iteration, model, optimizer, stats)

        return model

def test_model(model, stats, test_loader, args, save_imgs_dir=None, epoch=None, 
                                                                writer=None):
    with torch.no_grad():
        test.marginal_likelihoods(model, stats, test_loader, args, 
                                            save_imgs_dir, epoch, writer, k=1,
                                            dataset_name=args.dataset)

def get_model_named_params(args, guide, gen):
    '''Return the trainable parameters of the models
    '''
    if args.model_type == 'Sequential' and args.prior_dist == 'Sequential':
            named_params = itertools.chain(
                    guide.named_parameters(), 
                    guide.internal_decoder.no_img_dist_named_params(),
                    generative_model.img_dist_named_params())
    elif args.model_type in ['AIR']:
        if not args.execution_guided and args.prior_dist == 'Sequential':
            named_params = itertools.chain(
                            guide.non_decoder_named_params(),
                            generative_model.no_img_dist_named_params()
                                        ) 
        elif args.execution_guided or args.prior_dist == 'Sequential':
            named_params = itertools.chain(
                                # guide.non_decoder_named_params(),
                                # guide.non_decoder_named_params(),
                                # generative_model.decoder_named_params()
                                        )
        elif args.prior_dist == 'Independent':
            named_params = itertools.chain(guide.named_parameters(),
                                    generative_model.named_parameters()) 
        elif not args.execution_guided:
            named_params = itertools.chain(
                                guide.non_decoder_named_params(),
                                generative_model.decoder_named_params())
        else:
            named_params = itertools.chain(
                                    guide.named_parameters(),
                                    generative_model.named_parameters())
    elif args.model_type in ['VAE']:
        named_params = itertools.chain(
                                guide.named_parameters(),
                                generative_model.named_parameters())
    else:
        named_params = guide.named_parameters()
    return named_params

MWSLoss = namedtuple('MWSLoss', ["neg_elbo", "theta_loss", "phi_loss", 
                                "prior_loss","accuracy", "novel_proportion", 
                                "new_map", "overall_loss"])
def get_loss_tuple(args, generative_model, guide, iteration, imgs, writer, 
                obs_id):
    if args.model_type == 'Base':
        base.schedule_model_parameters(generative_model, guide, 
                                    iteration, args.loss, args.device)
        loss_tuple = losses.get_loss_base(
                                generative_model, guide, imgs, 
                                loss=args.loss,)
    elif args.model_type == 'Sequential':
        loss_tuple = losses.get_loss_sequential(
                                generative_model=generative_model, 
                                guide=guide,
                                imgs=imgs, 
                                loss_type=args.loss, 
                                k=1,
                                iteration=iteration,
                                writer=writer)
    elif args.model_type == 'AIR':
        loss_tuple = losses.get_loss_air(
                                generative_model=generative_model, 
                                guide=guide,
                                imgs=imgs, 
                                loss_type=args.loss, 
                                iteration=iteration,
                                writer=writer)
    elif args.model_type == 'VAE':
        loss_tuple = losses.get_loss_vae(
                                generative_model=generative_model,
                                guide=guide,
                                imgs=imgs,
                                iteration=iteration,
                                writer=writer)
    elif args.model_type == 'MWS':
        loss_tuple = get_mws_loss_tuple(generative_model,
                                                guide,
                                                memory,
                                                imgs.squeeze(1).round(),
                                                obs_id,
                                                args.num_particles,)
    else:
        raise NotImplementedError

def get_mws_loss_tuple(generative_model,
                        guide,
                        memory,
                        imgs,
                        obs_id,
                        num_particles,):
    (loss,
    theta_loss,
    phi_loss,
    prior_loss,
    accuracy,
    novel_proportion,
    new_map) = get_mws_loss(
                            generative_model,
                            guide,
                            memory,
                            imgs.squeeze(1).round(),
                            obs_id,
                            args.num_particles,
                            )
    return MWSLoss(neg_elbo=loss, theta_loss=theta_loss, phi_loss=phi_loss, 
                    prior_loss=prior_loss, accuracy=accuracy, 
                    novel_proportion=novel_proportion, 
                    new_map=new_map, overall_loss=loss)

def log_stat(args, loss, loss_tuple):
    '''Log to stats and generate output for display
    '''
    # Record stats
    if args.model_type != 'MWS':
        stats.trn_losses.append(loss.item())
    else:
        stats.theta_losses.append(loss_tuple.theta_loss)
        stats.phi_losses.append(loss_tuple.phi_loss)
        stats.prior_losses.append(loss_tuple.prior_loss)
        if accuracy is not None:
            stats.accuracies.append(loss_tuple.accuracy)
        if novel_proportion is not None:
            stats.novel_proportions.append(loss_tuple.novel_proportion)
        if new_map is not None:
            stats.new_maps.append(loss_tuple.new_map)
    
    # Log
    if iteration % args.log_interval == 0:
        if args.model_type == 'MWS':
            util.logging.info(
                "it. {}/{} | prior loss = {:.2f} | theta loss = {:.2f} | "
                "phi loss = {:.2f} | accuracy = {}% | novel = {}% | new map = {}% "
                "| last log_p = {} | last kl = {} | GPU memory = {:.2f} MB".format(
                    iteration,
                    args.num_iterations,
                    loss_tuple.prior_loss,
                    loss_tuple.theta_loss,
                    loss_tuple.phi_loss,
                    loss_tuple.accuracy * 100 if loss_tuple.accuracy is not None else None,
                    loss_tuple.novel_proportion * 100 if loss_tuple.novel_proportion is not None else None,
                    loss_tuple.new_map * 100 if loss_tuple.new_map is not None else None,
                    "N/A" if len(stats.log_ps) == 0 else stats.log_ps[-1],
                    "N/A" if len(stats.kls) == 0 else stats.kls[-1],
                    (
                        torch.cuda.max_memory_allocated(device=args.device) / 1e6
                        if args.device.type == "cuda"
                        else 0
                    ),
                )
            )
        else:
            util.logging.info(f"Iteration {iteration} | Loss = {stats.trn_losses[-1]:.3f}")
    return stats

def save(args, iteration, model, optimizer, stats):
    # save iteration.pt
    util.save_checkpoint(
        util.get_checkpoint_path(args, 
        checkpoint_iteration=iteration),
        model,
        optimizer,
        stats,
        run_args=args,
    )
    # save latest.pt
    util.save_checkpoint(
        util.get_checkpoint_path(args),
        model,
        optimizer,
        stats,
        run_args=args,
    )

def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=
                                        argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=4)

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
    parser.add_argument(
        '-eg', "--execution_guided",
        action='store_true', 
        # default=True,
        help="if not declared, False"
    )
    parser.add_argument(
        '--exec_guid_type',
        # default='canvas_so_far',
        default='canvas', choices=['residual', 'canvas', 'target+residual'], 
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
        help='if specified then False'
    )
    parser.add_argument(
        '--intermediate_likelihood',
        type=str,
        default=None,
        choices=[None, 'Mean', 'Geom'],
        help='''Which intermediate likelihood to use for computing the final
        likelihood'''
    )

    # Baseline network
    parser.add_argument('--num_baseline_layers', default=3, type=int, help='')
    parser.add_argument('--bl_mlp_hid_dim', default=256, type=int, help='')
    parser.add_argument('--bl_rnn_hid_dim', default=256, type=int, help='')
    parser.add_argument('--no_maxnorm', action='store_true', 
                                     help='if specified then True.')
    parser.add_argument('--no_strk_tanh', action='store_true', 
                                     help='if specified then True.')
    # parser.add_argument('--maxnorm', default=True, type=bool,
    #                                  help='if not specified then True.')

    # Optimization
    parser.add_argument("--continue_training", action="store_true", help=" ")
    parser.add_argument("--num-iterations", default=50000, type=int, help=" ")
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
                    choices=['MNIST', 'omniglot', 'generative_model', 
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
    pprint(vars(args))
    Lite(strategy="ddp", devices=1, accelerator="gpu", precision="bf16"
         ).run(args)