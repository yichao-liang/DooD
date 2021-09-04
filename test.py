"""test.py: Load the models and test it throught classification and marginal 
likelihood
"""
import os

import itertools
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image


import util, plot, losses, train

def marginal_likelihoods(model, stats, test_loader, args, save_imgs_dir=None,
                            epoch=None, writer=None, k=1, 
                            train_loader=None, optimizer=None, dataset_name=None):
    '''Compute the marginal likelihood through IWAE's k samples and log the 
    reconstruction through `writer` and `save_imgs_dir`.
        If `train_loader` and `optimizer` is not None, do some finetuning
    Args:
        test_loader (DataLoader): testset dataloader
    '''
    # Fining tunning
    if train_loader is not None and optimizer is not None:
        args.num_iterations = len(stats.trn_losses) + 200 # Finetuning iterations
        model = train.train(model, optimizer, stats, (train_loader, None), args, 
                            writer=writer)

    if args.model_type in ['Sequential', 'AIR']:
        cum_losses = [0]*8
    elif args.model_type in ['Base', 'VAE']:
        cum_losses = [0]*3

    generative_model, guide = model
    generative_model.eval(); guide.eval()

    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(args.device)

            if args.model_type == 'Base':
                loss_tuple = losses.get_loss_base(
                                        generative_model, guide, imgs, 
                                        loss=args.loss,)
            elif args.model_type == 'Sequential':
                loss_tuple = losses.get_loss_sequential(
                                                generative_model, guide,
                                                imgs, args.loss, k=k)
            elif args.model_type == 'AIR':
                loss_tuple = losses.get_loss_air(
                                                generative_model, 
                                                guide,
                                                imgs, 
                                                args.loss)
            elif args.model_type == 'VAE':
                loss_tuple = losses.get_loss_vae(
                        generative_model=generative_model,
                        guide=guide,
                        imgs=imgs)
            else:
                raise NotImplementedError
                
            for i in range(len(loss_tuple)):
                cum_losses[i] += loss_tuple[i].sum()      
        
        # Logging
        data_size = len(test_loader.dataset)
        for i in range(len(cum_losses)):
            cum_losses[i] /= data_size

        if args.model_type in ['Sequential', 'AIR']:
            loss_tuple = losses.SequentialLoss(*cum_losses)
        elif args.model_type in ['Base', 'VAE']:
            loss_tuple = losses.BaseLoss(*cum_losses)

        if stats is not None:
            stats.tst_losses.append(loss_tuple.overall_loss)
        
        for n, l in zip(loss_tuple._fields, loss_tuple):
            # writer.add_scalar(f"{test_loader.dataset.__name__}/Test curves/"+n, 
            if dataset_name is not None:
                writer.add_scalar(f"{dataset_name}/Test curves/"+n, l, epoch)
            else:
                writer.add_scalar("Test curves/"+n, l, epoch)
        # writer.add_scalars("Test curves", {n:l for n, l in 
        #                         zip(loss_tuple._fields, loss_tuple)}, epoch)   
        util.logging.info(f"Epoch {epoch} Test loss | Loss = {loss_tuple.overall_loss:.3f}")

        plot.plot_reconstructions(imgs=imgs, 
                                    guide=guide, 
                                    generative_model=generative_model, 
                                    args=args, 
                                    writer=writer, 
                                    epoch=epoch,
                                    is_train=False)

        writer.flush()

def classification_evaluation(guide, args, writer, dataset):
    '''Train two classification models:
    one using the `guide` output; 
    the other using the raw image output
    and log both's accuracy throught `writer`
    '''
    guide_classifier, raw_classifier, dataloaders, optimizer, stats = \
                                init_classification_nets(guide, args, dataset)
    guide_clf_stats, raw_clf_stats = stats
    train_loader, test_loader = dataloaders
    log_interval = 100
    num_iterations = 10000
    iteration = 0 # iterations-so-far
    epoch = 0
    while iteration < num_iterations:
        # Train
        for imgs, target in train_loader:
            imgs, target = imgs.to(args.device), target.to(args.device)
            # Get the z through guide
            with torch.no_grad():
                zs = guide(imgs).z_smpl
                bs = imgs.shape[0]
                zs = torch.cat([rvs.view(bs, -1) for rvs in zs], dim=-1)
           
            # Forward
            guide_clf_pred = guide_classifier(zs)
            raw_clf_pred = raw_classifier(imgs.view(bs, -1))

            guide_clf_loss = F.nll_loss(guide_clf_pred, target, )
            raw_clf_loss = F.nll_loss(raw_clf_pred, target)

            # Backward, optimize
            guide_clf_loss.backward()
            raw_clf_loss.backward()
            optimizer.step()

            # Log
            if iteration % log_interval == 0:
                util.logging.info('Train iteration: {}/{}\tguide_clf_Loss: {:.6f}, raw_clf_Loss: {:.6f}'.format(
                    iteration, num_iterations, guide_clf_loss, raw_clf_loss))
            writer.add_scalar(f"{dataset}/Classification.GuideClassifier.Train.Accuracy", 
                                                      guide_clf_loss, iteration)
            writer.add_scalar(f"{dataset}/Classification.RawClassifier.Train.Accuracy", 
                                                        raw_clf_loss, iteration)
            guide_clf_stats.trn_accuracy.append(guide_clf_loss.item())
            raw_clf_stats.trn_accuracy.append(raw_clf_loss.item())

            iteration += 1

        epoch += 1 
        # Test
        with torch.no_grad():
            guide_clf_loss_sum, raw_clf_loss_sum = 0, 0
            for imgs, target in test_loader:
                bs = imgs.shape[0]
                imgs, target = imgs.to(args.device), target.to(args.device)

                # Get the z through guide
                zs = guide(imgs).z_smpl
                zs = torch.cat([rvs.view(bs, -1) for rvs in zs], dim=-1)
                
                # Forward
                guide_clf_pred = guide_classifier(zs)
                raw_clf_pred = raw_classifier(imgs.view(bs, -1))

                guide_clf_loss = F.nll_loss(guide_clf_pred, target)
                raw_clf_loss = F.nll_loss(raw_clf_pred, target)

                guide_clf_loss_sum += guide_clf_loss
                raw_clf_loss_sum += raw_clf_loss

            n = len(test_loader.dataset)
            guide_clf_loss = guide_clf_loss_sum / n
            raw_clf_loss = raw_clf_loss_sum / n
            # Log
            util.logging.info('Test epoch {}: guide_clf_Loss: {:.6f}, raw_clf_Loss: {:.6f}'.format(
                    epoch, guide_clf_loss, raw_clf_loss))
            writer.add_scalar(f"{dataset}.Classification.GuideClassifier.Test.Accuracy", 
                                                    guide_clf_loss, epoch)
            writer.add_scalar(f"{dataset}.Classification.RawClassifier.Test.Accuracy", 
                                                        raw_clf_loss, epoch)
            guide_clf_stats.tst_accuracy.append(guide_clf_loss.item())
            raw_clf_stats.tst_accuracy.append(raw_clf_loss.item())

def init_classification_nets(guide, args, dataset):
    # Dataset
    train_loader, test_loader = init_dataloader(guide.img_dim[-1], dataset)

    # Models
    guide_classifier_in_dim = guide.max_strks * (guide.z_where_dim + 
                                                  guide.z_what_dim + 1)
    raw_classifier_in_dim = np.prod(guide.img_dim)
    classifier_out_dim = len(set(test_loader.dataset.targets.numpy()))
    
    guide_classifier = util.init_mlp(in_dim=guide_classifier_in_dim, 
                                     out_dim=classifier_out_dim,
                                     hidden_dim=256,
                                     num_layers=3,).to(args.device)
    raw_classifier = util.init_mlp(in_dim=raw_classifier_in_dim,
                                   out_dim=classifier_out_dim,
                                   hidden_dim=256,
                                   num_layers=3,).to(args.device)
    
    # Optimizer
    params = itertools.chain(guide_classifier.parameters(),
                             raw_classifier.parameters())
    optimizer = torch.optim.Adam(params, 
                                 lr=0.001)
    
    # Stats
    stats = (util.ClfStats([], []), util.ClfStats([], []))

    return (guide_classifier, raw_classifier, (train_loader, test_loader), 
            optimizer, stats)

def init_dataloader(res, dataset):
    # Dataloader
    # Train dataset
    res = guide.img_dim[-1]
    transform = transforms.Compose([
                            transforms.Resize([res,res], antialias=True),
                            transforms.ToTensor(),
                            ])
    if dataset == "EMNIST":
        trn_dataset = datasets.EMNIST(root='./data', train=True, split='balanced',
                            transform=transform, download=True)
        tst_dataset = datasets.EMNIST(root='./data', train=False, split='balanced',
                            transform=transform, download=True)
    elif dataset == 'KMNIST':
        trn_dataset = datasets.KMNIST(root='./data', train=True,
                            transform=transform, download=True)

        tst_dataset = datasets.KMNIST(root='./data', train=False,
                            transform=transform, download=True)
    elif dataset == 'QMNIST':
        trn_dataset = datasets.QMNIST(root='./data', train=True,
                            transform=transform, download=True)
        tst_dataset = datasets.QMNIST(root='./data', train=False,
                            transform=transform, download=True)
    elif dataset == 'Omniglot':
        trn_dataset = datasets.Omniglot(root='./data', train=True,
                            transform=transform, download=True)
        tst_dataset = datasets.Omniglot(root='./data', train=False,
                            transform=transform, download=True)        
    elif dataset == 'Quickdraw':
        data_dirs = {
            'train': './data/quickdraw/train/',
            'valid': './data/quickdraw/valid/',
            'test': './data/quickdraw/test/',
        }  

        data_transforms = {
            'train':transforms.Compose([
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
        #             transforms.Normalize([0.155], [0.316])
                ]),
            'valid':transforms.Compose([
                    transforms.ToTensor(),
        #             transforms.Normalize([0.155], [0.316])
                ]),
            'test':transforms.Compose([
                    transforms.ToTensor(),
        #             transforms.Normalize([0.155], [0.316])
                ])
        }
        files = os.listdir(data_dirs['train'])
        idx_to_class = sorted([f.split('_')[-1].split('.')[0] for f in files])

        class_to_idx = {idx_to_class[i]: i for i in range(len(idx_to_class))}
        dataset = {}
        # mean, std = None, None

        for d in ['train', 'valid', 'test']:
            data_x = []
            data_y = []
            for path, _, files in os.walk(data_dirs[d]):
                for f in files:
                    c = f.split('_')[-1].split('.')[0] # get class name from file name
                    x = np.load(path + f).reshape(-1, 28, 28) / 255
                    y = np.ones((len(x), 1), dtype=np.int64) * class_to_idx[c]
                    
                    data_x.extend(x)
                    data_y.extend(y)
        #     if d == 'train':
        #         mean = np.mean(data_x)
        #         std = np.std(data_x)
            dataset[d] = torch.utils.data.TensorDataset(torch.stack(
                [data_transforms[d](Image.fromarray(np.uint8(i*255))) for i in 
                data_x]), torch.stack([torch.Tensor(j) for j in data_y]))
        trn_dataset = dataset['train']
        tst_dataset = dataset['test']
    else: raise NotImplementedError



    train_loader = DataLoader(trn_dataset, batch_size=64, shuffle=True, 
                                                            num_workers=4)
    test_loader = DataLoader(tst_dataset, batch_size=64, shuffle=True, 
                                                            num_workers=4)
    
    return train_loader, test_loader

def get_args_parser():
    parser = argparse.ArgumentParser(formatter_class=
                                        argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt_path", 
        default="./save/sequential_air_grad-constrain/checkpoints/143000.pt",
        type=str,
        help="Path to checkpoint for evaluation")
    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    test_datasets = ["EMNIST", "KMNIST", "QMNIST", "Omniglot", "Quickdraw"]
    
    # Init
    device = util.get_device()
    args.device = device
    (gen, guide), optimizer, stats, _, trn_args = util.load_checkpoint(
                                                path=args.ckpt_path,
                                                device=device)
    model = (gen, guide)
    writer = SummaryWriter(log_dir="./log/classification",)

    for dataset in test_datasets:
        # Evaluation marginal likelihood
        train_loader, test_loader = init_dataloader(gen.res, dataset)
        marginal_likelihoods(model=model, stats=stats, test_loader=test_loader, 
                            args=trn_args, save_imgs_dir=None, epoch=None, 
                            writer=writer, k=1,
                            # train_loader=None, optimizer=None)
                            train_loader=train_loader, optimizer=optimizer)

        # Evaluation classification
        classification_evaluation(guide, args, writer, dataset)    
