import numpy as np
import os
from scipy.io import loadmat
#  from scipy.misc import imresize
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
import bezier

from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import kornia

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS


ORDER = 10
IMSIZE = (64, 64)
ORIGINAL_SPLIT = False

files = "data_background", "data_evaluation"

example_c = torch.tensor([[[[0.3867, 0.9570],
                            [0.0318, 0.2978],
                            [0.0449, 0.3531],
                            [0.8481, 0.6929]],
                           [[0.5208, 0.9020],
                            [0.6040, 0.4645],
                            [0.0715, 0.0193],
                            [0.2173, 0.9927]]]])


omniglot_stroke = torch.tensor([[0.2502, 0.4032],
                                [0.2590, 0.3930],
                                [0.2678, 0.3828],
                                [0.2765, 0.3828],
                                [0.2765, 0.3624],
                                [0.2853, 0.3624],
                                [0.2940, 0.3522],
                                [0.2940, 0.3420],
                                [0.3028, 0.3318],
                                [0.3116, 0.3318],
                                [0.3203, 0.3318],
                                [0.3203, 0.3114],
                                [0.3291, 0.3114],
                                [0.3379, 0.3114],
                                [0.3466, 0.3012],
                                [0.3554, 0.3012],
                                [0.3642, 0.3012],
                                [0.3642, 0.2910],
                                [0.3729, 0.2910],
                                [0.3817, 0.2910],
                                [0.3904, 0.2910],
                                [0.3992, 0.2910],
                                [0.4167, 0.2808],
                                [0.4255, 0.2808],
                                [0.4255, 0.2706],
                                [0.4343, 0.2706],
                                [0.4430, 0.2706],
                                [0.4430, 0.2605],
                                [0.4518, 0.2605],
                                [0.4606, 0.2605],
                                [0.4693, 0.2503],
                                [0.4781, 0.2503],
                                [0.4869, 0.2503],
                                [0.4956, 0.2503],
                                [0.5044, 0.2503],
                                [0.5131, 0.2503],
                                [0.5219, 0.2503],
                                [0.5307, 0.2503],
                                [0.5394, 0.2503],
                                [0.5482, 0.2605],
                                [0.5570, 0.2605],
                                [0.5657, 0.2605],
                                [0.5657, 0.2706],
                                [0.5745, 0.2706],
                                [0.5745, 0.2808],
                                [0.5745, 0.2910],
                                [0.5833, 0.3012],
                                [0.5833, 0.3114],
                                [0.5920, 0.3216],
                                [0.6008, 0.3420],
                                [0.6008, 0.3522],
                                [0.6096, 0.3624],
                                [0.6096, 0.3726],
                                [0.6096, 0.3624],
                                [0.6271, 0.3624],
                                [0.6271, 0.3726],
                                [0.6358, 0.3726],
                                [0.6358, 0.3828],
                                [0.6446, 0.3930],
                                [0.6446, 0.4032],
                                [0.6534, 0.4032],
                                [0.6534, 0.4235],
                                [0.6534, 0.4337],
                                [0.6534, 0.4541],
                                [0.6534, 0.4643],
                                [0.6534, 0.4745],
                                [0.6534, 0.4949],
                                [0.6534, 0.5051],
                                [0.6534, 0.5153],
                                [0.6534, 0.5255],
                                [0.6534, 0.5357],
                                [0.6534, 0.5459],
                                [0.6534, 0.5561],
                                [0.6534, 0.5663],
                                [0.6534, 0.5765],
                                [0.6534, 0.5866],
                                [0.6534, 0.5968],
                                [0.6534, 0.6070],
                                [0.6534, 0.6172],
                                [0.6446, 0.6376],
                                [0.6358, 0.6580],
                                [0.6358, 0.6682],
                                [0.6358, 0.6784],
                                [0.6271, 0.6886],
                                [0.6096, 0.6988],
                                [0.6096, 0.7090],
                                [0.6096, 0.7192],
                                [0.5920, 0.7395],
                                [0.5833, 0.7497],
                                [0.5657, 0.7497],
                                [0.5570, 0.7497],
                                [0.5482, 0.7497],
                                [0.5307, 0.7497],
                                [0.5131, 0.7497],
                                [0.4956, 0.7497],
                                [0.4869, 0.7497],
                                [0.4869, 0.7395],
                                [0.4869, 0.7192],
                                [0.4869, 0.6988],
                                [0.4869, 0.6784],
                                [0.4869, 0.6682],
                                [0.4869, 0.6580],
                                [0.4869, 0.6376],
                                [0.4869, 0.6172],
                                [0.4956, 0.6172],
                                [0.5044, 0.6070],
                                [0.5219, 0.5968],
                                [0.5307, 0.5866],
                                [0.5394, 0.5765],
                                [0.5657, 0.5663],
                                [0.5833, 0.5561],
                                [0.5920, 0.5459],
                                [0.6096, 0.5459],
                                [0.6183, 0.5459],
                                [0.6358, 0.5459],
                                [0.6446, 0.5459],
                                [0.6534, 0.5459],
                                [0.6621, 0.5459],
                                [0.6797, 0.5459],
                                [0.6797, 0.5561],
                                [0.6884, 0.5561],
                                [0.6972, 0.5561],
                                [0.7060, 0.5663],
                                [0.7147, 0.5765],
                                [0.7235, 0.5765],
                                [0.7322, 0.5866],
                                [0.7410, 0.6070],
                                [0.7498, 0.6070],
                                [0.2502, 0.4032],
                                [0.2502, 0.4032]])

omniglot_control_points = torch.tensor(
    [[0.7548, 0.6123],
     [0.6443, 0.4919],
     [0.7105, 0.7078],
     [0.8238, 0.2596],
     [0.0130, 0.3619],
     [0.0307, 1.1353],
     [0.7727, 1.3094],
     [1.0870, 0.7478],
     [0.7578, 0.2820],
     [0.4565, 0.2969],
     [0.5482, 0.5052],
     [0.7421, 0.5812],
     [0.7294, 0.4376],
     [0.5894, 0.1612],
     [0.5040, 0.0048],
     [0.4795, 0.1801],
     [0.4538, 0.4300],
     [0.4004, 0.3149],
     [0.2909, 0.1986],
     [0.3116, 0.3730],
     [0.2507, 0.4009]]).unsqueeze(0).unsqueeze(0)

omniglot_img = bezier.Bezier(32, 2000, method="bounded")(omniglot_control_points)


def generate_random_image(res=28,
                          steps=16,
                          n_strokes=2,
                          n_control_points=4,
                          noise_control_points=False,
                          noise_img=False,
                          true_c=None,
                          true_img=None):
    "Generate an image using the forward model and then see if gradients stay put"
    b = bezier.Bezier(res, steps, method="bounded").cuda()
    if true_c is None:
        true_c = torch.rand(1, n_strokes, n_control_points, 2).cuda()

    if true_img is None:
        true_img = b(true_c)

    if noise_control_points:
        # Add some Gaussian noise
        true_c += torch.randn(true_c.size()).cuda() * 0.01

    if noise_img:
        # Add some Gaussian noise
        true_img += torch.randn(true_img.size()).cuda() * 0.01

    c = torch.tensor(true_c, requires_grad=True)
    opt = torch.optim.Adam([c], lr=1e-3)

    img = b(c)
    loss = F.mse_loss(img, true_img, reduction='sum')
    loss.backward()

    opt.step()
    return true_c, c, c.grad, loss


def test_decoder(true_img, true_control_points, n_iters, alpha=100, interval=100):
    true_img = true_img  # .cuda()

    n_strokes = 1
    n_control_points = 10

    b = bezier.Bezier(32, 2000, method="bounded")  # .cuda()
    # init_c = true_control_points + 0.2
    init_c = torch.rand(1, n_strokes, n_control_points, 2)  # .cuda()
    gauss = kornia.filters.GaussianBlur2d((7, 7), (5.5, 5.5))

    c = torch.tensor(init_c, requires_grad=True)
    opt = torch.optim.Adam([c], lr=1e-3)

    recons = []
    for i in range(n_iters):
        opt.zero_grad()
        img = b(c)

        loss = F.l1_loss(gauss(img), gauss(true_img), reduction='sum') / img.sum() + alpha*torch.abs(c - 0.5).sum()
        # loss = F.binary_cross_entropy_with_logits(img, true_img, reduction='sum') / img.sum()
        loss.backward()

        opt.step()
        if i % interval == 0:
            recons.append(img)
            print(loss)
    save_image(make_grid(torch.cat(recons), nrow=10, pad_value=255), "../results/debug_recon/recons0.png")
    return None #c, c.grad, loss

def test_decoder_conv(true_img, true_control_points, n_iters, alpha=100, interval=100):
    b = bezier.Bezier(32, 2000, method="bounded")
    init_c = true_control_points - 0.2
    # torch.rand(1, n_strokes, 5, 2).cuda()
    c = torch.tensor(init_c, requires_grad=True)
    opt = torch.optim.Adam([c], lr=1e-3)

    recons = []
    loss_graph = []
    for i in range(n_iters):
        img = b(c)

        conv_grid = -F.conv2d(true_img,img,padding=32) #Similitude map between generated and target images
        idx = (conv_grid-torch.min(conv_grid)==0).nonzero() #index of best similitude
        idxx, idxy = idx[0,2].item(), idx[0,3].item()
        
        # Shifting generated image to maximize similitude with target
        img_replace = torch.zeros_like(img)
        img_replace[:,:,max(0,idxx-32):idxx,max(0,idxy-32):idxy] = img[:,:,max(0,32-idxx):64-idxx,max(0,32-idxy):64-idxy]

        loss = F.l1_loss(img_replace, true_img, reduction='mean') / img_replace.sum()  + alpha*torch.abs(c - 0.5).sum()
        # loss = F.binary_cross_entropy_with_logits(img_replace, true_img, reduction='sum') / img_replace.sum()


        # Bluring shifted image
        # blur_radius = 7
        # blur_kernel = torch.zeros(1,1,2*blur_radius+1,2*blur_radius+1)
        # blur_kernel[0,0,blur_radius,blur_radius] = 1
        # blur_kernel = torch.from_numpy(gaussian_filter(blur_kernel.detach().numpy(),0.01))
        # true_img_blur = F.conv2d(true_img,blur_kernel,padding=blur_radius)
        # img_replace = F.conv2d(img_replace,blur_kernel,padding=blur_radius)

        # loss = F.l1_loss(img_replace, true_img_blur, reduction='mean') / img_replace.sum()  + alpha*torch.abs(c - 0.5).sum()
        # loss = F.binary_cross_entropy_with_logits(img_replace, true_img_blur, reduction='sum') / img_replace.sum()

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_graph.append(loss.item())
        if i % interval == 0:
            recons.append(img_replace)
            print(loss)
    save_image(make_grid(torch.cat(recons), nrow=10, pad_value=255), "../results/debug_recon/conv_recons0.png")

    plt.plot(loss_graph)
    plt.show()

    return None #c, c.grad, loss

def hmc_model(true_img, true_control_points, b):
    n_strokes = 1
    
    c_size = true_control_points.size()
    c = pyro.sample("c", dist.Beta(torch.ones(c_size), torch.ones(c_size)))
    # img = torch.nn.Linear(c_size[-1]*c_size[-2], 32*32)(c.view(-1)).reshape(32, 32)
    img = b(1.8*c - 0.9)
    pyro.sample("obs", dist.Laplace(img, 1e-2 * img.sum()), obs=true_img)


def test_decoder_hmc(model, n_iters, true_img, true_control_points):
    b = bezier.Bezier(32, 2000, method="bounded")
    nuts_kernel = NUTS(model)

    mcmc = MCMC(nuts_kernel, num_samples=n_iters, warmup_steps=n_iters//2)
    mcmc.run(true_img, true_control_points, b)

    hmc_samples = {k: v.detach() for k, v in mcmc.get_samples().items()}

    recons = []
    for i, c in enumerate(hmc_samples['c']):
        if i % 10 == 0:
            recons.append(b(c))

    save_image(make_grid(torch.cat(recons), nrow=10, pad_value=255), "../results/debug_recon/recons0.png")
    return None
    

def generate_from_omniglot(strokes, control_points, border=0.05):
    "Generate an image using ground truth control points fitted from Omniglot stroke data"
    s_x = strokes[:, 0]
    s_y = strokes[:, 1]

    c_x = control_points[:, 0]
    c_x = normalise(c_x, min(s_x), max(s_x), border)

    c_y = control_points[:, 1]
    c_y = normalise(c_y, min(s_y), max(s_y), border)

    return torch.stack((c_x, c_y)).T


def normalise(x, min_, max_, border=0.05):
    min_ -= border
    max_ += border
    return (x - min_) / (max_ - min_)


def comb(n, k):
    c = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    return c.exp()


def get_design_matrix(points, order=3):
    t = torch.linspace(0, 1, points)
    feat = [comb(order, k) * t**(order - k) * (1 - t)**(k) for k in range(order + 1)]
    return torch.stack(feat)


def shrinkage_est(X, stroke, alpha=0.001):
    S, _ = torch.solve(X, (X @ X.T) + alpha * torch.eye(X.size()[0]))
    return S @ stroke.T


def process(rawdata, imsize=None, order=3):
    images = []
    resized = []
    for alphabet in rawdata['images']:
        for symbol in alphabet[0]:
            for artist in symbol[0]:
                if imsize is not None:
                    resized.append(imresize(artist[0], imsize) < 0.5)
                images.append(artist[0] < 0.5)
    images = np.array(images, dtype=bool)
    if len(resized) > 0:
        resized = np.array(resized, dtype=bool)

    strokes = []
    for alphabet in rawdata['drawings']:
        for symbol in alphabet[0]:
            for artist in symbol[0]:
                character = []
                for stroke in artist[0]:
                    stroke = np.atleast_2d(stroke[0]) * np.array([1, 1])
                    X = get_design_matrix(len(stroke), order=order)
                    w = shrinkage_est(X, stroke).reshape((1, 2 * (order + 1)))
                    character.append(np.concatenate(([[len(stroke)]], w), 1))
                strokes.append(np.vstack(character))
    strokes = np.array(strokes)

    if imsize is not None:
        return images, strokes, resized
    else:
        return images, strokes, images


if __name__ == '__main__':
    pass
    
# if __name__ == '__main__':
#     path = os.path.dirname(__file__)
#     if ORIGINAL_SPLIT:
#         for file in files:
#             print("Parsing", file)
#             rawdata = loadmat(os.path.join(path, file+".mat"))
#             print("... loaded data,", len(rawdata['images']), "distinct characters; processing")
#             _, strokes, resized = process(rawdata, imsize=IMSIZE, order=ORDER)
#             print("... saving ...")
#             np.savez_compressed(os.path.join(path, file+".npz"), resized, strokes)
#             print("... saved downsampled images and bezier curves\n")
#     else:
#         print("Constructing randomized train/test split.")
#         all_strokes = None
#         all_images = None
#         for file in files:
#             print("Parsing", file)
#             rawdata = loadmat(os.path.join(path, file+".mat"))
#             print("... loaded data,", len(rawdata['images']), "distinct characters; processing")
#             _, strokes, resized = process(rawdata, imsize=IMSIZE, order=ORDER)
#             if all_strokes is None:
#                 all_strokes = strokes
#                 all_images = resized
#             else:
#                 all_strokes = np.concatenate((all_strokes, strokes))
#                 all_images = np.concatenate((all_images, resized))
#         # np.random.seed(0)
#         # N = all_strokes.shape[0]
#         # ordering = np.random.permutation(N)
#         # training_ratio = 0.8
#         # cutoff = int(training_ratio*N)
#         # print("Split: %d training images, %d test images" % (cutoff, N-cutoff))
#         # print("Saving training data ...")
#         # train_strokes = all_strokes[ordering][:cutoff]
#         # train_images = all_images[ordering][:cutoff]
#         # np.savez_compressed(os.path.join(path, "train_split.npz"), train_images, train_strokes)
#         # print("Saving test data ...")
#         # test_strokes = all_strokes[ordering][cutoff:]
#         # test_images = all_images[ordering][cutoff:]
#         # np.savez_compressed(os.path.join(path, "test_split.npz"), test_images, test_strokes)
#         print("Done")
