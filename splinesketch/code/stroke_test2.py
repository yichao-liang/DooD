import numpy as np

import torch
import torch.nn.functional as F
import bezier


example_c = torch.tensor([[[[0.3867, 0.9570],
                            [0.0318, 0.2978],
                            [0.0449, 0.3531],
                            [0.8481, 0.6929]],
                           [[0.5208, 0.9020],
                            [0.6040, 0.4645],
                            [0.0715, 0.0193],
                            [0.2173, 0.9927]]]])


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
    return true_c, c, c.grad


if __name__ == '__main__':
    np.random.seed(0)

    # 1. x_hat = render(c); loss(x, x_hat)
    print("Case 1")
    r = generate_random_image(true_c=example_c.cuda(), noise_control_points=False, noise_img=False)
    print(*r, sep="\n")

    # 2. x_hat = render(c); loss(x_hat, x + noise)
    print("Case 2")
    r = generate_random_image(true_c=example_c.cuda(), noise_control_points=False, noise_img=True)
    print(*r, sep="\n")

    # 3. x_hat = render(c + noise); loss(x, x_hat)
    print("Case 3")
    r = generate_random_image(true_c=example_c.cuda(), noise_control_points=True, noise_img=False)
    print(*r, sep="\n")

    # 4. x_hat = render(c + noise); loss(x_hat, x + noise)
    print("Case 4")
    r = generate_random_image(true_c=example_c.cuda(), noise_control_points=True, noise_img=True)
    print(*r, sep="\n")

    print("Pixel is distance of:", 1./28)
