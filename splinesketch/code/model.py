import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.distributions as dist
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
import bezier
import kornia

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training dataset
train_loader = DataLoader(
    datasets.MNIST(root='../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.RandomRotation(30, fill=(0,)),
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=32, shuffle=True, num_workers=4
)
# Test dataset
test_loader = DataLoader(
    datasets.MNIST(root='../data', train=False,
                   transform=transforms.Compose([
                       transforms.RandomRotation(30, fill=(0,)),
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=32, shuffle=True, num_workers=4
)


def to_logit(x):
    eta = 1e-6
    renorm = ((x * 255) + torch.rand(x.shape)) / 256
    renorm = renorm.clamp(min=eta, max=(1. - eta))
    return renorm.log() - (1. - renorm).log()


def _get_kwargs(device):
    return {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}


def mnist_loaders(batch_size, shuffle=True, device="cuda", logit_tx=False):
    tx = transforms.ToTensor()
    if logit_tx:
        tx = transforms.Compose([tx, transforms.Lambda(to_logit)])
    train = DataLoader(datasets.MNIST('../data', train=True, download=True, transform=tx),
                       batch_size=batch_size, shuffle=shuffle, **_get_kwargs(device))
    test = DataLoader(datasets.MNIST('../data', train=False, download=True, transform=tx),
                      batch_size=batch_size, shuffle=shuffle, **_get_kwargs(device))
    return train, test

# Laplace instead of Gaussian for NN params


doc = """
The first pass for me will be can we tackle one glimpse and one layout
This will use MNIST first and then maybe Omniglot
"""


def get_design_matrix(points, order=3):
    t = torch.linspace(0, 1, points)
    feat = [comb(order, k) * t**(order - k) * (1 - t)**(k) for k in range(order + 1)]
    return torch.stack(feat)


def shrinkage_est(X, stroke, alpha=0.001):
    S, _ = torch.solve(X, (X @ X.T) + alpha * torch.eye(X.size()[0]))
    return S @ stroke.T


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 70)
        self.bz = bezier.Bezier(res=28, steps=400, method='bounded')

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x):
        # transform the input
        #x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        z = self.fc2(x)
        # print(crappyhist(x.cpu().detach().numpy().flatten()))
        x = self.bz(z.view((-1, 7, 5, 2)))
        return x


model = Net().to(device)
#optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

gauss = kornia.filters.GaussianBlur2d((7, 7), (5.5, 5.5))

def train(epoch):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        
        output = model(data)
        loss = F.l1_loss(gauss(output), gauss(data), reduction='sum') / gauss(output).sum()
        #loss = (torch.abs(output - data)/ 0.01).sum()

        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # save_image(make_grid(data, nrow=8), "original"+str(batch_idx)+".jpg")
            # save_image(make_grid(output, nrow=8), "reconstruct"+str(batch_idx)+".jpg")

def debug_one_datapoint(img, steps=100):
    "print control point, grad control point, image, and loss"
    save_image(img.cpu(), "../results/debug_recon/orig.png")

    losses = []
    control_points_ = []
    grad_control_points = []
    recons = []

    def update_grads(grad_cp):
        grad_control_points.append(grad_cp)

    for i in range(steps):
        img = img.to(device)
        optimizer.zero_grad()
        control_points, output = model(img)
        control_points.register_hook(update_grads)
        loss = F.l1_loss(output, img, reduction='sum') + 20*((control_points - 0.5) ** 2).sum()
        #loss = dist.Laplace(output, output.sum()).log_prob(img).sum() + 20*((control_points - 0.5) ** 2).sum()

        loss.backward()
        optimizer.step()

        losses.append(loss)
        control_points_.append(control_points)
        #if i % 20 == 0:
        recons.append(output.cpu())

    plot_figure(range(1, steps + 1), losses, "Gradient Steps", "Loss", "loss_plot.png")
    plot_summary(range(1, steps + 1),
                 control_points_,
                 "Gradient Steps",
                 "Control Points",
                 "../results/control_point.png")
    plot_summary(range(1, steps + 1),
                 grad_control_points,
                 "Gradient Steps",
                 "Grad Control Points",
                 "../results/control_point_grad_plot.png")
    save_image(make_grid(torch.cat(recons), nrow=10, pad_value=255), "../results/debug_recon/recons.png")



def plot_summary(x, y, xlabel, ylabel, out_file):
    plt.figure()
    plt.plot(x, list(map(torch.min,  y)), label="Min")
    plt.plot(x, list(map(torch.max,  y)), label="Max")
    plt.plot(x, list(map(torch.mean, y)), label="Mean")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(out_file)


def plot_figure(x, y, xlabel, ylabel, out_file):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(out_file)

def test_image():
    image, _ = train_loader.__iter__().next()
    return image

def test():
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.l1_loss(output, data, reduction='sum').item()

        n = min(data.size(0), 8)
        comparison = torch.cat([data[:n],
                                output.view(16, 1, 28, 28)[:n]])
        save_image(comparison.cpu(),
                   '../results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}\n'
              .format(test_loss))


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def crappyhist(a, bins=50, width=140):
    h, b = np.histogram(a, bins)

    for i in range (0, bins):
        print('{:12.5f}  | {:{width}s} {}'.format(
            b[i],
            '#'*int(width*h[i]/np.amax(h)),
            h[i],
            width=width))
    print('{:12.5f}  |'.format(b[bins]))

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


for epoch in range(1, 100 + 1):
    train(epoch)
    test()

