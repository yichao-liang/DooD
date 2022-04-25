#!/bin/python3

import argparse
import torch
import numpy as np
from time import time
from einops import rearrange

def comb(n, k):
    c = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    return c.exp()

class Bezier(torch.nn.Module):
    """
    Takes in tensors shaped (batches, curves, control_points, x/y)
    and returns tensors representing greyscale images shaped (batches, height, width)
    """
    def __init__(self, res=512, steps=128, method='base', debug=False):
        super().__init__()
        self.res = res
        self.steps = steps
        self.method = method
        self.debug = debug

        # C.shape: [res, res]
        C, D = torch.meshgrid(torch.Tensor(range(self.res)), torch.Tensor(range(self.res)))
        # c.shape [1, 1, res, res]
        self.c = torch.nn.Parameter(C.unsqueeze(0).unsqueeze(0) / res, requires_grad=False)
        self.d = torch.nn.Parameter(D.unsqueeze(0).unsqueeze(0) / res, requires_grad=False)

        if method == 'base':
            self.raster = self._raster_base
        if method == 'half':
            self.raster = self._raster_half
        elif method == 'bounded':
            self.raster = self._raster_bounded
        elif method == 'bounded_tight':
            self.raster = self._raster_bounded_tight
        elif method == 'shrunk':
            self.raster = self._raster_shrunk

    def sample_curve(self, control_points, t):
        '''
        Args: 
            control_points [batch_shape, n_strks, n_points, 2]
            t [self.steps]
        '''
        order = control_points.size()[2] - 1
        order = torch.tensor(order).float().to(t.device)
        
        # feat: [n_points, steps]
        feat = torch.stack([comb(order, k) * t**(order - k) * (1 - t)**(k)
                            for k in torch.arange(order + 1).float().to(t.device)])
        # curve: [bs, n_strks, 2, steps]
        curve = torch.einsum('bcki,kt->bcit', control_points, feat)
        return curve
    
    def get_sample_curve(self, control_points, n_steps):
        '''Get [b, strk, 2 (x, y), n_steps] sample curve coordinates
        '''
        steps = torch.linspace(0, 1, n_steps).to(control_points.device)

        return self.sample_curve(control_points, steps)
        
    def forward(self, control_points, sigma:float, keep_strk_dim:bool):
        '''
        Args:
            control_points: [batch_shape, n_strks, n_points, 2]
            sigma (float): Default: 1e-2. Controls the bandwidth of the Gaussian
                kernel for rendering. The higher, the larger range of curve
                points that it takes into consideration.
            keep_strk_dim: If `True`, return each stroke's rendering.
        Return:
            if keep_strk_dim: [batch_size, n_strk, n_channel (1), H, W]
            else: [batch_size, n_channels (1), H, W]
        '''
        # BCKXY -> BHW
        # steps.shape: [self.steps]
        steps = torch.linspace(0, 1, self.steps).to(control_points.device)
        shape = control_points.shape[:-2]

        if keep_strk_dim:
            # curve coordinates at each of the step t out of the e.g. 100 steps.
            # [b, n_strks, 2, steps] -> [(b * n_strks), 2, steps]
            curve = rearrange(self.sample_curve(control_points, steps),
                            'b strk xy pts -> (b strk) xy pts')
            return self.raster(curve, sigma).view(*shape, 1, self.res, self.res)
        else:
            # [b, n_strks, xy, n_pts] -> [b, xy, (n_strks * n_pts)]
            curve = rearrange(self.sample_curve(control_points, steps),
                            'b c k s -> b k (c s)')
            breakpoint()
            return self.raster(curve, sigma)


    def _raster_base(self, curve, sigma=5e-2):
        '''Raster image from curve sample points
        Args:
            curve [bs, 2, steps]
            sigma float or [bs, 1]
        Return:
            img [bs, 1, res, res]
        '''
        tic = time()

        # x: [bs, 1, 1, steps]
        x = curve[:, 0].unsqueeze(1).unsqueeze(1)
        y = curve[:, 1].unsqueeze(1).unsqueeze(1)
        batch_size = curve.size()[0]
        steps = curve.size()[-1]
        
        # x_ = x.expand(self.res, self.res, steps)
        # y_ = y.expand(self.res, self.res, steps)
        # c: [1, 1, res, res] -> [bs, steps, res, res] -> [bs, res, res, steps]
        c = torch.transpose(self.c.expand(batch_size, steps, self.res, self.res)
                                                                        ,1, 3)
        d = torch.transpose(self.d.expand(batch_size, steps, self.res, self.res)
                                                                        ,1, 3)

        if self.debug:
            print(time() - tic)

        if torch.is_tensor(sigma):
            # size [bs, 1, 1, 1]
            sigma = sigma.flatten()[:, None, None, None]
        raster = torch.exp((-(x - c)**2 - (y - d)**2) / (2*sigma**2))
        # raster = torch.mean(raster, dim=2)
        # raster = torch.min(torch.sum(raster, dim=2), torch.Tensor([1]).to(self.device))

        # [bs, res, res]
        raster = torch.sum(raster, dim=3)
        #  raster = torch.max(raster, dim=2)[0]
        if self.debug:
            print(time() - tic)

        # return torch.transpose(torch.squeeze(raster), 0, 1)
        return raster.float().unsqueeze(1)

    def _raster_half(self, curve, sigma=1e-2):
        tic = time()

        x = curve[0]
        y = curve[1]

        steps = curve.size()[1]
        x_ = x.half()
        y_ = y.half()
        c = torch.transpose(self.c.expand(steps, self.res, self.res), 0, 2).half()
        d = torch.transpose(self.d.expand(steps, self.res, self.res), 0, 2).half()

        if self.debug:
            print(time() - tic)

        raster = torch.exp((-(x_ - c)**2 - (y_ - d)**2) / (2*sigma**2))
        #  raster = torch.mean(raster, dim=2)
        raster = torch.sum(raster, dim=2)
        if self.debug:
            print(time() - tic)

        return torch.transpose(torch.squeeze(raster.float()), 0, 1)

    def _raster_bounded(self, curve, sigma=1e-2 * 2):
        '''
        Args:
            curve (tensor): [batch_size, curves_per_img, steps]
            sigma (float): Default: 1e-2. Controls the bandwidth of the Gaussian
                kernel for rendering. The higher, the larger range of curve
                points that it takes into consideration.
                **For the loss with prior work, sigma should be at least 0.02**

        Return:
            imgs [batch_size, 1, res, res]
        '''
        # has shape [batch_size, 1, 1, steps]
        x = curve[:, 0].unsqueeze(1).unsqueeze(1)
        y = curve[:, 1].unsqueeze(1).unsqueeze(1)
        batch_size = curve.size()[0]
        steps = curve.size()[-1]
        # TODO figure out how to do per-image bounding
        # start of modification
        # xmin = x.min(-1, True)[0]  #1).values
        # ymin = y.min(-1, True)[0]  #1).values
        # # normalize x, y before rendering
        # x = x - xmin
        # y = y - ymin
        # xmax = x.max(-1, True)[0].unsqueeze(0)  #1).values
        # ymax = y.max(-1, True)[0].unsqueeze(0)  #1).values
        # scale = torch.cat([xmax,ymax], 0).max(0)[0]
        # x =  x / scale
        # y = y / scale
        # end to modification
        xmax = x.max()  #1).values
        ymax = y.max()  #1).values 
        xmin = x.min()  #1).values
        ymin = y.min()  #1).values
        # assert xmax <= 1. and ymax <= 1. and xmin >= 0., ymin >= 0.
        if torch.is_tensor(sigma):
            sig_max = sigma.detach().max()
            xmax = torch.clamp((self.res * (xmax + 3*sig_max)).ceil(), 0, 
                                                                self.res).int()
            ymax = torch.clamp((self.res * (ymax + 3*sig_max)).ceil(), 0, 
                                                                self.res).int()
            xmin = torch.clamp((self.res * (xmin - 3*sig_max)).floor(), 0, 
                                                                self.res).int()
            ymin = torch.clamp((self.res * (ymin - 3*sig_max)).floor(), 0, 
                                                                self.res).int()
        else:
            xmax = torch.clamp((self.res * (xmax + 3*sigma)).ceil(), 0, 
                                                                self.res).int()
            ymax = torch.clamp((self.res * (ymax + 3*sigma)).ceil(), 0, 
                                                                self.res).int()
            xmin = torch.clamp((self.res * (xmin - 3*sigma)).floor(), 0, 
                                                                self.res).int()
            ymin = torch.clamp((self.res * (ymin - 3*sigma)).floor(), 0, 
                                                                self.res).int()

        # if x.is_cuda:
        #     x = x.half()
        #     y = y.half()

        #     c = torch.transpose(self.c.half().expand(batch_size, steps, 
        #                     self.res, self.res), 1, 3)[:, xmin:xmax, ymin:ymax]
        #     d = torch.transpose(self.d.half().expand(batch_size, steps, 
        #                     self.res, self.res), 1, 3)[:, xmin:xmax, ymin:ymax]
        # else:

        # # has shape [batch_size, res, res, steps]
        c = torch.transpose(self.c.expand(batch_size, steps, self.res, 
                            self.res), 1, 3)[:, xmin:xmax, ymin:ymax]
        d = torch.transpose(self.d.expand(batch_size, steps, self.res, 
                            self.res), 1, 3)[:, xmin:xmax, ymin:ymax]
        # breakpoint()
        
        '''
        x, y: curve-point coords, c, d: meshgrid coords (copied for each batch 
        element and each step)
        The first line put a Guassian kernel on each point in the meshgrid
        '''
        if torch.is_tensor(sigma):
            sigma = sigma.flatten()[:, None, None, None]
        raster_ = torch.exp((-(x - c)**2 - (y - d)**2) / (2*sigma**2))
        #  raster_ = torch.mean(raster_, dim=2)
        raster_ = torch.sum(raster_, dim=3)
        raster = torch.zeros([batch_size, self.res, self.res]).to(curve.device)
        raster[:, xmin:xmax, ymin:ymax] = raster_

        return raster.float().unsqueeze(1)

    def _raster_bounded_tight(self, curve, sigma=1e-2):
        tic = time()
        print(curve)
        # align start and end points
        theta = torch.atan(curve[1, -1] / curve[0, -1])
        print(theta)
        R = torch.Tensor([[theta.cos(), theta.sin()], [-theta.sin(), theta.cos()]]).to(curve.device)

        T = curve[:, 0].expand(steps, 2).transpose(0, 1)
        curve -= T
        print(curve)
        curve = R.matmul(curve)
        print(R)
        print(curve)
        x = curve[0]
        y = curve[1]
        xmax, ymax = [(self.res * (i.max() + 3*sigma)).ceil().int().item() for i in (x, y)]
        xmin, ymin = [(self.res * (i.min() - 3*sigma)).floor().int().item() for i in (x, y)]
        w = xmax-xmin
        h = ymax-ymin
        print(xmin, xmax)
        print(ymin, ymax)
        x_ = x.expand(w, h, steps)
        y_ = y.expand(w, h, steps)
        c = self.c[self.res+xmin:self.res+xmax, self.res+ymin:self.res+ymax]
        d = self.d[self.res+xmin:self.res+xmax, self.res+ymin:self.res+ymax]
        print(c)
        print(d)
        raster_ = torch.exp((-(x_ - c)**2 - (y_ - d) ** 2) / (2*sigma**2))
        #  raster_ = torch.mean(raster_, dim=2)
        raster_ = torch.sum(raster_, dim=2)
        raster = torch.zeros([2*self.res, 2*self.res])
        raster[self.res+xmin:self.res+xmax, self.res+ymin:self.res+ymax] = raster_

        if self.debug:
            print(time() - tic)

        return torch.transpose(torch.squeeze(raster), 0, 1)

    def _raster_shrunk(self, curve, sigma=1e-2):
        tic = time()

        x = curve[0]
        y = curve[1]

        steps = curve.size()[1]

        raster = torch.zeros([self.res, self.res], requires_grad=False).to(curve.device)
        spread = 2 * sigma
        # nextpow2 above 2 standard deviations in both x and y
        w = 2*int(2**np.ceil(np.log2(self.res*spread)))
        if self.debug:
            print(w)
        # lower left corner of a w*w block centered on each point of the curve
        blocks = torch.clamp((self.res * curve).floor().int() - w // 2, 0,  self.res - w)

        #  blocks = []
        #  mask = torch.zeros([self.res, self.res]).byte()
        #  # non overlapping blocks instead
        #  for point in torch.t(curve):
            #  x, y = torch.clamp((self.res * point).floor().int() - w // 2, 0,  self.res - w)

            #  mask[x:x+w, y:y+w] = 1
            #  blocks.append([x, y])

        # chunked
        # xmax, ymax = (self.res * (curve + spread)).ceil().int()
        # xlim = torch.stack([xmin, xmax], 1)
        # ylim = torch.stack([ymin, ymax], 1)
        # print(x.size())
        #  for point in torch.t(curve)[::]:
            #  xmax, ymax = (self.res * (point + spread)).ceil().int().tolist()
            #  xmin, ymin = (self.res * (point - spread)).floor().int().tolist()
            #  chunks.append([xmin, xmax, ymin, ymax])
        #  xmax, ymax = [(self.res * (i.max() + 3*sigma)).ceil().int().item() for i in (x, y)]
        #  xmin, ymin = [(self.res * (i.min() - 3*sigma)).floor().int().item() for i in (x, y)]

        # w * w * steps
        #  c = torch.zeros([w, w, steps])
        #  d = torch.zeros([w, w, steps])
        #  for t, (px, py) in enumerate(torch.t(blocks)):
            #  c[:,:,t] = self.c[px:px+w, py:py+w, t]
            #  d[:,:,t] = self.d[px:px+w, py:py+w, t]
        if self.debug:
            print('{}: Bounding rectangles found.'.format(time() - tic))
        x_ = x.expand(w, w, steps)
        y_ = y.expand(w, w, steps)
        c_ = torch.transpose(self.c.expand(steps, self.res, self.res), 0, 2)
        d_ = torch.transpose(self.d.expand(steps, self.res, self.res), 0, 2)
        c = torch.stack([c_[px:px+w, py:py+w, t] for t, (px, py) in enumerate(torch.t(blocks))], dim=2)
        d = torch.stack([d_[px:px+w, py:py+w, t] for t, (px, py) in enumerate(torch.t(blocks))], dim=2)
        if self.debug:
            print('{}: Bounding rectangles found.'.format(time() - tic))
        if self.debug:
            print('{}: Dims expanded.'.format(time() - tic))
        raster_ = torch.exp((-(x_ - c)**2 - (y_ - d)**2) / (2*sigma**2))
        # raster_ = (x_ - c)**2 + (y_ - d)**2
        if self.debug:
            print('{}: Gradient generated.'.format(time() - tic))
        #  idx = torch.LongTensor
        #  self.r.scatter_(2, raster_)
        for t, (x, y) in enumerate(torch.t(blocks)):
            raster[x:x+w, y:y+w] += raster_[:,:,t]
        # raster = torch.mean(self.r, dim=2)

        raster = torch.min(raster, torch.Tensor([1]).to(curve.device))
        #  for xmin, xmax, ymin, ymax in segments:
            #  w = xmax-xmin
            #  h = ymax-ymin
            #  print(w, h)
            #  x_ = x.expand(w, h, steps)
            #  y_ = y.expand(w, h, steps)
            #  #  x_ = x.expand(self.res, self.res, steps)
            #  #  y_ = y.expand(self.res, self.res, steps)
            #  # this is the slow part
            #  c = self.c[xmin:xmax, ymin:ymax]
            #  d = self.d[xmin:xmax, ymin:ymax]
            #  raster_ = torch.exp((-(x_ - c)**2 - (y_ - d) ** 2) / (2*sigma**2))
            #  raster_ = torch.mean(raster_, dim=2)
            #  raster[xmin:xmax, ymin:ymax] = raster_
        if self.debug:
            print('{}: Rasterized.'.format(time() - tic))

        return torch.transpose(torch.squeeze(raster), 0, 1)
