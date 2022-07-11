import math

import numpy as np
from scipy.fftpack import dct

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# based on DCT matrix functions from: https://github.com/qbx2/pytorch-dct-notebooks/blob/master/pytorch_dct_2d.ipynb

def dft_dct_matrix(n):
    """DCT-I (equal to DFT on real numbers with even symmetry
    https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-I)"""
    ret = torch.Tensor(n, n)

    for k in range(n):
        for i in range(n):
            if i == 0:
                ret[k, i] = 1.
            elif i == (n - 1):
                ret[k, i] = (-1.) ** k
            else:
                ret[k, i] = 2. * math.cos(math.pi * k * i / (n - 1.))

    return ret


def dumb_dct_loop(x):
    """Expect input vector, then does DCT in a loop, according to the
    definition in the scipy docs."""
    y = np.zeros_like(x)
    n = len(x)

    for k in range(n):
        for i in range(n):
            f = math.sqrt(1. / (2. * n)) if k > 0 else math.sqrt(1. / (4. * n))
            y[k] += x[i] * f * math.cos(math.pi * k * (2. * i + 1.) / (2. * n))

    return 2. * y


def dct_matrix(n):
    """DCT-II"""
    ret = torch.Tensor(n, n)

    for k in range(n):
        for i in range(n):
            f = math.sqrt(1. / (2. * n)) if k > 0 else math.sqrt(1. / (4. * n))
            ret[k, i] = f * math.cos(math.pi * k * (2. * i + 1.) / (2. * n))

    return 2. * ret


def dumb_idct_loop(x):
    """Expect input vector, then does IDCT in a loop, according to the definition in the scipy docs."""
    y = np.zeros_like(x)
    n = len(x)

    for k in range(n):
        for i in range(n):
            f = math.sqrt(2. / n)
            z = 1. / math.sqrt(1. * n) if i == 0 else f * math.cos(math.pi * (k + .5) * float(i) / float(n))
            y[k] += x[i] * z

    return y


def idct_matrix(n):
    """DCT-III"""
    ret = torch.Tensor(n, n)

    for k in range(n):
        for i in range(n):
            f = math.sqrt(2. / n)
            ret[k, i] = 1. / math.sqrt(1. * n) if i == 0 else f * math.cos(math.pi * (k + .5) * float(i) / float(n))

    return ret


def symmetric_dct_matrix(n):
    """DCT-IV"""
    ret = torch.Tensor(n, n)

    for k in range(n):
        for i in range(n):
            ret[k, i] = math.sqrt(2. / n) * math.cos((math.pi / n) * (i + .5) * (k + .5))

    return ret


class DCTlayer(nn.Linear):
    """A linear layer with no bias, and fixed transformation using the DCT
    coefficients."""

    def __init__(self, in_features, type='II'):
        if type == 'I':
            self.coef = dft_dct_matrix(in_features)  # dct coefficients
        elif type == 'II':
            self.coef = dct_matrix(in_features)  # dct coefficients
        elif type == 'III':
            self.coef = idct_matrix(in_features)  # dct coefficients
        elif type == 'IV':
            self.coef = symmetric_dct_matrix(in_features)  # dct coefficients
        super().__init__(in_features, in_features, bias=False)

    def reset_parameters(self):
        self.weight.data = self.coef  # .permute(1,0)
        self.weight.requires_grad = False  # never update this parameter

    def forward(self, input):
        """Expecting 4D standard image tensor input, deal with colour channels
        independently."""
        n, c, w, _ = input.size()
        input = input.view(n * c * w, w)
        # 2D DCT decomposes into two linear operations
        dct_1 = F.linear(input, self.weight, None)
        dct_1 = dct_1.view(n * c, w, w).permute(0, 2, 1).contiguous().view(n * c * w, w)
        dct_2 = F.linear(dct_1, self.weight, None)
        dct_2 = dct_2.view(n * c, w, w).permute(0, 2, 1).contiguous()
        # unpack the channels again
        dct_out = dct_2.view(n, c, w, w)
        return dct_out

    def extra_repr(self):
        return 'in_features/out_features={}'.format(self.in_features)


# def test():
#     # sample random matrix, take DCT with fftpack
#     D = 4
#     rng = np.random.RandomState(1)
#
#     # test without layer
#     X = rng.randn(D).astype(np.float32)
#     dctmat = dct_matrix(D)
#     Y_pytorch = torch.mm(dctmat, torch.from_numpy(X).view(D, 1))
#     Y_scipy = dct(X, axis=0, norm='ortho')
#     assert np.abs(Y_scipy - dumb_dct_loop(X)).sum() < 0.001
#     assert np.abs(Y_pytorch.squeeze().data.numpy() - Y_scipy).sum() < 0.001
#
#     # check type IV is symmetric in 1D
#     dctmat = symmetric_dct_matrix(D)
#     Y_pytorch = torch.mm(dctmat, torch.from_numpy(X).view(D, 1))
#     X_recon = torch.mm(dctmat, Y_pytorch)
#     assert np.abs(X_recon.squeeze().data.numpy() - X).sum()
#
#     # now 2D checks
#     X = rng.randn(D, D).astype(np.float32)
#
#     print("Type I")
#     torch_dct = DCTlayer(D, type='I')
#     Y_pytorch = torch_dct(torch.from_numpy(X).view(1, 1, D, D))
#     # DCT type 2 with scipy
#     Y_scipy = dct(dct(X, type=1, axis=1).T, type=1, axis=1).T
#     error = np.abs(Y_scipy - Y_pytorch.view(D, D).data.numpy()).sum()
#     print("  Error between scipy and pytorch implementation: {}".format(error))
#     assert error < 0.001
#
#     print("Type II")
#     torch_dct = DCTlayer(D)
#     Y_pytorch = torch_dct(torch.from_numpy(X).view(1, 1, D, D))
#     # DCT type 2 with scipy
#     Y_scipy = dct(dct(X, axis=1, norm='ortho').T, axis=1, norm='ortho').T
#     error = np.abs(Y_scipy - Y_pytorch.view(D, D).data.numpy()).sum()
#     print("  Error between scipy and pytorch implementation: {}".format(error))
#     assert error < 0.001
#
#     print("Type III")
#     torch_dct = DCTlayer(D, type='III')
#     Y_pytorch = torch_dct(torch.from_numpy(X).view(1, 1, D, D))
#     # DCT type 2 with scipy
#     Y_scipy = dct(dct(X, type=3, axis=1, norm='ortho').T, type=3, axis=1, norm='ortho').T
#     error = np.abs(Y_scipy - Y_pytorch.view(D, D).data.numpy()).sum()
#     print("  Error between scipy and pytorch implementation: {}".format(error))
#     assert error < 0.001
#
#     print("Error Type II -> Type III reconstruction")
#     # Test if III can reconstruct from result of III
#     X = torch.from_numpy(X).view(1, 1, D, D)
#     torch_dct = DCTlayer(D, type='II')
#     torch_idct = DCTlayer(D, type='III')
#     Y_pytorch = torch_dct(X)
#     error = np.abs((X - torch_idct(Y_pytorch)).data.numpy()).sum()
#     print("  Reconstruction error: {}".format(error))
#     assert error < 0.001
#
#     # Can only test that IV reconstructs from itself
#     print("Type IV")
#     torch_dct = DCTlayer(D, type='IV')
#     Y_pytorch = torch_dct(X)
#     error = np.abs((X - torch_dct(Y_pytorch)).data.numpy()).sum()
#     print("  Reconstruction error: {}".format(error))
#     assert error < 0.001
#
#
# if __name__ == '__main__':
#     test()