import torch
from torch import nn
from torch.nn import functional as F
import pywt
import math
from iUNet.layers import InvertibleDownsampling2D
from iUNet.dct import dct_matrix

class LearnSplit(nn.Module):
    def __init__(self, in_channels, channel_multiplier, dilate, c_channels, stride=1):
        super(LearnSplit, self).__init__()

        self.stride = stride

        self.downsampling = InvertibleDownsampling2D(
            in_channels=in_channels,
            channel_multiplier=channel_multiplier,
            dilate=dilate,
            stride=(stride, stride),
            method='cayley',  # 'exp', 'cayley', 'householder'
            init='dct',  # dct haar random
            learnable=True
        )

        self.in_channels = in_channels
        self.channel_multiplier = channel_multiplier
        self.c_channels = c_channels  # number of coarse channels
        self.normalize = self.stride / math.sqrt(self.channel_multiplier * self.in_channels)

    def forward(self, x):
        fx = self.downsampling.forward(x) * self.normalize
        xc = fx[:, :self.c_channels, :, :].contiguous()
        xd = fx[:, self.c_channels:, :, :].contiguous()
        return xc, xd

    def inverse(self, xc, xd):
        x = torch.cat((xc, xd), 1)
        x = self.downsampling.inverse(x) * self.normalize
        return x

    def get_kernel(self):
        return self.downsampling.get_kernel()

class DCTSplit(nn.Module):
    def __init__(self, in_channels, channel_multiplier, dilate, c_channels, stride=1):
        super(DCTSplit, self).__init__()

        self.in_channels = in_channels
        self.channel_multiplier = channel_multiplier
        self.dilate = dilate
        self.stride = stride
        self.c_channels = c_channels

        self.out_channels = in_channels * channel_multiplier

        kernel_size = int(self.out_channels ** 0.5)
        kernel_shape = ((self.out_channels, in_channels) + (kernel_size, kernel_size))
        kernel_transposed_shape = ((in_channels, self.out_channels) + (kernel_size, kernel_size))

        self.kernel_matrix = dct_matrix(self.out_channels).cuda()
        self.kernel_matrix_transposed = torch.flip(self.kernel_matrix, [1])
        self.kernel = self.kernel_matrix.reshape(kernel_shape)
        self.kernel_transposed = self.kernel_matrix_transposed.reshape(kernel_transposed_shape)

        self.psize = int(self.kernel.size(3) / 2) * self.dilate
        self.paddsz = math.floor(self.kernel.size(3) / 2) * self.dilate
        self.normalize = self.stride ** 2 / (self.channel_multiplier * self.in_channels)

    def forward(self, x):
        if self.kernel.size(3) % 2 == 0:
            x = F.pad(x, (self.psize - self.dilate, self.psize, self.psize - self.dilate, self.psize), mode='replicate')
        else:
            x = F.pad(x, (self.paddsz, self.paddsz, self.paddsz, self.paddsz), mode='replicate')

        fx = F.conv2d(x, self.kernel, stride=self.stride, dilation=self.dilate)

        xc = fx[:, :self.c_channels, :, :].contiguous()
        xd = fx[:, self.c_channels:, :, :].contiguous()
        return xc, xd

    def inverse(self, xc, xd):
        x = torch.cat((xc, xd), 1)

        if self.stride == 2:
            # Apply transposed convolution in order to invert the downsampling.
            x = F.conv_transpose2d(x, self.kernel, stride=self.stride, groups=self.in_channels)
        else:
            if self.kernel.size(3) % 2 == 0:
                x = F.pad(x, (self.psize, self.psize - self.dilate, self.psize, self.psize - self.dilate), mode='replicate')
            else:
                x = F.pad(x, (self.paddsz, self.paddsz, self.paddsz, self.paddsz), mode='replicate')
            x = F.conv2d(x, self.kernel_transposed, stride=self.stride, dilation=self.dilate)

        x = x * self.normalize
        return x

    def get_kernel(self):
        return self.kernel_matrix

class waveletDecomp(nn.Module):
    def __init__(self, stride=2, c_channels=1):
        super(waveletDecomp, self).__init__()

        self.stride = stride
        self.c_channels = c_channels

        wavelet = pywt.Wavelet('haar')
        dec_hi = torch.tensor(wavelet.dec_hi[::-1])
        dec_lo = torch.tensor(wavelet.dec_lo[::-1])

        self.filters_dec = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                                        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                                        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                                        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0).cuda()
        self.filters_dec = self.filters_dec.unsqueeze(1)
        self.psize = int(self.filters_dec.size(3) / 2)

        rec_hi = torch.tensor(wavelet.rec_hi[::-1])
        rec_lo = torch.tensor(wavelet.rec_lo[::-1])
        self.filters_rec = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                                        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                                        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                                        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0).cuda()
        self.filters_rec = self.filters_rec.unsqueeze(0)

        self.filters_rec_transposed = torch.flip(self.filters_rec.permute(1, 0, 2, 3), [2, 3])

    def forward(self, x):
        if self.stride == 1:
            x = F.pad(x, (self.psize - 1, self.psize, self.psize - 1, self.psize), mode='replicate')

        coeff = F.conv2d(x, self.filters_dec, stride=self.stride, bias=None, padding=0)

        out = coeff / 2
        xc = out[:, :self.c_channels, :, :].contiguous()
        xd = out[:, self.c_channels:, :, :].contiguous()
        return xc, xd

    def inverse(self, xc, xd):
        x = torch.cat((xc, xd), 1)
        if self.stride == 1:
            x = F.pad(x, (self.psize, self.psize - 1, self.psize, self.psize - 1), mode='replicate')

        if self.stride == 1:
            coeff = F.conv2d(x, self.filters_rec, stride=self.stride, bias=None, padding=0)
        else:
            coeff = F.conv_transpose2d(x, self.filters_rec_transposed, stride=self.stride,
                                                       bias=None, padding=0)
        out = coeff

        return (out * self.stride ** 2) / 2
