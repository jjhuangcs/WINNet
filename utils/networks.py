import torch
from torch import nn
from torch.nn import Parameter
import math
from utils.splitmerge import *


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

    def inverse(self, *inputs):
        for module in reversed(self._modules.values()):
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class STnoise(nn.Module):
    def __init__(self, f_ch):
        super(STnoise, self).__init__()
        self.thre = Parameter(-0.1 * torch.rand(f_ch))
        self.softplus = nn.Softplus(beta=20)
        self.relu = nn.ReLU()

    def forward(self, x, noiseL):
        sgn = torch.sign(x)
        thre = self.softplus(self.thre)
        ##
        thre = thre.repeat(x.size(0), x.size(2), x.size(3), 1).permute(0, 3, 1, 2).contiguous()
        thre = thre * (noiseL ** 1 / 50 ** 1)
        tmp = torch.abs(x) - thre
        out = sgn * (tmp + torch.abs(tmp)) / 2
        ##
        return out


class Conv2dSTnoise(nn.Module):
    def __init__(self, in_ch, out_ch, f_sz, dilate):
        super(Conv2dSTnoise, self).__init__()
        if_bias = False

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=f_sz,
                              padding=math.floor(f_sz / 2) + dilate - 1, dilation=dilate, bias=if_bias)
        torch.nn.init.kaiming_uniform_(self.conv.weight, a=0, mode='fan_out', nonlinearity='relu')
        self.soft = STnoise(out_ch)

    def forward(self, x, noiseL):
        x = self.conv(x)
        x = self.soft(x, noiseL)
        return x, noiseL

    def linear(self):
        P_mat = self.conv.weight.reshape(self.conv.weight.shape[0], -1)
        _, sP, _ = torch.svd(P_mat)
        sv = sP[0]

        return self.conv.weight, sv


class SepConv(nn.Module):
    def __init__(self, in_ch, f_ch, f_sz, dilate=1):
        super(SepConv, self).__init__()

        if_bias = False
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=f_sz,
                               padding=math.floor(f_sz / 2) + dilate - 1, dilation=dilate, bias=if_bias,
                               groups=in_ch)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=0.0, mode='fan_out', nonlinearity='relu')

        self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=f_ch, kernel_size=1,
                               padding=math.floor(1 / 2) + dilate - 1, dilation=dilate, bias=if_bias,
                               groups=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=0.0, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.conv2(self.conv1(x))

    def linear(self):
        chals = self.conv1.weight.shape[0]
        psz = self.conv1.weight.shape[-1]
        conv1Full = torch.zeros([chals, chals, psz, psz], device=torch.device('cuda:0'))
        for i in range(chals):
            conv1Full[i, i, :, :] = self.conv1.weight[i, 0, :, :]

        conv21 = nn.functional.conv2d(conv1Full, self.conv2.weight, padding=self.conv2.weight.shape[-1] - 1)
        return conv21.permute([1, 0, 2, 3])


class ResBlockSepConvST(nn.Module):
    def __init__(self, in_ch, f_ch, f_sz, dilate=1):
        super(ResBlockSepConvST, self).__init__()

        self.conv1 = SepConv(in_ch, f_ch, f_sz, dilate)
        self.conv2 = SepConv(f_ch, in_ch, f_sz, dilate)
        self.soft1 = STnoise(f_ch)
        self.soft2 = STnoise(in_ch)

        self.identity = torch.zeros([in_ch, in_ch, 2 * f_sz - 1, 2 * f_sz - 1], device=torch.device('cuda:0'))
        for i in range(in_ch):
            self.identity[i, i, int(f_sz) - 1, int(f_sz) - 1] = 1

    def forward(self, x, noiseL):
        return self.soft2(x + self.conv2(self.soft1(self.conv1(x), noiseL)), noiseL), noiseL

    def linear(self):
        conv21 = nn.functional.conv2d(self.conv1.linear().permute([1, 0, 2, 3]),
                                      torch.rot90(self.conv2.linear(), 2, [2, 3]),
                                      padding=self.conv2.linear().shape[-1] - 1)
        conv21 = conv21 + self.identity

        P_mat = conv21.reshape(conv21.shape[0], -1)
        _, sP, _ = torch.svd(P_mat)
        sv = sP[0]
        return conv21.permute([1, 0, 2, 3]), sv


class PUNet(nn.Module):
    def __init__(self, in_ch, out_ch, f_ch, f_sz, num_layers, dilate):
        super(PUNet, self).__init__()

        if_bias = False
        self.layers = []
        self.layers.append(Conv2dSTnoise(in_ch, f_ch, f_sz, dilate))
        for _ in range(int(num_layers)):
            self.layers.append(ResBlockSepConvST(f_ch, int(f_ch / 1), 2 * f_sz - 1, dilate))
        self.net = mySequential(*self.layers)

        self.convOut = nn.Conv2d(f_ch, out_ch, f_sz, stride=1, padding=math.floor(f_sz / 2) + dilate - 1,
                                 dilation=dilate, bias=if_bias)
        self.convOut.weight.data.fill_(0.)

    def forward(self, x, noiseL):
        x, noiseL = self.net(x, noiseL)
        out = self.convOut(x)
        return out

    def linear(self):
        for i in range(len(self.net)):
            if i == 0:
                conv0, sP0 = self.net[i].linear()
            else:
                conv, sP = self.net[i].linear()
                conv0 = nn.functional.conv2d(conv0.permute([1, 0, 2, 3]), torch.rot90(conv, 2, [2, 3]),
                                             padding=int((conv.shape[-1] - 1))).permute([1, 0, 2, 3])
                sP0 = sP0 + sP
        out = conv0
        return out, sP0


class LiftingStep(nn.Module):
    def __init__(self, pin_ch, f_ch, uin_ch, f_sz, dilate, num_layers):
        super(LiftingStep, self).__init__()

        self.dilate = dilate

        pf_ch = int(f_ch)
        uf_ch = int(f_ch)
        self.predictor = PUNet(pin_ch, uin_ch, pf_ch, f_sz, num_layers, dilate)
        self.updator = PUNet(uin_ch, pin_ch, uf_ch, f_sz, num_layers, dilate)

    def forward(self, xc, xd, noiseL):
        Fxc = self.predictor(xc, noiseL)
        xd = - Fxc + xd
        Fxd = self.updator(xd, noiseL)
        xc = xc + Fxd

        return xc, xd, noiseL

    def inverse(self, xc, xd, noiseL):
        Fxd = self.updator(xd, noiseL)
        xc = xc - Fxd
        Fxc = self.predictor(xc, noiseL)
        xd = xd + Fxc

        return xc, xd, noiseL

    def linear(self):
        linearconvP, sP = self.predictor.linear()
        linearconvU, sU = self.updator.linear()
        normPU = (sP + sU) / 2
        return linearconvP, linearconvU, normPU


class LINN(nn.Module):
    def __init__(self, in_ch, pin_ch, f_ch, uin_ch, f_sz, dilate, num_step, num_layers, lvl, mode='dct'):
        super(LINN, self).__init__()
        self.layers = []
        for _ in range(num_step):
            self.layers.append(LiftingStep(pin_ch, f_ch, uin_ch, f_sz, dilate, num_layers))
        self.net = mySequential(*self.layers)

        stride = 1
        dilate = 1

        if mode == 'learn':
            self.learnDownUp = LearnSplit(in_channels=in_ch, channel_multiplier=int((uin_ch + pin_ch) / in_ch),
                                          dilate=dilate, c_channels=pin_ch, stride=stride)
        elif mode == 'dct':
            self.learnDownUp = DCTSplit(in_channels=in_ch, channel_multiplier=int((uin_ch + pin_ch) / in_ch),
                                        dilate=dilate, c_channels=pin_ch, stride=stride)
        elif mode == 'wavelet':
            self.learnDownUp = waveletDecomp(stride=1, c_channels=pin_ch)

    def forward(self, x, noiseL):
        xc0, xd0 = self.learnDownUp.forward(x)
        xc, xd = xc0, xd0
        for i in range(len(self.net)):
            xc, xd, noiseL = self.net[i].forward(xc, xd, noiseL)
        return xc, xd, xc0, xd0

    def inverse(self, xc, xd, noiseL):
        for i in reversed(range(len(self.net))):
            xc, xd, noiseL = self.net[i].inverse(xc, xd, noiseL)
        x = self.learnDownUp.inverse(xc, xd)
        return x

    def get_kernel(self):
        return self.learnDownUp.get_kernel()

    def linear(self):
        norm_total = 0
        for i in range(len(self.net)):
            _, _, normPU = self.net[i].linear()
            norm_total += normPU
        norm_total = norm_total / len(self.net)
        return norm_total


class DnBlock_CLISTAShare(nn.Module):
    def __init__(self, f_ch, uin_ch, f_sz, num_of_layers):
        super(DnBlock_CLISTAShare, self).__init__()
        self.num_of_layers = num_of_layers

        self.A = nn.Conv2d(uin_ch, f_ch, f_sz, stride=1, padding=math.floor(f_sz / 2), bias=False)
        torch.nn.init.kaiming_uniform_(self.A.weight, a=0.0, mode='fan_out', nonlinearity='relu')
        self.S = nn.Conv2d(f_ch, uin_ch, f_sz, stride=1, padding=math.floor(f_sz / 2), bias=False)
        torch.nn.init.kaiming_uniform_(self.S.weight, a=0.0, mode='fan_out', nonlinearity='relu')

        self.layers = []
        for _ in range(num_of_layers):
            self.layers.append(STnoise(f_ch))
        self.net = mySequential(*self.layers)

        self.identity = torch.zeros([uin_ch, uin_ch, 2 * f_sz - 1, 2 * f_sz - 1], device=torch.device('cuda:0'))
        for i in range(uin_ch):
            self.identity[i, i, int(f_sz - 1), int(f_sz - 1)] = 1

        self.mse = nn.MSELoss(size_average=False)

    def forward(self, x, noiseL):
        g = self.A(x)
        for i in range(self.num_of_layers):
            f = g + self.A(x - self.S(g))
            g = self.net[i](f, noiseL)
        out = self.S(g)

        return out

    def orthogonal(self):
        convSA = nn.functional.conv2d(self.A.weight.permute([1, 0, 2, 3]), torch.rot90(self.S.weight, 2, [2, 3]),
                                      padding=self.S.weight.shape[-1] - 1)
        loss = self.mse(self.identity, convSA)
        return loss



#############################################################################################################
# J.J. Huang and P.L. Dragotti, "WINNet: Wavelet-Inspired Invertible Network for Image Denoising,"
# IEEE Trans. on Image Processing, vol. 31, pp.4377-4392, June 2022.
#############################################################################################################
class WINNetklvl(nn.Module):
    def __init__(self, steps=4, layers=4, channels=32, klvl=3, mode='dct', dnlayers=3):
        super(WINNetklvl, self).__init__()
        pin_chs = 1
        uint_chs = 4 ** 2 - pin_chs
        nstep = steps
        nlayer = layers
        Dnchanls = 64
        self.mode = mode

        if mode == 'wavelet':
            uint_chs = 2 ** 2 - 1

        self.innlayers = []
        for ii in range(klvl):
            dilate = 2 ** ii
            if ii > 1:
                dilate = 2
            self.innlayers.append(LINN(in_ch=1, pin_ch=pin_chs, f_ch=channels, uin_ch=uint_chs, f_sz=3,
                                       dilate=dilate, num_step=nstep, num_layers=nlayer, lvl=ii, mode=mode))
        self.innnet = mySequential(*self.innlayers)

        self.ddnlayers = []
        for ii in range(klvl):
            self.ddnlayers.append(DnBlock_CLISTAShare(uin_ch=uint_chs, f_ch=Dnchanls, f_sz=3, num_of_layers=dnlayers))
        self.ddnnet = mySequential(*self.ddnlayers)

    def forward(self, x, noiseL, noiseT=None):
        if noiseT is None:
            noiseT = noiseL

        xc, xd, xc_, xd_, K = [], [], [], [], []
        loss = 0

        for i in range(len(self.innnet)):
            if i == 0:
                tmpxc, tmpxd, _, _ = self.innnet[i].forward(x, noiseL if noiseL >= noiseT else noiseT)
            else:
                tmpxc, tmpxd, _, _ = self.innnet[i].forward(xc[i - 1], noiseL if noiseL >= noiseT else noiseT)

            xc.append(tmpxc)
            xd.append(tmpxd)

            tmpxd_ = self.ddnnet[i].forward(xd[i], noiseL)

            '''Orthogonal Loss'''
            loss = loss + 1e1 * self.ddnnet[i].orthogonal()
            if self.mode == 'orth':
                loss = loss + self.innnet[i].learnDownUp.orthogonal()

            xd_.append(tmpxd_)
            xc_.append(tmpxc)

        for i in reversed(range(len(self.innnet))):
            if i > 0:
                xc_[i - 1] = self.innnet[i].inverse(xc_[i], xd_[i], noiseL if noiseL >= noiseT else noiseT)
            else:
                out = self.innnet[i].inverse(xc_[i], xd_[i], noiseL if noiseL >= noiseT else noiseT)

        loss = loss / len(self.innnet)
        return out, loss

    def linear(self):
        norm_total = 0
        for i in range(len(self.innnet)):
            normlvl = self.innnet[i].linear()
            norm_total += normlvl

        return norm_total / len(self.innnet)


class noiseEst(nn.Module):
    def __init__(self, psz=8, stride=1, num_layers=5, f_ch=16, fsz=3):
        super(noiseEst, self).__init__()
        self.psz = psz
        self.stride = stride

        self.fsz = fsz

        self.num_layers = num_layers
        self.f_ch = f_ch

        self.unfold = nn.Unfold(kernel_size=(self.psz, self.psz), stride=self.stride)

        if_bias = False

        layers = []
        layers.append(nn.Conv1d(in_channels=1, out_channels=f_ch, kernel_size=self.fsz, padding=0, bias=if_bias))
        layers.append(nn.ReLU())
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels=f_ch, out_channels=f_ch, kernel_size=self.fsz, padding=0, bias=if_bias))
            layers.append(nn.ReLU())
        layers.append(nn.Conv1d(in_channels=f_ch, out_channels=1, kernel_size=self.fsz, padding=0, bias=if_bias))
        self.net = nn.Sequential(*layers)

        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        '''Unfold input image into patches'''
        P = self.unfold(img)
        P = P.permute(0, 2, 1)

        '''Reshape patches to vectors'''
        V = torch.reshape(P, (-1, 1, P.size(2)))
        V = self.net(V)
        W = self.sigmoid(torch.mean(V, 2))
        W = torch.reshape(W, (P.size(0), P.size(1), -1))

        '''Reshapeing the W into image size for Visualization'''
        if 1:
            WImg = torch.reshape(W, (P.size(0), P.size(1), -1))
            WImg = WImg.permute(0, 2, 1)
            WImg = nn.functional.fold(WImg, output_size=(img.shape[-2] - self.psz + 1,
                                                         img.shape[-1] - self.psz + 1),
                                      kernel_size=(1, 1), stride=self.stride)
        else:
            WImg = []

        pw = P * W
        pw = torch.reshape(pw, (1, -1, self.psz ** 2))
        p = torch.reshape(P, (1, -1, self.psz ** 2))

        '''Implement SVD(P x diag(w) x P^T)'''
        PWPT = torch.matmul(p.transpose(-2, -1), pw)
        _, S, _ = torch.svd(PWPT, some=True)

        out = S[:, -1]

        out = (out / torch.sum(W)) ** 0.5  # standard deviation
        ratio = torch.sum(W) / pw.size(1)  # Percentage of Selected Patches
        out = torch.mean(out) * 255  # rescale to [0,255]
        return out, ratio, WImg



