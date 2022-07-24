import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.networks import WINNetklvl, noiseEst
from utils.dataset import *
from utils import utils_sisr as sr
import hdf5storage
from scipy import ndimage
import math
from PIL import Image

########################################################################################################################
def get_rho_sigma(sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, w=1.0):
    '''
    One can change the sigma to implicitly change the trade-off parameter
    between fidelity term and prior term
    '''
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
    rhos = list(map(lambda x: 0.23*(sigma**2)/(x**2), sigmas))
    return rhos, sigmas
########################################################################################################################

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="WINNet with Noise Estimation for Deblur")
parser.add_argument("--num_of_steps", type=int, default=4, help="Number of steps")
parser.add_argument("--num_of_layers", type=int, default=4, help="Number of layers")
parser.add_argument("--num_of_channels", type=int, default=32, help="Number of channels")
parser.add_argument("--lvl", type=int, default=1, help="number of levels")
parser.add_argument("--split", type=str, default="dct", help='splitting operator')
parser.add_argument("--dnlayers", type=int, default=4, help="Number of denoising layers")
parser.add_argument("--logdirdn", type=str, default="logs", help='path of log files')
parser.add_argument("--logdirne", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--show_results", type=bool, default=False, help="show results")

opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    if not os.path.exists('results/WINNet_deblur'):
        os.makedirs('results/WINNet_deblur')
    if not os.path.exists('results/Noisy_deblur'):
        os.makedirs('results/Noisy_deblur')

    device_ids = [0]
    # Build model
    net = WINNetklvl(steps=opt.num_of_steps, layers=opt.num_of_layers, channels=opt.num_of_channels, klvl=opt.lvl)
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdirdn, 'net_WINNet.pth')))
    model.eval()

    net_NE = noiseEst(psz=8, stride=1, num_layers=5, f_ch=16, fsz=5)
    model_NE = nn.DataParallel(net_NE, device_ids=device_ids).cuda()
    model_NE.load_state_dict(torch.load(os.path.join(opt.logdirne, 'net_NENet.pth')))
    model_NE.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()

    kernels = hdf5storage.loadmat(os.path.join('data/kernels', 'Levin09.mat'))['kernels'] # kernels_12.mat  Levin09

    for k_index in range(kernels.shape[1]):
        torch.manual_seed(12345)
        psnr_test = 0
        print('Kernel Number %d' % (k_index))
        k = kernels[0, k_index].astype(np.float64)

        k_tensor = torch.Tensor(k).unsqueeze(0).unsqueeze(0)
        print(k_tensor.shape)

        img_index = 0
        for f in files_source:
            img_index += 1
            # image
            img_H = cv2.imread(f)
            img_H = normalize(np.float32(img_H[:, :, 0]))
            img_H = np.expand_dims(img_H, 0)
            img_H = np.expand_dims(img_H, 1)
            img_H_tensor = torch.Tensor(img_H)
            if img_H_tensor.size(2) % 2 == 1:
                img_H_tensor = img_H_tensor[:, :, 0:img_H_tensor.size(2) - 1, :]
            if img_H_tensor.size(3) % 2 == 1:
                img_H_tensor = img_H_tensor[:, :, :, 0:img_H_tensor.size(3) - 1]

            f_sz = math.floor(k_tensor.shape[-1] / 2)
            k_exp = np.expand_dims(k, 0)
            k_exp = np.expand_dims(k_exp, 0)
            img_B = ndimage.filters.convolve(img_H, k_exp, mode='wrap')#circulant convolution
            img_B_tensor = torch.Tensor(img_B)

            np.random.seed(seed=0)  # for reproducibility
            img_N = img_B + np.random.normal(0, opt.test_noiseL / 255., img_B.shape)  # add AWGN
            img_N_tensor = torch.Tensor(img_N)

            # blurred and noisy image
            img_B_tensor, x = Variable(img_B_tensor.cuda()), Variable(img_N_tensor.cuda())

            k_tensor = k_tensor.cuda()
            FB, FBC, F2B, FBFy = sr.pre_calculate(x, k_tensor, sf=1)

            amp = 2
            lambd = 0.23
            i = 0

            with torch.no_grad():
                stdNv_, _, _ = model_NE(x)
                stdNv_ = stdNv_.item()

            stdNv = stdNv_ * 10

            while stdNv > stdNv_:
                # step 1, data fidelity term
                tau = lambd * (stdNv_**2 / stdNv**2 * torch.ones(1)).repeat(1, 1, 1, 1).cuda()

                xd = sr.data_solution(x, FB, FBC, F2B, FBFy, tau, sf=1)
                psnr_xd = batch_PSNR(xd, img_H_tensor, 1., 2*f_sz)

                # step 2, denoise estimation
                with torch.no_grad():
                    stdNv, _, _ = model_NE(xd)
                    stdNv = stdNv.item()

                # step 3, denoiser
                with torch.no_grad():
                    out, _ = model(xd, stdNv * amp, noiseT=0)
                    x = out

                psnr_x = batch_PSNR(out, img_H_tensor, 1., 2*f_sz)
                print('xk %f, zk %f, std %f, tau %f' % (psnr_xd, psnr_x, stdNv, tau))
                i += 1

                if opt.show_results:
                    save_out_path = "results/WINNet_deblur/{}_x_denoise_img_{}.png".format(k_index, i)
                    out_ = torch.clamp(x[:, :, 2 * f_sz:-1 - 2 * f_sz, 2 * f_sz:-1 - 2 * f_sz], 0, 1).cpu()
                    save_img(save_out_path, out_)

                    save_out_path = "results/WINNet_deblur/{}_x_data_img_{}.png".format(k_index, i)
                    out_ = torch.clamp(xd[:, :, 2*f_sz:-1-2*f_sz, 2*f_sz:-1-2*f_sz], 0, 1).cpu()
                    save_img(save_out_path, out_)

            out = torch.clamp(out, 0., 1.)
            psnr = batch_PSNR(out, img_H_tensor, 1., 2*f_sz)

            if opt.show_results:
                save_out_path = "results/WINNet_deblur/{}_noise_img.png".format(k_index)
                out_ = torch.clamp(img_N_tensor[:, :, 2 * f_sz:-1 - 2 * f_sz, 2 * f_sz:-1 - 2 * f_sz], 0, 1).cpu()
                save_img(save_out_path, out_)

                save_out_path = "results/WINNet_deblur/{}_GT_img.png".format(k_index)
                out_ = torch.clamp(img_H_tensor[:, :, 2 * f_sz:-1 - 2 * f_sz, 2 * f_sz:-1 - 2 * f_sz], 0, 1).cpu()
                save_img(save_out_path, out_)

                save_out_path = "results/WINNet_deblur/{}_kernel_img.png".format(k_index)
                out_ = k_tensor.cpu()
                save_img(save_out_path, out_[0])

            psnr_test += psnr

            print("Image %d Kernel %d   PSNR  %f" % (img_index, k_index, psnr))

if __name__ == "__main__":
    main()
