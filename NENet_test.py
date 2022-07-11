import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.networks import noiseEst
from utils.dataset import *
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="NENet_Test")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--epoch", type=int, default=1, help="test epoch")
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--show_results", type=bool, default=False, help="show results")
parser.add_argument("--psz", type=int, default=8, help="patch size")
parser.add_argument("--layers", type=int, default=5, help="number of layers")
parser.add_argument("--fch", type=int, default=16, help="feature channels")
parser.add_argument("--fsz", type=int, default=5, help="feature size")

opt = parser.parse_args()

def main():
    if not os.path.exists('Results/noiseEst'):
        os.makedirs('Results/noiseEst')

    # Build model
    print('Loading model ...\n')
    net = noiseEst(psz=opt.psz, stride=1, num_layers=opt.layers, f_ch=opt.fch, fsz=opt.fsz)

    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_NENet_epoch_{}.pth'.format(opt.epoch))))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Number of Parameters: ', pytorch_total_params)

    model.eval()
    torch.manual_seed(123)
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    MAE = 0
    i = 1
    sigmaL = 1
    sigmaH = 100
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img).cuda()
        if ISource.size(2) % 2 == 1:
            ISource = ISource[:, :, 0:ISource.size(2) - 1, :]
        if ISource.size(3) % 2 ==1:
            ISource = ISource[:, :, :, 0:ISource.size(3) - 1]

        # noise
        for i in range(sigmaL, sigmaH+1):
            noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=(i)/255.).cuda()
            # noisy image
            INoisy = ISource + noise
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())

            with torch.no_grad(): 
                nEst, ratio, gate_ = model(INoisy)

            if opt.show_results:
                save_out_path = "Results/noiseEst/out_img_{}.png".format(i+1)
                save_img(save_out_path, gate_)
            abs_err = torch.abs(i - nEst)
            MAE += abs_err
        i = i + 1

    MAE /= len(files_source) * (sigmaH + 1 - sigmaL)
    print("\nMean Absolute Error: %f" % MAE)

if __name__ == "__main__":
    main()
