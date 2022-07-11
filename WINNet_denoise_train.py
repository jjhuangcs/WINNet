import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.networks import WINNetklvl
from utils.dataset import prepare_data, Dataset
import time
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="WINNet")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--num_of_steps", type=int, default=4, help="Number of steps")
parser.add_argument("--num_of_layers", type=int, default=4, help="Number of layers")
parser.add_argument("--num_of_channels", type=int, default=32, help="Number of channels")
parser.add_argument("--lvl", type=int, default=1, help="number of levels")
parser.add_argument("--split", type=str, default="dct", help='splitting operator')
parser.add_argument("--dnlayers", type=int, default=4, help="Number of denoising layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--start_epoch", type=int, default=0, help="start epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--decay_rate", type=float, default=0.1, help="decay rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')

opt = parser.parse_args()
print(opt)

def main():
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True, pin_memory=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = WINNetklvl(steps=opt.num_of_steps, layers=opt.num_of_layers, channels=opt.num_of_channels, klvl=opt.lvl,
                     mode=opt.split, dnlayers=opt.dnlayers)
    criterion = nn.MSELoss(size_average=False).cuda()
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    torch.backends.cudnn.benchmark = True

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Number of Parameters: ', pytorch_total_params)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    noiseL_B = [0, 55]  # ingnored when opt.mode=='S'
    start_epoch = opt.start_epoch
    if start_epoch > 0:
        print('Start Epoch: ', start_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.outf, 'net_WINNet_epoch_{}.pth'.format(start_epoch))))
        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]
            current_lr = current_lr * (opt.decay_rate ** (start_epoch // opt.milestone))
            param_group["lr"] = current_lr
            print('Learning rate: ', current_lr)
    else:
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net_WINNet_epoch_{}.pth'.format(start_epoch)))
    epoch = start_epoch + 1
    torch.cuda.synchronize()
    t1 = time.time()
    print('Epoch: ', epoch)
    while epoch <= opt.epochs:
        with tqdm(loader_train, unit='batch') as tepoch:
            i = 0
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                # training step
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                img_train = data.cuda()
                ########################################################################################################
                if opt.mode == 'S':
                    stdN = opt.noiseL
                if opt.mode == 'B':
                    stdRD = np.random.uniform(noiseL_B[0], noiseL_B[1], size=1)
                    stdN = stdRD[0]
                stdNv = Variable(stdN*torch.ones(1, device=torch.device('cuda:0')))
                noise = torch.cuda.FloatTensor(img_train.size()).normal_(mean=0, std=stdN / 255.)
                ########################################################################################################
                imgn_train = img_train + noise
                img_train, imgn_train = Variable(img_train), Variable(imgn_train)

                outn_train, oloss = model(imgn_train, stdNv)
                loss = criterion(outn_train, img_train) / (imgn_train.size()[0] * 2)
                loss = loss + oloss
                Loss = loss.detach()
                if i % 10 == 0:
                    norm_PU = net.linear()
                    loss = loss + 1 * 1e-1 * norm_PU
                ########################################################################################################
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)# Gradient Clipping
                optimizer.step()

                tepoch.set_postfix(loss=Loss)
                time.sleep(0.0001)
                i += 1
        torch.cuda.synchronize()
        t2 = time.time()
        print('Time:' f'{(t2 - t1) / 60: .3e} mins')
        print('Orthogonal Loss: ', oloss)
        print('Norm PU', norm_PU)

        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]
            print('Learning rate: ', current_lr)
        if torch.isnan(Loss) or torch.isinf(Loss):
            if epoch > 0:
                print('Load Model Epoch: ', epoch-1)
                model.load_state_dict(torch.load(os.path.join(opt.outf, 'net_WINNet_epoch_{}.pth'.format(epoch-1))))
            else:
                net = WINNetklvl(steps=opt.num_of_steps, layers=opt.num_of_layers, channels=opt.num_of_channels, klvl=opt.lvl,
                                 mode=opt.modeSM, dnlayers=opt.dnlayers)
                model = nn.DataParallel(net, device_ids=device_ids).cuda()

            for param_group in optimizer.param_groups:
                current_lr = param_group["lr"]
                current_lr = current_lr * 0.8
                param_group["lr"] = current_lr
                print('Learning rate: ', current_lr)
            continue

        torch.cuda.synchronize()
        t1 = time.time()

        print("Save Model Epoch: %d" % (epoch))
        print('\n')
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net_WINNet_epoch_{}.pth'.format(epoch)))
        epoch += 1

        print('Epoch: ', epoch)
        if (epoch > 0) and (epoch % opt.milestone == 0):
            for param_group in optimizer.param_groups:
                current_lr = param_group["lr"]
                current_lr = current_lr * opt.decay_rate
                param_group["lr"] = current_lr
                print('Learning rate: ', current_lr)

        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
