# [WINNet: Wavelet-inspired Invertible Network for Image Denoising](https://ieeexplore.ieee.org/document/9807636)
by [Jun-Jie Huang](https://jjhuangcs.github.io/) (jjhuang@nudt.edu.cn) and [Pier Luigi Dragotti](http://www.commsp.ee.ic.ac.uk/~pld/)

Pytorch implementation for "WINNet: Wavelet-inspired Invertible Network for Image Denoising" (TIP'2022).

<img width="654" alt="WINNet" src="https://user-images.githubusercontent.com/89965355/178172283-b6b9e7da-add2-44ad-b83d-3b87918a8c5b.png">

Overview of the proposed wavelet-inspired invertible network (WINNet). It consists of K levels of lifting inspired invertible neural networks (LINN) and denoising network. The forward transform of LINN non-linearly converts the input noisy image into coarse part (green) and detail parts (black). Denoising network will perform denoising operation on the detail part while the coarse version is decomposed again using a second level and the decomposition and denoising steps are repeated K times. The backward transform of the LINN will reconstruct the denoised image using the denoised detail parts and the original coarse part. The estimated noise level from the noise estimation network will be used to adjust the soft-thresholds of the soft-thresholding non-linearity to make the WINNet to adapt well to the current noise level.

# 1. Dependencies
* Python
* torchvision
* PyTorch>=1.0
* OpenCV for Python
* HDF5 for Python
* tensorboardX (Tensorboard for Python)

# 2. Usage

## 2.1. Training

The 400 training images from the Berkeley segmentation dataset (BSD) of size 180 × 180 are used for training. The training patch size for
non-blind and blind denoising setting is 40×40 and 50×50, respectively.

### Training WINNet with known noise level
```python WINNet_denoise_train.py --mode S --lr 0.001 --outf logs/WINNet_lvl_1_nlvl_25 --lvl 1 --noiseL 25```

Note: For training with a single noise level, three noise levels are considered, i.e., $\sigma_N$ = 15, 25 and 50.

### Training WINNet with blind noise level [0, 55]
```python WINNet_denoise_train.py --mode B --lr 0.001 --outf logs/WINNet_lvl_1_nlvl_0_55 --lvl 1```

Note: For blind image denoising scenario, the training noise level $\sigma_N$ is uniformly drawn from [0, 55].

### Training Noise Estimation Network (NENet)
```python NENet_train.py --lr 0.001 --outf logs/NENet```

Note: For training NENet, the training noise level $\sigma_N$ is uniformly drawn from [0, 55].

## 2.2. Testing

### Testing for denoising with known noise level
```python WINNet_denoise_test.py --logdir logs/WINNet_lvl_1_nlvl_25 --lvl 1 --test_noiseL 25 --start_epoch 50 --end_epoch 50 --test_data Set12```

### Testing for blind denoising
```python WINNet_wNE_blind_denoise_test.py --logdirdn logs/WINNet_lvl_1_nlvl_0_55 --logdirne logs/NENet --lvl 1 --epochdn 50 --epochne 1 --test_data Set12 --test_noiseL 85```

### Testing for deblurring
```python WINNet_deblur_test.py --logdirdn logs/WINNet_lvl_1_nlvl_0_55 --logdirne logs/NENet --lvl 1 --epochdn 50 --epochne 1 --test_data Set12 --test_noiseL 2.55```

# Citation

If you use any part of this code in your research, please cite our paper:


```
@article{huang2022WINNet,
  author={Huang, Jun-Jie and Dragotti, Pier Luigi},
  journal={IEEE Transactions on Image Processing},
  title={{WINNet}: Wavelet-Inspired Invertible Network for Image Denoising},
  year={2022},
  volume={31},
  number={},
  pages={4377-4392},
  doi={10.1109/TIP.2022.3184845}
}
```
```
@inproceedings{huang2021linn,
  author={Huang, Jun-Jie and Dragotti, Pier Luigi},
  title={{LINN}: Lifting Inspired Invertible Neural Networks for Image Denoising},
  booktitle={2021 29th European Signal Processing Conference (EUSIPCO)}, 
  year={2021},
  pages={636-640},
  organization={EURASIP},
  doi={10.23919/EUSIPCO54536.2021.9615931}
}
```

