# [WINNet: Wavelet-inspired Invertible Network for Image Denoising](https://ieeexplore.ieee.org/document/9807636)
by Jun-Jie Huang and Pier Luigi Dragotti

<img width="654" alt="WINNet" src="https://user-images.githubusercontent.com/89965355/178172283-b6b9e7da-add2-44ad-b83d-3b87918a8c5b.png">

# 1. Dependencies
* Python
* torchvision
* PyTorch>=1.0
* OpenCV for Python
* HDF5 for Python
* tensorboardX (Tensorboard for Python)

# 2. Usage

## 2.1. Training

### Training WINNet with known noise level
```python WINNet_denoise_train.py --mode S --lr 0.001 --outf logs/WINNet_lvl_1_nlvl_25 --lvl 1 --noiseL 25```

### Training WINNet with blind noise level [0, 55]
```python WINNet_denoise_train.py --mode B --lr 0.001 --outf logs/WINNet_lvl_1_nlvl_0_55 --lvl 1```

### Training Noise Estimation Network (NENet)
```python NENet_train.py --lr 0.001 --outf logs/NENet```

## 2.2. Testing

### Testing for denoising with known noise level
```python WINNet_denoise_test.py --logdir logs/WINNet_lvl_1_nlvl_25 --lvl 1 --test_noiseL 25 --start_epoch 50 --end_epoch 50 --test_data Set12```

### Testing for blind denoising
```python WINNet_wNE_blind_denoise_test.py --logdirdn logs/WINNet_lvl_1_nlvl_0_55 --logdirne logs/NENet --lvl 1 --epochdn 50 --epochne 1 --test_data Set12 --test_noiseL 85```

### Testing for deblurring
```python WINNet_deblur_test.py --logdirdn logs/WINNet_lvl_1_nlvl_0_55 --logdirne logs/NENet --lvl 1 --epochdn 50 --epochne 1 --test_data Set12 --test_noiseL 2.55```

# BibTex
```
@ARTICLE{huang22WINNet,
  author={Huang, Jun-Jie and Dragotti, Pier Luigi},
  journal={IEEE Transactions on Image Processing},
  title={WINNet: Wavelet-Inspired Invertible Network for Image Denoising},
  year={2022},
  volume={31},
  number={},
  pages={4377-4392},
  doi={10.1109/TIP.2022.3184845}
}
```

