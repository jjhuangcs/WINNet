# [WINNet: Wavelet-inspired Invertible Network for Image Denoising](https://arxiv.org/pdf/2109.06381.pdf)
by Jun-Jie Huang and Pier Luigi Dragotti

### 1. Dependencies
* Python
* torchvision
* PyTorch>=1.0
* OpenCV for Python
* HDF5 for Python
* tensorboardX (Tensorboard for Python)

### 2. Usage
#### 2.1. Training WINNet with known noise level
```python WINNet_denoise_train.py --mode S --lr 0.001 --outf logs/WINNet_lvl_1_nlvl_25 --lvl 1 --noiseL 25```

#### 2.2. Training WINNet with blind noise level [0, 55]
```python WINNet_denoise_train.py --mode B --lr 0.001 --outf logs/WINNet_lvl_1_nlvl_0_55 --lvl 1```

#### 2.3. Training Noise Estimation Network (NENet)
```python NENet_train.py --lr 0.001 --outf logs/NENet```

#### 2.4 Testing for denoising with known noise level
```python WINNet_denoise_test.py --logdir logs/WINNet_lvl_1_nlvl_25 --lvl 1 --test_noiseL 25 --start_epoch 50 --end_epoch 50 --test_data Set12```

#### 2.5 Testing for blind denoising
```python WINNet_wNE_blind_denoise_test.py --logdirdn logs/WINNet_lvl_1_nlvl_0_55 --logdirne logs/NENet --lvl 1 --epochdn 50 --epochne 1 --test_data Set12 --test_noiseL 85```

#### 2.6 Testing for deblurring
```python WINNet_deblur_test.py --logdirdn logs/WINNet_lvl_1_nlvl_0_55 --logdirne logs/NENet --lvl 1 --epochdn 50 --epochne 1 --test_data Set12 --test_noiseL 2.55```

