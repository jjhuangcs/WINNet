## [WINNet: Wavelet-inspired Invertible Network for Image Denoising](https://arxiv.org/pdf/2109.06381.pdf)
by Jun-Jie Huang and Pier Luigi Dragotti

### Dependencies
* Python
* torchvision
* PyTorch>=1.0
* OpenCV for Python
* HDF5 for Python
* tensorboardX (Tensorboard for Python)

### Usage

#### Training
```python WINNet_denoise_train.py --mode S --lr 0.001 --outf logs/WINNet_lvl_1_nlvl_25 --lvl 1 --noiseL 25```

#### Testing
```python WINNet_denoise_test.py --logdir logs/WINNet_lvl_1_nlvl_25 --lvl 1 --test_noiseL 25 --start_epoch 50 --end_epoch 50  --test_data Set12```

