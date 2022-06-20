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
```python WINNet_train_iter.py --mode S --noiseL 15 --val_noiseL 15 --lr 0.001 --outf logs/WINNet_noiseL_15 --lvl 1 --start_epoch 0 --preprocess True```

#### Testing


