# BigGAN-tensorflow
Reimplementation of the Paper: Large Scale GAN Training for High Fidelity Natural Image Synthesis

# Introduction
Simply implement the great paper [(BigGAN)Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/pdf/1809.11096), which can generate very realistic images. However, due to my poor device :sob:, I just train the image of size 32x32 of cifar-10 and the image of size 64x64 of Imagenet64. By the way, the training procedure is really slow.

From the paper:
![](https://github.com/MingtaoGuo/BigGAN-tensorflow/blob/master/IMGS/paper.jpg)

# Dataset
1. Image 32x32: cifar-10: http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz
2. Image 64x64: ImageNet64: https://drive.google.com/open?id=1uN9O69eeqJEPV797d05ZuUmJ23kGVtfU

Just download the dataset, and put them into the folder 'dataset'

# Architecture
![](https://github.com/MingtaoGuo/BigGAN-tensorflow/blob/master/IMGS/architecture.jpg)

# Results
32x32 Cifar-10
--------------
#### Configuration:
Training iteration: 100,000

||Discriminator|Generator|
|-|-|-|
|Update step|2|1|
|Learning rate|4e-4|1e-4|
|Orthogonal reg|:heavy_check_mark:|:heavy_check_mark:|
|Orthogonal init|:heavy_check_mark:|:heavy_check_mark:|
|Hierarchical latent|:x:|:heavy_check_mark:|
|Projection batchnorm|:heavy_check_mark:|:x:|

![](https://github.com/MingtaoGuo/BigGAN-tensorflow/blob/master/IMGS/cifar10.jpg)

64x64 ImageNet
--------------
#### Configuration:
Training iteration: 100,000

||Discriminator|Generator|
|-|-|-|
|Update step|2|1|
|Learning rate|4e-4|1e-4|
|Orthogonal reg|:heavy_check_mark:|:heavy_check_mark:|
|Orthogonal init|:heavy_check_mark:|:heavy_check_mark:|
|Hierarchical latent|:x:|:heavy_check_mark:|
|Projection batchnorm|:heavy_check_mark:|:x:|

Under training ..........
-----------
To be continue.
-----------------
