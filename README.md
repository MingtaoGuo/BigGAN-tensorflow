# BigGAN-tensorflow
Reimplementation of the Paper: Large Scale GAN Training for High Fidelity Natural Image Synthesis

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
|Orthogonal reg|True|True|
|Orthogonal init|True|True|
|Hierarchical latent|X|True|
|Projection batchnorm|True|X|

![](https://github.com/MingtaoGuo/BigGAN-tensorflow/blob/master/IMGS/cifar10.jpg)

64x64 ImageNet
--------------
#### Configuration:
Training iteration: 100,000

||Discriminator|Generator|
|-|-|-|
|Update step|2|1|
|Learning rate|4e-4|1e-4|
|Orthogonal reg|True|True|
|Orthogonal init|True|True|
|Hierarchical latent|X|True|
|Projection batchnorm|True|X|

To be continue.
-----------------
