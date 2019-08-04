# A Bio-inspired Quaternion Local Phase CNN Layer with Contrast Invariance and Linear Sensitivity to Rotation Angles

## Abstract

Deep learning models have been particularly successful with image recognition using Convolutional Neural Networks (CNN). However, the learning of a contrast invariance and rotation equivariance response may fail even with very deep CNNs or by large data augmentations in training.  

We were inspired by the V1 visual features of the mammalian visual system.  To emulate as much as possible the early visual system and add more equivariant capacities to the CNN, we present a  quaternion local phase convolutional neural network layer encoding  three local phases. We present two experimental setups: An image classification task with three contrast levels, and a linear regression task that predicts the rotation angle of an image.  In sum, we obtain new patterns and feature representations for deep learning, which capture illumination invariance and a linear response
to rotation angles.

## Requirements

- Python (3.7)
- keras (2.2.4)
- tensorflow (1.13)
- opencv-python (4.0.0.21)
- pandas (0.24.2)
- argparse (1.4.0)

## Usage
There are three different test cases:
1) 100 MNIST digits with plane rotations and a CNN.
2) 100 MNIST digits with plane rotations, application of the Q9 and a MLP.
3) Comparation of a CNN and Q9 with contrast degradation.

Case # 1
```
$ python regression_conv_100.py
```

Case # 2
```
$ python regression_dense_100.py
```

Case # 3
```
$ python q9vsconlayer.py --batchsize 128 --epochs 100 --eta 0.001
```
