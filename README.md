# A Bio-inspired Quaternion Local Phase CNN Layer with Contrast Invariance and Linear Sensitivity to Rotation Angles

## Abstract

Deep learning models have been particularly successful with image recognition using Convolutional Neural Networks (CNN). However, the learning of a contrast invariance and rotation equivariance response may fail even with very deep CNNs or by large data augmentations in training.  

We were inspired by the V1 visual features of the mammalian visual system.  To emulate as much as possible the early visual system and add more equivariant capacities to the CNN, we present a  quaternion local phase convolutional neural network layer encoding  three local phases. We present two experimental setups: An image classification task with three contrast levels, and a linear regression task that predicts the rotation angle of an image.  In sum, we obtain new patterns and feature representations for deep learning, which capture illumination invariance and a linear response
to rotation angles.

## Requirements

- Python (>=3.7)
- keras (>=2.2.4)
- tensorflow (1.15)
- opencv (>=4.1.2)
- pandas (>=0.25.3)
- scikit-image (>=0.15.0)
- matplotlib (>=3.1.2)

## Usage
There are six different test cases:
1) 100 MNIST digits and 100 CIFAR-10 images with plane rotations and a CNN.
2) 100 MNIST digits and 100 CIFAR-10 images with plane rotations, application of the Q9 and a MLP.
3) Comparation of a CNN and Q9 with contrast degradation.

### Case # 1
```
$ python regression_conv_100.py
```
**Output**

*Note: these file can be used by the jupyter notebook file prl_rotation_fig_v2.ipynb.*

 *  *loss_conv_cifar10.csv* (loss of convolution layer with CIFAR-10 dataset)
 *  *loss_conv_mnist.csv* (loss of convolution layer with MNIST dataset)
 *  *val_loss_conv_cifar10.csv* (validation loss of convolution layer with CIFAR-10 dataset)
 *  *val_loss_conv_mnist.csv* (validation loss of convolution layer with MNIST dataset)
 *  *prediction_conv_cifar10.csv* (prediction/inference of convolution layer with CIFAR-10 dataset)
 *  *prediction_conv_mnist.csv* (prediction/inference of convolution layer with MNIST dataset)

### Case # 2
```
$ python regression_q9_100.py
```
**Output**:

*Note: these file can be used by the jupyter notebook file prl_rotation_fig_v2.ipynb.*

 *  *loss_q9_cifar10.csv* (loss of Q9 layer with CIFAR-10 dataset)
 *  *loss_q9_mnist.csv* (loss of Q9 layer with MNIST dataset)
 *  *val_loss_q9_cifar10.csv* (validation loss of Q9 layer with CIFAR-10 dataset)
 *  *val_loss_q9_mnist.csv* (validation loss of Q9 layer with MNIST dataset)
 *  *prediction_q9_cifar10.csv* (prediction/inference of Q9 layer with CIFAR-10 dataset)
 *  *prediction_q9_mnist.csv* (prediction/inference of Q9 layer with MNIST dataset)

### Case # 3
```
---------------------------------------------------------------------
usage: q9vsconlayer.py [-h] -b BATCHSIZE -e EPOCHS -l ETA -d DATA

Q9/CNN MNIST/CIFAR-10

optional arguments:
  -h, --help            show this help message and exit
  -b BATCHSIZE, --batchsize BATCHSIZE
                        Batch size
  -e EPOCHS, --epochs EPOCHS
                        Epochs
  -l ETA, --eta ETA     Learning rate
  -d DATA, --data DATA  Dataset (0) MNIST, (1) CIFAR-10
---------------------------------------------------------------------

$ python q9vsconlayer.py --batchsize 128 --epochs 100 --eta 0.0001 -d 0
```
