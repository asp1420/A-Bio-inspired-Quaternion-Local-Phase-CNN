#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = ["E. Ulises Moya", " Sebastian Salazar-Colores", "Abraham Sanchez", "Sebastian Xambò", "Ulises Cortes"]
__copyright__ = "Copyright 2019, Gobierno de Jalisco"
__credits__ = ["E. Ulises Moya"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = ["E. Ulises Moya", "Abraham Sanchez"]
__email__ = "eduardo.moya@jalisco.gob.mx"
__status__ = "Development"

import argparse
import os
import sys
from os.path import join

import gc
import time
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import load_model

from models import get_conv_model
from models import get_dense_model

try:
    sys.path.insert(0, 'tools')
    from quaternionImage import qph9
    from transform import change_contrast
    from utils import create_directory
except:
    raise

parser = argparse.ArgumentParser(description='Q9/CNN MNIST/CIFAR-10')
parser.add_argument('-b', '--batchsize', required=True, type=int, help='Batch size')
parser.add_argument('-e', '--epochs', required=True, type=int, help='Epochs')
parser.add_argument('-l', '--eta', required=True, type=float, help='Learning rate')
parser.add_argument('-d', '--data', required=True, type=int, help='Dataset (0) MNIST, (1) CIFAR-10')
args = parser.parse_args()

path_models = 'models'

create_directory(path_models)

contrast_levels = [.0, .3, .7, .9]
test_types = ['normal', 'q9']

batch_size = args.batchsize
num_classes = 10
lr = args.eta
epochs = args.epochs

if args.data == 0:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('Open MNIST')
else:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

img_rows, img_cols = x_train.shape[1:3]

train_elements = x_train.shape[0] - 10000

x_val = x_train[train_elements:, :]
y_val = y_train[train_elements:]
x_train = x_train[:train_elements, :]
y_train = y_train[:train_elements]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Train -> (x)', x_train.shape, '- (y)', y_train.shape)
print('Validation -> (x)', x_val.shape, '- (y)', y_val.shape)
print('Test -> (x)', x_test.shape, '- (y)', y_test.shape)

print('\nTraining ...')

# Training in diferent contrast values
for level in contrast_levels:

    train = change_contrast(x_train, level)
    validation = change_contrast(x_val, level)
    if train[0].ndim == 2:
        train = np.expand_dims(train, axis=-1)
        validation = np.expand_dims(validation, axis=-1)
    print('='*80)
    print('Contrast Level     ', level)
    print('='*80)
    for ttest in test_types:
        creation_time = time.time()
        if ttest == 'normal':
            input_shape = train[0].shape
            model = get_conv_model(input_shape, num_classes)
            print(model.summary())
        elif ttest == 'q9':
            
            print('\nCalculating Q9 ...')
            train = qph9(train, img_rows, phi=0, ch=.5, cv=.5, sh=4, sv=4)
            train = train.reshape((len(x_train), img_rows*img_rows*9))
            validation = qph9(validation, img_rows, phi=0, ch=.5, cv=.5, sh=4, sv=4)
            validation = validation.reshape((len(x_val), img_rows*img_rows*9))
            input_shape = train[0].shape
            model = get_dense_model(input_shape, num_classes)
            print(model.summary())
        print('Creation Time %s'% (time.time()-creation_time))
        train = train.astype('float32') / 255
        validation = validation.astype('float32') / 255

        opt = keras.optimizers.Adam(lr=lr)
        model_name = 'contrast_%i_%s.h5' % (int(level * 100), ttest)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        checkpoint = ModelCheckpoint(join(path_models, model_name),
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto',
                                     period=1)
        callbacks_list = [checkpoint, History()]
        train_time= time.time()
        history = model.fit(train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(validation, y_val),
                         callbacks=callbacks_list,
                         verbose=1)
        print('Training Time %s'% (time.time()-train_time))
        print('Total Time %s'% (time.time()-creation_time))
        ph_hist = pd.DataFrame(history.history) 
        ph_hist.to_csv('history_contrast_%i_%s_%s.csv' % (int(level * 100), ttest,args.data))
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('acc_contrast_%i_%s_%s.png' % (int(level * 100), ttest,args.data))
        plt.close()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss_contrast_%i_%s_%s.png' % (int(level * 100), ttest,args.data))
        plt.close()
        del model
        K.clear_session()
        gc.collect()
    del train
    del validation
    gc.collect()


print('\nTest ...')

# Evaluation
evaluation = [['model', 'contlvl', 'test', 'tstacc', 'tstloss']]
for level in contrast_levels:
    print('='*80)
    print('Contrast Level     ', level)
    print('='*80)
    test = change_contrast(x_test, level)
    if test[0].ndim == 2:
        test = np.expand_dims(test, axis=-1)
    list_models = sorted(os.listdir(path_models))

    for ttest in test_types:
        if ttest == 'q9':
            print('Calculating Q9 ...')
            test = qph9(test, img_rows, phi=0, ch=.5, cv=.5, sh=4, sv=4)
            test = test.reshape((len(x_test), img_rows*img_rows*9))
        test = test.astype('float32') / 255

        for model_name in list_models:
            if ttest in model_name:
                model = load_model(join(path_models, model_name))
                print(model.summary())
                loss, acc = model.evaluate(x=test,
                                           y=y_test,
                                           batch_size=batch_size,
                                           verbose=1,
                                           sample_weight=None,
                                           steps=None)
                evaluation.append([model_name, str(level), ttest, str(acc), str(loss)])
                print('model:', model_name, 'cont lvl:', level, 'test type:', ttest, 'acc:', acc, 'loss:', loss)
                del model
                K.clear_session()
                gc.collect()

df = pd.DataFrame(evaluation)
df.to_csv('evaluation_clasification_%s.csv'%(args.data), index=False, header=False)

print('Done!')
