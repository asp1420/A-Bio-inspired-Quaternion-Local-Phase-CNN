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
import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
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

parser = argparse.ArgumentParser(description='Q9 MNIST')
parser.add_argument('-b', '--batchsize', required=True, type=int, help='Batch size')
parser.add_argument('-e', '--epochs', required=True, type=int, help='Epochs')
parser.add_argument('-l', '--eta', required=True, type=float, help='Learning rate')
args = parser.parse_args()

path_models = 'models'

create_directory(path_models)

contrast_levels = [.0, .3, .6, .9]
test_types = ['normal', 'q9']

batch_size = args.batchsize
num_classes = 10
lr = args.eta
epochs = args.epochs

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_val = x_train[50000:, :]
y_val = y_train[50000:]
x_train = x_train[:50000, :]
y_train = y_train[:50000]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Train -> (x)', x_train.shape, '- (y)', y_train.shape)
print('Validation -> (x)', x_val.shape, '- (y)', y_val.shape)
print('Test -> (x)', x_test.shape, '- (y)', y_test.shape)

# Training
for level in contrast_levels:

    train = np.expand_dims(change_contrast(x_train, level), axis=3)
    validation = np.expand_dims(change_contrast(x_val, level), axis=3)
    print('Level     ', level)
    for ttest in test_types:

        if ttest == 'normal':
            input_shape = train[0].shape
            model = get_conv_model(input_shape, num_classes)
        elif ttest == 'q9':
            print('Calculating Q9 ...')
            train = qph9(train, 28, phi=0, ch=.25, cv=.25, sh=4, sv=4)
            train = train.reshape((50000, 7056))
            validation = qph9(validation, 28, phi=0, ch=.25, cv=.25, sh=4, sv=4)
            validation = validation.reshape((10000, 7056))
            input_shape = train[0].shape
            model = get_dense_model(input_shape, num_classes)
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

        hist = model.fit(train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(validation, y_val),
                         callbacks=callbacks_list,
                         verbose=1)
        del model
        K.clear_session()
        gc.collect()

# Evaluation
evaluation = [['model', 'contlvl', 'test', 'tstacc', 'tstloss']]
for level in contrast_levels:
    test = np.expand_dims(change_contrast(x_test, level), axis=3)
    list_models = sorted(os.listdir(path_models))

    for ttest in test_types:
        if ttest == 'q9':
            print('Calculating Q9 ...')
            test = qph9(test, 28, phi=0, ch=.25, cv=.25, sh=4, sv=4)
            test = test.reshape((10000, 7056))
        test = test.astype('float32') / 255

        for model_name in list_models:
            if ttest in model_name:
                model = load_model(join(path_models, model_name))
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
df.to_csv('evaluation.csv', index=False, header=False)
