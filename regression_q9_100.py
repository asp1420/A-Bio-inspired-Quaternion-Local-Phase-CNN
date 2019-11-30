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

import sys
import gc
import numpy as np

from keras.layers import Dense
from keras.layers import Input
from keras import backend as K
try:
    sys.path.insert(0, 'tools')
    from transform import plane_rotation
    from quaternionImage import qph9_one
except:
    raise
from keras.models import Model
from keras.datasets import cifar10
from keras.datasets import mnist

datasets = ['mnist', 'cifar10']

for dataset in datasets:

    print('='*70)
    print(dataset)
    print('='*70)

    if dataset is 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset is 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    height, width = x_train.shape[1:3]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    steps = 3
    total_images = 100
    curr_proc = 0
    epochs = 1000
    losses = []
    val_losses = []
    predictions = []

    angle_train = np.arange(0, 46, steps)
    angle_val = np.arange(1, 46, steps)
    angle_test = np.arange(2, 46, steps)

    train_x = np.zeros((len(angle_train), height*width*9))
    train_y = np.zeros((len(angle_train), 1))
    val_x = np.zeros((len(angle_val), height*width*9))
    val_y = np.zeros((len(angle_val), 1))
    test_x = np.zeros((len(angle_test), height*width*9))
    test_y = np.zeros((len(angle_test), 1))

    for (image, label) in zip(x_train, y_train):
        if curr_proc > total_images - 1: break
        if label == 0: continue
        print('------------------ (', curr_proc, ') ------------------')
        # Rotation train
        for i in range(0, len(angle_train)):
            rot_img = plane_rotation(image, angle_train[i])
            p = qph9_one(img=rot_img, out_size=height, phi=.0, ch=.5, cv=.5, sh=4, sv=4)
            train_x[i] = p.flatten()
            train_y[i] = angle_train[i] / 45
        # Rotation validation
        for i in range(0, len(angle_val)):
            rot_img = plane_rotation(image, angle_val[i])
            p = qph9_one(img=rot_img, out_size=height, phi=.0, ch=.5, cv=.5, sh=4, sv=4)
            val_x[i] = p.flatten()
            val_y[i] = angle_val[i] / 45
        # Rotation test
        for i in range(0, len(angle_test)):
            rot_img = plane_rotation(image, angle_test[i])
            p = qph9_one(img=rot_img, out_size=height, phi=.0, ch=.5, cv=.5, sh=4, sv=4)
            test_x[i] = p.flatten()
            test_y[i] = angle_test[i] / 45

        inputs = Input((height*width*9,))
        dens = Dense(units=1000, input_shape=(height*width*9,), activation='sigmoid')(inputs)
        dens = Dense(units=64, activation='sigmoid')(dens)
        dens = Dense(units=32, activation='sigmoid')(dens)
        dens = Dense(units=8, activation='sigmoid')(dens)
        dens = Dense(units=1, activation='sigmoid')(dens)
        model = Model(inputs=inputs, outputs=dens)
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        # Train and validation
        history = model.fit(train_x, train_y, epochs=epochs, batch_size=10, verbose=1, validation_data=(val_x, val_y))

        # Evaluation
        pred = []
        for i in range(0, len(angle_test)):
            test = np.expand_dims(test_x[i], axis=0)
            prediction = model.predict(test)
            pred.append(prediction)
            print(angle_test[i], '° =', prediction*45, 'label:', label)

        p = np.array(pred)
        p = np.append(p, label)
        predictions.append(p)

        loss = history.history['loss']
        loss = np.append(loss, label)
        losses.append(loss)

        val_loss = history.history['val_loss']
        val_loss = np.append(val_loss, label)
        val_losses.append(val_loss)
        curr_proc += 1
        K.clear_session()
        gc.collect()

    np.savetxt("prediction_q9_%s.csv" % (dataset), np.array(predictions), delimiter=',')
    np.savetxt("loss_q9_%s.csv" % (dataset), np.array(losses), delimiter=',')
    np.savetxt("val_loss_q9_%s.csv" % (dataset), np.array(val_losses), delimiter=',')

