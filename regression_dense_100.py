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

from keras.layers import Dense
from keras.layers import Input

try:
    sys.path.insert(0, 'tools')
    from transform import plane_rotation
    from quaternionImage import qph9_one
except:
    raise
from keras.models import Model
from keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

steps = 5
total_digits = 100
angle_qft = 0
angle = np.arange(0, 46, steps)

x = np.zeros((len(angle), 7056))
y = np.zeros((len(angle), 1))

curr_proc = 0
losses = []
predictions = []

for (image, label) in zip(x_train, y_train):
    if curr_proc > total_digits - 1: break
    if label == 0: continue
    print('------------------ (', curr_proc, ') ------------------')
    for i in range(0, len(angle)):
        rot_img = plane_rotation(image, angle[i])
        p = qph9_one(img=rot_img, out_size=28, phi=.0, ch=.5, cv=.5, sh=4, sv=4)
        x[i] = p.flatten()
        y[i] = angle[i] / 45
        print(angle[i], '° =', y[i], ', digit:', label)

    inputs = Input((7056,))
    dens = Dense(units=1000, input_shape=(7056,), activation='sigmoid')(inputs)
    dens = Dense(units=64, activation='sigmoid')(dens)
    dens = Dense(units=32, activation='sigmoid')(dens)
    dens = Dense(units=8, activation='sigmoid')(dens)
    dens = Dense(units=1, activation='sigmoid')(dens)
    model = Model(inputs=inputs, outputs=dens)

    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    history = model.fit(x, y, epochs=1000, batch_size=10, verbose=1)

    angle = np.arange(0, 46, 5)

    test = np.zeros((len(angle), 7056))

    for i in range(0, len(angle)):
        rot_img = plane_rotation(image, angle[i])
        p = qph9_one(img=rot_img, out_size=28, phi=.0, ch=.5, cv=.5, sh=4, sv=4)
        test[i] = p.flatten()

    pred = []
    for i in range(0, len(angle)):
        test2 = np.expand_dims(test[i], axis=0)
        prediction = model.predict(test2)
        pred.append(prediction)
        print(angle[i], '° =', prediction, 'digit:', label)

    p = np.array(pred)
    p = np.append(p, label)
    loss = history.history['loss']
    loss = np.append(loss, label)
    losses.append(loss)
    predictions.append(p)
    curr_proc += 1

np.savetxt("predictions_dense.csv", np.array(predictions), delimiter=',')
np.savetxt('losses_dense.csv', np.array(losses), delimiter=',')
