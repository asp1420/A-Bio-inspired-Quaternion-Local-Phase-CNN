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

import numpy as np
import cv2


def change_contrast_one(image, level=.0):
    zeros = np.zeros(image.shape, dtype=np.float32)
    cv2.normalize(image, zeros, 1, level, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    return zeros*255


def change_contrast(images, level=.0):
    changes = []
    for image in images:
        changes.append(change_contrast_one(image, level))
    return np.array(changes)


def plane_rotation(img, theta):
    ch = len(img.shape)
    if ch == 3:
        cols, rows, ch = img.shape
    if ch == 2:
        cols, rows = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
    im = cv2.warpAffine(img, M, (cols, rows))
    return im
