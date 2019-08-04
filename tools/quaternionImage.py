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

# function to create a Quaternion Gabor Function
# The imput imaage  has to be  in gray scale and  is better if is square image
# I belive that nos square image  could have some possible errors

import cv2
import numpy as np


def qgabor(L1, L2, phi=0, ch=2, cv=2, sh=2, sv=2):
    X = np.arange(-L1 / 2, L1 / 2, 1)
    Y = np.arange(-L2 / 2, L2 / 2, 1)
    X, Y = np.meshgrid(X, Y)
    x = X * np.cos(phi) + Y * np.sin(phi)
    y = -X * np.sin(phi) + Y * np.cos(phi)
    g = np.exp((-(x ** 2)) / (2 * sh ** 2)) * np.exp((-(y ** 2)) / (2 * sv ** 2))
    r = g * np.cos(ch * x / sh) * np.cos(cv * y / sv)
    i = g * np.sin(ch * x / sh) * np.cos(cv * y / sv)
    j = g * np.cos(ch * x / sh) * np.sin(cv * y / sv)
    k = g * np.sin(ch * x / sh) * np.sin(cv * y / sv)
    return r, i, j, k


def qfilter(img, L1, L2, phi=0, ch=2, cv=2, sh=2, sv=2):
    if img.ndim == 3:
        img = img.mean(2)
    b = 1
    img = cv2.add(img, b)  # some bias to avoid zero in the artag

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    r, i, j, k = qgabor(L1, L2, phi, ch, cv, sh, sv)
    m0 = np.dstack((r, r))
    m1 = np.dstack((i, i))
    m2 = np.dstack((j, j))
    m3 = np.dstack((k, k))
    m0 = np.dstack((r, r))
    m1 = np.dstack((i, i))
    m2 = np.dstack((j, j))
    m3 = np.dstack((k, k))
    fshift0 = dft_shift * m0
    fshift1 = dft_shift * m1
    fshift2 = dft_shift * m2
    fshift3 = dft_shift * m3

    f_ishift0 = np.fft.ifftshift(fshift0)
    img_back0 = cv2.idft(f_ishift0)
    img_back0 = cv2.magnitude(img_back0[:, :, 0], img_back0[:, :, 1])

    f_ishift1 = np.fft.ifftshift(fshift1)
    img_back1 = cv2.idft(f_ishift1)
    img_back1 = cv2.magnitude(img_back1[:, :, 0], img_back1[:, :, 1])

    f_ishift2 = np.fft.ifftshift(fshift2)
    img_back2 = cv2.idft(f_ishift2)
    img_back2 = cv2.magnitude(img_back2[:, :, 0], img_back2[:, :, 1])

    f_ishift3 = np.fft.ifftshift(fshift3)
    img_back3 = cv2.idft(f_ishift3)
    img_back3 = cv2.magnitude(img_back3[:, :, 0], img_back3[:, :, 1])

    return img_back0, img_back1, img_back2, img_back3


def qphase(image, L1, L2, phi, ch, cv, sh, sv):
    a, b, c, d = qfilter(image, L1, L2, phi, ch, cv, sh, sv)
    mag = np.sqrt(np.multiply(a, a) + np.multiply(b, b) + np.multiply(c, c) + np.multiply(d, d))
    mag_ = mag / mag.max()  # Normalize to one
    # Normalizacion.
    a = a / mag
    b = b / mag
    c = c / mag
    d = d / mag
    e0 = a
    e1 = b
    e2 = c
    e3 = d

    a11 = np.multiply(e0, e0) + np.multiply(e1, e1) - np.multiply(e2, e2) - np.multiply(e3, e3)
    a21 = 2 * (np.multiply(e1, e2) + np.multiply(e0, e3))
    a31 = 2 * (np.multiply(e1, e3) - np.multiply(e0, e2))
    a12 = 2 * (np.multiply(e1, e2) - np.multiply(e0, e3))
    a22 = np.multiply(e0, e0) - np.multiply(e1, e1) + np.multiply(e2, e2) - np.multiply(e3, e3)
    a32 = 2 * (np.multiply(e2, e3) + np.multiply(e0, e1))
    a13 = 2 * (np.multiply(e1, e3) + np.multiply(e0, e2))

    PSI = -np.arcsin(a12)
    PHI = -np.arctan2(-a32 / np.cos(PSI), a22 / np.cos(PSI))
    THETA = -np.arctan2(-a13 / np.cos(PSI), a11 / np.cos(PSI))

    eps = 1e-10
    INDX = abs(a12) >= 1 - eps
    PHI[INDX] = 0
    PSI[INDX] = -a12[INDX] * np.pi / 2
    THETA[INDX] = -a12[INDX] * np.arctan2(a31[INDX] * a12[INDX], -a21[INDX] * a12[INDX])

    PHI = PHI / 2
    THETA = THETA / 2
    PSI = PSI / 2

    return mag_, PHI, THETA, PSI


def qph333(img, phi, ch, cv, sh, sv):
    if img.ndim == 3:
        img = img.mean(2)

    L1, L2 = img.shape
    mag, phi, theta, psi = qphase(img, L1, L2, phi, ch, cv, sh, sv)
    a = L1
    b = L2

    Magnitud = np.zeros((a, b))
    Phi = np.zeros((a, b))
    Theta = np.zeros((a, b))
    Psi = np.zeros((a, b))

    cv2.normalize(mag, Magnitud, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    cv2.normalize(phi, Phi, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    cv2.normalize(theta, Theta, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    cv2.normalize(psi, Psi, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    Phi *= 179
    Phi.astype('uint8')
    Phi = cv2.convertScaleAbs(Phi)

    Theta *= 179
    Theta.astype('uint8')
    Theta = cv2.convertScaleAbs(Theta)

    Psi *= 179
    Psi.astype('uint8')
    Psi = cv2.convertScaleAbs(Psi)

    Magnitud *= 255
    Magnitud.astype('uint8')
    Magnitud = cv2.convertScaleAbs(Magnitud)

    zeros = np.zeros(Magnitud.shape)
    zeros.astype('uint8')
    zeros = cv2.convertScaleAbs(zeros)

    ones = np.zeros(Magnitud.shape)
    ones[:, :] = 255
    ones.astype('uint8')
    ones = cv2.convertScaleAbs(ones)

    Phi_img = cv2.merge((Phi, ones, Magnitud))
    Phi_bgr = cv2.cvtColor(Phi_img, cv2.COLOR_HSV2BGR)
    Phi_rgb = cv2.cvtColor(Phi_bgr, cv2.COLOR_BGR2RGB)

    Theta_img = cv2.merge((Theta, ones, Magnitud))
    Theta_bgr = cv2.cvtColor(Theta_img, cv2.COLOR_HSV2BGR)
    Theta_rgb = cv2.cvtColor(Theta_bgr, cv2.COLOR_BGR2RGB)

    Psi_img = cv2.merge((Psi, ones, Magnitud))
    Psi_bgr = cv2.cvtColor(Psi_img, cv2.COLOR_HSV2BGR)
    Psi_rgb = cv2.cvtColor(Psi_bgr, cv2.COLOR_BGR2RGB)

    return Phi_rgb, Theta_rgb, Psi_rgb


def qph9_one(img, out_size, phi, ch, cv, sh, sv):
    if img.ndim == 3:
        img = img.mean(2)
    img = cv2.resize(img, (out_size, out_size))
    phi_3, theta_3, psi_3 = qph333(img, phi, ch, cv, sh, sv)
    ch_num = 9
    qftrgb9 = np.zeros((out_size, out_size, ch_num), dtype='uint8')
    qftrgb9[:, :, 0] = phi_3[:, :, 0]
    qftrgb9[:, :, 1] = phi_3[:, :, 1]
    qftrgb9[:, :, 2] = phi_3[:, :, 2]
    qftrgb9[:, :, 3] = theta_3[:, :, 0]
    qftrgb9[:, :, 4] = theta_3[:, :, 1]
    qftrgb9[:, :, 5] = theta_3[:, :, 2]
    qftrgb9[:, :, 6] = psi_3[:, :, 0]
    qftrgb9[:, :, 7] = psi_3[:, :, 1]
    qftrgb9[:, :, 8] = psi_3[:, :, 2]
    return qftrgb9


def qph9(images, out_size, phi, ch, cv, sh, sv):
    changes = []
    for image in images:
        changes.append(qph9_one(image, out_size, phi, ch, cv, sh, sv))
    return np.array(changes)
