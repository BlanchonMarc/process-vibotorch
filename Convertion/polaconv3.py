#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:57:02 2018

@author: olivierm

compute AOP movie or image from a set of raw images

"""
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


data = np.genfromtxt('flatfield.csv',
                     dtype=float, delimiter=',', skip_header=9)

# %% Pixel order

# only valid for super pixel

# PO = (0, 0)
# I0   I45
# I135 I90

# PO = (0, 1)
# I45  I0
# I90  I135

# PO = (1, 0)
# I135  I90
# I0    I45

# PO = (1, 1)
# I90   I135
# I45   I0

PO = (0, 1)

# %% Read Polar

def read_polar(filename):
    """ Read a raw image"""
    # --- Open the image
    imCAM = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    imCAM = imCAM * data
    # --- Extract superpixels
    images = np.array([imCAM[PO[0]::2, PO[1]::2],
                       imCAM[PO[0]::2, 1-PO[1]::2],
                       imCAM[1-PO[0]::2, 1-PO[1]::2],
                       imCAM[1-PO[0]::2, PO[1]::2]])

    # --- Saturated pixels
    #satur = reduce(np.bitwise_and, images) == 255

    # --- Stokes parameters
    M = np.array([[0.5, 0.5, 0.5, 0.5],
                  [1.0, 0.0, -1., 0.0],
                  [0.0, 1.0, 0.0, -1.]])

    stokes = np.tensordot(M, images, 1)

    # --- Other parameters
    compl = stokes[1]+stokes[2]*1j
    aop = np.angle(compl) / 2.
    dop = np.divide(abs(compl), stokes[0], out=np.zeros_like(stokes[0]), where=stokes[0] != 0)



#    A = 0.5 * np.array([[1, 1, 0.],
#                        [1, 0, 1.],
#                        [1, -1, 0],
#                        [1, 0, -1]])
#
#    error = sum((images - np.tensordot(np.dot(A, M), images, 1))**2,0)

    # --- show some results
    #aop[satur] = np.nan
    aop[dop > 1] = np.nan
    aop[dop < 0.2] = np.nan
    cmap = plt.get_cmap('prism')
    cmap.set_bad((0, 0, 0))
    cmap.set_under((0, 0, 0))
    aop_RGB = cmap(np.mod(aop, np.pi)/np.pi)

    return np.uint8(aop_RGB[:, :, [2, 1, 0]]*255)



#fname = 'Datasets/2017-03-14-move1/frame-0000.tiff'
#cwd = os.getcwd()
#image = read_polar(fname)
#base, first = os.path.split(fname)
#name, ext = first.split('.')
#plt.imshow(image)
#cv2.imwrite(os.path.join(cwd, name + '-aop.png'), image)


# %% Main
if __name__ == '__main__':
    #Direct call of the script


    parser = argparse.ArgumentParser(prog='polaconv3',
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    parser.add_argument("fname", help="image to be converted or first image to be converted as video")
    parser.add_argument('--v', help="V = video output", action='store_true')


    args = parser.parse_args()

    cwd = os.getcwd()  # Get the current working directory


    if args.v:  # Video
        #--- Initialization
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        base, first = os.path.split(args.fname)
        init, ext = first.lstrip('frame-').split('.')
        name = os.path.join(cwd, os.path.split(base)[1])
        video = cv2.VideoWriter(name + '.avi', fourcc, 10, (320, 230), True)

        #--- Write
        count = int(init)

        while True:
            fname = 'frame-{:04d}'.format(count) + '.' + ext
            if os.path.exists(os.path.join(base, fname)):
                image = read_polar(os.path.join(base, fname))
                video.write(image)
                count += 1
            else:
                break

        #--- Close
        video.release()


    else:
        # -- image
        image = read_polar(args.fname)
        base, first = os.path.split(args.fname)
        name, ext = first.split('.')
        cv2.imwrite(os.path.join(cwd, name + '-aop.png'), image)
