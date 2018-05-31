#!/usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:11:15 2016

Try some algorithms with pixelated camera

@author: olivierm
"""

from os.path import join, exists, split
from os import makedirs
import argparse
import numpy as np
from scipy.signal import convolve2d
import cv2

# Definition of the Polaim class


class Polaim(object):
    """Class that describes pixelated images"""

    def __init__(self, image, method='superpixel'):
        """Method for initialization"""
        self.raw = image
        self.quad = np.vstack((np.hstack((image[0::2, 0::2],
                                          image[0::2, 1::2])),
                               np.hstack((image[1::2, 1::2],
                                          image[1::2, 0::2]))))
        self.images = [] # list of images I0, I45, I90, I135
        self._set_method(method)

    def _get_method(self):
        """ getter method for the method parameter"""
        return self._method

    def _set_method(self, method):
        """ setter method for the method parameter
        Interpolation strategies for reducing IFOV artifacts in miccrogrid ...
        Ratliff 2009"""
        self._method = method
        if method[:7] == 'ratliff' and 1 <= int(method[7]) <= 4:
            if method[7] == '1':
                kernels = [np.array([[1, 0], [0, 0.]]),
                           np.array([[0, 1], [0, 0.]]),
                           np.array([[0, 0], [0, 1.]]),
                           np.array([[0, 0], [1, 0.]])]
            elif method[7] == '2':
                kernels = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0.]]),
                           np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])/2.,
                           np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])/4.,
                           np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])/2.]
            elif method[7] == '3':
                b = np.sqrt(2)/2/(np.sqrt(2)+np.sqrt(10)/2)
                a = np.sqrt(10)/2/(np.sqrt(2)+np.sqrt(10)/2)

                kernels = [np.array([[0, b, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, a, 0, b],
                                     [0, 0, 0, 0]]),
                           np.array([[0, 0, b, 0],
                                     [0, 0, 0, 0],
                                     [b, 0, a, 0],
                                     [0, 0, 0, 0]]),
                           np.array([[0, 0, 0, 0],
                                     [b, 0, a, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, b, 0]]),
                           np.array([[0, 0, 0, 0],
                                     [0, a, 0, b],
                                     [0, 0, 0, 0],
                                     [0, b, 0, 0]])]
            elif method[7] == '4':
                c = np.sqrt(2)/2/(3*np.sqrt(2)/2+np.sqrt(2)/2+np.sqrt(10))
                b = np.sqrt(10)/2/(3*np.sqrt(2)/2+np.sqrt(2)/2+np.sqrt(10))
                a = 3*np.sqrt(2)/2/(3*np.sqrt(2)/2+np.sqrt(2)/2+np.sqrt(10))

                kernels = [np.array([[0, b, 0, c],
                                     [0, 0, 0, 0],
                                     [0, a, 0, b],
                                     [0, 0, 0, 0]]),
                           np.array([[c, 0, b, 0],
                                     [0, 0, 0, 0],
                                     [b, 0, a, 0],
                                     [0, 0, 0, 0]]),
                           np.array([[0, 0, 0, 0],
                                     [b, 0, a, 0],
                                     [0, 0, 0, 0],
                                     [c, 0, b, 0]]),
                           np.array([[0, 0, 0, 0],
                                     [0, a, 0, b],
                                     [0, 0, 0, 0],
                                     [0, b, 0, c]])]
            Is = []
            for k in kernels:
                Is.append(convolve2d(self.raw, k, mode='same'))

            offsets = [[(0, 0), (0, 1), (1, 1), (1, 0)],
                       [(0, 1), (0, 0), (1, 0), (1, 1)],
                       [(1, 1), (1, 0), (0, 0), (0, 1)],
                       [(1, 0), (1, 1), (0, 1), (0, 0)]]

            self.images = []
            for (j, o) in enumerate(offsets):
                self.images.append(np.zeros(self.raw.shape))
                for ide in range(4):
                    self.images[j][o[ide][0]::2, o[ide][1]::2] = Is[ide][o[ide][0]::2, o[ide][1]::2]

        elif method == 'superpixel':
            self.images = [self.raw[0::2, 0::2].astype(float),
                           self.raw[0::2, 1::2].astype(float),
                           self.raw[1::2, 1::2].astype(float),
                           self.raw[1::2, 0::2].astype(float)]

        else:
            self.images = []

    method = property(_get_method, _set_method)

    @property
    def polarization(self):
        """ Property that computes the polar params from the 4 images """
        Js = self.images
        inten = (Js[0]+Js[1]+Js[2]+Js[3])/2.
        aop = (0.5*np.arctan2(Js[1]-Js[3], Js[0]-Js[2]))
        dop = np.sqrt((Js[1]-Js[3])**2+(Js[0]-Js[2])**2)/(Js[0]+Js[1]+Js[2]+Js[3]+np.finfo(float).eps)*2
        return (inten, aop, dop)

    @property
    def rgb(self):
        """ Property that return the RGB representation of the pola params """
        (inten, aop, dop) = self.polarization
        hsv = np.uint8(cv2.merge(((aop+np.pi/2)/np.pi*180,
                                  dop*255,
                                  inten/inten.max()*255)))
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    def __repr__(self):
        """Representation function of the Polaim class"""
        return "Polacam image ({})".format(self.method)

    @property
    def export(self):
        """Image representation for export"""
        (inten, aop, dop) = self.polarization
        nbr, nbc = self.rgb.shape[0], self.rgb.shape[1]
        fina = np.zeros((nbr*2, nbc*2, 3), dtype='uint8')
        aop_colorHSV = np.uint8(cv2.merge(((aop+np.pi/2)/np.pi*180,
                                           np.ones(aop.shape)*255,
                                           np.ones(aop.shape)*255)))
        aop_colorRGB = cv2.cvtColor(aop_colorHSV, cv2.COLOR_HSV2RGB)

        for c in range(3):
            fina[:nbr, :nbc, c] = np.uint8(inten/inten.max()*255)
            fina[:nbr, nbc:, c] = aop_colorRGB[:, :, c]
            fina[nbr:, :nbc, c] = np.uint8(dop*255)
            fina[nbr:, nbc:, c] = self.rgb[:, :, c]
        return fina


if __name__=='__main__':
    #Direct call of the script
    # main

    # suppose we have
    #000 045 000 045
    #135 090 135 090
    #000 045 000 045
    #135 090 135 090

    # imCAM = synthetic()
    parser = argparse.ArgumentParser(prog='polaconv',
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    parser.add_argument("folder", help="folder that contains the image files")
    parser.add_argument('--m', help="M = method: superpixel, ratliff1, ratliff2, ratliff3, ratliff4", default='superpixel')
    parser.add_argument('--e', help="E = extension: tiff, bmp", default='tiff')
    parser.add_argument('--v', help="V = video output", action='store_true')


    args = parser.parse_args()
    rep_in = args.folder

    path_in = rep_in
    if exists(path_in):
        if path_in.endswith('/'):  # to remove the last slash if present
            path_in = path_in[:-1]
        path, folder = split(path_in)
        path_out = join(path, folder + 'rgb')
        # initialization
        if args.v:
            # video
            # set the video size
            if args.m == 'superpixel':
                size = (640, 460)
            else:
                size = (1280, 920)
            video = cv2.VideoWriter(folder + '.avi', 0, 10, size)
        else:
            # sequence
            if not exists(path_out):
                makedirs(path_out)
            else:
                pass

        # reading files
        count = 0
        while True:
            count += 1
            print(count)
            fname = 'image_{:05d}'.format(count) + '.' + args.e
            print(fname)
            if exists(join(path_in, fname)):
                imCAM = cv2.imread(join(path_in, fname), -1)
                print(imCAM.min(), imCAM.max())
                imp = Polaim(imCAM, args.m)
                print("Processing " + join(path_out, fname))
                if args.v:
                    video.write(imp.export)
                else:
                    testim = imp.export
                    print(testim.shape)
                    cv2.imwrite(join(path_out, fname), imp.export)
                    cv2.imwrite(join(path_out + '/1', fname), testim[0:459, 0:639, :])
                    cv2.imwrite(join(path_out + '/2', fname), testim[0:459, 640:1279, :])
                    cv2.imwrite(join(path_out + '/3', fname), testim[460:919, 640:1279, :])
                    cv2.imwrite(join(path_out + '/4', fname), testim[460:919, 0:639, :])
            else:
                break
        if args.v:
            # close the video
            video.release()
