#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import nibabel as nib
import skimage.transform as sktf

##################################
def resizeNii(pathNii, newSize=(33, 33, 33)):
    if isinstance(pathNii,str) or isinstance(pathNii,unicode):
        tnii = nib.load(pathNii)
    else:
        tnii = pathNii
    timg = tnii.get_data()
    oldSize = timg.shape
    dataNew = sktf.resize(timg, newSize, order=4, preserve_range=True, mode='edge')
    affineOld = tnii.affine.copy()
    affineNew = tnii.affine.copy()
    k20_Old = float(oldSize[2]) / float(oldSize[0])
    k20_New = float(newSize[2]) / float(newSize[0])
    for ii in range(3):
        tCoeff = float(newSize[ii]) / float(oldSize[ii])
        if ii == 2:
            tCoeff = (affineNew[0, 0] / affineOld[0, 0]) * (k20_Old / k20_New)
        affineNew[ii, ii] *= tCoeff
        affineNew[ii,  3] *= tCoeff
    retNii = nib.Nifti1Image(dataNew, affineNew, header=tnii.header)
    return retNii

##################################
if __name__ == '__main__':
    # wdir = '/mnt/data1T2/datasets2/kaggle_Bowl_2017/tmp/_data_2/_ones'
    # lstPath = sorted(glob.glob('%s/*.hdr' % wdir))
    # fidx = '/mnt/data1T2/datasets2/kaggle_Bowl_2017/tmp/_data_2/idx.txt'
    fidx = '/Users/alexanderkalinovsky/data/Lab225/kaggle_DSB_2017_ar/original/rois_three_classes/idx.txt'
    with open(fidx, 'r') as f:
        lstPath = sorted(f.read().splitlines())
    numPath = len(lstPath)
    for ii,pp in enumerate(lstPath):
        inpNii = nib.load(pp)
        outNii = resizeNii(inpNii, newSize=(33,33,33))
        foutNii = '%s.nii.gz' % os.path.splitext(pp)[0]
        nib.save(outNii, foutNii)
        # plt.figure()
        # #1
        # plt.subplot(3, 2, 1)
        # plt.imshow(inpNii.get_data()[:, :, 3 * inpNii.shape[-1] / 10])
        # plt.subplot(3, 2, 2)
        # plt.imshow(outNii.get_data()[:, :, 3 * outNii.shape[-1] / 10])
        # #2
        # plt.subplot(3, 2, 3)
        # plt.imshow(inpNii.get_data()[:, :, 5 * inpNii.shape[-1] / 10])
        # plt.subplot(3, 2, 4)
        # plt.imshow(outNii.get_data()[:, :, 5 * outNii.shape[-1] / 10])
        # #3
        # plt.subplot(3, 2, 5)
        # plt.imshow(inpNii.get_data()[:, :, 7 * inpNii.shape[-1] / 10])
        # plt.title('%s' % list(inpNii.shape))
        # plt.subplot(3, 2, 6)
        # plt.imshow(outNii.get_data()[:, :, 7 * outNii.shape[-1] / 10])
        # plt.title('%s' % list(outNii.shape))
        # plt.show()
        print ('[%d/%d] : %s : %s -> %s ' % (ii, numPath, os.path.basename(foutNii), list(inpNii.shape), list(outNii.shape)))
