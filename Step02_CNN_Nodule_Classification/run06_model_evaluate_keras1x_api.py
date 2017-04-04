#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import time
import shutil
import os
import nibabel as nib
import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg') #FIXME: Fix for bug in OrthoSlicer3D() viewer with matplotlib 2.0.0

import matplotlib.pyplot as plt
import skimage.io as skio

import keras
import keras.backend as K
import keras.optimizers as kopt
import keras.callbacks as kall
from keras.utils.visualize_util import plot as plot_model
from keras.layers import Input, Flatten, Activation, MaxPooling3D, Dense, Convolution3D
from keras.models import Model
from keras.utils import np_utils

try:
   import cPickle as pickle
except:
   import pickle

#####################################################
prefMean='mean.pkl'

#####################################################
def buildMeanData(lstPath, pathMean, numMean = 100, isDebug=False):
    if os.path.isfile(pathMean):
        print (':: Loading mean data... [%s]' % pathMean)
        with open(pathMean, 'r') as f:
            ret = pickle.load(f)
    else:
        print (':: Build mean data...')
        numData  = len(lstPath)
        if numData < numMean:
            numMean = numData
        lstPath = np.array(lstPath)
        lstPath = np.random.permutation(lstPath)[:numMean]
        numSplit = 5
        numStep  = numMean/numSplit
        if numStep<1:
            numStep = 1
        meanImg = None
        for ii, pathNii in enumerate(lstPath):
            timg = nib.load(pathNii).get_data().astype(np.float)
            if meanImg is None:
                meanImg  = timg.copy()
            else:
                meanImg += timg
            if (ii%numStep)==0:
                print ('\t[%d/%d]' % (ii, numMean))
        meanImg /= numMean
        ret = {
            'meanImg': meanImg,
            'meanVal': meanImg.mean(),
            'meanStd': meanImg.std()
        }
        with open(pathMean, 'w') as f:
            pickle.dump(ret, f)
    print (':: mean/std = %0.4f/%0.4f' % (ret['meanVal'], ret['meanStd']))
    if isDebug:
        nib.viewers.OrthoSlicer3D(ret['meanImg']).show()
    return ret

#####################################################
def loadTrainData(pathIdx, pathMean=None, isRemoveMean=True):
    wdir = os.path.dirname(pathIdx)
    with open(pathIdx, 'r') as f:
        lstPath = f.read().splitlines()
    arrLbl  = np.array([int(os.path.basename(xx)[0]) for xx in lstPath])
    numLbl = len(np.unique(arrLbl))
    arrPath = np.array([os.path.join(wdir, xx) for xx in lstPath])
    if pathMean is None:
        pathMean = '%s-%s' % (pathIdx, pathMean)
    meanInfo = buildMeanData(arrPath.copy().tolist(), pathMean, isDebug=False)
    meanVal = meanInfo['meanVal']
    meanStd = meanInfo['meanStd']
    retX = None
    retY = np_utils.to_categorical(arrLbl, nb_classes=numLbl)
    numData = len(arrPath)
    numSplit = 7
    numStep = numData/numSplit
    if numStep<1:
        numStep=1
    print (':: Loading 3d-images: [%s]' % os.path.basename(pathIdx))
    lstIdx = []
    for ii,pathNii in enumerate(arrPath):
        lstIdx.append(os.path.basename(pathNii)[:-7].split('_')[-1])
        timg = nib.load(pathNii).get_data().astype(np.float)
        if isRemoveMean:
            timg -= meanVal
            timg /= 3.*meanStd
        if K.image_dim_ordering()=='th':
            timg = timg.reshape([1] + list(timg.shape))
        else:
            timg = timg.reshape(list(timg.shape) + [1])
        if retX is None:
            retX = np.zeros([numData] + list(timg.shape))
        retX[ii] = timg
        if (ii%numStep)==0:
            print ('\t[%d/%d] : %s' % (ii, numData, pathNii))
    return (retX, retY, lstIdx)

#####################################################
if __name__ == '__main__':
    # pathIdxTrn = '/Users/alexanderkalinovsky/data/Lab225/kaggle_DSB_2017_ar/rois_three_classes/idx-trn.txt'
    # pathIdxVal = '/Users/alexanderkalinovsky/data/Lab225/kaggle_DSB_2017_ar/rois_three_classes/idx-val.txt'
    pathIdxTrn  = '/home/ar/datasets/kaggle_DSB_2017_ar/idx-trn.txt'
    pathIdxData = '/home/ar/datasets/kaggle_DSB_2017_ar/rois_val_nii_33x33x33/idx.txt'
    pathMean = '%s-%s' % (pathIdxTrn, prefMean)
    pathModel = 'model_cnn_ct3d.h5'
    pathModelPlot = '%s-plot.png' % pathModel
    pathLog = '%s-log.csv' % pathModel
    #
    # trnX, trnY = loadTrainData(pathIdxTrn, pathMean=pathMean, isRemoveMean=True)
    dataX, dataY, dataIdx = loadTrainData(pathIdxData, pathMean=pathMean, isRemoveMean=True)
    # numCls = trnY.shape[-1]
    # numTrn = trnY.shape[ 0]
    #
    if not os.path.isfile(pathModel):
        raise Exception('Cant find model [%s]' % pathModel)
    else:
        # pathModelBk = '%s-%s.bk' % (pathModel, time.strftime('%Y.%m.%d-%H.%M.%S'))
        # shutil.copy(pathModel, pathModelBk)
        model = keras.models.load_model(pathModel)
    # plot_model(model, to_file=pathModelPlot, show_shapes=True)
    # plt.imshow(skio.imread(pathModelPlot))
    # plt.show()
    model.summary()
    retProb = model.predict(dataX, batch_size=32)
    fout = '%s-prob.csv' % pathIdxData
    with open(fout, 'w') as f:
        for ii, (idx, prob) in enumerate(zip(dataIdx, retProb)):
            f.write('%s,%0.5f,%0.5f,%0.5f\n' % (idx, prob[0], prob[1], prob[2]))
    print ('---')

    # batchSize = 8
    # numEpochs = 300
    # numIterPerEpoch = numTrn/(numCls*batchSize)
    # model.fit(trnX, trnY, nb_epoch=10, validation_data=(valX, valY))
    # model.fit_generator(
    #     generator=train_generator(dataX=trnX, dataY=trnY, batchSize=batchSize, isRandomize=True),
    #     samples_per_epoch=numIterPerEpoch,
    #     nb_epoch=numEpochs, validation_data=(valX, valY),
    #     callbacks=[
    #         kall.ModelCheckpoint(pathModel, verbose=True, save_best_only=True),
    #         kall.CSVLogger(pathLog, append=True)
    #     ])