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
from keras.utils.vis_utils import plot_model
from keras.layers import Input, Conv3D, Flatten, Activation, MaxPooling3D, Dense
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
    retY = np_utils.to_categorical(arrLbl, num_classes=numLbl)
    numData = len(arrPath)
    numSplit = 7
    numStep = numData/numSplit
    if numStep<1:
        numStep=1
    print (':: Loading 3d-images: [%s]' % os.path.basename(pathIdx))
    for ii,pathNii in enumerate(arrPath):
        timg = nib.load(pathNii).get_data().astype(np.float)
        if isRemoveMean:
            timg -= meanVal
            timg /= meanStd
        if K.image_data_format()=='channels_first':
            timg = timg.reshape([1] + list(timg.shape))
        else:
            timg = timg.reshape(list(timg.shape) + [1])
        if retX is None:
            retX = np.zeros([numData] + list(timg.shape))
        retX[ii] = timg
        if (ii%numStep)==0:
            print ('\t[%d/%d] : %s' % (ii, numData, pathNii))
    return (retX, retY)

#####################################################
def buildModel_SimpleCNN3D(inpShape=(64, 64, 64, 1), numCls=2, sizFlt=3, numHiddenDense=128):
    dataInput = Input(shape=inpShape)
    # -------- Encoder --------
    # Conv1
    kernelSize = (sizFlt, sizFlt, sizFlt)
    x = Conv3D(filters=16, kernel_size=kernelSize, padding='same', activation='relu')(dataInput)
    # x = Conv3D(filters=16, kernel_size=kernelSize, padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    # Conv2
    x = Conv3D(filters=32, kernel_size=kernelSize, padding='same', activation='relu')(x)
    # x = Conv3D(filters=32, kernel_size=kernelSize, padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    # Conv3
    x = Conv3D(filters=64, kernel_size=kernelSize, padding='same', activation='relu')(x)
    # x = Conv3D(filters=64, kernel_size=kernelSize, padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    #
    # Dense1 (hidden)
    x = Flatten()(x)
    if numHiddenDense>0:
        x = Dense(units=numHiddenDense, activation='relu')(x)
    # Dense2
    x = Dense(units=numCls, activation='softmax')(x)
    retModel = Model(dataInput, x)
    return retModel

#####################################################
def _getRand():
    return 2. * (np.random.rand() - 0.5)

def train_generator(dataX, dataY, batchSize=32, isRandomize=False, meanShift=0.1, meaStd=0.01):
    numLbl = dataY.shape[-1]
    dictLblIdx = {xx:np.where(dataY[:,xx]>0.5)[0] for xx in range(numLbl)}
    while True:
        retX = np.zeros([batchSize * numLbl] + list(dataX.shape[1:]))
        retY = np.zeros([batchSize * numLbl] + list(dataY.shape[1:]))
        dataIdx = []
        for ii in range(numLbl):
            dataIdx.append(np.random.permutation(dictLblIdx[ii].copy())[:batchSize])
        dataIdx = np.random.permutation(np.array(dataIdx).reshape(-1))
        for ii, idx in enumerate(dataIdx):
            if isRandomize:
                rndShiftMean = meanShift*_getRand()
                rndShiftStd  = meaStd*_getRand()
                retX[ii] = (dataX[idx] - rndShiftMean)/(1.0 + rndShiftStd)
            else:
                retX[ii] = dataX[idx]
            retY[ii] = dataY[idx]
        yield (retX, retY)

#####################################################
if __name__ == '__main__':
    # pathIdxTrn = '/Users/alexanderkalinovsky/data/Lab225/kaggle_DSB_2017_ar/rois_three_classes/idx-trn.txt'
    # pathIdxVal = '/Users/alexanderkalinovsky/data/Lab225/kaggle_DSB_2017_ar/rois_three_classes/idx-val.txt'
    pathIdxTrn = '/home/ar/datasets/kaggle_DSB_2017_ar/idx-trn.txt'
    pathIdxVal = '/home/ar/datasets/kaggle_DSB_2017_ar/idx-val.txt'
    pathMean = '%s-%s' % (pathIdxTrn, prefMean)
    pathModel = 'model_cnn_ct3d.h5'
    pathModelPlot = '%s-plot.png' % pathModel
    pathLog = '%s-log.csv' % pathModel
    #
    trnX, trnY = loadTrainData(pathIdxTrn, pathMean=pathMean, isRemoveMean=False)
    valX, valY = loadTrainData(pathIdxVal, pathMean=pathMean, isRemoveMean=False)
    numCls = trnY.shape[-1]
    numTrn = trnY.shape[ 0]
    #
    if not os.path.isfile(pathModel):
        model = buildModel_SimpleCNN3D(inpShape=valX.shape[1:],
                                       numCls=numCls,
                                       numHiddenDense=-1)
        # popt = kopt.Adam(lr=0.00001)
        popt = 'adam'
        model.compile(optimizer=popt,
                      loss='categorical_crossentropy',
                      # loss='binary_crossentropy',
                      metrics=['accuracy'])
    else:
        pathModelBk = '%s-%s.bk' % (pathModel, time.strftime('%Y.%m.%d-%H.%M.%S'))
        shutil.copy(pathModel, pathModelBk)
        model = keras.models.load_model(pathModel)
    plot_model(model, to_file=pathModelPlot, show_shapes=True)
    # plt.imshow(skio.imread(pathModelPlot))
    # plt.show()
    model.summary()
    batchSize = 8
    numEpochs = 100
    numIterPerEpoch = numTrn/(numCls*batchSize)
    model.fit(trnX, trnY, epochs=10, validation_data=(valX, valY))
    # model.fit_generator(
    #     generator=train_generator(dataX=trnX, dataY=trnY, batchSize=16, isRandomize=False),
    #     steps_per_epoch=numIterPerEpoch,
    #     epochs=numEpochs, validation_data=(valX, valY),
    #     callbacks=[
    #         kall.ModelCheckpoint(pathModel, verbose=True, save_best_only=True),
    #         kall.CSVLogger(pathLog, append=True)
    #     ])