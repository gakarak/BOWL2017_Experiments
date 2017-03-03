#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import time

import keras
import keras.applications as kapp
from keras import backend as K
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot as kplot

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from run00_common import BatcherImage3D, split_list_by_blocks, buildModel_SimpleCNN3D

#####################################################
if __name__ == '__main__':
    if len(sys.argv) > 5:
        parNumEpoch  = int(sys.argv[1])
        parBatchSize = int(sys.argv[2])
        parOptimizer = sys.argv[3]
        fidxTrn      = sys.argv[4]
        fidxVal      = sys.argv[5]
    else:
        parNumEpoch  = 30
        parBatchSize = 128
        parOptimizer = 'adam'
        fidxTrn = '../data/01_nodules_classification/idx.txt-train.txt'
        fidxVal = '../data/01_nodules_classification/idx.txt-val.txt'
    # (0) Basic configs
    parModelType = 'SimpleCNN3D'
    parBatchSizeTrn = parBatchSize
    parBatchSizeVal = parBatchSize
    # (1) Load data into Batchers
    if K.image_dim_ordering()=='th':
        batcherTrn = BatcherImage3D(fidxTrn, isTheanoShape=True,
                                    isLoadIntoMemory=True)
        batcherVal = BatcherImage3D(fidxVal, isTheanoShape=True,
                                    isLoadIntoMemory=True,
                                    pathMeanData=batcherTrn.pathMeanVal)
    else:
        batcherTrn = BatcherImage3D(fidxTrn, isTheanoShape=False,
                                    isLoadIntoMemory=True)
        batcherVal = BatcherImage3D(fidxVal, isTheanoShape=False,
                                    isLoadIntoMemory=True,
                                    pathMeanData=batcherTrn.pathMeanVal)
    if K.image_dim_ordering() == 'tf':
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.96
        set_session(tf.Session(config=config))
    print (batcherTrn)
    print (batcherVal)
    # (2) Configure training process
    parExportInfo = 'opt.%s.%s' % (parOptimizer, parModelType)
    parNumIterPerEpochTrn = batcherTrn.numImg / parBatchSizeTrn
    parNumIterPerEpochVal = batcherVal.numImg / parBatchSizeTrn
    stepPrintTrn = int(parNumIterPerEpochTrn / 20)
    stepPrintVal = int(parNumIterPerEpochVal / 20)
    if stepPrintTrn < 1:
        stepPrintTrn = parNumIterPerEpochTrn
    if stepPrintVal < 1:
        stepPrintVal = parNumIterPerEpochVal
    #
    print ('*** Train params *** : #Epoch=%d, #BatchSize=%d, #IterPerEpoch=%d' % (parNumEpoch, parBatchSizeTrn, parNumIterPerEpochTrn))
    # (3) Build & visualize model
    if parModelType=='SimpleCNN3D':
        model = buildModel_SimpleCNN3D(inpShape=batcherTrn.shapeImg, numCls=batcherTrn.numCls)
    else:
        raise Exception('Unknown model type: [%s]' % parModelType)
    model.compile(loss='categorical_crossentropy',
                  optimizer=parOptimizer,
                  metrics=['accuracy'])
    fimgModel = 'model-cnn-simple.jpg'
    kplot(model, fimgModel, show_shapes=True)
    plt.imshow(skio.imread(fimgModel))
    plt.show()
    model.summary()
    # (4) Train model
    t0 = time.time()
    for eei in range(parNumEpoch):
        print ('[TRAIN] Epoch [%d/%d]' % (eei, parNumEpoch))
        # (1) train step
        tmpT1 = time.time()
        for ii in range(parNumIterPerEpochTrn):
            dataX, dataY = batcherTrn.getBatchData(parBatchSize=parBatchSizeTrn, isRemoveMean=True)
            tret = model.train_on_batch(dataX, dataY)
            if (ii % stepPrintTrn) == 0:
                print ('\t[train] epoch [%d/%d], iter = [%d/%d] : loss[iter]/acc[iter] = %0.3f/%0.2f%%'
                       % (eei, parNumEpoch, ii, parNumIterPerEpochTrn, tret[0], 100. * tret[1]))
        tmpDT = time.time() - tmpT1
        print ('\t*** train-time for epoch #%d is %0.2fs' % (eei, tmpDT))
        # (2) model validation step
        if ((eei + 1) % 3) == 0:
            tmpIdxList = range(batcherVal.numImg)
            lstIdxSplit = split_list_by_blocks(tmpIdxList, parBatchSizeVal)
            tmpVal = []
            for ii, ll in enumerate(lstIdxSplit):
                dataX, dataY = batcherVal.getBatchDataByIdx(parBatchIdx=ll, isRemoveMean=True)
                tret = model.evaluate(dataX, dataY, verbose=False)
                tmpVal.append(tret)
                if (ii % stepPrintVal) == 0:
                    print ('\t\t[val] epoch [%d/%d], iter = [%d/%d] : loss[iter]/acc[iter] = %0.3f/%0.2f%%'
                           % (eei, parNumEpoch, ii, parNumIterPerEpochVal, tret[0], 100. * tret[1]))
            tmpVal = np.array(tmpVal)
            tmeanValLoss = float(np.mean(tmpVal[:, 0]))
            tmeanValAcc  = float(np.mean(tmpVal[:, 1]))
            print ('\t::validation: mean-losss/mean-acc = %0.3f/%0.3f' % (tmeanValLoss, tmeanValAcc))
        # (3) export model step
        if ((eei + 1) % 3) == 0:
            tmpT1 = time.time()
            tmpFoutModel = batcherTrn.exportModel(model, eei + 1, extInfo=parExportInfo)
            tmpDT = time.time() - tmpT1
            print ('[EXPORT] Epoch [%d/%d], export to [%s], time is %0.3fs' % (eei, parNumEpoch, tmpFoutModel, tmpDT))
    dt = time.time() - t0
    print ('Time for #%d Epochs is %0.3fs, T/Epoch=%0.3fs' % (parNumEpoch, dt, dt / parNumEpoch))
