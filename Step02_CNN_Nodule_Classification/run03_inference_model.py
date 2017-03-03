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

from sklearn import metrics

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from run00_common import BatcherImage3D, split_list_by_blocks, buildModel_SimpleCNN3D

#####################################################
if __name__ == '__main__':
    # (0) Basic configs
    parBatchSize = 64
    parOptimizer = 'adam'
    fidxTrn = '../data/01_nodules_classification/idx.txt-train.txt'
    fidxVal = '../data/01_nodules_classification/idx.txt-val.txt'
    wdir = os.path.dirname(fidxTrn)
    parModelType = 'SimpleCNN3D'
    # (1) Load data into Batchers
    parIsTheanoShape = True if (K.image_dim_ordering()=='th') else False
    batcherTrn = BatcherImage3D(fidxTrn, isTheanoShape=parIsTheanoShape, isLoadIntoMemory=False)
    batcherVal = BatcherImage3D(fidxVal, isTheanoShape=parIsTheanoShape, isLoadIntoMemory=False,
                                pathMeanData=batcherTrn.pathMeanVal)
    if K.image_dim_ordering() == 'tf':
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.96
        set_session(tf.Session(config=config))
    print (batcherTrn)
    print (batcherVal)
    # (2) Configure training process
    parExportInfo = 'opt.%s.%s' % (parOptimizer, parModelType)
    # (3) Loading model
    modelTrained = BatcherImage3D.loadModelFromDir(wdir)
    # modelTrained.summary()
    # (4) Iterate & Inference
    pathSplit = split_list_by_blocks(range(batcherVal.numImg), parBatchSize)
    arrResults = None
    for ii,pp in enumerate(pathSplit):
        dataX, dataY = batcherVal.getBatchDataByIdx(pp)
        tret = modelTrained.predict_on_batch(dataX)
        if arrResults is None:
            arrResults = tret
        else:
            arrResults = np.concatenate((arrResults,tret))
        # print (pp)
    tmpFPR, tmpTPR, tmpThresh = metrics.roc_curve(batcherVal.arrClsOneHot[:,1], arrResults[:,1])
    tauc = metrics.roc_auc_score(batcherVal.arrClsOneHot[:,1], arrResults[:,1])
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.hold(True)
    plt.plot(tmpFPR, tmpTPR)
    plt.plot([0,1],[0,1])
    plt.hold(False)
    plt.grid(True)
    plt.title('AUC: %0.3f' % tauc)
    #
    plt.subplot(1, 3, 2)
    plt.plot(tmpThresh, tmpFPR), plt.title('FPR')
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(tmpThresh, tmpTPR), plt.title('TPR')
    plt.grid(True)
    plt.show()
    print ('----')


