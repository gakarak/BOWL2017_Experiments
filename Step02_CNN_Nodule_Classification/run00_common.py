#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import glob
import os
import sys
import time
import numpy as np
import json
import nibabel as nib

try:
   import cPickle as pickle
except:
   import pickle

import skimage.io as skio
import skimage.transform as sktf
import skimage.color as skolor
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.utils import np_utils

from keras.models import Model
from keras.layers import Dense, Convolution3D, Activation, MaxPooling3D,\
    Flatten, BatchNormalization, InputLayer, Dropout, Reshape, Permute, Input, UpSampling3D, Lambda
from keras.layers.normalization import BatchNormalization

#####################################################
def split_list_by_blocks(lst, psiz):
    tret = [lst[x:x + psiz] for x in range(0, len(lst), psiz)]
    return tret

#####################################################
class BatcherImage3D:
    meanPrefix  = 'mean.pkl'
    wdir        = None
    pathCSV     = None
    pathMeanVal = None
    arrPathImg  = None
    arrClsIdx   = None
    arrClsOneHot= None
    numCls  = 0
    numImg  = 0
    shapeImg    = None
    #
    isDataInMemory = False
    dataImg     = None
    meanData    = None
    #
    imgScale    = 1.
    modelPrefix = None
    def __init__(self, pathCSV, pathMeanData=None, isRecalculateMeanIfExist=False, isTheanoShape=True, isLoadIntoMemory=False):
        self.isTheanoShape=isTheanoShape
        if not os.path.isfile(pathCSV):
            raise Exception('Cant find CSV file [%s]' % pathCSV)
        self.pathCSV    = os.path.abspath(pathCSV)
        self.wdir = os.path.dirname(self.pathCSV)
        # (1) load images info
        tdataCSV = pd.read_csv(self.pathCSV, sep=',')
        tmpPathsList = tdataCSV['path'].tolist()
        self.arrPathImg  = np.array([os.path.join(self.wdir, xx) for xx in tmpPathsList])
        self.arrClsIdx   = tdataCSV['label'].as_matrix()
        self.numImg = len(self.arrPathImg)
        self.numCls = len(np.unique(self.arrClsIdx))
        self.arrClsOneHot   = np_utils.to_categorical(self.arrClsIdx, self.numCls)
        # (2) initialize shape parameter
        tdataImg = nib.load(self.arrPathImg[0]).get_data()
        tdataImg = self.transformImageFromOriginal(tdataImg, isRemoveMean=False)
        self.shapeImg = tdataImg.shape
        # (3) load mean-data
        if pathMeanData is None:
            self.pathMeanVal = '%s-%s' % (self.pathCSV, self.meanPrefix)
            self.precalculateAndLoadMean(isRecalculateMean=isRecalculateMeanIfExist)
        else:
            self.pathMeanVal = os.path.abspath(pathMeanData)
            if not os.path.isfile(self.pathMeanVal):
                raise Exception('Cant find MEAN-data file [%s]' % self.pathMeanVal)
            self.precalculateAndLoadMean(isRecalculateMean=isRecalculateMeanIfExist)
        # (4) Load into memory
        if isLoadIntoMemory:
            self.isDataInMemory = True
            self.dataImg = np.zeros([self.numImg] + list(self.shapeImg), dtype=np.float)
            print (':: Loading data into memory:')
            for ii in range(self.numImg):
                tpathImg = self.arrPathImg[ii]
                tdataImg = nib.load(tpathImg).get_data()
                tdataImg = self.transformImageFromOriginal(tdataImg, isRemoveMean=True)
                self.dataImg[ii] = tdataImg
                if (ii % 10) == 0:
                    print ('\t[%d/%d] ...' % (ii, self.numImg))
            print ('\t... [done]')
        else:
            self.isDataInMemory = False
            self.dataImg = None
    def toString(self):
        if self.isInitialized():
            tstr = 'BatcherImage3D(#Images=%d, #Cls=%d, mean=%0.2f, shape=%s, inMemory=%s)' \
                   % (self.numImg, self.numCls, self.meanData['meanCh'], list(self.shapeImg), self.isDataInMemory)
        else:
            tstr = 'BatcherImage3D is not initialized'
        return tstr
    def __str__(self):
        return self.toString()
    def __repr__(self):
        return self.toString()
    def isInitialized(self):
        return (self.numImg>0) and (self.numCls > 0) and (self.wdir is not None)
    def checkIsInitialized(self):
        if not self.isInitialized():
            raise Exception('class Batcher() is not correctly initialized')
    def precalculateAndLoadMean(self, isRecalculateMean=False):
        if os.path.isfile(self.pathMeanVal) and (not isRecalculateMean):
            print (':: found mean-value file, try to load from it [%s] ...' % self.pathMeanVal)
            with open(self.pathMeanVal, 'r') as f:
                self.meanData = pickle.load(f)
            tmpMeanKeys = ('meanImg', 'meanCh', 'meanStd', 'meanImgCh')
            for ii in tmpMeanKeys:
                if ii not in self.meanData.keys():
                    raise Exception('Mean-file is invalid. Cant find key-value in mean-file [%s]' % self.pathMeanVal)
        else:
            self.meanData = {}
            self.meanData['meanImg'] = None
            self.meanData['meanImgCh'] = None
            maxNumImages = 1000
            if len(self.arrPathImg) < maxNumImages:
                maxNumImages = len(self.arrPathImg)
            rndIdx = np.random.permutation(range(len(self.arrPathImg)))[:maxNumImages]
            print ('*** Precalculate mean-info:')
            for ii, idx in enumerate(rndIdx):
                tpathImg = self.arrPathImg[idx]
                tdataImg = nib.load(tpathImg).get_data()
                tdataImg = self.transformImageFromOriginal(tdataImg, isRemoveMean=False)
                if self.meanData['meanImg'] is None:
                    self.meanData['meanImg'] = tdataImg
                else:
                    self.meanData['meanImg'] += tdataImg
                if (ii % 10) == 0:
                    print ('\t[%d/%d] ...' % (ii, len(rndIdx)))
            self.meanData['meanImg'] /= len(rndIdx)
            self.meanData['meanCh']  = np.mean(self.meanData['meanImg'])
            self.meanData['meanStd'] = np.std(self.meanData['meanImg'])
            print (':: mean-image %s mean/std channels value is [%s/%s], saved to [%s]'
                   % (self.meanData['meanImg'].shape, self.meanData['meanCh'], self.meanData['meanStd'], self.pathMeanVal))
            with open(self.pathMeanVal, 'wb') as f:
                pickle.dump(self.meanData, f)
    def preprocImageShape(self, img):
        if self.isTheanoShape:
            return img.reshape([1] + list(img.shape))
        else:
            return img.reshape(list(img.shape) + [1])
    def removeMean(self, img):
        ret = img
        ret -= self.meanData['meanCh']
        ret /= 3*self.meanData['meanStd']
        # ret -= self.meanData['meanImg']
        return ret
    def transformImageFromOriginal(self, pimg, isRemoveMean=True):
        tmp = self.preprocImageShape(pimg)
        tmp = tmp.astype(np.float) / self.imgScale
        if isRemoveMean:
            tmp = self.removeMean(tmp)
        return tmp
    def getBatchDataByIdx(self, parBatchIdx, isRemoveMean=True):
        rndIdx = parBatchIdx
        parBatchSize = len(rndIdx)
        dataX = np.zeros([parBatchSize] + list(self.shapeImg), dtype=np.float)
        dataY = np.zeros([parBatchSize] + [self.numCls], dtype=np.float)
        for ii, tidx in enumerate(rndIdx):
            if self.isDataInMemory:
                dataX[ii] = self.dataImg[tidx]
            else:
                tpathImg = self.arrPathImg[tidx]
                timg = nib.load(tpathImg).get_data()
                tdataImg = self.transformImageFromOriginal(timg, isRemoveMean=isRemoveMean)
                dataX[ii] = tdataImg
            dataY[ii] = self.arrClsOneHot[tidx, :]
        return (dataX, dataY)
    def getBatchData(self, parBatchSize=64, isRemoveMean=True):
        rndIdx=np.random.permutation(range(self.numImg))[:parBatchSize]
        return self.getBatchDataByIdx(parBatchIdx=rndIdx, isRemoveMean=isRemoveMean)
    def exportModel(self, model, epochId, extInfo=None):
        if extInfo is not None:
            tmpModelPrefix = '%s-%s' % (self.modelPrefix, extInfo)
        else:
            tmpModelPrefix = self.modelPrefix
        foutModel = "%s-e%03d.json" % (tmpModelPrefix, epochId)
        foutWeights = "%s-e%03d.h5" % (tmpModelPrefix, epochId)
        foutModel = '%s-%s' % (self.pathCSV, foutModel)
        foutWeights = '%s-%s' % (self.pathCSV, foutWeights)
        with open(foutModel, 'w') as f:
            str = json.dumps(json.loads(model.to_json()), indent=3)
            f.write(str)
        model.save_weights(foutWeights, overwrite=True)
        return foutModel
    @staticmethod
    def loadModelFromJson(pathModelJson):
        if not os.path.isfile(pathModelJson):
            raise Exception('Cant find JSON-file [%s]' % pathModelJson)
        tpathBase = os.path.splitext(pathModelJson)[0]
        tpathModelWeights = '%s.h5' % tpathBase
        if not os.path.isfile(tpathModelWeights):
            raise Exception('Cant find h5-Weights-file [%s]' % tpathModelWeights)
        with open(pathModelJson, 'r') as f:
            tmpStr = f.read()
            model = keras.models.model_from_json(tmpStr)
            model.load_weights(tpathModelWeights)
        return model
    @staticmethod
    def loadModelFromDir(pathDirWithModels, paramFilter=None):
        if paramFilter is None:
            lstModels = glob.glob('%s/*.json' % pathDirWithModels)
        else:
            lstModels = glob.glob('%s/*%s*.json' % (pathDirWithModels, paramFilter))
        pathJson = os.path.abspath(sorted(lstModels)[-1])
        print (':: found model [%s] in directory [%s]' % (os.path.basename(pathJson), pathDirWithModels))
        return BatcherImage3D.loadModelFromJson(pathJson)

#####################################################
def buildModel_SimpleCNN3D(inpShape=(1, 64, 64, 64), numCls=2, sizFlt=3, numHiddenDense=128):
    dataInput = Input(shape=inpShape)
    # -------- Encoder --------
    # Conv1
    x = Convolution3D(nb_filter=16, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
                      border_mode='same', activation='relu')(dataInput)
    x = Convolution3D(nb_filter=16, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
                      border_mode='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    # Conv2
    x = Convolution3D(nb_filter=32, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
                      border_mode='same', activation='relu')(x)
    x = Convolution3D(nb_filter=32, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
                      border_mode='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    # Conv3
    x = Convolution3D(nb_filter=64, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
                      border_mode='same', activation='relu')(x)
    x = Convolution3D(nb_filter=64, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
                      border_mode='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    #
    # Dense1 (hidden)
    x = Flatten()(x)
    if numHiddenDense>0:
        x = Dense(output_dim=numHiddenDense, activation='relu')(x)
    # Dense2
    x = Dense(output_dim=numCls, activation='softmax')(x)
    retModel = Model(dataInput, x)
    return retModel

#####################################################
if __name__=='__main__':
    fcsvTrn = '/mnt/data1T2/datasets2/kaggle_Bowl_2017/tmp/data_nii/idx.txt-train.txt'
    fcsvVal = '/mnt/data1T2/datasets2/kaggle_Bowl_2017/tmp/data_nii/idx.txt-val.txt'
    wdir = os.path.dirname(fcsvTrn)
    batcherTrain = BatcherImage3D(pathCSV=fcsvTrn,
                                  isTheanoShape=True,
                                  isLoadIntoMemory=False)
    batcherVal = BatcherImage3D(pathCSV=fcsvVal,
                                pathMeanData=batcherTrain.pathMeanVal,
                                isTheanoShape=True,
                                isLoadIntoMemory=True)
    print (batcherTrain)
    print (batcherVal)
    dataX, dataY = batcherTrain.getBatchData(parBatchSize=8, isRemoveMean=True)
    print ('dataX.shape = %s, dataY.shape = %s' % ( list(dataX.shape), list(dataY.shape) ))
    print ('-------')


