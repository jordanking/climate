#!/usr/bin/env python
# coding: utf-8

# author: Jordan King

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import csv as csv
import pickle
import os.path

from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint
from keras.utils.generic_utils import Progbar, printv

from random import randint, uniform

from matplotlib import pyplot
from skimage.io import imshow
from skimage.util import crop
from skimage import transform, filters, exposure

from scipy.ndimage.filters import gaussian_filter
import math as math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import interactive

class Predictor:
    """
    predicts classes from mnist images
    configure params in the __init__ section
    call run once instantiated and parameterized correctly.
    model architecture is established in build_keras()
    preprocessing params are not yet in the __init__ list, but
    will slowly be added.

    'issues':
    xval mode doesn't make new models each fold
    random displacement is too expensive still
    batch iterators are a bit of a mess if you look too closely
    """

    def __init__(self):
        """
        sets the parameters for the Predictor:
        mode:
            pred - trains on full train, tests on full test, writes out predictions
            xval - cross validation of model (see issues) and is very expensive
            resu - resume training if interrupted by loading a checkpoint model, writes predictions after
            load - load checkpoint model and write predictions
        folds: for xval
        subset: percent of training data to use during xval or model fitting
        """
        self.mode = 'pred'

        self.models = [{'path': 'data/model1_train.csv', 'resolution':[192,288], 'y':1},
                        {'path': 'data/model2_train.csv', 'resolution':[128,256], 'y':2},
                        {'path': 'data/model3_train.csv', 'resolution':[160,320], 'y':3}]
        self.test_data = {'path': 'data/model_test.csv', 'resolution':[72,144], 'y':[]}
        self.nb_classes = len(self.models)

        self.nb_epoch = 16
        self.batch_size = 128
        self.subset = 1

        self.folds = 5

        self.load_weights_file = 'tmp/checkpoint_weights.hdf5'
        self.save_weights_file = 'tmp/checkpoint_weights.hdf5'
        self.resume_from_epoch = 0

        self.out_file = 'solutions/answers1.csv'

    def load_data(self):
        """ 
        returns X and y - data and target - as numpy arrays, X normalized
        and y made categorical.
        """
        self.observations = [model['resolution'][0]*model['resolution'][1]*4 for model in self.models]
        self.X = np.empty([sum(self.observations), 1275])

        idx = 0
        for m in range(len(self.models)):
            print('Loading model', m+1, '...')
            # print(self.models[m])

            self.X[idx:idx+self.observations[m], :1274] = pd.read_csv(self.models[m]['path'], delimiter=',', header=0, dtype='float32').values[:,1:]
            self.X[idx:idx+self.observations[m], 1274] = self.models[m]['y']
            idx += self.observations[m]

        print('Shuffling data order...')
        np.random.shuffle(self.X)

        if self.subset != 1:
            #     np.random.shuffle(model['X'])
            self.X = self.X[0:int(self.subset*data.shape[0]):, ::]

            # model['X'] = model['X'].reshape(model['X'].shape[0], 1, model['resolution'][0], model['resolution'][1])
            # to do : normalize

        self.y = self.X[:,[1274]]
        self.X = self.X[:, :1274]
        print(self.y.shape, self.y)
        self.y = np_utils.to_categorical(self.y, self.nb_classes)

        # preprocess?


    def batch_warp(self, X_batch, y_batch):
        """
        Data augmentation
        """

        return [X_batch, y_batch]

    def build_keras(self):
        """
        constructs the neural network model
        """

        model = Sequential()

        model.add(Dense(1274, 1274))
        model.add(Activation('relu'))
        # model.add(Dropout(0.5))

        # model.add(Dense(1274, 512))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))

        model.add(Dense(1274, self.nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adadelta')

        self.model = model

    # def fit_model(self, X, y):
    #     """
    #     fits a model to some data
    #     """

    #     for e in range(self.nb_epoch):
    #         print('Epoch: ', e, ' of ', self.nb_epoch)
    #         progbar = Progbar(target=X.shape[0], verbose=True)

    #         # batch train with realtime data augmentation
    #         total_accuracy = 0
    #         total_loss = 0
    #         current = 0

    #         for X_batch, y_batch in self.datagen.flow(X, y, self.batch_size):

    #             # prepare the batch with random augmentations
    #             X_batch, y_batch = self.batch_warp(X_batch, y_batch)

    #             # train on the batch
    #             loss, accuracy = self.model.train(X_batch, y_batch, accuracy = True)
                
    #             # update the progress bar
    #             total_loss += loss * self.batch_size
    #             total_accuracy += accuracy * self.batch_size
    #             current += self.batch_size
    #             if current > self.X.shape[0]:
    #                 current = self.X.shape[0]
    #             else:
    #                 progbar.update(current, [('loss', loss), ('acc.', accuracy)])
    #         progbar.update(current, [('loss', total_loss/current), ('acc.', total_accuracy/current)])
            
    #         # checkpoints between epochs
    #         self.model.save_weights(self.save_weights_file, overwrite = True)

    def cross_validate(self):
        """
        provides a simple cross validation measurement. It doen't make a new
        model for each fold though, so it isn't actually cross validation... the
        model just gets better with time for now. This is pretty expensive to run.
        """

        kf = KFold(self.X.shape[0], self.folds)
        scores = []

        for train, test in kf:
            self.build_keras()
            X_train, X_test, y_train, y_test = self.X[train], self.X[test], self.y[train], self.y[test]

            self.model.fit(X_train, y_train, batch_size = self.batch_size, 
                            nb_epoch = self.nb_epoch, verbose = 1,
                            validation_data = (X_test, y_test), show_accuracy = True)
            
            loss, score = self.model.evaluate(X_test, y_test, show_accuracy=True, verbose=1)
            print ('Loss: ' + str(loss))
            print ('Score: ' + str(score))
            scores.append(score)
        
        scores = np.array(scores)
        print("Accuracy: " + str(scores.mean()) + " (+/- " + str(scores.std()/2) + ")")

    def get_predictions(self):
        """
        trains and predicts on the mnist data
        """

        test_data = pd.read_csv(self.test_file, delimiter=',', skiprows=1, dtype='float32')
        # test_data = test_data.reshape(test_data.shape[0], 1, 28, 28)
        # test_data /= 255

        return self.model.predict_classes(test_data, batch_size = self.batch_size)

    def save_predictions(self, predictions):
        """
        saves the predictions to file in a format that kaggle likes.
        :param predictions: A single dimensional list of classifications
        :p
        """

        predictions_file = open(self.out_file, "wb")
        open_file_object = csv.writer(predictions_file, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        open_file_object.writerow(['id','model'])
        for i in range(0,predictions.shape[0]):
            open_file_object.writerow([i+1, predictions[i]])
        predictions_file.close()

    def resume(self):
        """
        resumes training if training is interrupted.
        """
        self.nb_epoch = self.nb_epoch - self.resume_from_epoch

        self.model.load_weights(self.load_weights_file)

        self.model.self.fit(X_train, y_train, batch_size = self.batch_size, nb_epoch = self.nb_epoch, verbose = 1, show_accuracy = True)

    def run(self):
        """
        set up the test here!
        """

        if self.mode != "load":
            print('loading data...')
            self.load_data()
        
        print('building model...')
        if self.mode != 'xval':
            self.build_keras()

        if self.mode == 'xval':
            print('evaluating model...')
            self.cross_validate()

        if self.mode == 'pred':
            print('training model...')
            self.model.fit(self.X, self.y, batch_size = self.batch_size, nb_epoch = self.nb_epoch, verbose = 1, show_accuracy = True)

            print('obtaining predictions...')
            self.save_predictions(self.get_predictions())

        if self.mode == 'resu':
            print('resuming training...')
            self.resume()
            
            print('obtaining predictions...')
            self.save_predictions(self.get_predictions())

        if self.mode == 'load':
            print('loading data...')
            self.model.load_weights(self.load_weights_file)

            print('obtaining predictions...')
            self.save_predictions(self.get_predictions())

def main():
    network = Predictor()
    network.run()

if __name__ == '__main__':
    main()
