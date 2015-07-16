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
        self.mode = 'kmnn'

        self.models = [{'path': 'data/model1_train.csv', 'resolution':[192,288], 'y':0},
                        {'path': 'data/model2_train.csv', 'resolution':[128,256], 'y':1},
                        {'path': 'data/model3_train.csv', 'resolution':[160,320], 'y':2}]

        # self.models = [{'path': 'small_data/model1_train.csv', 'resolution':[99,1], 'y':0},
        #                 {'path': 'small_data/model2_train.csv', 'resolution':[99,1], 'y':1},
        #                 {'path': 'small_data/model3_train.csv', 'resolution':[99,1], 'y':2}]

        self.test_data = {'path': 'data/model_test.csv', 'resolution':[72,144], 'y':[]}
        self.nb_classes = len(self.models)

        self.nb_epoch = 20
        self.batch_size = 128
        self.subset = 1
        self.columns = 1274

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

        # add a column for y values to shuffle
        self.X = np.empty([sum(self.observations), self.columns+1])

        idx = 0
        for m in range(len(self.models)):
            print('Loading model', m+1, '...')
            # print(self.models[m])
            self.X[idx:idx+self.observations[m], :self.columns] = pd.read_csv(self.models[m]['path'], delimiter=',', header=0, dtype='float32').values[:,1:self.columns+1]
            self.X[idx:idx+self.observations[m], self.columns] = self.models[m]['y']
            idx += self.observations[m]

        print('Shuffling data order...')
        np.random.shuffle(self.X)

        if self.subset != 1:
            #     np.random.shuffle(model['X'])
            self.X = self.X[0:int(self.subset*self.X.shape[0]):, ::]

            # model['X'] = model['X'].reshape(model['X'].shape[0], 1, model['resolution'][0], model['resolution'][1])
            # to do : normalize

        self.y = self.X[:,[self.columns]]
        self.X = self.X[:, :self.columns]

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

        model.add(Dense(self.columns, self.columns))
        model.add(Activation('relu'))
        # model.add(Dropout(0.5))

        model.add(Dense(self.columns, self.columns))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.columns, self.columns))
        model.add(Activation('relu'))
        # model.add(Dropout(0.5))

        model.add(Dense(self.columns, self.nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adadelta')

        self.model = model

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

    def kmeans_xval(self):
        """
        provides a simple cross validation measurement. It doen't make a new
        model for each fold though, so it isn't actually cross validation... the
        model just gets better with time for now. This is pretty expensive to run.
        """

        kf = KFold(self.X.shape[0], self.folds)
        scores = []

        for train, test in kf:
            X_train, X_test, y_train, y_test = self.X[train], self.X[test], self.y[train], self.y[test]

            correct = 0.0
            total = 0.0

            for i in range(X_test.shape[0]):
                total += 1
                best_class, best_dist, next_best_class, next_best_dist, n = self.find_best_class(X_test[i], X_train, y_train)

                if (best_class == y_test[i]).all():
                    correct += 1

                print('Seen:',total, 'Pred:', np.argmax(best_class)+1,'Truth:',np.argmax(y_test[i])+1, 'Dist', best_dist, 'Corr:', correct, 'Acc:', correct/total)
                print('Test Obs:', X_test[i], 'Nearest Neighbor:', X_train[n])
            
            score = correct / X_test.shape[0]
            print ('Score: ' + str(score))
            scores.append(score)
        
        scores = np.array(scores)
        print("Accuracy: " + str(scores.mean()) + " (+/- " + str(scores.std()/2) + ")")

    def kmeans(self):
        """
        provides a simple cross validation measurement. It doen't make a new
        model for each fold though, so it isn't actually cross validation... the
        model just gets better with time for now. This is pretty expensive to run.
        """

        X_test = pd.read_csv(self.test_data['path'], delimiter=',', skiprows=1, dtype='float32').values[:,1:self.columns+1]
        predictions = np.empty([X_test.shape[0]], dtype = 'int32')

        X_train, y_train = self.X, self.y

        for i in range(X_test.shape[0]):

            best_class, best_dist, next_best_class, next_best_dist, n = self.find_best_class(X_test[i], X_train, y_train)
            # print('best class',best_class,'best_dist',best_dist)
            predictions[i] = int(np.argmax(best_class)) + 1
            print('ID:', i, 'Pred:', predictions[i], 'Dist:', best_dist, '2nd Pred:', int(np.argmax(next_best_class))+1, '2nd dist:', next_best_dist)
            print('Test Obs:', X_test[i], 'Nearest Neighbor:', X_train[n])

        return predictions

    def find_best_class(self, current, X_train, y_train):
        best_class, next_best_class, best_dist, next_best_dist, best_idx = [0,0,0], [0,0,0], 3.4028235e+38, 3.4028235e+38, 0
        for n in range(X_train.shape[0]):
            if abs(X_train[n][0] - current[0]) < 2 and abs(X_train[n][1]- current[1]) < 2:
                # print(X_train[n])
                dist = np.linalg.norm(X_train[n][2:] - current[2:])
                if dist < best_dist:
                    next_best_dist = best_dist
                    best_dist = dist
                    next_best_class = best_class
                    best_class = y_train[n]
                    best_idx = n
        # return np.argmax(best_class) - 1
        return best_class, best_dist, next_best_class, next_best_dist, best_idx

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
            print('Loading data...')
            self.load_data()
        
        if self.mode != 'xval' and self.mode != 'kmnn':
            print('building model...')
            self.build_keras()

        if self.mode == 'xval':
            print('evaluating model...')
            self.cross_validate()

        if self.mode == 'kmxv':
            print('Performing k-means xval')
            self.kmeans_xval()

        if self.mode == 'kmnn':
            print('Performing k-means nearest neighbor search')
            self.save_predictions(self.kmeans())

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