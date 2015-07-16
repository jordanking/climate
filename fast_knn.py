#!/usr/bin/env python
# coding: utf-8

# author: Jordan King

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import csv as csv

from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

from random import randint, uniform
import math as math

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

        self.subset = 1
        self.columns = 1274

        self.folds = 5

        self.out_file = 'solutions/answersknn.csv'

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
            np.random.shuffle(model['X'])
            self.X = self.X[0:int(self.subset*self.X.shape[0]):, ::]

            # to do : normalize?

        self.y = self.X[:,[self.columns]]
        self.X = self.X[:, :self.columns]

        self.y = np_utils.to_categorical(self.y, self.nb_classes)



   

    def kmeans_xval(self):
        """
        provides a simple cross validation measurement. It doen't make a new
        model for each fold though, so it isn't actually cross validation... the
        model just gets better with time for now. This is pretty expensive to run.
        """

    def kmeans(self):
        """
        provides a simple cross validation measurement. It doen't make a new
        model for each fold though, so it isn't actually cross validation... the
        model just gets better with time for now. This is pretty expensive to run.
        """

        X_test = pd.read_csv(self.test_data['path'], delimiter=',', skiprows=1, dtype='float32').values[:,1:self.columns+1]
        predictions = np.empty([X_test.shape[0]], dtype = 'int32')

        kdtree = spatial.cKDTree(self.X[:,0:2], leafsize=10)

        for i in range(X_test.shape[0]):

            best_class, next_best_class, best_dist, next_best_dist, best_idx = [0,0,0], [0,0,0], 3.4028235e+38, 3.4028235e+38, 0

            # find points within +- 2 degrees lat long
            neighbors = kdtree.query_ball_point(X_test[i][0:2], 2)
            
            # find closest neighbor among these
            for n in range(len(neighbors)):
                dist = np.linalg.norm(X_test[i][2:] - self.X[n][2:])
                if dist < best_dist:
                    next_best_dist = best_dist
                    best_dist = dist
                    next_best_class = best_class
                    best_class = self.y[n]
                    best_idx = n

            predictions[i] = np.argmax(best_class) + 1

            print('ID:', i, 'Pred:', predictions[i], 'Dist:', best_dist, '2nd Pred:', np.argmax(next_best_class)+1, '2nd dist:', next_best_dist)
            print('Test Obs:', X_test[i], 'Nearest Neighbor:', self.X[best_idx])

        return predictions

    

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

    
    def run(self):
        """
        set up the test here!
        """


        print('Loading data...')
        self.load_data()

        if self.mode == 'kmxv':
            print('Performing k-means xval')
            self.kmeans_xval()

        if self.mode == 'kmnn':
            print('Performing k-means nearest neighbor search')
            self.save_predictions(self.kmeans())

def main():
    p = Predictor()
    p.run()

if __name__ == '__main__':
    main()