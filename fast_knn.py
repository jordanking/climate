#!/usr/bin/env python
# coding: utf-8

# author: Jordan King

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import csv as csv
from scipy import spatial
from scipy import stats


from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

from random import randint, uniform
import math as math

class Predictor:
    """
    Classifies model observations based on nearest neighbor search
    """

    def __init__(self):
        """
        sets the parameters for the Predictor:
        mode:
            kmxv - xval
            kmnn - predict and save predictions
        folds: for xval
        subset: percent of training data to use during xval or model fitting
        """
        self.mode = 'kmnn'

        self.models = [{'path': 'data/model1_train.csv', 'resolution':[192,288], 'y':1},
                        {'path': 'data/model2_train.csv', 'resolution':[128,256], 'y':2},
                        {'path': 'data/model3_train.csv', 'resolution':[160,320], 'y':3}]

        # self.models = [{'path': 'small_data/model1_train.csv', 'resolution':[99,1], 'y':1},
        #                 {'path': 'small_data/model2_train.csv', 'resolution':[99,1], 'y':2},
        #                 {'path': 'small_data/model3_train.csv', 'resolution':[99,1], 'y':3}]

        self.test_data = {'path': 'data/model_test.csv', 'resolution':[72,144], 'y':[]}
        self.nb_classes = len(self.models)

        self.subset = 1
        self.columns = 1274
        self.topn = 5

        self.folds = 5

        self.out_file = 'solutions/answersknnv.csv'

    def load_data(self):
        """ 
        returns X and y - data and target - as numpy arrays, y made categorical.
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

        print('Finding mean observations...')
        for i in range(self.X.shape[0]):
            self.X[i,:] = np.mean(self.X[i*4:i+4:], axis=0)
        self.X = self.X[0:self.X.shape[0]/4,:]

        if self.subset != 1:
            np.random.shuffle(model['X'])
            self.X = self.X[0:int(self.subset*self.X.shape[0]):, ::]

        self.y = self.X[:,[self.columns]]
        self.X = self.X[:, :self.columns]

    def kmeans(self):
        """
        runs scipy kmeans
        """

        print('Loading test data...')
        X_test = pd.read_csv(self.test_data['path'], delimiter=',', header=0, dtype='float32').values[:,1:self.columns+1]
        predictions = np.empty([X_test.shape[0]], dtype = 'int32')

        print('Building tree...')
        kdtree = spatial.cKDTree(self.X[:,0:2], leafsize=10)

        print('Developing predictions...')
        for i in range(X_test.shape[0]):

            # find points within +- x degrees lat long of test point
            neighbors = kdtree.query_ball_point(X_test[i,0:2], 4)
            neighbor_data = np.empty([len(neighbors), 3])
            neighbor_data[:,0] = neighbors
            
            # Identify stats about these points
            for neighbor in neighbor_data:
                # find climate dist difference and class of neighbor
                neighbor[1] = np.linalg.norm(X_test[i,2:] - self.X[neighbor[0],2:])
                neighbor[2] = self.y[neighbor[0]]

            # sort the data by dist
            neighbor_data = neighbor_data[neighbor_data[:,1].argsort()]

            topx = self.topn
            while True:
                if neighbor_data[0,2] == neighbor_data[1,2] or neighbor_data[0,1] * 2 < neighbor_data[1,1]:
                    predictions[i] = neighbor_data[0,2]
                    print('No vote at:', i, 'with outcome:', predictions[i], 'and dist:', neighbor_data[0,1])
                    break

                classes, counts = stats.mode(neighbor_data[:topx,2])
                if counts[0] > (topx/3)+1:
                    predictions[i] = classes[0]

                    if not counts[0] >= topx:
                        print('Vote resolved at:', i, 'with outcome:', predictions[i], 'and votes:')
                        for x in range(topx):
                            print('Class:', neighbor_data[x,2], 'Dist:', neighbor_data[x,1])
                    break

                else:
                    print('Disputed point with', topx, 'votes:', i)
                    # print('Point:', X_test[i])
                    for x in range(topx):
                        print('Class:', neighbor_data[x,2], 'Dist:', neighbor_data[x,1])
                        # print('Point:', self.X[neighbor_data[x,0]])
                    topx+=2
        print(np.bincount(predictions))

        return predictions

    def save_predictions(self, predictions):
        """
        saves the predictions to file in a format that kaggle likes.
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

        if self.mode == 'kmnn':
            print('Beginning evaluation...')
            self.save_predictions(self.kmeans())

def main():
    p = Predictor()
    p.run()

if __name__ == '__main__':
    main()