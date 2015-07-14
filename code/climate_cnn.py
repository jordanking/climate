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

        self.train_file = 'data/train.csv'
        self.test_file = 'data/test.csv'

        self.nb_epoch = 16
        self.batch_size = 128
        self.nb_classes = 10
        self.subset = 1

        self.folds = 5

        self.load_weights_file = 'tmp/checkpoint_weights.hdf5'
        self.save_weights_file = 'tmp/checkpoint_weights.hdf5'
        self.resume_from_epoch = 35

        self.out_file = 'solutions/answers_el_1_16.csv'

    def load_data(self):
        """ 
        returns X and y - data and target - as numpy arrays, X normalized
        and y made categorical.
        """

        if (os.path.exists('data/train.pickle')):
            with open('data/train.pickle', 'rb') as f:
                data = pickle.load(f)
        else:
            data = np.genfromtxt(self.train_file, delimiter=',', skip_header=1, dtype='float32')
            with open('data/train.pickle', 'wb') as f:
                pickle.dump(data, f)

        if self.subset != 1:
            np.random.shuffle(data)
            data = data[0:int(self.subset*data.shape[0]):, ::]

        X = data[0::,1::]
        X = X.reshape(X.shape[0], 1, 28, 28)
        X /= 255

        y = data[0::,0]
        y = np_utils.to_categorical(y, self.nb_classes)

        self.datagen = ImageDataGenerator(featurewise_center=False,
                                            samplewise_center=False,
                                            featurewise_std_normalization=False,
                                            samplewise_std_normalization=False,
                                            zca_whitening=False,
                                            rotation_range=0,
                                            width_shift_range=0.,
                                            height_shift_range=0.,
                                            horizontal_flip=False,
                                            vertical_flip=False)
        self.datagen.fit(X)

        self.X = X
        self.y = y

    def image_warp(self, image, displacement_field_x, displacement_field_y):
        """
        expensive warping of an image given displacement field.
        """
        result = np.zeros(image.shape)

        for row in xrange(image.shape[1]):
            for col in xrange(image.shape[0]):
                low_ii = row + int(math.floor(displacement_field_x[row, col]))
                high_ii = row + int(math.ceil(displacement_field_x[row, col]))

                low_jj = col + int(math.floor(displacement_field_y[row, col]))
                high_jj = col + int(math.ceil(displacement_field_y[row, col]))

                if low_ii < 0 or low_jj < 0 or high_ii >= image.shape[1] -1 \
                   or high_jj >= image.shape[0] - 1:
                    continue

                res = image[low_ii, low_jj]/4 + image[low_ii, high_jj]/4 + \
                        image[high_ii, low_jj]/4 + image[high_ii, high_jj]/4

                result[row, col] = res
        return result

    #### adapted from https://github.com/FlorianMuellerklein/lasagne_mnist/blob/master/helpers.py

    def batch_warp(self, X_batch, y_batch):
        """
        Data augmentation for feeding images into CNN.
        This example will randomly rotate all images in a given batch between -10 and 10 degrees
        and to random translations between -5 and 5 pixels in all directions.
        Random zooms between 1 and 1.3.
        Random shearing between -20 and 20 degrees.
        Randomly applies sobel edge detector to 1/4th of the images in each batch.
        Randomly inverts 1/2 of the images in each batch.
        """
        PIXELS = 28

        # set empty copy to hold augmented images so that we don't overwrite
        X_batch_aug = np.empty(shape = (X_batch.shape[0], 1, PIXELS, PIXELS), dtype = 'float32')

        # random rotations betweein -8 and 8 degrees
        dorotate = randint(-5,5)

        # random translations
        trans_1 = randint(-3,3)
        trans_2 = randint(-3,3)

        # random zooms
        zoom = uniform(0.8, 1.2)

        # shearing
        shear_deg = uniform(-10, 10)

        # set the transform parameters for skimage.transform.warp
        # have to shift to center and then shift back after transformation otherwise
        # rotations will make image go out of frame
        center_shift   = np.array((PIXELS, PIXELS)) / 2. - 0.5
        tform_center   = transform.SimilarityTransform(translation=-center_shift)
        tform_uncenter = transform.SimilarityTransform(translation=center_shift)

        tform_aug = transform.AffineTransform(rotation = np.deg2rad(dorotate),
                                              scale =(1/zoom, 1/zoom),
                                              shear = np.deg2rad(shear_deg),
                                              translation = (trans_1, trans_2))

        tform = tform_center + tform_aug + tform_uncenter

        # random distortions
        sigma = 4
        alpha = 34

        d_x = np.random.uniform(low = -1, high = 1, size = (28,28))
        d_y = np.random.uniform(low = -1, high = 1, size = (28,28))

        d_x = gaussian_filter(d_x, sigma) * alpha
        d_y = gaussian_filter(d_y, sigma) * alpha

        # elastic_distortion_field = np.empty((2, 28, 28))
        # for r in range(28):
        #     for c in range(28):
        #         elastic_distortion_field[0,r,c] = d_x[r,c]
        #         elastic_distortion_field[1,r,c] = d_y[r,c]

        # images in the batch do the augmentation
        # interactive(True)

        for j in range(X_batch.shape[0]):

            # if j == 0:
            #     plt.imshow(X_batch[j][0], cmap='Greys')
            #     raw_input('press return to continue or q to end...')

            # X_batch_aug[j][0] = transform._warps_cy._warp_fast(X_batch[j][0], elastic_distortion_field, (PIXELS, PIXELS))
            X_batch_aug[j][0] = self.image_warp(X_batch[j][0], d_x, d_y)

            X_batch_aug[j][0] = transform._warps_cy._warp_fast(X_batch[j][0], tform.params, (PIXELS, PIXELS))
            
            # if j == 0:
            #     plt.imshow(X_batch_aug[j][0], cmap='Greys')
            #     raw_input('press return to continue or q to end...')

        # use sobel edge detector filter on one quarter of the images
        indices_sobel = np.random.choice(X_batch_aug.shape[0], X_batch_aug.shape[0] / 4, replace = False)
        for k in indices_sobel:
            img = X_batch_aug[k][0]
            X_batch_aug[k][0] = filters.sobel(img)

        return [X_batch_aug, y_batch]

    #### end adaption from https://github.com/FlorianMuellerklein/lasagne_mnist/blob/master/helpers.py


    def build_keras(self):
        """
        constructs the neural network model
        """

        model = Sequential()

        model.add(Convolution2D(64, 1, 3, 3, border_mode='full')) 
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 64, 3, 3))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(poolsize=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(128, 64, 3, 3, border_mode='full')) 
        model.add(Activation('relu'))

        model.add(Convolution2D(128, 128, 3, 3))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(poolsize=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(16*196*2, 512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(512, 512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(512, self.nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adadelta')

        self.model = model

    def fit_model(self, X, y):
        """
        fits a model to some data
        """

        for e in range(self.nb_epoch):
            print('Epoch: ', e, ' of ', self.nb_epoch)
            progbar = Progbar(target=X.shape[0], verbose=True)

            # batch train with realtime data augmentation
            total_accuracy = 0
            total_loss = 0
            current = 0
            for X_batch, y_batch in self.datagen.flow(X, y, self.batch_size):

                # prepare the batch with random augmentations
                X_batch, y_batch = self.batch_warp(X_batch, y_batch)

                # train on the batch
                loss, accuracy = self.model.train(X_batch, y_batch, accuracy = True)
                
                # update the progress bar
                total_loss += loss * self.batch_size
                total_accuracy += accuracy * self.batch_size
                current += self.batch_size
                if current > self.X.shape[0]:
                    current = self.X.shape[0]
                else:
                    progbar.update(current, [('loss', loss), ('acc.', accuracy)])
            progbar.update(current, [('loss', total_loss/current), ('acc.', total_accuracy/current)])
            
            # checkpoints between epochs
            self.model.save_weights(self.save_weights_file, overwrite = True)

    def cross_validate(self):
        """
        provides a simple cross validation measurement. It doen't make a new
        model for each fold though, so it isn't actually cross validation... the
        model just gets better with time for now. This is pretty expensive to run.
        """

        kf = KFold(self.X.shape[0], self.folds)
        scores = []

        for train, test in kf:
            X_train, X_test, y_train, y_test = self.X[train], self.X[test], self.y[train], self.y[test]

            self.model = self.fit_model(X_train, y_train)
            
            loss, score = self.model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
            print ('Loss: ' + str(loss))
            print ('Score: ' + str(score))
            scores.append(score)
        
        scores = np.array(scores)
        print("Accuracy: " + str(scores.mean()) + " (+/- " + str(scores.std()/2) + ")")

    def get_predictions(self):
        """
        trains and predicts on the mnist data
        """

        test_data = np.genfromtxt(self.test_file, delimiter=',', skip_header=1, dtype='float32')
        test_data = test_data.reshape(test_data.shape[0], 1, 28, 28)
        test_data /= 255

        return self.model.predict_classes(test_data, batch_size = self.batch_size)

    def save_predictions(self, predictions):
        """
        saves the predictions to file in a format that kaggle likes.
        :param predictions: A single dimensional list of classifications
        :p
        """

        predictions_file = open(self.out_file, "wb")
        open_file_object = csv.writer(predictions_file, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        open_file_object.writerow(['ImageId','Label'])
        for i in range(0,predictions.shape[0]):
            open_file_object.writerow([i+1, predictions[i]])
        predictions_file.close()

    def resume(self):
        """
        resumes training if training is interrupted.
        """
        self.nb_epoch = self.nb_epoch - self.resume_from_epoch

        self.model.load_weights(self.load_weights_file)

        self.fit_model(self.X, self.y)

    def run(self):
        """
        set up the test here!
        """

        if self.mode != "load":
            print('loading data...')
            self.load_data()
        
        print('building model...')
        self.build_keras()

        if self.mode == 'xval':
            print('evaluating model...')
            self.cross_validate()

        if self.mode == 'pred':
            print('training model...')
            self.model = self.fit_model(self.X, self.y)

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
