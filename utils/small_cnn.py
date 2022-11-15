# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements Small CNN model in keras using tensorflow backend."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import keras
import keras.backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop 

import numpy as np
import tensorflow as tf


class SmallCNN(object):
    """Small convnet that matches sklearn api.

    Implements model from
    https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
    Adapts for inputs of variable size, expects data to be 4d tensor, with
    # of obserations as first dimension and other dimensions to correspond to
    length width and # of channels in image.
    """

    def __init__(self,
                 random_state=1,
                 epochs=50,
                 batch_size=32,
                 solver='rmsprop',
                 learning_rate=0.001,
                 lr_decay=0.):
        # params
        self.solver = solver
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        # data
        self.encode_map = None
        self.decode_map = None
        self.model = None
        self.random_state = random_state
        self.n_classes = None

    def build_model(self, X):
        # assumes that data axis order is same as the backend
        input_shape = X.shape[1:]
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        # model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(4, activation='softmax'))



        # model.add(Conv1D(32, (1), padding='same', input_shape=input_shape, name='conv1'))
        # model.add(Activation('relu'))
        # model.add(Conv1D(32, (1), name='conv2'))
        # model.add(Activation('relu'))
        # model.add(MaxPooling1D(pool_size=(2)))
        # model.add(Dropout(0.25))

        # model.add(Conv1D(64, (1), padding='same', name='conv3'))
        # model.add(Activation('relu'))
        # model.add(Conv1D(64, (1), name='conv4'))
        # model.add(Activation('relu'))
        # model.add(MaxPooling1D(pool_size=(2)))
        # model.add(Dropout(0.25))

        # model.add(Flatten())
        # model.add(Dense(512, name='dense1'))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(self.n_classes, name='dense2'))
        # model.add(Activation('softmax'))

        # try:
        #   optimizer = getattr(keras.optimizers, self.solver)
        # except:
        #   raise NotImplementedError('optimizer not implemented in keras')
        # All optimizers with the exception of nadam take decay as named arg
        # try:
        #   opt = optimizer(lr=self.learning_rate, decay=self.lr_decay)
        # except:
        #   opt = optimizer(lr=self.learning_rate, schedule_decay=self.lr_decay)

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        # Save initial weights so that model can be retrained with same
        # initialization
        model.build()
        self.initial_weights = copy.deepcopy(model.get_weights())

        self.model = model

    def create_y_mat(self, y):
        y_encode = self.encode_y(y)
        y_encode = np.reshape(y_encode, (y_encode.size, 1))
        y_mat = keras.utils.to_categorical(y_encode, self.n_classes)
        return np.array(y_mat)

    # Add handling for classes that do not start counting from 0
    def encode_y(self, y):
        if self.encode_map is None:
            self.classes_ = sorted(list(set(y)))
            self.n_classes = len(self.classes_)
            self.encode_map = dict(
                zip(self.classes_, range(len(self.classes_))))
            self.decode_map = dict(
                zip(range(len(self.classes_)), self.classes_))
        transformed_y = np.array([self.encode_map[i] for i in y])
        return transformed_y

    def decode_y(self, y):
        def mapper(x): return self.decode_map[x]
        transformed_y = np.array(map(mapper, y))
        return transformed_y

    def fit(self, X_train, y_train, sample_weight=None):
        y_mat = tf.stack(self.create_y_mat(y_train))
        X_train = tf.stack(X_train)
        if self.model is None:
            self.build_model(X_train)

        # We don't want incremental fit so reset learning rate and weights
        K.set_value(self.model.optimizer.lr, self.learning_rate)
        self.model.set_weights(self.initial_weights)
        self.model.fit(
            X_train,
            y_mat,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            sample_weight=sample_weight,
            verbose=0)

    def predict(self, X_val):
        predicted = self.model.predict(X_val)
        return predicted

    def score(self, X_val, val_y):
        y_mat = self.create_y_mat(val_y)
        val_acc = self.model.evaluate(X_val, y_mat)[1]
        return val_acc

    def decision_function(self, X):
        return self.predict(X)

    def transform(self, X):
        model = self.model
        inp = [model.input]
        activations = []

        # Get activations of the first dense layer.
        output = [layer.output for layer in model.layers if
                  layer.name == 'dense1'][0]
        func = K.function(inp + [K.learning_phase()], [output])
        for i in range(int(X.shape[0]/self.batch_size) + 1):
            minibatch = X[i *
                          self.batch_size: min(X.shape[0], (i+1) * self.batch_size)]
            list_inputs = [minibatch, 0.]
            # Learning phase. 0 = Test mode (no dropout or batch normalization)
            layer_output = func(list_inputs)[0]
            activations.append(layer_output)
        output = np.vstack(tuple(activations))
        return output

    def get_params(self, deep=False):
        params = {}
        params['solver'] = self.solver
        params['epochs'] = self.epochs
        params['batch_size'] = self.batch_size
        params['learning_rate'] = self.learning_rate
        params['weight_decay'] = self.lr_decay
        if deep:
            return copy.deepcopy(params)
        return copy.copy(params)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
