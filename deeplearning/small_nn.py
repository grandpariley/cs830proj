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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import keras
import keras.backend as K
from keras.optimizers import RMSprop

import numpy as np
import tensorflow as tf


class SmallNN(object):
    def __init__(self,
                 random_state=1,
                 epochs=20,
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
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Embedding(5000, 4, input_length=5000),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(20, return_sequences=True)
                ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(20)
                ),
                tf.keras.layers.Dense(4, activation='softmax')
            ]
        )

        model.compile(
            loss='categorical_crossentropy',
            optimizer=RMSprop(),
            metrics=['accuracy']
        )

        model.build()
        self.initial_weights = copy.deepcopy(model.get_weights())

        self.model = model

    def create_y_mat(self, y):
        y_encode = self.encode_y(y)
        y_encode = np.reshape(y_encode, (y_encode.size, 1))
        y_mat = keras.utils.to_categorical(y_encode, self.n_classes)
        return np.array(y_mat)

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

        K.set_value(self.model.optimizer.lr, self.learning_rate)
        self.model.set_weights(self.initial_weights)
        self.model.fit(
            X_train,
            y_mat,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            sample_weight=sample_weight,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=2)]
        )

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

        output = [layer.output for layer in model.layers if
                  layer.name == 'dense1'][0]
        func = K.function(inp + [K.learning_phase()], [output])
        for i in range(int(X.shape[0]/self.batch_size) + 1):
            minibatch = X[i *
                          self.batch_size: min(X.shape[0], (i+1) * self.batch_size)]
            list_inputs = [minibatch, 0.]
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
