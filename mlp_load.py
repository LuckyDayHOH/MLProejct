import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models
from keras import layers
from keras import Input, Model
from keras import optimizers
from keras import regularizers
from keras import Sequential
from keras.utils.np_utils import to_categorical
from setting import Setting

setting = Setting()

class MLP_load():

    def create_model(self, onehot_length):
        inputs = Input(shape=(len(setting.input_index),))
        x = layers.Dense(setting.nnode, activation=setting.actv_name)(inputs)

        if setting.nlayer != 1:
            for i in range(0, setting.nlayer - 1, 1):
                x = layers.Dense(setting.nnode, activation=setting.actv_name)(x)
        else:
            pass

        output1 = layers.Dense(onehot_length, activation='softmax', name='avg_road')(x)
        output2 = layers.Dense(onehot_length, activation='softmax', name='trinspct_road')(x)
        output3 = layers.Dense(onehot_length, activation='softmax', name='surfc_gripfc')(x)

        model = Model(inputs=inputs, outputs=[output1, output2, output3])

        model.compile(optimizer='adam',
                      loss={
                          'avg_road': setting.loss,
                          'trinspct_road': setting.loss,
                          'surfc_gripfc': setting.loss
                      },
                      metrics=['accuracy']
        )

        return model

    def to_onehot(self, train, val, test):

        train_label = to_categorical(train)
        val_label = to_categorical(val)
        test_label = to_categorical(test)

        return train_label, val_label, test_label
