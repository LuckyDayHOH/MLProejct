import sys
import os
import numpy as np
import pandas as pd
from tensorflow import keras
from setting import Setting

setting = Setting()

class MLP_load():

    def create_model(self, onehot_length):
        inputs = keras.Input(shape=(len(setting.input_index),))
        x = keras.layers.Dense(setting.nnode, activation=setting.actv_name)(inputs)

        if setting.nlayer != 1:
            for i in range(0, setting.nlayer - 1, 1):
                x = keras.layers.Dense(setting.nnode, activation=setting.actv_name)(x)
        else:
            pass

        output1 = keras.layers.Dense(onehot_length, activation='softmax', name='avg_road')(x)
        output2 = keras.layers.Dense(onehot_length, activation='softmax', name='trinspct_road')(x)
        output3 = keras.layers.Dense(onehot_length, activation='softmax', name='surfc_gripfc')(x)

        model = keras.Model(inputs=inputs, outputs=[output1, output2, output3])

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

        train_label = keras.utils.to_categorical(train)
        val_label = keras.utils.to_categorical(val)
        test_label = keras.utils.to_categorical(test)

        return train_label, val_label, test_label
