import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from opts import parser
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras

class VGG_layer(keras.layers.Layer):
    def __init__(self, filters, kernel_s, pad, drop_out, drop_rate):
        super(VGG_layer, self).__init__()
        
        self.d = drop_out
        
        self.conv = keras.layers.Conv2D(filters, kernel_s, padding=pad, kernel_initializer=tf.keras.initializers.HeNormal())
        self.bn = keras.layers.BatchNormalization()
        self.drop = keras.layers.Dropout(drop_rate)
        self.relu = keras.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        if self.d:
            x = self.drop(x)
        
        return x

class FC_layer(keras.layers.Layer):
    def __init__(self, units, drop_out, drop_rate):
        super(FC_layer, self).__init__()
        
        self.d = drop_out
        
        self.fc = keras.layers.Dense(units)
        self.bn = keras.layers.BatchNormalization()
        self.drop = keras.layers.Dropout(drop_rate)
        self.relu = keras.layers.ReLU()
        
    def call(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        
        if self.d:
            x = self.drop(x)
        
        return x

class VGG(keras.Model):
    def __init__(self, layers, num_class, drop_rate, before_softmax=True):
        super(VGG, self).__init__()
        self.num_class = num_class
        self.drop_rate = drop_rate

        # define new layers 
        self.layer1 = self.Construct_VGG_layer(layers[0], 64, 3, 'same', [False, False], drop_rate)
        self.layer2 = self.Construct_VGG_layer(layers[1], 128, 3, 'same', [False, True], drop_rate)
        self.layer3 = self.Construct_VGG_layer(layers[2], 256, 3, 'same', [True, True, False], drop_rate)
        self.layer4 = self.Construct_VGG_layer(layers[3], 512, 3, 'same', [True, True, False], drop_rate)
        self.layer5 = self.Construct_VGG_layer(layers[4], 512, 3, 'same', [True, True, False], drop_rate)
        
        self.pool = keras.layers.MaxPooling2D((2, 2))
        self.flatten = keras.layers.Flatten()
        
        self.fc1 = FC_layer(512, True, drop_rate)
        self.fc2 = FC_layer(256, False, drop_rate)
        if before_softmax:
            self.fc3 = tf.keras.layers.Dense(num_class, activation=None)
        else:
            self.fc3 = tf.keras.layers.Dense(num_class, activation='softmax')

    def Construct_VGG_layer(self, num_layers, num_filter, kernel_s, padding, activation_f, use_drop, drop_rate):
        return_layers = []

        if num_layers == 2 or num_layers == 3:
            for i in range(num_layers):
                return_layers.append(VGG_layer(num_filter, kernel_s, padding, activation_f, use_drop[i], drop_rate))

        if num_layers == 4:
            use_drop.append(False)
            for i in range(num_layers):
                return_layers.append(VGG_layer(num_filter, kernel_s, padding, activation_f, use_drop[i], drop_rate))

        return tf.keras.Sequential(return_layers)

    def call(self, x):     
        # convolution layers
        x = self.layer1(x)
        x = self.pool(x)
        
        x = self.layer2(x)
        x = self.pool(x)
        
        x = self.layer3(x)
        x = self.pool(x)
        
        x = self.layer4(x)
        x = self.pool(x)
        
        x = self.layer5(x)
        x = self.pool(x)
        
        x = self.flatten(x)
        
        # fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x