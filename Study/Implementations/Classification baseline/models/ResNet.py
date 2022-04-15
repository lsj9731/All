import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from opts import parser
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras

class Plain_layer(keras.layers.Layer):
    def __init__(self, filters, kernel_s, stride, pad, downsample=None):
        super(Plain_layer, self).__init__()
        
        self.conv1 = keras.layers.Conv2D(filters, kernel_s, strides = stride, padding=pad, kernel_initializer=tf.keras.initializers.HeNormal(), use_bias = False)
        self.bn1 = keras.layers.BatchNormalization()

        self.conv2 = keras.layers.Conv2D(filters, kernel_s, strides = 1, padding=pad, kernel_initializer=tf.keras.initializers.HeNormal(), use_bias = False)
        self.bn2 = keras.layers.BatchNormalization()

        self.relu = keras.layers.ReLU()
        self.down = downsample


    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.down is not None:
            identity = self.down(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class Bottle_layer(keras.layers.Layer):
    def __init__(self, filters, kernel_s, stride, pad, downsample=None):
        super(Bottle_layer, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters, kernel_size = (1, 1), strides = stride, padding='valid', kernel_initializer=tf.keras.initializers.HeNormal(), use_bias = False)
        self.bn1 = keras.layers.BatchNormalization()

        self.conv2 = keras.layers.Conv2D(filters, kernel_size = kernel_s, strides = 1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), use_bias = False)
        self.bn2 = keras.layers.BatchNormalization()

        self.conv3 = keras.layers.Conv2D(filters*4, kernel_size = (1, 1), strides = 1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), use_bias = False)
        self.bn3 = keras.layers.BatchNormalization()

        self.relu = keras.layers.ReLU()
        self.down = downsample
        
    def call(self, x):

        # (16, 55, 55, 64)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.down is not None:
            identity = self.down(identity)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(keras.Model):
    def __init__(self, layers, num_class, model_version, before_softmax=True):
        super(ResNet, self).__init__()
        self.model_version = model_version
        self.num_class = num_class

        # define new layers 
        self.zero_padding = tf.keras.layers.ZeroPadding2D(padding=(3, 3))
        self.conv = keras.layers.Conv2D(64, 7, strides=2, padding='valid', activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), use_bias = False)
        self.pool = keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2))

        if self.model_version == 'ResNet18' or self.model_version == 'ResNet34':
            self.layer1 = self.Construct_plain_layer(layers[0], 64, 3, 1, 'same', False, False, False)
            self.layer2 = self.Construct_plain_layer(layers[1], 128, 3, 1, 'same', True, True, False)
            self.layer3 = self.Construct_plain_layer(layers[2], 256, 3, 1, 'same', True, True, False)
            self.layer4 = self.Construct_plain_layer(layers[3], 512, 3, 1, 'same', True, True, False)
        elif self.model_version == 'ResNet50' or self.model_version == 'ResNet101' or self.model_version == 'ResNet101':
            self.layer1 = self.Construct_plain_layer(layers[0], 64, 3, 1, 'valid', False, False, True)
            self.layer2 = self.Construct_plain_layer(layers[1], 128, 3, 1, 'valid', True, True, True)
            self.layer3 = self.Construct_plain_layer(layers[2], 256, 3, 1, 'valid', True, True, True)
            self.layer4 = self.Construct_plain_layer(layers[3], 512, 3, 1, 'valid', True, True, True)

        self.avgpool = keras.layers.AveragePooling2D((7, 7))

        if before_softmax:
            self.fc = tf.keras.layers.Dense(num_class, activation=None)
        else:
            self.fc = tf.keras.layers.Dense(num_class, activation='softmax')

    def Construct_plain_layer(self, num_layers, num_filter, kernel_s, strides, padding, downsample, get_stride, if_bottle):
        return_layers = []
        for n in range(num_layers):
            if n == 0 and downsample == True and if_bottle == False:
                down_sample = keras.layers.Conv2D(num_filter, kernel_size=1, strides = 2, padding = padding, use_bias = False)
            elif n == 0 and downsample == True:
                down_sample = keras.layers.Conv2D(num_filter * 4, kernel_size=1, strides = 2, padding = padding, use_bias = False)    
            elif if_bottle:
                down_sample = keras.layers.Conv2D(num_filter * 4, kernel_size=1, strides = 1, padding = padding, use_bias = False)
            else:
                down_sample = None

            if n == 0 and get_stride == True:
                strides = 2
            else:
                strides = 1

            if if_bottle:
                return_layers.append(Bottle_layer(num_filter, kernel_s, strides, padding, down_sample))
            else:
                return_layers.append(Plain_layer(num_filter, kernel_s, strides, padding, down_sample))

        return tf.keras.Sequential(return_layers)

    def call(self, x):     
        x = self.zero_padding(x)
        x = self.conv(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = tf.squeeze(x)
        x = self.fc(x)

        return x