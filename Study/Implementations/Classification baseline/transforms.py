import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from opts import parser
from tensorflow.keras.layers.experimental import preprocessing
from opts import parser

class Normalize(object):
    def __init__(self): 
        self.func = preprocessing.Normalization()

    def __call__(self, data):
        pre_data = func(data)

        return pre_data

class Resizing(object):
    def __init__(self, size): 
        self.func = tf.image.resize(size = size)

    def __call__(self, img_group):
        pre_data = func(img_group)
    
        return pre_data

class Rescaling(object):
    def __init__(self): 
        self.func = tf.image.per_image_standardization()

    def __call__(self, img_group):
        pre_data = func(img_group)

        return pre_data

class CenterCrop(object):
    def __init__(self, size):
        height, width = size 
        self.func = preprocessing.CenterCrop(height, width)
        
    def __call__(self, img_group):
        pre_data = func(img_group)
    
        return pre_data