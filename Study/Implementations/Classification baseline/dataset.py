import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from opts import parser
import math
from tensorflow.keras.utils import Sequence
from transforms import *

IMG_SIZE = 224

class Dataloader(Sequence):
    def __init__(self, dataset_dir, batch_size, preprocessing, mode):
        print('initialization of Dataloader {} set.'.format(mode))
        self.batch_size = batch_size
        self.transform = preprocessing
        self.Loader(dataset_dir, mode)
        self.shuffle = False
        self.on_epoch_end()
        self.mode = mode
        print('Done.\n')

    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
        
    def Loader(self, dataset_dir, mode):
        if mode == 'train':
            self.x, self.y = np.load(os.path.join(dataset_dir, 'X_train.npy')), np.load(os.path.join(dataset_dir, 'y_train.npy'))
        elif mode == 'valid':
            self.x, self.y = np.load(os.path.join(dataset_dir, 'X_valid.npy')), np.load(os.path.join(dataset_dir, 'y_valid.npy'))
        elif mode == 'test':
            self.x, self.y = np.load(os.path.join(dataset_dir, 'X_test.npy')), np.load(os.path.join(dataset_dir, 'y_test.npy'))

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_x = [self.x[i] for i in indices]
        batch_y = [self.y[i] for i in indices]

        # preprocessing & augmentations
        if self.mode == 'train':
            batch_x, batch_y = self._augmemntation(batch_x, batch_y)
        else:
            batch_x, batch_y = self.resize_and_rescale(batch_x, batch_y)
        
        return np.array(batch_x), np.array(batch_y)

    def _augmemntation(self, images, labels):
        # Define List
        image, label = self.resize_and_rescale(images, labels)
        
        new_image, new_label = image, label

        # Random brightness
        brightness_image = tf.image.random_brightness(image, max_delta=0.5) 
        new_image, new_label = tf.concat([new_image, brightness_image], axis=0), tf.concat([new_label, label], axis=0)

        # Random rotate
        rotated_image = tf.image.rot90(image)
        new_image, new_label = tf.concat([new_image, rotated_image], axis=0), tf.concat([new_label, label], axis=0)

        # Random Flip left rights
        fliped_image = tf.image.random_flip_left_right(image)
        new_image, new_label = tf.concat([new_image, fliped_image], axis=0), tf.concat([new_label, label], axis=0)

        return new_image, new_label

    def resize_and_rescale(self, image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = (image / 255.0)
        return image, label

