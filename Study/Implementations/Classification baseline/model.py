import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from opts import parser
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras

class Counstructor(keras.Model):
    def __init__(self, num_class, drop_rate, model_version = None):
        super(Counstructor, self).__init__()
        self.num_class = num_class
        self.drop_rate = drop_rate
        self.model_version = model_version
        self._prepare_base_model(num_class, drop_rate, model_version)

        print(("""
        Initializing Model.
        Model Configurations:
            model version:      {}
            num_class:          {}
            dropout_ratio:      {}
        """.format(self.model_version, self.num_class, self.drop_rate)))

    def _prepare_base_model(self, n_classes, drops, base_model):
        if 'VGG16' in base_model:
            import models.VGG as VGG
            self.base_model = VGG.VGG(layers = [2, 2, 3, 3, 3], num_class=n_classes, drop_rate=drops)
        elif 'VGG19' in base_model:
            import models.VGG as VGG
            self.base_model = VGG.VGG(layers = [2, 2, 4, 4, 4], num_class=n_classes, drop_rate=drops)
        elif 'ResNet18' in base_model:
            import models.ResNet as ResNet
            self.base_model = ResNet.ResNet(layers = [2, 2, 2, 2], num_class=n_classes, model_version = base_model)
        elif 'ResNet34' in base_model:
            import models.ResNet as ResNet
            self.base_model = ResNet.ResNet(layers = [3, 4, 6, 3], num_class=n_classes, model_version = base_model)
        elif 'ResNet50' in base_model:
            import models.ResNet as ResNet
            self.base_model = ResNet.ResNet(layers = [3, 4, 6, 3], num_class=n_classes, model_version = base_model)
        elif 'ResNet101' in base_model:
            import models.ResNet as ResNet
            self.base_model = ResNet.ResNet(layers = [3, 4, 23, 3], num_class=n_classes, model_version = base_model)
        elif 'ResNet152' in base_model:
            import models.ResNet as ResNet
            self.base_model = ResNet.ResNet(layers = [3, 8, 36, 3], num_class=n_classes, model_version = base_model)    


    def call(self, inputs):
        x = self.base_model(inputs)

        return x