import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from opts import parser
import math
from tensorflow.keras.utils import Sequence
import transformer


class Positionalencoding(tf.keras.layers.Layer):
    def __init__(self, max_time = 10000, n_dim=10):
        super(Positionalencoding, self).__init__()
        # define
        self.max_time = max_time
        self.n_dim = n_dim
        self._num_timescales = self.n_dim // 2

    def get_timescales(self):
        # This is a bit hacky, but works
        timescales = self.max_time ** np.linspace(0, 1, self._num_timescales)
        return timescales

    # Keras API에서는 레이어의 build(self, inputs_shape) 메서드에서 레이어 가중치를 만드는 것이 좋다.
    def build(self, input_shape):
        self.timescales = self.add_weight(
            'timescales',
            (self._num_timescales, ),
            trainable=False,
            initializer=tf.keras.initializers.Constant(self.get_timescales())
        )
        
    def call(self, input_time):
        return_time = []
        for t in input_time:
            t = tf.expand_dims(t, axis = -1)
            scaled_time = t / self.timescales[None, None, :]
            signal = tf.concat(
                [
                    tf.sin(scaled_time),
                    tf.cos(scaled_time)
                ],
                axis=-1)
            return_time.append(signal)

        return return_time


class PaddedToSegments(tf.keras.layers.Layer):
    """Convert a padded tensor with mask to a stacked tensor with segments."""
    def call(self, inputs, mask):
        collected_list, valid_list = [], []
        for i in range(len(inputs)):
            valid_observations = tf.where(tf.reshape(mask[i], (1, -1)))
            collected_values = tf.gather_nd(inputs[i], valid_observations)
            collected_list.append(collected_values)
            valid_list.append(valid_observations[:, 0])

        return collected_list, valid_list


class SegmentAggregation(tf.keras.layers.Layer):
    def __init__(self, aggregation_fn='sum'):
        super().__init__()
        self.aggregation_fn = self._get_aggregation_fn(aggregation_fn)

    def _get_aggregation_fn(self, aggregation_fn):
        if aggregation_fn == 'sum':
            return tf.math.segment_sum
        elif aggregation_fn == 'mean':
            return tf.math.segment_mean
        elif aggregation_fn == 'max':
            return tf.math.segment_max
        else:
            raise ValueError('Invalid aggregation function')

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, data, segment_ids):
        assert segment_ids is not None
        return_list = []
        for i in range(len(data)):
            output = self.aggregation_fn(data[i], segment_ids[i])
            return_list.append(output)

        return return_list

class TransformerModel(tf.keras.Model):
    def __init__(self, output_activation, output_dims, n_dims, n_heads, n_layers, dropout, attn_dropout, aggregation_fn, max_timescale):
        super(TransformerModel, self).__init__()

        # first positional encoding, max_timescale = 100.0, n_dim=128
        self.positional = Positionalencoding(max_timescale, n_dim=n_dims)

        # for linear transformation
        self.demo_linear = tf.keras.layers.Dense(n_dims, activation=None)
        self.value_linear = tf.keras.layers.Dense(n_dims, activation=None)

        # adding value, time
        self.adding = tf.keras.layers.Add()

        # Transformer block
        self.transformer_blocks = []
        for _ in range(n_layers):
            transformer_block = transformer.TransformerBlock(n_dims, n_heads, dropout)
            self.transformer_blocks.append(transformer_block)

        self.to_segments = PaddedToSegments()
        self.aggregation = SegmentAggregation(aggregation_fn)
        self.drop1 = tf.keras.layers.Dropout(0.5)
        self.drop2 = tf.keras.layers.Dropout(0.2)
        self.out_mlp1 = tf.keras.layers.Dense(256, activation = 'relu')
        self.out_mlp2 = tf.keras.layers.Dense(1, activation = 'sigmoid')

    def _value_transform(self, inputs):
        total_value = []
        for data in inputs:
            linears = self.value_linear(tf.reshape(data, (1, data.shape[-2], data.shape[-1])))
            total_value.append(linears)

        return total_value

    def _time_adding(self, in_time, in_val):
        total_adding = []
        for i in range(len(in_time)):
            added = self.adding([in_time[i], in_val[i]])
            total_adding.append(added)
        
        return total_adding

    def _comdined(self, in_de, in_val):
        total_com = []
        for i in range(len(in_val)):
            reshaped = tf.expand_dims(in_de[i], 0)
            reshaped = tf.expand_dims(reshaped, 0)

            comdined = tf.concat([reshaped, in_val[i]], axis = 1)
            total_com.append(comdined)

        return total_com

    def _get_mask(self, d_length):
        mask_list = []
        for i in range(len(d_length)):
            mask = tf.sequence_mask(d_length[i]+1, name='mask')
            mask = tf.squeeze(mask)
            mask_list.append(mask)

        return mask_list

    def _out_mlp(self, inputs, mlps):
        output_list = []
        for i in range(len(inputs)):
            # out = mlps(self.drop1(inputs[i]))
            out = mlps(inputs[i])
            output_list.append(out)

        return tf.concat(output_list, axis=0)

    def call(self, inputs):
        # input step
        demo, times, value_modality_embedding, length = inputs

        # positional encoding
        transformed_times = self.positional(times)

        # get mask
        mask = self._get_mask(length)

        # get linear operation
        t_demo = self.demo_linear(tf.reshape(demo, (len(demo), -1)))
        t_value = self._value_transform(value_modality_embedding)

        # element wise adding value, time
        a_value = self._time_adding(transformed_times, t_value)

        # concatenate demographics and values
        comdined = self._comdined(t_demo, a_value)

        # Transformer operation
        transformer_out = comdined
        for block in self.transformer_blocks:
            transformer_out = block(transformer_out, mask)

        collected_values, segment_ids = self.to_segments(transformer_out, mask)

        aggregated_values = self.aggregation(collected_values, segment_ids)

        output1 = self._out_mlp(aggregated_values, self.out_mlp1)
        out_drop = self.drop2(output1)
        output = self.out_mlp2(out_drop)
        
        return output