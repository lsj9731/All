import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from opts import parser
import math
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence


class LayerNormalization(tf.keras.layers.Layer):
    """
    Implementation of Layer Normalization (https://arxiv.org/abs/1607.06450).

    "Unlike batch normalization, layer normalization performs exactly
    the same computation at training and test times."
    """
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        dim = input_shape[-1]
        self.gain = self.add_weight(
            name='gain',
            shape=(dim,),
            initializer='ones',
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(dim,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        variance = K.mean(
            K.square(inputs - mean), axis=self.axis, keepdims=True)
        epsilon = K.constant(1e-5, dtype=K.floatx())
        normalized_inputs = (inputs - mean) / K.sqrt(variance + epsilon)
        result = self.gain * normalized_inputs + self.bias
        return result


class Attenion_operation(tf.keras.layers.Layer):
    def __init__(self, n_dim=128, n_head=4, drop=0.):
        super(Attenion_operation, self).__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = drop
        
        self.attention_dropout_layer = tf.keras.layers.Dropout(self.dropout)

        self.addition_layer = tf.keras.layers.Add()
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

        self.norm1_layer = LayerNormalization()
        self.norm2_layer = LayerNormalization()

        self.transition_layer1 = tf.keras.layers.Dense(384, activation='relu')
        self.transition_layer2 = tf.keras.layers.Dense(128, activation='relu')

    def build(self, input_shape):
        self.qkv_weight = self.add_weight(
            'qkv_weight',
            (self.n_dim, self.n_dim*3),
            trainable=True,
            initializer='glorot_uniform')

        self.output_weights = self.add_weight(
            name='output_weights',
            shape=(self.n_dim, self.n_dim),
            initializer='glorot_uniform',
            trainable=True)

    def mask_attention_if_needed(self, dot_product, mask=None):
        results = []
        for i in range(len(mask)):
            expanded_mask = tf.cast(tf.expand_dims(mask[i], -1), tf.float32)
            input_shape = tf.shape(dot_product[i])
            attention_mask = tf.reshape(tf.matmul(expanded_mask, expanded_mask, transpose_b=True), (1, 1, mask[i].shape[-1], mask[i].shape[-1]))

            shape_with_attn_heads = tf.stack([-1, self.n_head, input_shape[-2], input_shape[-1]])
            input_reshaped = tf.reshape(dot_product[i], shape_with_attn_heads)
            close_to_negative_inf = -1e9
            
            result = (
                attention_mask * input_reshaped
                + (attention_mask - 1) * close_to_negative_inf
            )
            output = tf.nn.softmax(tf.reshape(result, input_shape))
            results.append(output)

        return results

    def _get_permutation(self, inputs, t_shape):
        return_list = []
        return_shape = [] 
        for i in range(len(inputs)):
            _transformation = K.permute_dimensions(inputs[i], t_shape)
            return_list.append(_transformation)
            return_shape.append(tf.shape(_transformation))

        return return_list, return_shape

    def _qk_matmul(self, in_q, in_k, q_shape, k_t_shape, sqrt_dim):
        return_list = []
        for i in range(len(in_q)):
            mm = tf.matmul(tf.reshape(in_q[i], (-1, q_shape[i][-2], q_shape[i][-1])), tf.reshape(in_k[i], (-1, k_t_shape[i][-2], k_t_shape[i][-1])))
            return_list.append(mm / sqrt_dim)
        
        return return_list

    def _list_dropout(self, arrays, drop_layer):
        return_list = [drop_layer(arr) for arr in arrays]

        return return_list

    def _reshaped_matmul(self, in_q, in_v, v_shape, q_shape, model_dim, keys):
        return_list = []
        for i in range(len(in_q)):
            qkv_mm = tf.matmul(in_q[i], tf.reshape(in_v[i], (-1, v_shape[i][-2], v_shape[i][-1])))
            input_reshaped = tf.reshape(qkv_mm, (-1, self.n_head, q_shape[i][-2], v_shape[i][-1]))
            input_reshaped_merged = K.reshape(K.permute_dimensions(input_reshaped, [0, 2, 1, 3]),(-1, model_dim))

            output_shape = tf.stack([-1, tf.shape(keys[i])[1], model_dim])
            attention_out = tf.reshape(
                tf.matmul(input_reshaped_merged, self.output_weights),
                output_shape)

            return_list.append(attention_out)

        return return_list

    def attention(self, query, value, key, out_len, model_dim, mask):
        q, q_shape = self._get_permutation(query, [0, 2, 1, 3])
        v, v_shape = self._get_permutation(value, [0, 2, 1, 3])
        k_transposed, k_t_shape = self._get_permutation(key, [0, 2, 3, 1])

        # MatMul & Scaling
        sqrt_d = K.constant(np.sqrt(model_dim // self.n_head), dtype=K.floatx())
        scaled_qk = self._qk_matmul(q, k_transposed, q_shape, k_t_shape, sqrt_d)

        pre_softmax = self.mask_attention_if_needed(scaled_qk, mask=mask)
        drop_qk = self._list_dropout(pre_softmax, self.attention_dropout_layer)

        qkv_mm = self._reshaped_matmul(drop_qk, v, v_shape, q_shape, model_dim, key)
        
        return qkv_mm

    def _get_shape(self, inputs):
        totals = [tf.shape(data) for data in inputs]
        return totals

    def _qkv_matmul(self, inputs, dim, r_shape):
        # tf.matmul(tf.reshape(inputs, (-1, model_dim)), self.qkv_weight)
        # 
        totals, shapes = [], []
        for i in range(len(inputs)):
            cast_input = tf.cast(inputs[i], dtype=tf.float32)
            _temp = tf.matmul(tf.reshape(cast_input, (-1, dim)), self.qkv_weight)
            temp_s = tf.stack([-1, r_shape[i][1], self.n_head, dim // self.n_head])

            totals.append(_temp)
            shapes.append(temp_s)
        return totals, shapes

    def _split(self, inputs, shapes):
        q_list, k_list, v_list = [], [], []
        for i in range(len(inputs)):
            q, k, v = tf.split(inputs[i], 3, axis=-1)
            q_list.append(tf.reshape(q, (1, -1, 4, 32)))
            k_list.append(tf.reshape(k, (1, -1, 4, 32)))
            v_list.append(tf.reshape(v, (1, -1, 4, 32)))

        return q_list, k_list, v_list

    def _addition_loop(self, att_out, inputs):
        return_list = []
        for i in range(len(inputs)):
            added = self.addition_layer([att_out[i], inputs[i]])
            return_list.append(added)

        return return_list

    def _layer_norm(self, inputs, norms):
        return_list = []
        for i in range(len(inputs)):
            after_norm = norms(inputs[i])
            return_list.append(after_norm)

        return return_list

    def _feedforward(self, inputs, net):
        return_list = []
        for i in range(len(inputs)):
            outputs = net(inputs[i])
            return_list.append(outputs)

        return return_list

    def call(self, inputs, mask):
        # save to inputs shape

        input_shape = self._get_shape(inputs)
        model_dim = input_shape[0][-1]

        # The first thing we need to do is to perform affine transformations -> 가장 먼저 아핀 변환을 수행
        # of the inputs to get the Queries, the Keys and the Values.
        qkv, qkv_shape = self._qkv_matmul(inputs, model_dim, input_shape)

        # split query, keys and values
        pre_q, pre_k, pre_v = self._split(qkv, qkv_shape)

        # operations attention
        attention_out = self.attention(pre_q, pre_v, pre_k, None, model_dim, mask=mask)

        # residual learning & layer normalization
        post_residual1 = self._addition_loop(attention_out, inputs)

        if 0. < self.dropout < 1.0:
            post_residual1 = self._list_dropout(post_residual1, self.dropout_layer)

        norm1_output = self._layer_norm(post_residual1, self.norm1_layer)
        
        # Feed Forward layers
        output = self._feedforward(norm1_output, self.transition_layer1)
        output = self._feedforward(output, self.transition_layer2)

        # residual learning & layer normalization
        post_residual2 = self._addition_loop(output, norm1_output)

        if 0. < self.dropout < 1.0:
            post_residual2 = self._list_dropout(post_residual2, self.dropout_layer)

        norm2_output = self._layer_norm(post_residual2, self.norm2_layer)

        return norm2_output

        
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, n_dims, n_head, dropout):
        super(TransformerBlock, self).__init__()
        self._attention = Attenion_operation(n_dim=n_dims, n_head=n_head, drop=dropout)


    def call(self, inputs, mask):
        op_attention = self._attention(inputs, mask)

        return op_attention