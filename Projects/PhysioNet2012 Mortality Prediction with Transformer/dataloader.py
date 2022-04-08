import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import logging
from opts import parser
import math
from tensorflow.keras.utils import Sequence

class Dataloader(Sequence):
    def __init__(self, dataset_dir, batch_size, mode, dataset_name):
        print('initialization of Dataloader {} set.'.format(mode))
        self.batch_size = batch_size
        self.names = dataset_name
        self.data_load(dataset_dir, mode)
        self.mode = mode

        print(("""
        Initializing Dataset.
        Model Configurations:
            Dataset Name:       {}
            Batch size:         {}
            Dataset mode:       {}
        """.format(self.names, self.batch_size, self.mode)))

    def data_load(self, dataset_dir, mode):
        if mode == 'Training':
            self.data = os.listdir(os.path.join(dataset_dir, 'train'))
            self.annotation = pd.read_csv(os.path.join(dataset_dir, 'train_gt.csv'))
            d_mode = '/train'
        elif mode == 'Validation':
            self.data = os.listdir(os.path.join(dataset_dir, 'valid'))
            self.annotation = pd.read_csv(os.path.join(dataset_dir, 'valid_gt.csv'))
            d_mode = '/valid'
        elif mode == 'Testing':
            self.data = os.listdir(os.path.join(dataset_dir, 'test'))
            self.annotation = pd.read_csv(os.path.join(dataset_dir, 'test_gt.csv'))
            d_mode = '/test'

        self.data = sorted(self.data)
        self.x = [pd.read_csv(os.path.join(dataset_dir+d_mode, data)) for data in self.data]
        self.annotation = self.annotation.sort_values('RecordID')
        self.y = list(np.array(self.annotation['In-hospital_death']).reshape(-1))

    def preprocessing(self, inputs, labels):
        static_features = [
        'Age', 'Gender', 'Height', 'ICUType'
        ]
        ts_features = [
            'Weight', 'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
            'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
            'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg',
            'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
            'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
            'Urine', 'WBC', 'pH'
        ]

        # preprocessing to demo, time, value, value onehot, time length feafures
        # demographics
        total_demo = []
        for i, dd in enumerate(inputs):
            statics_indicator = dd['Parameter'].isin(static_features)
            statics = dd[statics_indicator]
            statics = statics.sort_values(by='Parameter' ,ascending=True)

            demo_info = statics.to_numpy()
            demo = []
            demo.append(demo_info[0][2])
            demo.append(demo_info[2][2])

            if demo_info[1][2] == 0.0:  
                demo.extend([1, 0])
            else:
                demo.extend([0, 1])

            icu = np.zeros((4))
            icu[int(demo_info[3][2])-1] = 1
            demo.extend(icu)

            demo = np.array(demo)

            demo_mean = np.mean(demo, axis=0)
            demo_std = np.std(demo, axis=0)
            demo = (demo - demo_mean) / demo_std

            total_demo.append(demo)
        
        total_demo = np.array(total_demo)

        # generate dataframe
        total_values, total_times = [], []
        for dd in inputs:
            statics_indicator = dd['Parameter'].isin(['RecordID'] + static_features)
            data = dd[~statics_indicator]
            duplicated_ts = data[['Time', 'Parameter']].duplicated()

            if duplicated_ts.sum() > 0:
                logging.debug(
                    'Got duplicated time series variables for RecordID')
                data = data.groupby(['Time', 'Parameter'], as_index=False).mean().reset_index()

            time_series = data.pivot(index='Time', columns='Parameter', values='Value')
            time_series = time_series.reindex(columns=ts_features).dropna(how='all').reset_index()

            # time
            time = time_series['Time']
            time = time.to_numpy()
            p_time = [float(t.split(':')[0]) + float(t.split(':')[1]) * 0.01 for t in time]

            total_times.append(p_time)

            # value
            value = time_series[ts_features]
            value = value.fillna(0)
            value = value.to_numpy()

            value_mean = np.mean(value, axis=0)
            value_std = np.std(demo, axis=0)
            value = (value - value_mean) / value_std
            value_mask = tf.sequence_mask(value)[:, :, 0]

            value_modality_embedding = tf.concat((
                    value,
                    tf.cast(value_mask, tf.float32)
                ), axis=-1)

            total_values.append(value_modality_embedding)

        # lengths
        total_length = []
        for dd in total_times:
            total_length.append(np.array([len(dd)]))

        total_length = np.array(total_length)

        return (total_demo, total_times, total_values, total_length), labels

    def _data_sampling(self, data, labels):
        # 5:5 ratio data sampling
        clip_length = self.batch_size // 2
        sample_pos, sample_neg = [], []
        sample_pos_label, sample_neg_label = [], []

        for i in range(2):
            cnt = 0
            if i == 0:
                for p_ids, v in enumerate(labels):
                    if cnt < clip_length:
                        if v == 0:
                            sample_pos_label.append(v)
                            sample_pos.append(data[p_ids])
                            cnt += 1
            elif i == 1:
                for n_ids, v in enumerate(labels):
                    if cnt < clip_length:
                        if v == 1:
                            sample_neg_label.append(v)
                            sample_neg.append(data[n_ids])
                            cnt += 1

        sample_pos.extend(sample_neg)
        sample_pos_label.extend(sample_neg_label)

        return sample_pos, sample_pos_label

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        if self.mode == 'Training':
            s = np.arange(len(self.x))
            np.random.shuffle(s)

            shuffled_x, shuffled_y = [], []

            for i in s:
                shuffled_x.append(self.x[i])
                shuffled_y.append(self.y[i])
            
            batch_x, batch_y = self._data_sampling(shuffled_x, shuffled_y)

            ss = np.arange(self.batch_size)
            np.random.shuffle(ss)

            return_a, return_b = [], []

            for ii in ss:
                return_a.append(batch_x[ii])
                return_b.append(batch_y[ii])

            p_batch_x, p_batch_y = self.preprocessing(return_a, return_b)
        else:
            self.indices = np.arange(len(self.x))
            indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

            batch_x = [self.x[i] for i in indices]
            batch_y = [self.y[i] for i in indices]

            p_batch_x, p_batch_y = self.preprocessing(batch_x, batch_y)

        return p_batch_x, np.array(p_batch_y).reshape(-1, 1)