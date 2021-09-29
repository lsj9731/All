import numpy as np
import csv
import pandas as pd
import os
import tensorflow as tf
import sys
import scipy
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale
from tensorflow import keras
import random
import json
import itertools

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# function
def Split_Data(numpy_data):
    # 생체 데이터가 들어와서 성별이랑 구분됨
    Total_Data_vital = numpy_data[:-1]
    Total_Data_Gender = []
    Total_Data_Gender.append(numpy_data[-1])
    

    Total_Data_vital = np.array(Total_Data_vital)
    Total_Data_Gender = np.array(Total_Data_Gender)

    return Total_Data_vital, Total_Data_Gender

def Calculation_mean_std(vital_data):
    # 실 데이터의 평균과 편차를 구함
    Data_mean = np.mean(vital_data)
    Data_std = np.std(vital_data)

    return Data_mean, Data_std

def Transform_Vital_Gender(Vital_Data, Gender_Data, mean_, std_, real_mean, real_std):
    # 0인 데이터를 제외하고 zscore 진행
    # 훈련에 사용한 데이터의 각각의 평균과 편차를 각 데이터 컬럼마다 적용

    normalization_vital = []
    i = 0
    for value in Vital_Data:
        _temp_ = []
        if value == 0 or value == 0.0:
            normalization_vital.append(0)
            i += 1
        # 변경한 부분
        else :
            _temp_ = (value - real_mean[i]) / real_std[i]
            normalization_vital.append(_temp_)
            i += 1


    # 성별 데이터 onehot
    Gender_Data_numpy = np.array(Gender_Data)
    onehot_class = []
    if Gender_Data_numpy == [0]:
        onehot_class = [1, 0]
    else :
        onehot_class = [0, 1]

    
    vital_add_gender = []

    # 두 데이터를 묶음
    concat = []
    value1 = normalization_vital
    value2 = onehot_class
    concat = np.concatenate((value1, value2), axis = 0)
    vital_add_gender.append(concat)
        
    vital_add_gender = np.array(vital_add_gender)

    return vital_add_gender


def Impute_Data(vital_d, sample_d):
    # 0인 부분만 생성된 데이터로 변경
    vital_d = list(itertools.chain.from_iterable(vital_d))
    sample_d = list(itertools.chain.from_iterable(sample_d))
    i = 0

    for value in vital_d:
        if value == 0:
            vital_d[i] = sample_d[i]
        i += 1
    
    return vital_d


def Rescaling(data_array, genders , _mean, _std, real_mean, real_std):
    data_array = np.array(data_array)
    data_array = data_array.reshape(-1, 11)
    # 원핫 적용된 데이터를 전체의 평균 편차로 다시 되돌려야함
    Total_Data_add_Gender_Test_ = np.delete(data_array, (9, 10), axis = 1)
    Total_Data_add_Gender_Test_ = list(itertools.chain.from_iterable(Total_Data_add_Gender_Test_))

    # onehot 성별 삭제 후 원래의 성별 결합
    value_1 = Total_Data_add_Gender_Test_
    value_2 = genders[0]
    _temp_ = []

    for value in value_1:
        _temp_.append(value)
    _temp_.append(value_2)

    Total_Data_add_Gender = np.array(_temp_)



    # 리스케일링 하기위해 성별 제외 데이터 저장
    new_data = []
    _temp_ = []
    data_value = Total_Data_add_Gender[:-1]

    # 전체 데이터의 평균과 편차를 이용
    i = 0
    for value in data_value:
        _temp = value * real_std[i] + real_mean[i]
        new_data.append(_temp)
        i += 1

    # 리스케일링 된 데이터와 성별을 결합
    new_data.append(value_2)

    # np 데이터로 변환 / 절대값 / 2자리까지 표시
    rescaling_data = new_data
    rescaling_data = np.array(rescaling_data)
    rescaling_data = abs(rescaling_data)
    rescaling_data = np.around(rescaling_data, 2)
    
    # 예측에 넣기위해 10열로 맞춰줌
    Total_Data_add_Gender = np.array(Total_Data_add_Gender)
    Total_Data_add_Gender = Total_Data_add_Gender.reshape(-1, 10)

    return Total_Data_add_Gender, rescaling_data


def Classification_Sepsis_or_Normal(data):
    data = list(itertools.chain.from_iterable(data))
    print('데이터 : ', data)
    m_data = []
    m_data.append(np.around(data[1], 2))

    return m_data




# json 데이터 load 및 preprocessing 
def get_dict(json_Data_):
    # dict_Data = json.loads(json_Data_)
    # np_Data = np.array(list(dict_Data.items()))

    # test
    np_Data = json_Data_
    Total_mean = np.load('./Total_Data_mean.npy')
    Total_std = np.load('./Total_Data_std.npy')
    
    vital_Data, Gender_Data = Split_Data(np_Data)
    print('생체 데이터 : ', vital_Data, '\n성별 데이터 : ', Gender_Data)
    vital_Data_mean, vital_Data_std = Calculation_mean_std(vital_Data)
    print('생체 데이터 평균 : ', vital_Data_mean, '\n생체 데이터 편차 : ', vital_Data_std)
    vital_add_gender_data = Transform_Vital_Gender(vital_Data, Gender_Data, vital_Data_mean, vital_Data_std, Total_mean, Total_std)
    print('생체 데이터 + 성별 데이터 : ', vital_add_gender_data)

    # gan = load_model('C:/jupyter/Human_Deep/Human_Deep/gan_saved_model/')
    # discriminator = load_model('C:/jupyter/Human_Deep/Human_Deep/g_saved_model/')
    generator = load_model('g_saved_model/')
    generator.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    G_model = generator
 
    Samples_Data = G_model(vital_add_gender_data, training = False)
    print('생성된 샘플 데이터 : ', Samples_Data)

    imputed_data = Impute_Data(vital_add_gender_data, Samples_Data)
    print('결측치를 채운 데이터 : ', imputed_data)

    vital_add_gender_data_v, rescaling_data = Rescaling(imputed_data, Gender_Data , vital_Data_mean, vital_Data_std, Total_mean, Total_std)
    print('예측에 넣을 데이터 : ', vital_add_gender_data_v, '\n리스케일링 된 출력에 쓸 데이터 : ', rescaling_data)
    
    C_model = load_model('Classification_model/')
    sepsis_predict = C_model.predict(vital_add_gender_data_v)
    sepsis_percent = Classification_Sepsis_or_Normal(sepsis_predict)

    return rescaling_data, sepsis_percent


if __name__ == '__main__':
    test_data = np.load('./no_test2.npy')
    # 모델 정확도 ( 성능 부분 ) / percent
    model_accuracy = np.load('./Classification_model_accuracy.npy')
    model_accuracy = np.around(model_accuracy * 100, 1)
    data1, percent = get_dict(test_data)
    print('생체정보 + 혈액정보 : ', data1, '패혈증 확률 : ', percent)
    print('     원본 데이터    : ', test_data)
    print('     모델 정확도    : ', model_accuracy, '%')