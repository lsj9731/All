from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import json
import pandas as pd
import os
import requests
import seaborn as sns

def get_dir_segment_result():
    s_data_dir = 'C:/Users/82107/Desktop/Python/Data/Skin_cancer_data/Task_1\ISIC2018_Task1-2_Validation_Input'
    s_annotation_dir = 'C:/Users/82107/Desktop/Python/Data/Skin_cancer_data/Task_1/ISIC2018_Task1_Validation_GroundTruth'

    data_list = os.listdir(s_data_dir)
    annotation_list = os.listdir(s_annotation_dir)

    image_data, image_gt = [], []
    for annotation in annotation_list:
        s_a = annotation.split('_')[:-1]
        anno = s_a[0]+'_'+s_a[1]
        get_image = np.array(Image.open(os.path.join(s_data_dir, anno+'.jpg')).resize((224,224))).reshape(-1, 224, 224, 3)
        get_annotation = np.array(Image.open(os.path.join(s_annotation_dir, annotation)).resize((224,224))).reshape(-1, 224, 224, 1)
        image_data.append(get_image)
        image_gt.append(get_annotation)

    image_data = np.concatenate(image_data)
    image_gt = np.concatenate(image_gt)

    # scale the values to 0.0 to 1.0
    data = image_data / 255.0

    # npy data -> tolist()
    data = json.dumps({"signature_name": "serving_default", "instances": data.tolist()})

    # send data using POST request and receive prediction result
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://169.56.89.210:8503/v1/models/sc_model/versions/4:predict', data=data, headers=headers)
    print('json_response : ', json_response)
    predictions = json.loads(json_response.text)['predictions']
    predictions = np.array(predictions)

    # get segmented pixel
    predictions = np.argmax(predictions, axis = -1)
    predictions = predictions.reshape(-1, 224, 224, 1)

    print('predictions shape : ', predictions.shape, 'image_gt shape : ', image_gt.shape)

    # sample for mask iou
    iou_list = []
    for i in range(len(predictions)):
        intersection = np.logical_and(image_gt[i], predictions[i])
        union = np.logical_or(image_gt[i], predictions[i])
        iou_score = np.sum(intersection) / np.sum(union)
        iou_list.append(iou_score)

    threshold = 0.1
    sorted_list = [[] for _ in range(9)]
    for i in range(len(sorted_list)):
        temp = []
        for piou in iou_list:
            if piou >= threshold:
                temp.append(piou)
        sorted_list[i].append(len(temp) * 0.01)
        threshold += 0.1

    return print('sorted_list : ', sorted_list)
    
def get_image_segment_result():
    s_data_dir = 'C:/Users/82107/Desktop/Python/Data/Skin_cancer_data/Task_1/ISIC2018_Task1-2_Validation_Input/ISIC_0017460.jpg'
    data = np.array(Image.open(s_data_dir).resize((224,224))).reshape(-1, 224, 224, 3)

    # scale the values to 0.0 to 1.0
    data = data / 255.0

    # npy data -> tolist()
    data = json.dumps({"signature_name": "serving_default", "instances": data.tolist()})

    # send data using POST request and receive prediction result
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://169.56.89.210:8503/v1/models/sc_model/versions/4:predict', data=data, headers=headers)
    print('json_response : ', json_response)
    predictions = json.loads(json_response.text)['predictions']
    predictions = np.array(predictions)

    # get segmented pixel
    predictions = np.argmax(predictions, axis = -1)
    predictions = predictions.reshape(224, 224)

    plt.figure(figsize=(12, 3))
    plt.imshow(predictions)
    plt.axis('off')
    plt.title('\n\n{}'.format('Segmentation model prediction'), fontdict={'size': 16})
    plt.show()

    return print('Done')

get_image_segment_result()