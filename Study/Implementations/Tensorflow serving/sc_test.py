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
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from tensorflow.keras.utils import to_categorical


def get_dir_result():
  # load skin cancer images
  sc_data_dir = 'C:/Users/82107/Desktop/Python/Data/Skin_cancer_data/Task_3/ISIC2018_Task3_Validation_Input'
  sc_annotation = pd.read_csv('C:/Users/82107/Desktop/Python/Data/Skin_cancer_data/Task_3/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv')

  data, label = [], []
  for idx in range(len(sc_annotation)):
    e_annotation = sc_annotation.iloc[idx]
    image_name, labels = e_annotation[0], np.argmax(e_annotation[1:])
    get_image = np.array(Image.open(os.path.join(sc_data_dir, image_name+'.jpg')).resize((224,224))).reshape(-1, 224, 224, 3)
    data.append(get_image)
    label.append(labels)
  data = np.concatenate(data, axis=0)


  # scale the values to 0.0 to 1.0
  data = data / 255.0

  class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

  print('images.shape: {}, of {}'.format(data.shape, data.dtype))

  # npy data -> tolist()
  data = json.dumps({"signature_name": "serving_default", "instances": data.tolist()})

  # send data using POST request and receive prediction result
  headers = {"content-type": "application/json"}
  json_response = requests.post('http://169.56.89.210:8502/v1/models/sc_model/versions/2:predict', data=data, headers=headers)
  print('json_response : ', json_response)
  predictions = json.loads(json_response.text)['predictions']
  print('all prediction length : ', len(predictions))
  # show first prediction result
  print('The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
    class_names[np.argmax(predictions[0])], np.argmax(predictions[0]), class_names[label[0]], label[0]))

  label_c = to_categorical(label)
  max_predictions = np.argmax(predictions, axis = 1)

  print(classification_report(label, max_predictions))

  # get acc, recall, precision, f1 score
  accuracy = accuracy_score(label, max_predictions)
  precision = precision_score(label, max_predictions, average='weighted')
  recall = recall_score(label, max_predictions, average='weighted')
  f1score = f1_score(label, max_predictions, average='weighted')
  weighted_auc_ovo = roc_auc_score(label_c, predictions, multi_class="ovo", average="weighted")

  # get roc curve image
  n_classes = 7
  fpr = dict()
  tpr = dict()
  roc_auc = dict()

  predictions = np.array(predictions)
  print('label_c : ', label_c.shape, 'predictions : ', predictions.shape)

  for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(label_c[:, i], predictions[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])
      
  fpr["micro"], tpr["micro"], _ = roc_curve(label_c.ravel(), predictions.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_classes):
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
  mean_tpr /= n_classes
  lw = 2

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # Plot all ROC curves
  plt.figure()
  plt.plot(
      fpr["micro"],
      tpr["micro"],
      label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
      color="deeppink",
      linestyle=":",
      linewidth=4,
  )

  plt.plot(
      fpr["macro"],
      tpr["macro"],
      label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
      color="navy",
      linestyle=":",
      linewidth=4,
  )

  colors = cycle(["aqua", "darkorange", "cornflowerblue"])
  for i, color in zip(range(n_classes), colors):
      plt.plot(
          fpr[i],
          tpr[i],
          color=color,
          lw=lw,
          label="ROC curve of class {0} (area = {1:0.2f})".format(class_names[i], roc_auc[i]),
      )

  plt.plot([0, 1], [0, 1], "k--", lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Some extension of Receiver operating characteristic to multiclass")
  plt.legend(loc="lower right")
  plt.savefig('images/roc_curve.png')

  # get confusion matrix image
  cf_matrix = confusion_matrix(label, max_predictions)
  cf_plot = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
  fig = cf_plot.get_figure()
  fig.savefig("images/confusion_matrix.png") 
  print('Done.')

  roc_image = Image.open('images/roc_curve.png')
  cf_image = Image.open('images/confusion_matrix.png')

  return accuracy, recall, precision, f1score, weighted_auc_ovo, roc_image, cf_image


def get_image_result():
  # load skin cancer images
  sc_image_dir = 'C:/Users/82107/Desktop/Python/Data/Skin_cancer_data/Task_3/ISIC2018_Task3_Validation_Input/ISIC_0034321.jpg'
  get_image = np.array(Image.open(sc_image_dir).resize((224,224))).reshape(-1, 224, 224, 3)

  # scale the values to 0.0 to 1.0
  data = get_image / 255.0

  # npy data -> tolist()
  data = json.dumps({"signature_name": "serving_default", "instances": data.tolist()})

  # send data using POST request and receive prediction result
  headers = {"content-type": "application/json"}
  json_response = requests.post('http://169.56.89.210:8502/v1/models/sc_model/versions/2:predict', data=data, headers=headers)
  print('json_response : ', json_response)
  predictions = json.loads(json_response.text)['predictions']
  predictions = np.array(predictions)
  class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

  proba = []
  for p in predictions.ravel():
    proba.append(round(p, 2))

  sorted_proba = []
  sorted_label = []
  for i in range(len(proba)):
    max_idx = proba.index(max(proba))
    pop_value = proba.pop(max_idx)
    pop_label = class_names.pop(max_idx)

    sorted_proba.append(pop_value)
    sorted_label.append(pop_label)

  return sorted_proba, sorted_label


acc, rec, pre, f1, auc, roc_image, cf_image = get_dir_result()
print('accuracy : ', round(acc, 2))
print('recall : ', round(rec, 2))
print('precision : ', round(pre, 2))
print('f1 score : ', round(f1, 2))
print('auc score : ', round(auc, 2))

sorted_probability, sorted_labels = get_image_result()
print('sorted_probability : ', sorted_probability)
print('sorted_labels : ', sorted_labels)