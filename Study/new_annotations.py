from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
import time
from tqdm import tqdm

annotation_path = 'C:/Users/82107/Downloads/new_annotations'
test_list = ['B2019-EM-01-0060', 'B2019-EM-01-0062', 'B2019-EM-01-0076', 'B2019-EM-01-0083', 'B2019-EM-01-0090', 'B2019-EM-01-0093']
train_list = ['B2019-EM-01-0007', 'B2019-EM-01-0008', 'B2019-EM-01-0009', 'B2019-EM-01-0011', 'B2019-EM-01-0012', 'B2019-EM-01-0014',
            'B2019-EM-01-0016', 'B2019-EM-01-0017', 'B2019-EM-01-0018', 'B2019-EM-01-0020', 'B2019-EM-01-0022', 'B2019-EM-01-0024',
            'B2019-EM-01-0025', 'B2019-EM-01-0030', 'B2019-EM-01-0032', 'B2019-EM-01-0037', 'B2019-EM-01-0042', 'B2019-EM-01-0043',
            'B2019-EM-01-0044', 'B2019-EM-01-0046', 'B2019-EM-01-0047', 'B2019-EM-01-0051', 'B2019-EM-01-0052', 'B2019-EM-01-0054',
            'B2019-EM-01-0056']
csv_list = os.listdir(annotation_path)

csvvvv = pd.read_csv('C:/Users/82107/Downloads/new_annotations/B2019-EM-01-0060.csv')
col_name = csvvvv.columns.tolist()

test_annotation = []
train_annotation = []
for csv in csv_list:
    csvs = csv.split('.')[0]
    if csvs in test_list:
        open_csv = pd.read_csv(os.path.join(annotation_path, csv))
        for i in range(len(open_csv)):
            test_annotation.append(open_csv.iloc[i])
    elif csvs in train_list:
        open_csv = pd.read_csv(os.path.join(annotation_path, csv))
        for j in range(len(open_csv)):
            train_annotation.append(open_csv.iloc[j])

test_annotation_df = pd.DataFrame(test_annotation, columns = col_name)
train_annotation_df = pd.DataFrame(train_annotation, columns = col_name)

test_annotation_df.to_csv('C:/Users/82107/Downloads/annotation/Test_Annotation.csv', index=False)
train_annotation_df.to_csv('C:/Users/82107/Downloads/annotation/Val_Annotation.csv', index=False)