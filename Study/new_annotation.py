import numpy as np
import pandas as pd
from collections import Counter
import json
from datetime import datetime, timedelta
import datetime
import os

csv_path = 'C:/Users/82107/Downloads/annotation_csv'
csv_list = os.listdir(csv_path)

event_list = ['Wake', 'N1', 'N2', 'N3']

for csv in csv_list:
    open_csv = pd.read_csv(os.path.join(csv_path, csv))
    col = open_csv.columns.tolist()
    print(col)
    count_csv = []
    for i in range(len(open_csv)):
        if open_csv.iloc[i]['event_type'] in event_list:
            count_csv.append(open_csv.iloc[i])
    count_csv_df = pd.DataFrame(count_csv, columns = col)
    count_csv_df.to_csv('C:/Users/82107/Downloads/new_annotations/'+csv, index=False)