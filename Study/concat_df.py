from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
import time
from tqdm import tqdm

csv_dir = './csv'
csv_out_dir = './csv_output'

csv_dir_names = os.listdir(csv_dir)

for dirs in csv_dir_names:
    csv_ = os.listdir(os.path.join(csv_dir, dirs))
    all_df = []
    for get_csv in csv_:
        dir = os.path.join(csv_dir, dirs)
        df = pd.read_csv(os.path.join(dir, get_csv))
        df = np.array(df)
        all_df.append(df)
    all_df = np.concatenate(all_df, axis=0)
    col_name = []
    for i in range(101):
        names = 'f'+str(i)
        col_name.append(names)
    all_df = pd.DataFrame(all_df, columns=col_name)
    all_df.to_csv(csv_out_dir+'/'+dirs+'.csv', index=False)