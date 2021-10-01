from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd

# 영상 데이터들이 담긴 공간 경로
all_data_dir = 'C:/Users/82107/Downloads/new_jpg/datas'
data_dir_list = os.listdir(all_data_dir)

for i, data_dir in enumerate(data_dir_list):
    data = os.path.join(all_data_dir, data_dir)
    open_data = os.listdir(data)
    gen_dir = data+'/results'
    if len(open_data) != 1:
        if not os.path.exists(gen_dir):
            os.mkdir(gen_dir)
        else:
            raise OSError("already folder.")
        rgb_all = []
        flow_x_all = []
        flow_y_all = []
        for datas in open_data:
            load_data_dir = os.path.join(data, datas)
            image_data = os.listdir(load_data_dir)
            for img in image_data:
                if 'img' in img:
                    rgb = Image.open(os.path.join(load_data_dir, img)).convert('RGB')
                    rgb_all.append(rgb)
                elif 'flow_x' in img:
                    flow_x = Image.open(os.path.join(load_data_dir, img)).convert('L')
                    flow_x_all.append(flow_x)
                elif 'flow_y' in img:
                    flow_y = Image.open(os.path.join(load_data_dir, img)).convert('L')
                    flow_y_all.append(flow_y)
                    
        for ii in range(len(rgb_all)):
            rgb_all[ii].save(gen_dir+'/'+"img_{:05d}.jpg".format(ii))

        for jj in range(len(flow_x_all)):
            flow_x_all[jj].save(gen_dir+'/'+"flow_x_{:05d}.jpg".format(jj))

        for kk in range(len(flow_y_all)):
            flow_y_all[kk].save(gen_dir+'/'+"flow_y_{:05d}.jpg".format(kk))