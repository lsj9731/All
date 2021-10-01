from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
import time
from tqdm import tqdm

# 영상 데이터들이 담긴 공간 경로
all_data_dir = './datas'
data_dir_list = os.listdir(all_data_dir)

for i, data_dir in enumerate(data_dir_list):
    start = time.time()
    data = os.path.join(all_data_dir, data_dir)
    open_data = os.listdir(data)
    # 저장할 위치는 데이터 경로에서 results란 디렉토리 내부
    gen_dir = data+'/results'
    if len(open_data) != 1:
        print(str(i)+'번째 작업 중 입니다.')
        # 디렉토리 생성 / 이미 있다면 OSerror 발생
        if not os.path.exists(gen_dir):
            os.mkdir(gen_dir)
        else:
            raise OSError("already folder.")
        rgb_all = []
        flow_x_all = []
        flow_y_all = []
        # 경로에서 이미지 누적
        for datas in tqdm(open_data):
            time.sleep(1 / len(open_data))
            load_data_dir = os.path.join(data, datas)
            image_data = os.listdir(load_data_dir)
            [rgb_all.append(Image.open(os.path.join(load_data_dir, img)).convert('RGB')) for img in image_data if 'img' in img]
            [flow_x_all.append(Image.open(os.path.join(load_data_dir, img)).convert('L')) for img in image_data if 'flow_x' in img]
            [flow_y_all.append(Image.open(os.path.join(load_data_dir, img)).convert('L')) for img in image_data if 'flow_y' in img]

        # 모든 이미지 누적 후 저장
        [rgb_all[ii].save(gen_dir+'/'+"img_{:05d}.jpg".format(ii)) for ii in range(len(rgb_all))]
        [flow_x_all[jj].save(gen_dir+'/'+"flow_x_{:05d}.jpg".format(jj)) for jj in range(len(flow_x_all))]
        [flow_y_all[kk].save(gen_dir+'/'+"flow_y_{:05d}.jpg".format(kk)) for kk in range(len(flow_y_all))]

        print('total time : ', time.time() - start)