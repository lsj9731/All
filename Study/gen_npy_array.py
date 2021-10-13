from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
import time
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt

# frame_dirs = 'C:/Users/82107/Downloads/raw_frame'
frame_dirs = './datas'
frame_folders = os.listdir(frame_dirs)

for folders in tqdm(frame_folders):
    start = time.time()
    time.sleep(1 / len(frame_folders))
    raw_frame_dir = os.path.join(frame_dirs, folders)
    raw_frame_folder = os.listdir(raw_frame_dir)
    rgb_all = []
    flow_x_all = []
    flow_y_all = []
    output_npy = raw_frame_dir

    print('load Image ...')
    for i, folder_name in enumerate(raw_frame_folder):
        img_list = os.listdir(os.path.join(raw_frame_dir, folder_name))
        img_list_dir = os.path.join(raw_frame_dir, folder_name)
        
        [rgb_all.append(Image.open(os.path.join(img_list_dir, img)).convert('RGB')) for img in img_list if 'img' in img]
        [flow_x_all.append(Image.open(os.path.join(img_list_dir, img)).convert('L')) for img in img_list if 'flow_x' in img]
        [flow_y_all.append(Image.open(os.path.join(img_list_dir, img)).convert('L')) for img in img_list if 'flow_y' in img]

    print('Done.\n')

    out_images_rgb = list()
    out_images_flow_X = list()
    out_images_flow_y = list()
    # 누적된 이미지를 처리
    # 정규화, 리사이즈

    print('preprocessing the Image ...')
    [out_images_rgb.append(np.array(rgb_image.resize((224, 224))) / 255.) for rgb_image in rgb_all]
    [out_images_flow_X.append(np.array(flow_x_image.resize((224, 224))) / 255.) for flow_x_image in flow_x_all]
    [out_images_flow_y.append(np.array(flow_y_image.resize((224, 224))) / 255.) for flow_y_image in flow_y_all]
    print('Done.\n')

    print('Concatenate npy array ...')
    out_images_rgb = np.concatenate(out_images_rgb).reshape(1, -1, 224, 224, 3)
    out_images_flow_X = np.concatenate(out_images_flow_X).reshape(1, -1, 224, 224, 1)
    out_images_flow_y = np.concatenate(out_images_flow_y).reshape(1, -1, 224, 224, 1)
    concat_flow = np.concatenate((out_images_flow_X, out_images_flow_y), axis = -1)
    print('Done.\n')

    print('Save npy array to npy file ...')
    np.save(raw_frame_dir+'/rgb.npy', out_images_rgb)
    np.save(raw_frame_dir+'/flow.npy', concat_flow)
    print('Done.\n')
    print('time : ', time.time() - start)