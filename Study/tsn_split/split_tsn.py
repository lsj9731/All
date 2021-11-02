import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from PIL import Image

# read data
data_dir = './data'
outout_dir = './data_output'
annotation_dir = './annotation'
annotation_list = os.listdir(annotation_dir)

data_list = os.listdir(data_dir)
for datas in tqdm(data_list):
    time.sleep(1 / len(data_list))

    json_annotation = json.load(open(annotation_dir+'/'+datas+'.json'))
    frame_rate = json_annotation['Video_Info'][0]['Frame_Rate']
    split_size = int(30 * frame_rate)

    one_data_path = os.path.join(data_dir, datas)
    one_data_list = os.listdir(one_data_path)

    if len(one_data_list) == 1:
        start = time.time()
        all_images = []
        all_flow_x = []
        all_flow_y = []

        image_data_path = os.path.join(one_data_path, one_data_list[0])
        image_list = os.listdir(image_data_path)

        [all_images.append(Image.open(os.path.join(image_data_path, img)).convert('RGB')) for img in image_list if 'img' in img]
        [all_flow_x.append(Image.open(os.path.join(image_data_path, img)).convert('L')) for img in image_list if 'flow_x' in img]
        [all_flow_y.append(Image.open(os.path.join(image_data_path, img)).convert('L')) for img in image_list if 'flow_y' in img]

        num_folder = len(all_images) // split_size

        one_data_path_split = one_data_path.split('\\')[-1]
        new_out_path = outout_dir+'/'+one_data_path_split
        os.mkdir(new_out_path)
        
        for i in range(num_folder+1):
            nums = all_images[i*split_size:(i+1)*split_size]
            flow_x = all_flow_x[i*split_size:(i+1)*split_size]
            flow_y = all_flow_y[i*split_size:(i+1)*split_size]
            len_nums = len(nums)
            new_path = new_out_path+'/split_{:04d}'.format(i)
            os.mkdir(new_path)

            [nums[rgb_idx].save(new_path+'/'+"img_{:05d}.jpg".format(rgb_idx)) for rgb_idx in range(len(nums))]
            [flow_x[flow_x_idx].save(new_path+'/'+"flow_x_{:05d}.jpg".format(flow_x_idx)) for flow_x_idx in range(len(flow_x))]
            [flow_y[flow_y_idx].save(new_path+'/'+"flow_y_{:05d}.jpg".format(flow_y_idx)) for flow_y_idx in range(len(flow_y))]

        print('end time : ', time.time() - start)

    elif len(one_data_list) != 1:
        start = time.time()
        all_images = []
        all_flow_x = []
        all_flow_y = []
        for load_dir in one_data_list:
            image_data_path = os.path.join(one_data_path, load_dir)
            image_list = os.listdir(image_data_path)
            [all_images.append(Image.open(os.path.join(image_data_path, img)).convert('RGB')) for img in image_list if 'img' in img]
            [all_flow_x.append(Image.open(os.path.join(image_data_path, img)).convert('L')) for img in image_list if 'flow_x' in img]
            [all_flow_y.append(Image.open(os.path.join(image_data_path, img)).convert('L')) for img in image_list if 'flow_y' in img]

        num_folder = len(all_images) // split_size

        one_data_path_split = one_data_path.split('\\')[-1]
        new_out_path = outout_dir+'/'+one_data_path_split
        os.mkdir(new_out_path)

        for i in range(num_folder+1):
            nums = all_images[i*split_size:(i+1)*split_size]
            flow_x = all_flow_x[i*split_size:(i+1)*split_size]
            flow_y = all_flow_y[i*split_size:(i+1)*split_size]
            len_nums = len(nums)
            new_path = new_out_path+'/split_{:04d}'.format(i)
            os.mkdir(new_path)

            [nums[rgb_idx].save(new_path+'/'+"img_{:05d}.jpg".format(rgb_idx)) for rgb_idx in range(len(nums))]
            [flow_x[flow_x_idx].save(new_path+'/'+"flow_x_{:05d}.jpg".format(flow_x_idx)) for flow_x_idx in range(len(flow_x))]
            [flow_y[flow_y_idx].save(new_path+'/'+"flow_y_{:05d}.jpg".format(flow_y_idx)) for flow_y_idx in range(len(flow_y))]

        print('end time : ', time.time() - start)