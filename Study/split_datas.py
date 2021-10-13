from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
import time
from tqdm import tqdm

# raw frame이 모여져있는 데이터 경로
all_data_dir = '../data'

# alls, cut1, cut2
data_dir_list = os.listdir(all_data_dir)

# 데이터 경로에서 폴더명을 하나씩 불러옴
# alls, cut1, cut2에서 데이터를 하나씩 load
for data_dir in tqdm(data_dir_list):
    print('Dir name : ', data_dir)
    start = time.time()
    # 경로 + 폴더명으로 데이터에 접근
    # /datas/alls, alls 아래에 있는 디렉토리를 읽음 / all_data1, all_data2
    data = os.path.join(all_data_dir, data_dir)
    open_data = os.listdir(data)

    # 읽어들인 경로에서 split이라는 이름의 디렉토리를 생성
    gen_dir1 = data+'/split1'
    gen_dir2 = data+'/split2'
    gen_dir3 = data+'/split3'
    # 영상이 1개인 풀영상이라면
    if len(open_data) == 1:
        # 디렉토리 생성 / 이미 있다면 OSerror 발생
        if not os.path.exists(gen_dir1) and not os.path.exists(gen_dir2) and not os.path.exists(gen_dir3):
            os.mkdir(gen_dir1)
            os.mkdir(gen_dir2)
            os.mkdir(gen_dir3)
        else:
            raise OSError("already folder.")

        rgb_all = []
        flow_x_all = []
        flow_y_all = []
        # 경로에서 이미지 누적
        print('load Image ...')
        for datas in open_data:
            load_data_dir = os.path.join(data, datas)
            image_data = os.listdir(load_data_dir)
            [rgb_all.append(Image.open(os.path.join(load_data_dir, img)).convert('RGB')) for img in image_data if 'img' in img]
            [flow_x_all.append(Image.open(os.path.join(load_data_dir, img)).convert('L')) for img in image_data if 'flow_x' in img]
            [flow_y_all.append(Image.open(os.path.join(load_data_dir, img)).convert('L')) for img in image_data if 'flow_y' in img]
        
        print('Done.\n')
        
        total_frame = len(rgb_all)
        split_frame = total_frame // 3
        split_list1 = []
        split_list2 = []
        split_list3 = []

        print('Split Image ...')

        split_list1.append(rgb_all[:split_frame])
        split_list1.append(flow_x_all[:split_frame])
        split_list1.append(flow_y_all[:split_frame])

        split_list2.append(rgb_all[split_frame:split_frame*2])
        split_list2.append(flow_x_all[split_frame:split_frame*2])
        split_list2.append(flow_y_all[split_frame:split_frame*2])

        split_list3.append(rgb_all[split_frame*2:])
        split_list3.append(flow_x_all[split_frame*2:])
        split_list3.append(flow_y_all[split_frame*2:])

        print('Done.\n')
        

        print('Save Image to split1 ...')
        # 이미지 저장
        # 이미지 저장
        data_frame = len(split_list1[0])
        [images1.save(gen_dir1+'/'+"img_{:05d}.jpg".format(r_idx)) for r_idx, images1 in enumerate(split_list1[0])]
        [images2.save(gen_dir1+'/'+"flow_x_{:05d}.jpg".format(x_idx)) for x_idx, images2 in enumerate(split_list1[1])]
        [images3.save(gen_dir1+'/'+"flow_y_{:05d}.jpg".format(y_idx)) for y_idx, images3 in enumerate(split_list1[2])]
        f = open(gen_dir1+'/frame.txt', "w")
        f.write(str(data_frame))

        print('Save Image to split2 ...')
        [images1.save(gen_dir2+'/'+"img_{:05d}.jpg".format(r_idx)) for r_idx, images1 in enumerate(split_list2[0])]
        [images2.save(gen_dir2+'/'+"flow_x_{:05d}.jpg".format(x_idx)) for x_idx, images2 in enumerate(split_list2[1])]
        [images3.save(gen_dir2+'/'+"flow_y_{:05d}.jpg".format(y_idx)) for y_idx, images3 in enumerate(split_list2[2])]    
        f = open(gen_dir2+'/frame.txt', "w")
        f.write(str(data_frame))

        print('Save Image to split3 ...')
        [images1.save(gen_dir3+'/'+"img_{:05d}.jpg".format(r_idx)) for r_idx, images1 in enumerate(split_list3[0])]
        [images2.save(gen_dir3+'/'+"flow_x_{:05d}.jpg".format(x_idx)) for x_idx, images2 in enumerate(split_list3[1])]
        [images3.save(gen_dir3+'/'+"flow_y_{:05d}.jpg".format(y_idx)) for y_idx, images3 in enumerate(split_list3[2])]
        f = open(gen_dir3+'/frame.txt', "w")
        f.write(str(data_frame))

        print('Time : ', time.time() - start)
        print('Done.\n')
    elif len(open_data) != 1:
        # open_data는 나눠진 전체 frame 디렉토리
        for data_dir in open_data:
            # 해당 디렉토리를 하나씩 열어서 사용
            load_data_dir = os.path.join(data, data_dir)
            image_data = os.listdir(load_data_dir)
            gen_dir1 = load_data_dir+'/split1'
            gen_dir2 = load_data_dir+'/split2'
            gen_dir3 = load_data_dir+'/split3'

            if not os.path.exists(gen_dir1) and not os.path.exists(gen_dir2) and not os.path.exists(gen_dir3):
                os.mkdir(gen_dir1)
                os.mkdir(gen_dir2)
                os.mkdir(gen_dir3)
            else:
                raise OSError("already folder.")

            rgb_all = []
            flow_x_all = []
            flow_y_all = []
            print('load Image ...')
            [rgb_all.append(Image.open(os.path.join(load_data_dir, img)).convert('RGB')) for img in image_data if 'img' in img]
            [flow_x_all.append(Image.open(os.path.join(load_data_dir, img)).convert('L')) for img in image_data if 'flow_x' in img]
            [flow_y_all.append(Image.open(os.path.join(load_data_dir, img)).convert('L')) for img in image_data if 'flow_y' in img]
            print('Done.\n')

            total_frame = len(rgb_all)
            split_frame = total_frame // 3
            split_list1 = []
            split_list2 = []
            split_list3 = []

            print('Split Image ...')

            split_list1.append(rgb_all[:split_frame])
            split_list1.append(flow_x_all[:split_frame])
            split_list1.append(flow_y_all[:split_frame])

            split_list2.append(rgb_all[split_frame:split_frame*2])
            split_list2.append(flow_x_all[split_frame:split_frame*2])
            split_list2.append(flow_y_all[split_frame:split_frame*2])

            split_list3.append(rgb_all[split_frame*2:])
            split_list3.append(flow_x_all[split_frame*2:])
            split_list3.append(flow_y_all[split_frame*2:])

            print('Done.\n')
            print('Save Image to split1 ...')
            # 이미지 저장
            data_frame = len(split_list1[0])
            [images1.save(gen_dir1+'/'+"img_{:05d}.jpg".format(r_idx)) for r_idx, images1 in enumerate(split_list1[0])]
            [images2.save(gen_dir1+'/'+"flow_x_{:05d}.jpg".format(x_idx)) for x_idx, images2 in enumerate(split_list1[1])]
            [images3.save(gen_dir1+'/'+"flow_y_{:05d}.jpg".format(y_idx)) for y_idx, images3 in enumerate(split_list1[2])]
            f = open(gen_dir1+'/frame.txt', "w")
            f.write(str(data_frame))

            print('Save Image to split2 ...')
            [images1.save(gen_dir2+'/'+"img_{:05d}.jpg".format(r_idx)) for r_idx, images1 in enumerate(split_list2[0])]
            [images2.save(gen_dir2+'/'+"flow_x_{:05d}.jpg".format(x_idx)) for x_idx, images2 in enumerate(split_list2[1])]
            [images3.save(gen_dir2+'/'+"flow_y_{:05d}.jpg".format(y_idx)) for y_idx, images3 in enumerate(split_list2[2])]    
            f = open(gen_dir2+'/frame.txt', "w")
            f.write(str(data_frame))

            print('Save Image to split3 ...')
            [images1.save(gen_dir3+'/'+"img_{:05d}.jpg".format(r_idx)) for r_idx, images1 in enumerate(split_list3[0])]
            [images2.save(gen_dir3+'/'+"flow_x_{:05d}.jpg".format(x_idx)) for x_idx, images2 in enumerate(split_list3[1])]
            [images3.save(gen_dir3+'/'+"flow_y_{:05d}.jpg".format(y_idx)) for y_idx, images3 in enumerate(split_list3[2])]
            f = open(gen_dir3+'/frame.txt', "w")
            f.write(str(data_frame))

            print('Time : ', time.time() - start)
            print('Done.\n')
print('All Done.')
