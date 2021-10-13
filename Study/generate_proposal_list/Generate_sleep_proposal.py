import numpy as np
import pandas as pd
import os
from itertools import groupby
from tqdm import tqdm 
import time
import warnings

warnings.filterwarnings(action='ignore')

def train_proposal(Val_name, PGM_dir, PEM_dir, Val_annotation):
    f = open("outputs/train_proposal.txt", 'w')
    for i, value in enumerate(tqdm(Val_name)):
        time.sleep(1/len(Val_name))
        # video index, name
        video_num = i
        video_name = value
        f.write('#'+str(video_num)+'\n')
        f.write(PGM_dir+video_name+'\n')
        
        # video frame 
        temp_idx = Val_annotation['video'] == video_name
        temp = Val_annotation[temp_idx]
        temp = list(temp['video-frame'])
        video_frame = temp[0]
        video_frame = int(video_frame) - 1
        f.write(str(video_frame)+'\n')
        f.write('1'+'\n')

        # num ground truth
        temp_idx = Val_annotation.video == video_name
        temp = Val_annotation[temp_idx]
        num_gt = len(temp)
        f.write(str(num_gt)+'\n')
        
        # get gt box
        for gt_index in range(num_gt):
            temp_idx = Val_annotation.video == video_name
            temp = Val_annotation[temp_idx]
            temp = temp.iloc[gt_index]
            temp_label = temp['label_idx']
            temp_label = int(temp_label) + 1
            
            temp_start = temp['startFrame']
            temp_end = temp['endFrame']
            
            f.write(str(temp_label)+' ')
            f.write(str(temp_start)+' ')
            f.write(str(temp_end)+'\n')
            
        # num proposal
        prop_df = pd.read_csv(os.path.join(PGM_dir+video_name+'.csv'))
        prop_df = prop_df.sample(frac=0.4, replace=True)
        num_prop = len(prop_df)
        f.write(str(num_prop)+'\n')
        
        # get proposal box
        for prop_index in range(num_prop):
            # get data
            prop_info = prop_df.iloc[prop_index]

            # get label
            match_min = int(prop_info['match_xmin'])
            match_max = int(prop_info['match_xmax'])
            get_label = Val_annotation[['video', 'startFrame', 'endFrame', 'label_idx']]
            get_index = get_label['video'] == video_name
            get_data = get_label[get_index]
            get_label_index = get_data['startFrame'] == match_min
            data_1 = get_data[get_label_index]
            get_label_index = get_data['endFrame'] == match_max
            data_2 = data_1[get_label_index]
            
            if len(data_2.label_idx.values) == 1:
                num_class = int(data_2.label_idx.values) + 1
                
                s_frame = int(prop_info['xmin'])
                e_frame = int(prop_info['xmax'])
                iou = float(prop_info['match_iou'])
                ioa = float(prop_info['match_ioa'])
                if prop_info['match_iou'] == 0 and prop_info['match_ioa'] ==  0:
                    num_class, iou, ioa = 0, 0, 0
                f.write(str(num_class)+' ')
                f.write(str(iou)+' ')
                f.write(str(ioa)+' ')
                f.write(str(s_frame)+' ')
                f.write(str(e_frame)+'\n')
            elif len(data_2.label_idx.values) >= 2:
                a_list = list(data_2.label_idx.values)
                for i in a_list:
                    num_class = int(data_2.label_idx.values) + 1
                    
                    s_frame = int(prop_info['xmin'])
                    e_frame = int(prop_info['xmax'])
                    iou = float(prop_info['match_iou'])
                    ioa = float(prop_info['match_ioa'])
                    if prop_info['match_iou'] == 0 and prop_info['match_ioa'] ==  0:
                        num_class, iou, ioa = 0, 0, 0
                
                    f.write(str(num_class)+' ')
                    f.write(str(iou)+' ')
                    f.write(str(ioa)+' ')
                    f.write(str(s_frame)+' ')
                    f.write(str(e_frame)+'\n')
            
    f.close()
    return print('Generated train proposal')


def test_proposal(Test_name, PGM_dir, PEM_dir, Test_annotation):
    f = open("outputs/test_proposal.txt", 'w')
    for i, value in enumerate(tqdm(Test_name)):
        time.sleep(1/len(Test_name))
        # video index, name
        video_num = i
        video_name = value
        f.write('#'+str(video_num)+'\n')
        f.write(PEM_dir+video_name+'\n')
        
        # video frame 
        temp_idx = Test_annotation['video'] == video_name
        temp = Test_annotation[temp_idx]
        temp = list(temp['video-frame'])
        video_frame = temp[0]
        video_frame = int(video_frame) - 1
        f.write(str(video_frame)+'\n')
        f.write('1'+'\n')
        
        # num ground truch
        temp_idx = Test_annotation['video'] == video_name
        temp = Test_annotation[temp_idx]
        num_gt = len(temp)
        f.write(str(num_gt)+'\n')
        
        # get gt box
        for gt_index in range(num_gt):
            temp_idx = Test_annotation['video'] == video_name
            temp = Test_annotation[temp_idx]
            temp = temp.iloc[gt_index]
            temp_label = temp['label_idx']
            temp_label = int(temp_label) + 1
            
            temp_start = temp['startFrame']
            temp_end = temp['endFrame']
            
            f.write(str(temp_label)+' ')
            f.write(str(temp_start)+' ')
            f.write(str(temp_end)+'\n')
            
        # num proposal
        prop_df = pd.read_csv(os.path.join(PEM_dir+video_name+'.csv'))
        prop_df = prop_df.sample(frac=0.3, replace=True)
        num_prop = len(prop_df)
        f.write(str(num_prop)+'\n')
        
        # get proposal box
        for prop_index in range(num_prop):
            # get data
            prop_info = prop_df.iloc[prop_index]
            # get label
            match_min = int(prop_info['match_xmin'])
            match_max = int(prop_info['match_xmax'])
            get_label = Test_annotation[['video', 'startFrame', 'endFrame', 'label_idx']]
            get_index = get_label['video'] == video_name
            get_data = get_label[get_index]
            get_label_index = get_data['startFrame'] == match_min
            data_1 = get_data[get_label_index]
            get_label_index = get_data['endFrame'] == match_max
            data_2 = data_1[get_label_index]
            
            if len(data_2.label_idx.values) == 1:
                num_class = int(data_2.label_idx.values) + 1
                s_frame = int(prop_info['xmin'])
                e_frame = int(prop_info['xmax'])
                iou = float(prop_info['match_iou'])
                ioa = float(prop_info['match_ioa'])
                if prop_info['match_iou'] == 0 and prop_info['match_ioa'] ==  0:
                    num_class, iou, ioa = 0, 0, 0
                f.write(str(num_class)+' ')
                f.write(str(iou)+' ')
                f.write(str(ioa)+' ')
                f.write(str(s_frame)+' ')
                f.write(str(e_frame)+'\n')
            elif len(data_2.label_idx.values) >= 2:
                a_list = list(data_2.label_idx.values)
                for i in a_list:
                    num_class = int(i) + 1
                    s_frame = int(prop_info['xmin'])
                    e_frame = int(prop_info['xmax'])
                    iou = float(prop_info['match_iou'])
                    ioa = float(prop_info['match_ioa'])
                    if prop_info['match_iou'] == 0 and prop_info['match_ioa'] ==  0:
                        num_class, iou, ioa = 0, 0, 0
                
                    f.write(str(num_class)+' ')
                    f.write(str(iou)+' ')
                    f.write(str(ioa)+' ')
                    f.write(str(s_frame)+' ')
                    f.write(str(e_frame)+'\n')
            
    f.close()
    return print('Generated test proposal')

if __name__ == "__main__":
    pgm_dir = 'data/PGM_results/'
    pem_dir = 'data/PEM_results/'
    V_annotation = pd.read_csv('data/Val_Annotation.csv')
    T_annotation = pd.read_csv('data/Test_Annotation.csv')

    V_name = list(set(V_annotation['video'][:]))
    T_name = list(set(T_annotation['video'][:]))

    train_proposal(V_name, pgm_dir, pem_dir, V_annotation)
    test_proposal(T_name, pgm_dir, pem_dir, T_annotation)

    print('All Done.')