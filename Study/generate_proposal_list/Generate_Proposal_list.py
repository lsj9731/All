import numpy as np
import pandas as pd
import os
from itertools import groupby
from tqdm import tqdm 
import time
import warnings

warnings.filterwarnings(action='ignore')

def get_frame(info):
    true_frame = []

    for i, data in enumerate(info):
        video_name = data[0].strip("\n").split("/")[-1]
        video_frame = data[1]
        true_frame.append([video_name, str(video_frame)])

    print('Done.')
    return true_frame

def train_proposal(Val_name, true_frame, existing_label, changed_label, PGM_dir, PEM_dir, Val_annotation):
    f = open("outputs/train_proposal.txt", 'w')
    for i, value in enumerate(tqdm(Val_name)):
        time.sleep(1/len(Val_name))
        # video index, name
        video_num = i
        video_name = value
        f.write('#'+str(video_num)+'\n')
        f.write(PGM_dir+video_name+'\n')
        
        # video frame 
        for i in range(len(Val_name)):
            if video_name in true_frame[i]:
                video_frame = true_frame[i][1]
                
        f.write(str(video_frame)+'\n')
        f.write('1'+'\n')


        
        # num ground truch
        temp_idx = Val_annotation.video == video_name
        temp = Val_annotation[temp_idx]
        num_gt = len(temp)
        f.write(str(num_gt)+'\n')
        
        # get gt box
        for gt_index in range(num_gt):
            temp_idx = Val_annotation.video == video_name
            temp = Val_annotation[temp_idx]
            temp = temp.iloc[gt_index]
            temp_label = temp['type_idx']
            temp_label = int(temp_label)
            
            label_index = existing_label.index(temp_label)
            temp_label = changed_label[label_index]
            
            temp_start = temp['startFrame']
            temp_end = temp['endFrame']
            
            f.write(str(temp_label)+' ')
            f.write(str(temp_start)+' ')
            f.write(str(temp_end)+'\n')
            
        # num proposal
        prop_df = pd.read_csv(os.path.join(PGM_dir+video_name+'.csv'))
        num_prop = len(prop_df)
        f.write(str(num_prop)+'\n')
        
        # get proposal box
        for prop_index in range(num_prop):
            # get data
            prop_info = prop_df.iloc[prop_index]

            # get label
            match_min = int(prop_info['match_xmin'])
            match_max = int(prop_info['match_xmax'])
            get_label = Val_annotation[['video', 'startFrame', 'endFrame', 'type_idx']]
            get_index = get_label['video'] == video_name
            get_data = get_label[get_index]
            get_label_index = get_data['startFrame'] == match_min
            data_1 = get_data[get_label_index]
            get_label_index = get_data['endFrame'] == match_max
            data_2 = data_1[get_label_index]
            
            if len(data_2.type_idx.values) == 1:
                num_class = int(data_2.type_idx.values)
                label_index = existing_label.index(num_class)
                num_class = changed_label[label_index]
                
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
            elif len(data_2.type_idx.values) >= 2:
                a_list = list(data_2.type_idx.values)
                for i in a_list:
                    num_class = int(i)
                    label_index = existing_label.index(num_class)
                    num_class = changed_label[label_index]
                    
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


def test_proposal(Test_name, PGM_dir, PEM_dir, Test_annotation, test_gt):
    f = open("outputs/test_proposal.txt", 'w')
    for i, value in enumerate(tqdm(Test_name)):
        time.sleep(1/len(Test_name))
        # video index, name
        video_num = i
        video_name = value
        f.write('#'+str(video_num)+'\n')
        f.write(PEM_dir+video_name+'\n')
        
        # video frame 
        temp_idx = test_gt['video-name'] == video_name
        temp = test_gt[temp_idx]
        temp = list(temp['video-frames'])
        video_frame = temp[0]
        video_frame = int(video_frame) - 1
        f.write(str(video_frame)+'\n')
        f.write('1'+'\n')
        
        # num ground truch
        temp_idx = test_gt['video-name'] == video_name
        temp = test_gt[temp_idx]
        num_gt = len(temp)
        f.write(str(num_gt)+'\n')
        
        # get gt box
        for gt_index in range(num_gt):
            temp_idx = test_gt['video-name'] == video_name
            temp = test_gt[temp_idx]
            temp = temp.iloc[gt_index]
            temp_label = temp['label_idx']
            temp_label = int(temp_label)
            
            temp_start = temp['f-init']
            temp_end = temp['f-end']
            
            f.write(str(temp_label)+' ')
            f.write(str(temp_start)+' ')
            f.write(str(temp_end)+'\n')
            
        # num proposal
        prop_df = pd.read_csv(os.path.join(PEM_dir+video_name+'.csv'))
        num_prop = len(prop_df)
        f.write(str(num_prop)+'\n')
        
        prop_df = prop_df.sort_values(by = ['iou_score'], ascending=False)
        if num_prop < 800:
            num_prop = num_prop
        elif num_prop >= 800:
            num_prop = 800
        
        # get proposal box
        for prop_index in range(num_prop):
            # get data
            prop_info = prop_df.iloc[prop_index]
            # get label
            match_min = int(prop_info['match_xmin'])
            match_max = int(prop_info['match_xmax'])
            get_label = test_gt[['video-name', 'f-init', 'f-end', 'label_idx']]
            get_index = get_label['video-name'] == video_name
            get_data = get_label[get_index]
            get_label_index = get_data['f-init'] == match_min
            data_1 = get_data[get_label_index]
            get_label_index = get_data['f-end'] == match_max
            data_2 = data_1[get_label_index]
            
            if len(data_2.label_idx.values) == 1:
                num_class = int(data_2.label_idx.values)
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
                    num_class = int(i)
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
    training_proposal = list(open("data/bsn_train_proposal_list.txt"))
    testing_proposal = list(open("data/bsn_test_proposal_list.txt"))
    pgm_dir = 'data/PGM_results/'
    pem_dir = 'data/PEM_results/'
    V_annotation = pd.read_csv('data/Val_Annotation.csv')
    T_annotation = pd.read_csv('data/Test_Annotation.csv')
    t_gt = pd.read_csv('data/thumos14_test_groundtruth.csv')

    V_name = list(set(V_annotation['video'][:]))
    T_name = list(set(T_annotation['video'][:]))

    train_prop = groupby(training_proposal, lambda x: x.startswith('#'))
    test_prop = groupby(testing_proposal, lambda x: x.startswith('#'))

    train_info = [[x.strip() for x in list(g)] for k, g in train_prop if not k]
    test_info = [[x.strip() for x in list(g)] for k, g in test_prop if not k]

    e_label = [7, 9, 12, 21, 22, 23, 24, 26, 31, 33, 36, 40, 45, 51, 68, 79, 85, 92, 93, 97]
    c_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    train_frame = get_frame(train_info)

    train_proposal(V_name, train_frame, e_label, c_label, pgm_dir, pem_dir, V_annotation)
    test_proposal(T_name, pgm_dir, pem_dir, T_annotation, t_gt)

    print('All Done.')