import numpy as np
import pandas as pd
from collections import Counter
import json
from datetime import datetime, timedelta
import datetime
import os

labels = ['Wake', 'N1', 'N2', 'N3', 'REM', 'Left', 'Prone', 'Right', 'Supine', 'Unknown Position', 'Upright', 'Apnea Central',
'Apnea Mixed', 'Apnea Obstructive', 'Desat', 'Hypopnea', 'Snore Train', 'LM Arousal', 'PLM Arousal', 'Rera', 'Respiratory Arousal', 
'Spontaneous Arousal', 'LM', 'PLM', 'REM Sleep Behavior', 'Artifact','Battery Critical','Battery Low','Device Connected','Device Disconnected',
'Error','Impedance Test Start','Lights Off','Lights On','Note','Patient Unit Connected','Patient Unit Disconnected','Power Loss','Recording Interrupted','Recording Resumed','Video Clip']


# Json 경로 지정
Json_list = os.listdir('./annotation')


for data in Json_list:
    count_start_end = []

    annotation = json.load(open('./annotation/'+data))
    video_name = annotation['Case_Info']['Case_Number']
    
    # 하나의 Annotation 일 때 수행
    if len(annotation['Video_Info']) == 1:
        video_frame = annotation['Video_Info'][0]['Frame_Count']
        frame_rate = annotation['Video_Info'][0]['Frame_Rate']
        
        # get video start time
        start_date = annotation['Video_Info'][0]['Start']
        end_date = annotation['Video_Info'][0]['End']
        s_datetime = datetime.datetime.strptime(start_date, '%Y/%m/%d %H:%M:%S.%f')
        e_datetime = datetime.datetime.strptime(end_date, '%Y/%m/%d %H:%M:%S.%f')
        
        # DataFrame으로 만들어주기 위한 데이터 누적
        all_datas = []
        temp = []
        for i in range(len(annotation['Event'])):
            
            if annotation['Event'][i]['Event_Label'] == 'Wake' or annotation['Event'][i]['Event_Label'] == 'N1' \
                or annotation['Event'][i]['Event_Label'] == 'N2'or annotation['Event'][i]['Event_Label'] == 'N3'\
                or annotation['Event'][i]['Event_Label'] == 'REM':
                
                if temp == []:
                    e_type = annotation['Event'][i]['Event_Label']
                    duration = annotation['Event'][i]['Duration(second)']
                
                    e_start_date = annotation['Event'][i]['Start_Time']
                    e_start_datetime = datetime.datetime.strptime(e_start_date, '%Y/%m/%d %H:%M:%S.%f')
                    e_end_date = annotation['Event'][i]['End_Time']
                    e_end_datetime = datetime.datetime.strptime(e_end_date, '%Y/%m/%d %H:%M:%S.%f')

                    if s_datetime <= e_start_datetime and e_datetime >= e_end_datetime:
                        action_start = e_start_datetime - s_datetime
                        action_end = e_end_datetime - s_datetime
                        if e_type in labels:
                            label_idx = labels.index(e_type)
                        temp = [video_name, e_type, label_idx, round(action_start.seconds+(action_start.microseconds/1000000), 1), round(action_end.seconds+(action_end.microseconds/1000000), 1), 
                                  int(action_start.seconds * frame_rate), int(action_end.seconds * frame_rate), duration, video_frame, frame_rate, 
                                 e_start_datetime, e_end_datetime]
                    
                
                else:
                        
                    if annotation['Event'][i]['Event_Label'] == e_type:
                        duration += annotation['Event'][i]['Duration(second)']
                
                        e_end_date = annotation['Event'][i]['End_Time']
                        e_end_datetime = datetime.datetime.strptime(e_end_date, '%Y/%m/%d %H:%M:%S.%f')
                        
                        if s_datetime <= e_start_datetime and e_datetime >= e_end_datetime:
                            action_start = e_start_datetime - s_datetime
                            action_end = e_end_datetime - s_datetime
                            if e_type in labels:
                                label_idx = labels.index(e_type)
                            temp = [video_name, e_type, label_idx, round(action_start.seconds+(action_start.microseconds/1000000), 1), round(action_end.seconds+(action_end.microseconds/1000000), 1), 
                                      int(action_start.seconds * frame_rate), int(action_end.seconds * frame_rate), duration, video_frame, frame_rate, 
                                     e_start_datetime, e_end_datetime]
                            
                            
                    else:
                        
                        all_datas.append(temp)
                        
                        e_type = annotation['Event'][i]['Event_Label']
                        duration = annotation['Event'][i]['Duration(second)']

                        e_start_date = annotation['Event'][i]['Start_Time']
                        e_start_datetime = datetime.datetime.strptime(e_start_date, '%Y/%m/%d %H:%M:%S.%f')

                        e_end_date = annotation['Event'][i]['End_Time']
                        e_end_datetime = datetime.datetime.strptime(e_end_date, '%Y/%m/%d %H:%M:%S.%f')
                        if s_datetime <= e_start_datetime and e_datetime >= e_end_datetime:
                            action_start = e_start_datetime - s_datetime
                            action_end = e_end_datetime - s_datetime
                            if e_type in labels:
                                label_idx = labels.index(e_type)
                            temp = [video_name, e_type, label_idx, round(action_start.seconds+(action_start.microseconds/1000000), 1), round(action_end.seconds+(action_end.microseconds/1000000), 1), 
                                      int(action_start.seconds * frame_rate), int(action_end.seconds * frame_rate), duration, video_frame, frame_rate, 
                                     e_start_datetime, e_end_datetime]
                            
            if annotation['Event'][i]['Event_Label'] == 'Analysis Stop':
                if all_datas[-1] != temp:   
                    all_datas.append(temp)
                    break
                else:
                    break
        
    else:
        video_frame = 0
        for i in range(len(annotation['Video_Info'])):
            video_frame += annotation['Video_Info'][i]['Frame_Count']
            count_start_end.append([annotation['Video_Info'][i]['Start'], annotation['Video_Info'][i]['End']])
    
        # get video start time
        start_date = annotation['Video_Info'][0]['Start']
        start_datetime = datetime.datetime.strptime(start_date, '%Y/%m/%d %H:%M:%S.%f')
        frame_rate = annotation['Video_Info'][0]['Frame_Rate']
        
        for j in range(len(annotation['Video_Info'])):
            c_start, c_end = count_start_end[0][0], count_start_end[-1][1]
            
            s_datetime = datetime.datetime.strptime(c_start, '%Y/%m/%d %H:%M:%S.%f')
            e_datetime = datetime.datetime.strptime(c_end, '%Y/%m/%d %H:%M:%S.%f')
            
            # DataFrame으로 만들어주기 위한 데이터 누적
            all_datas = []
            temp = []
            for i in range(len(annotation['Event'])):
            
                if annotation['Event'][i]['Event_Label'] == 'Wake' or annotation['Event'][i]['Event_Label'] == 'N1' \
                    or annotation['Event'][i]['Event_Label'] == 'N2'or annotation['Event'][i]['Event_Label'] == 'N3'\
                    or annotation['Event'][i]['Event_Label'] == 'REM':
                
                    if temp == []:
                        e_type = annotation['Event'][i]['Event_Label']
                        duration = annotation['Event'][i]['Duration(second)']

                        e_start_date = annotation['Event'][i]['Start_Time']
                        e_start_datetime = datetime.datetime.strptime(e_start_date, '%Y/%m/%d %H:%M:%S.%f')
                        e_end_date = annotation['Event'][i]['End_Time']
                        e_end_datetime = datetime.datetime.strptime(e_end_date, '%Y/%m/%d %H:%M:%S.%f')

                        if s_datetime <= e_start_datetime and e_datetime >= e_end_datetime:
                            action_start = e_start_datetime - s_datetime
                            action_end = e_end_datetime - s_datetime
                            if e_type in labels:
                                label_idx = labels.index(e_type)
                            temp = [video_name, e_type, label_idx, round(action_start.seconds+(action_start.microseconds/1000000), 1), round(action_end.seconds+(action_end.microseconds/1000000), 1), 
                                      int(action_start.seconds * frame_rate), int(action_end.seconds * frame_rate), duration, video_frame, frame_rate, 
                                     e_start_datetime, e_end_datetime]


                    else:
                        if annotation['Event'][i]['Event_Label'] == e_type:
                            duration += annotation['Event'][i]['Duration(second)']

                            e_end_date = annotation['Event'][i]['End_Time']
                            e_end_datetime = datetime.datetime.strptime(e_end_date, '%Y/%m/%d %H:%M:%S.%f')

                            if s_datetime <= e_start_datetime and e_datetime >= e_end_datetime:
                                action_start = e_start_datetime - s_datetime
                                action_end = e_end_datetime - s_datetime
                                if e_type in labels:
                                    label_idx = labels.index(e_type)
                                temp = [video_name, e_type, label_idx, round(action_start.seconds+(action_start.microseconds/1000000), 1), round(action_end.seconds+(action_end.microseconds/1000000), 1), 
                                          int(action_start.seconds * frame_rate), int(action_end.seconds * frame_rate), duration, video_frame, frame_rate, 
                                         e_start_datetime, e_end_datetime]


                        else:

                            all_datas.append(temp)

                            e_type = annotation['Event'][i]['Event_Label']
                            duration = annotation['Event'][i]['Duration(second)']

                            e_start_date = annotation['Event'][i]['Start_Time']
                            e_start_datetime = datetime.datetime.strptime(e_start_date, '%Y/%m/%d %H:%M:%S.%f')

                            e_end_date = annotation['Event'][i]['End_Time']
                            e_end_datetime = datetime.datetime.strptime(e_end_date, '%Y/%m/%d %H:%M:%S.%f')
                            if s_datetime <= e_start_datetime and e_datetime >= e_end_datetime:
                                action_start = e_start_datetime - s_datetime
                                action_end = e_end_datetime - s_datetime
                                if e_type in labels:
                                    label_idx = labels.index(e_type)
                                temp = [video_name, e_type, label_idx, round(action_start.seconds+(action_start.microseconds/1000000), 1), round(action_end.seconds+(action_end.microseconds/1000000), 1), 
                                          int(action_start.seconds * frame_rate), int(action_end.seconds * frame_rate), duration, video_frame, frame_rate, 
                                         e_start_datetime, e_end_datetime]
                if annotation['Event'][i]['Event_Label'] == 'Analysis Stop':
                    if all_datas[-1] != temp:
                        all_datas.append(temp)
                        break
                    else:
                        break
                                            
         # save to csv
    col = ['video-name', 'event_type', 'label_idx', 'start', 'end', 'startFrame', 'endFrame', 'duration', 'video-frame', 'frame-rate',
          'action-start', 'action-end']
    data_df = pd.DataFrame(all_datas, columns=col)
    
    # save path
    data_df.to_csv('./annotation_outputs/' + video_name+'.csv', index = False)
           
                
