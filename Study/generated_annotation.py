import numpy as np
import pandas as pd
from collections import Counter
import json
from datetime import datetime, timedelta
import datetime
import os

labels = ['Wake', 'N1', 'N2', 'N3', 'REM', 'REM Sleep Behavior', 'Apnea Central', 'Apnea Mixed', 'Apnea Obstructive', 'Hypopnea', 
          'Snore Train', 'Desat', 'Spontaneous Arousal', 'Respiratory Arousal', 'RERA', 'LM Arousal', 'PLM Arousal', 'LM', 'PLM', 
          'Supine', 'Prone', 'Left', 'Right', 'Upright','Artifact', 'Error', 'Note', 'Warning']

# Json 경로 지정
Json_list = os.listdir('./annotation/')


for data in Json_list:
    count_start_end = []
    annotation = json.load(open('./annotation/'+data))
    video_name = annotation['Case_Info']['Case_Number']
    
    # 하나의 Annotation 일 때 수행
    if len(annotation['Video_Info']) == 1:
        video_frame = annotation['Video_Info'][0]['Frame_Count']
        
        # get video start time
        start_date = annotation['Video_Info'][0]['Start']
        end_date = annotation['Video_Info'][0]['End']
        s_datetime = datetime.datetime.strptime(start_date, '%Y/%m/%d %H:%M:%S.%f')
        e_datetime = datetime.datetime.strptime(end_date, '%Y/%m/%d %H:%M:%S.%f')
        
        # DataFrame으로 만들어주기 위한 데이터 누적
        all_datas = []
        for i in range(len(annotation['Event'])):
            if annotation['Event'][i]['Event_Label'] == 'Analysis Stop':
                break

            e_type = annotation['Event'][i]['Event_Label']
            duration = annotation['Event'][i]['Duration(second)']

            e_start_date = annotation['Event'][i]['Start_Time']
            e_start_datetime = datetime.datetime.strptime(e_start_date, '%Y/%m/%d %H:%M:%S.%f')

            e_end_date = annotation['Event'][i]['End_Time']
            e_end_datetime = datetime.datetime.strptime(e_end_date, '%Y/%m/%d %H:%M:%S.%f')

            if s_datetime <= e_start_datetime and e_datetime >= e_end_datetime:
                action_start = e_start_datetime - start_datetime
                action_end = e_end_datetime - start_datetime
                if e_type in labels:
                    label_idx = labels.index(e_type)
                all_datas.append([video_name, e_type, label_idx, action_start.seconds, action_end.seconds, 
                                  action_start.seconds * 4.995, action_end.seconds * 4.995, duration, video_frame, 4.995, 
                                 e_start_datetime, e_end_datetime])
    else:
        video_frame = 0
        for i in range(len(annotation['Video_Info'])):
            video_frame += annotation['Video_Info'][i]['Frame_Count']
            count_start_end.append([annotation['Video_Info'][i]['Start'], annotation['Video_Info'][i]['End']])
    
        # get video start time
        start_date = annotation['Video_Info'][0]['Start']
        start_datetime = datetime.datetime.strptime(start_date, '%Y/%m/%d %H:%M:%S.%f')
        
        all_datas = []
        for j in range(len(annotation['Video_Info'])):
            c_start, c_end = count_start_end[j][0], count_start_end[j][1]
            
            s_datetime = datetime.datetime.strptime(c_start, '%Y/%m/%d %H:%M:%S.%f')
            e_datetime = datetime.datetime.strptime(c_end, '%Y/%m/%d %H:%M:%S.%f')
            
            # DataFrame으로 만들어주기 위한 데이터 누적
            for i in range(len(annotation['Event'])):
                if annotation['Event'][i]['Event_Label'] == 'Analysis Stop':
                    break
            
                e_type = annotation['Event'][i]['Event_Label']
                duration = annotation['Event'][i]['Duration(second)']

                e_start_date = annotation['Event'][i]['Start_Time']
                e_start_datetime = datetime.datetime.strptime(e_start_date, '%Y/%m/%d %H:%M:%S.%f')

                e_end_date = annotation['Event'][i]['End_Time']
                e_end_datetime = datetime.datetime.strptime(e_end_date, '%Y/%m/%d %H:%M:%S.%f')

                if s_datetime <= e_start_datetime and e_datetime >= e_end_datetime:
                    action_start = e_start_datetime - start_datetime
                    action_end = e_end_datetime - start_datetime
                    if e_type in labels:
                        label_idx = labels.index(e_type)
                    all_datas.append([video_name, e_type, label_idx, action_start.seconds, action_end.seconds, 
                                  action_start.seconds * 4.995, action_end.seconds * 4.995, duration, video_frame, 4.995, 
                                 e_start_datetime, e_end_datetime])
    
    # save to csv
    col = ['video-name', 'event_type', 'label_idx', 'start', 'end', 'startFrame', 'endFrame', 'duration', 'video-frame', 'frame-rate',
          'action-start', 'action-end']
    data_df = pd.DataFrame(all_datas, columns=col)
    
    # save path
    data_df.to_csv('./annotation_outputs_test/'+video_name+'.csv', index = False)