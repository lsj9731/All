import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import pickle
import time
from tqdm import tqdm
import argparse
from terminaltables import *

def iou_with_anchors(anchors_min,anchors_max,box_min,box_max):
    """Compute iou score between a ground truth and the predictions"""
    len_anchors=anchors_max-anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
  
    union_len = len_anchors - inter_len +box_max-box_min
    iou = np.divide(inter_len, union_len)
    return iou


def ioa_with_anchors(anchors_min,anchors_max,box_min,box_max):
    """Compute intersection between score a ground truth and the predictions"""
    len_anchors=anchors_max-anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    
    inter_len = np.maximum(int_xmax - int_xmin, 0.)

    scores = np.divide(inter_len, len_anchors)
    return scores

def get_result(num_class, class_length, annotation, pred):
    video_list = list(set(annotation['video-name']))
    n_label = len(list(set(annotation['label213123_idx'])))

    all_gt = [list() for i in range(class_length)]
    all_pred = [list() for i in range(class_length)]
    for i in tqdm(range(class_length)):
        time.sleep(1/n_label)
        gt_idx = annotation['label213123_idx'] == num_class[i]
        gt_video = annotation[gt_idx]

        for j in range(len(gt_video)):
            video_name, start, end, frame = gt_video.iloc[j]['video-name'], gt_video.iloc[j]['f-init'], gt_video.iloc[j]['f-end'], gt_video.iloc[j]['video-frames']
            selected_class = num_class[i]
            num_pred_idx = pred[selected_class]['video-id'] == video_name
            num_pred = pred[selected_class][num_pred_idx]
            ious = []
            matchs = []

            for k in range(len(num_pred)):
                p_start, p_end = num_pred.iloc[k]['t-start'] * frame, num_pred.iloc[k]['t-end'] * frame
                iou = iou_with_anchors(p_start, p_end, start, end)
                ioa = ioa_with_anchors(p_start, p_end, start, end)
                g_duration = end - start
                p_duration = p_end - p_start
                diff = np.abs(g_duration - p_duration)
                ious.append(iou)
                matchs.append([frame, p_start, p_end, diff, iou, ioa])
            ious = np.array(ious)
            max_idx = np.argmax(ious)
            match_datas = matchs[max_idx]
            all_gt[i].append(np.array([start, end]))
            all_pred[i].append(np.array(match_datas))
    pickle.dump(all_gt, open('ground_truth.pc', 'wb'))
    pickle.dump(all_pred, open('prediction.pc', 'wb'))

    return all_gt, all_pred    

def cal_sleep(df):
    g_start = df['G_start'].sum()
    g_end = df['G_end'].sum()
    g_duration = g_end - g_start
        
    p_start = df['P_start'].sum()
    p_end = df['P_end'].sum()
    p_duration = p_end - p_start

    return g_duration, p_duration

def print_df(gt, pred, length, N_sleep):
    col_name = ['G_start', 'G_end', 'Video_frame', 'P_start', 'P_end', 'Diff', 'IOU', 'IOA']

    g_sleep = 0
    p_sleep = 0
    TP, FP, FN = 0, 0, 0
    total_count = 0
    total_iou = 0
    total_ioa = 0
    a, b = 0, 0

    class_list = []
    iou_list = []
    ioa_list = []
    total_len_list = []
    zero_count_list = []

    all_list = []

    for i in range(length):
        total_sleep = []
        pred_arrays = pred[i]
        pred_arrays = np.around(pred_arrays, 2)
        gt_arrays = gt[i]
        gt_arrays = np.around(gt_arrays, 2)
        arrays = np.concatenate((gt_arrays, pred_arrays), axis = 1)
        arr_df = pd.DataFrame(arrays, columns=col_name)
        arr_df = arr_df.sort_values(by = ['G_start'])
        arr_df = arr_df.reindex(columns=['Video_frame', 'G_start', 'G_end', 'P_start', 'P_end', 'Diff', 'IOU', 'IOA'])
        arr_df = arr_df.reset_index(drop=True)
        
        for j in range(len(arr_df)):            
            if arr_df.iloc[j]['IOU'] == 0:
                FP += arr_df.iloc[j]['P_end'] - arr_df.iloc[j]['P_start']
                FN += arr_df.iloc[j]['G_end'] - arr_df.iloc[j]['G_start']
            else:
                if arr_df.iloc[j]['G_start'] <= arr_df.iloc[j]['P_start']:
                    FN += arr_df.iloc[j]['P_start'] - arr_df.iloc[j]['G_start']
                    if arr_df.iloc[j]['G_end'] <= arr_df.iloc[j]['P_end']:                 
                        TP += arr_df.iloc[j]['G_end'] - arr_df.iloc[j]['P_start']
                        FP += arr_df.iloc[j]['P_end'] - arr_df.iloc[j]['G_end']
                    else:
                        TP += arr_df.iloc[j]['P_end'] - arr_df.iloc[j]['P_start']
                        FN += arr_df.iloc[j]['G_end'] - arr_df.iloc[j]['P_end']
                else:
                    FP += arr_df.iloc[j]['G_start'] - arr_df.iloc[j]['P_start']
                    if arr_df.iloc[j]['G_end'] <= arr_df.iloc[j]['P_end']:
                        TP += arr_df.iloc[j]['G_end'] - arr_df.iloc[j]['G_start']
                        FP += arr_df.iloc[j]['P_end'] - arr_df.iloc[j]['G_end']
                    else:
                        TP += arr_df.iloc[j]['P_end'] - arr_df.iloc[j]['G_start']
                        FN += arr_df.iloc[j]['G_end'] - arr_df.iloc[j]['P_end']
        count = 0
        sum_iou = 0
        for ious in arr_df['IOU']:
            if ious == 0 :
                count += 1
            sum_iou += ious
        total_iou += sum_iou
        avg_iou = sum_iou / len(arr_df['IOU'])
        total_count += len(arr_df)
        
        sum_ioa = 0
        for ioas in arr_df['IOA']:
            sum_ioa += ioas
        avg_ioa = sum_ioa / len(arr_df['IOA'])
        total_ioa += sum_ioa
        
        t_frame = arr_df.iloc[0]['Video_frame'] * 50
        if i in N_sleep:
            g_sleeps, p_sleeps = cal_sleep(arr_df)
            g_sleep += g_sleeps
            p_sleep += p_sleeps
        
        class_list.append(i)
        iou_list.append(avg_iou)
        ioa_list.append(avg_ioa)
        total_len_list.append(len(arr_df))
        zero_count_list.append(count)

        a += avg_iou
        b += avg_ioa
    gt_sleep = (t_frame - g_sleep) / t_frame
    pred_sleep = (t_frame - p_sleep) / t_frame

    all_iou_avg = round(total_iou / total_count, 3)
    all_ioa_avg = round(total_ioa / total_count, 3)

    all_list.append(class_list)
    all_list.append(iou_list)
    all_list.append(ioa_list)
    all_list.append(total_len_list)
    all_list.append(zero_count_list)

    return gt_sleep, pred_sleep, all_iou_avg, all_ioa_avg ,TP, FN, FP, all_list

def main(n_class, length, N_sleep, ground_truth, prediction):
    print('DataFrame에서 값을 매칭하여 저장합니다.')
    
    if os.path.isfile('./ground_truth.pc'):
        gts = pickle.load(open('./ground_truth.pc', "rb"))
        preds = pickle.load(open('./prediction.pc', "rb"))
    else:
        gts, preds = get_result(n_class, length, ground_truth, prediction)

    print('\n 전체 통계를 가져옵니다.')
    annotation_efficiency, predicted_efficiency, IOU, IOA, tp, fn, fp, total_ = print_df(gts, preds, len(n_class), N_sleep)
    print('Done.\n')

    clss = np.array(total_[0])
    ious = np.array(total_[1])
    ioas = np.array(total_[2])
    p_len = np.array(total_[3])
    zeros = np.array(total_[4])

    accuracy = np.round((np.sum(p_len) - np.sum(zeros)) / np.sum(p_len), 3)
    recall = round(tp / (tp+fn),3)
    precision = round(tp / (tp+fp),3)
    f1_score = round(2 * ( precision * recall ) / ( precision + recall ), 3)
    print('+ ======= 최종 성능 지표 ======= +')
    print('|                                |')
    print('|  전체 평균 IOU  |   ', '{:.3f}'.format(IOU), '    |')
    print('|  전체 평균 IOA  |   ', '{:.3f}'.format(IOA), '    |')
    print('|                                |')
    print('+ =========== Score ============ +')
    print('|                                |')
    # print('|     Accuracy    |   ', '{:.3f}'.format(accuracy), '    |')
    print('|      Recall     |   ', '{:.3f}'.format(recall), '    |')
    print('|     Precision   |   ', '{:.3f}'.format(precision), '    |')
    print('|     F1-Score    |   ', '{:.3f}'.format(f1_score), '    |')
    print('|                                |') 
    print('+ ------------------------------ +')

    title = 'my_tables'
    data = [['class'], ['IOU'], ['IOA'], ['Length'], ['Zero_count'], ['clss_acc']]

    

    for i in range(len(clss)):
        data[0].append(clss[i])
        data[1].append(np.round(ious[i], 2))
        data[2].append(np.round(ioas[i], 2))
        data[3].append(p_len[i])
        data[4].append(zeros[i])
        acc = np.round(((p_len[i] - zeros[i]) / p_len[i]), 2)
        data[5].append(acc)
        

    table = AsciiTable(data, title)
    print(table.table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Output of PGCN's prediction results.")
    parser.add_argument('--gt', type=str, default='./data/Test_Annotation.csv')
    parser.add_argument('--pred', type=str, default='./data/pred_dump.pc')
    parser.add_argument('--none_sleep', nargs='+', type=int, default=[0])
    args = parser.parse_args()
    
    if args.none_sleep:
        print(f'제외할 클래스가 존재합니다. >> {args.none_sleep} << \n')
    else:
        print('제외할 클래스가 없습니다. \n')
    
    print('Annotation과 예측치를 불러옵니다.')
    gt = pd.read_csv(args.gt)
    pred = pickle.load(open(args.pred, "rb"))
    print('데이터 로드 완료.\n')
    
    all_class = list(set(gt['label_idx']))
    class_len = len(all_class)

    main(all_class, class_len, args.none_sleep, gt, pred)
    
    print("All Done.")