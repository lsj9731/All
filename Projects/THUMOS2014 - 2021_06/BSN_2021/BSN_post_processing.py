# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:21:39 2017

@author: wzmsltw
"""
import random
import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate
import scipy
import pandas as pd
import pandas
import numpy
import json
import time
import pp
import matplotlib.pyplot as plt


def IOU(s1,e1,s2,e2):
    if (s2>e1) or (s1>e2):
        return 0
    Aor=max(e1,e2)-min(s1,s2)
    Aand=min(e1,e2)-max(s1,s2)
    return float(Aand)/Aor

def NMS(df,nms_threshold):
    df=df.sort(columns="score",ascending=False)
    
    tstart=list(df.xmin.values[:])
    tend=list(df.xmax.values[:])
    tscore=list(df.score.values[:])
    rstart=[]
    rend=[]
    rscore=[]
    while len(tstart)>1:
        idx=1
        while idx<len(tstart):
            if IOU(tstart[0],tend[0],tstart[idx],tend[idx])>nms_threshold:
                tstart.pop(idx)
                tend.pop(idx)
                tscore.pop(idx)
            else:
                idx+=1
        rstart.append(tstart[0])
        rend.append(tend[0])
        rscore.append(tscore[0])
        tstart.pop(0)
        tend.pop(0)
        tscore.pop(0)
    newDf=pd.DataFrame()
    newDf['score']=rscore
    newDf['xmin']=rstart
    newDf['xmax']=rend
    return newDf

def Soft_NMS(df,nms_threshold):
    df=df.sort_values(by="score",ascending=False)
    
    tstart=list(df.xmin.values[:])
    tend=list(df.xmax.values[:])
    tscore=list(df.score.values[:])
    
    rstart=[]
    rend=[]
    rscore=[]

    while len(tscore)>1 and len(rscore)<1500:
        max_index=tscore.index(max(tscore))
        for idx in range(0,len(tscore)):
            if idx!=max_index:
                tmp_iou=IOU(tstart[max_index],tend[max_index],tstart[idx],tend[idx])
                tmp_width=tend[max_index]-tstart[max_index]
                tmp_width=tmp_width/300
                if tmp_iou>0.5+0.3*tmp_width:#*1/(1+np.exp(-max_index)):
                    tscore[idx]=tscore[idx]*np.exp(-np.square(tmp_iou)/0.75)
        
        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
                
    newDf=pd.DataFrame()
    newDf['score']=rscore
    newDf['xmin']=rstart
    newDf['xmax']=rend
    return newDf


def min_max(x):
    x=(x-min(x))/(max(x)-min(x))
    return x

dataSet="Test"
annoDf=pd.read_csv("./data/thumos_14_annotations/thumos14_test_groundtruth.csv")
videoNameList=list(set(annoDf["video-name"].values[:]))
random.shuffle(videoNameList)
nms_threshold=0.75

xmin_list=[]
xmax_list=[]
score_list=[]
frame_list=[]
video_list=[]

for videoName in videoNameList:
    videoAnno=annoDf[annoDf["video-name"]==videoName]
    videoFrame=videoAnno["video-frames"].values[0]
    #break
    df=pd.read_csv("./output/PEM_results/"+videoName+".csv")
    df['score']=df.iou_score.values[:]*df.xmin_score.values[:]*df.xmax_score.values[:]#


    sdf=Soft_NMS(df,0.5)

    proposal_list=[]

    #for j in range(min(200,len(sdf))):
    for j in range(min(1500,len(sdf))):
        xmin_list.append(sdf.xmin.values[j])
        xmax_list.append(sdf.xmax.values[j])
        score_list.append(sdf.score.values[j])
        frame_list.append(videoFrame)
        video_list.append(videoName)


outDf=pd.DataFrame()
outDf["f-end"]=xmax_list
outDf["f-init"]=xmin_list
outDf["score"]=score_list
outDf["video-frames"]=frame_list
outDf["video-name"]=video_list
    
outDf.to_csv("./output/thumos14_bsn_results.csv",index=False)


