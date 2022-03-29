# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:21:39 2017

@author: wzmsltw
"""
import random
import scipy.interpolate
import pandas
import numpy

    
def getBatchList(numProps,batch_size,shuffle=True):
    ## notice that there are some video appear twice in last two batch ##
    propList=list(range(numProps))
    batch_start_list=[i*batch_size for i in range(len(propList)//batch_size)]
    batch_start_list.append(len(propList)-batch_size)
    if shuffle==True:
        random.shuffle(propList)
    batch_prop_list=[]
    for bstart in batch_start_list:
        batch_prop_list.append(propList[bstart:(bstart+batch_size)])
    return batch_prop_list
    

def prop_dict_data(prop_dict):
    prop_name_list=prop_dict.keys()
    
    batch_feature_action=[]
    batch_feature_start=[]
    batch_feature_end=[]
    
    batch_iou_list=[]
    batch_ioa_list=[]

    batch_xmin_list=[]
    batch_xmax_list=[]
  
    for prop_name in prop_name_list:
        batch_feature_action.append(prop_dict[prop_name]["feature_action"])
        batch_feature_start.append(prop_dict[prop_name]["feature_start"])
        batch_feature_end.append(prop_dict[prop_name]["feature_end"])
        batch_iou_list.extend(list(prop_dict[prop_name]["match_iou"]))
        batch_ioa_list.extend(list(prop_dict[prop_name]["match_ioa"]))
        batch_xmin_list.extend(list(prop_dict[prop_name]["match_xmin"]))
        batch_xmax_list.extend(list(prop_dict[prop_name]["match_xmax"]))

    batch_feature_action=numpy.concatenate(batch_feature_action)
    batch_feature_start=numpy.concatenate(batch_feature_start)
    batch_feature_end=numpy.concatenate(batch_feature_end)
    batch_iou_list=numpy.array(batch_iou_list)
    batch_ioa_list=numpy.array(batch_ioa_list)
    batch_xmin_list=numpy.array(batch_xmin_list)
    batch_xmax_list=numpy.array(batch_xmax_list)
    fullData={"feature_action":batch_feature_action,"feature_start":batch_feature_start,"feature_end":batch_feature_end,
              "iou_list":batch_iou_list,"ioa_list":batch_ioa_list,"xmin_list":batch_xmin_list,"xmax_list":batch_xmax_list}
    return fullData

def getVideoProposalData(videoName):
    num_sample_start=8
    num_sample_end=8
    num_sample_action=16
        
    adf=pandas.read_csv("./output/TEM_results/"+videoName+".csv")
    
    snippets=adf.frame.values[:]
    score_action=adf.action.values[:]
    score_start =adf.start.values[:]
    score_end   =adf.end.values[:]
    
    pdf=pandas.read_csv("./output/PGM_results/"+videoName+".csv")

    tmp_zeros=numpy.zeros([20])    
    score_action=numpy.concatenate((tmp_zeros,score_action,tmp_zeros))
    score_start=numpy.concatenate((tmp_zeros,score_start,tmp_zeros))
    score_end=numpy.concatenate((tmp_zeros,score_end,tmp_zeros))
    
    tmp_x=[5*i-87. for i in range(20)]+list(snippets)+[5*i+5+snippets[-1] for i in range(20)]
    f_action=scipy.interpolate.interp1d(tmp_x,score_action,axis=0)
    f_start=scipy.interpolate.interp1d(tmp_x,score_start,axis=0)
    f_end=scipy.interpolate.interp1d(tmp_x,score_end,axis=0)
    #break
    feature_start=[]
    feature_end=[]
    feature_action=[]   
    
    for idx in range(len(pdf)):
        xmin=pdf.xmin.values[idx]
        xmax=pdf.xmax.values[idx]
        xlen=xmax-xmin
        xmin_0=xmin-xlen/5
        xmin_1=xmin+xlen/5
        xmax_0=xmax-xlen/5
        xmax_1=xmax+xlen/5
        #start
        plen_start=(xmin_1-xmin_0)/(num_sample_start-1)
        tmp_x_new=[xmin_0+plen_start*ii for ii in range(num_sample_start)] 
        tmp_y_new_start_action=f_action(tmp_x_new)

        tmp_y_new_start_start=f_start(tmp_x_new)
        tmp_y_new_start=numpy.concatenate((tmp_y_new_start_action,tmp_y_new_start_start))
        
        #end
        plen_end=(xmax_1-xmax_0)/(num_sample_end-1)
        tmp_x_new=[xmax_0+plen_end*ii for ii in range(num_sample_end)]
        tmp_y_new_end_action=f_action(tmp_x_new)
        
        tmp_y_new_end_end=f_end(tmp_x_new)
        tmp_y_new_end=numpy.concatenate((tmp_y_new_end_action,tmp_y_new_end_end))
        #action
        plen_action=(xmax-xmin)/(num_sample_action-1)
        tmp_x_new=[xmin+plen_action*ii for ii in range(num_sample_action)]
        tmp_y_new_action=f_action(tmp_x_new)
        tmp_y_new_action=numpy.reshape(tmp_y_new_action,[-1])
        #break
        feature_start.append(tmp_y_new_start)
        feature_end.append(tmp_y_new_end)
        feature_action.append(tmp_y_new_action)
    #break
    prop_dict={"match_iou":pdf.match_iou.values[:],"match_ioa":pdf.match_ioa.values[:],
              "xmin":pdf.xmin.values[:],"xmax":pdf.xmax.values[:],"xmin_score":pdf.xmin_score.values[:],"xmax_score":pdf.xmax_score.values[:],
              "match_xmin":pdf.match_xmin.values[:],"match_xmax":pdf.match_xmax.values[:],
              "feature_start":numpy.array(feature_start),"feature_end":numpy.array(feature_end),"feature_action":numpy.array(feature_action)}

    return prop_dict

def getBatchData(fullData,batch_props):

    batch_feature_action=fullData["feature_action"][batch_props]
    batch_feature_start=fullData["feature_start"][batch_props]
    batch_feature_end=fullData["feature_end"][batch_props]
    batch_iou_list=fullData["iou_list"][batch_props]
    batch_ioa_list=fullData["ioa_list"][batch_props]
    return batch_feature_action,batch_feature_start,batch_feature_end,batch_iou_list,batch_ioa_list

def getFullData(dataSet,flag_test=False):
    annoDf=pandas.read_csv("./data/annotations/"+dataSet+"_Annotation.csv")
    videoNameList=list(set(annoDf.video.values[:]))#[:20]
    VideoData={}
    for videoName in videoNameList:
        prop_dict = getVideoProposalData(videoName)
        VideoData[videoName]=prop_dict
    
    if flag_test==False:
        fullData=prop_dict_data(VideoData)
        return fullData
    else:
        return VideoData


