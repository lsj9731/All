# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 20:25:55 2017

@author: wzmsltw
"""

import tensorflow as tf
import numpy as np
import pandas as pd

import time
import PEM_load_data




def APN_Train(X_feature_action,X_feature_start,X_feature_end,Y_iou,LR,config):
    
    X=tf.concat((X_feature_action,X_feature_start,X_feature_end),axis=1)
    net=0.1*tf.matmul(X, config.W["iou_0"]) + config.biases["iou_0"]
    net=tf.nn.relu(net)
    net=0.1*tf.matmul(net, config.W["iou_1"]) + config.biases["iou_1"]
    net=tf.nn.sigmoid(net)
    
    anchors_iou=tf.reshape(net,[-1])

    return anchors_iou

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """
    def __init__(self):
        #common information
        self.training_epochs = 20
        
        self.input_steps=256
        self.learning_rates=[0.001]*10+[0.0001]*10
        #self.lambda_loss_amount = 0.000075
        
        self.num_random=10
        self.batch_size=16
        self.u_ratio_m=1
        self.u_ratio_l=2
        
        with tf.variable_scope("latent_net"):
            self.W = {
                'iou_0': tf.Variable(tf.truncated_normal([48, 256])),
                'iou_1': tf.Variable(tf.truncated_normal([256, 1]))}
            self.biases = {
                'iou_0': tf.Variable(tf.truncated_normal([256])),
                'iou_1': tf.Variable(tf.truncated_normal([1]))}

        with tf.variable_scope("comb_weight"):
            self.alpha=tf.Variable([1./3,1./3,1./3])




if __name__ == "__main__":
    config = Config()
    
    X_feature_action = tf.placeholder(tf.float32, [None,16])
    X_feature_start = tf.placeholder(tf.float32, [None,16])
    X_feature_end = tf.placeholder(tf.float32, [None,16])
    Y_iou=tf.placeholder(tf.float32,[None])
    LR= tf.placeholder(tf.float32)
    
    
    prop_score=APN_Train(X_feature_action,X_feature_start,X_feature_end,
                                                      Y_iou,LR,config)
    
    model_saver=tf.train.Saver(var_list=tf.trainable_variables(),max_to_keep=80)
    sess=tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    tf.global_variables_initializer().run()  
    model_saver.restore(sess,"models/PEM/pem_model_epoch-19")  

    dataSet="Test"
    videoDataTest=PEM_load_data.getFullData(dataSet,flag_test=True) 
    annoDf=pd.read_csv("./data/annotations/"+dataSet+"_Annotation.csv")
    videoNameList=list(set(annoDf.video.values[:]))

    for videoName in videoNameList:
        batch_feature_action=videoDataTest[videoName]["feature_action"]
        batch_feature_start=videoDataTest[videoName]["feature_start"]
        batch_feature_end=videoDataTest[videoName]["feature_end"]
        
        out_score=sess.run(prop_score,feed_dict={X_feature_action:batch_feature_action,
                                                          X_feature_start:batch_feature_start,
                                                          X_feature_end:batch_feature_end})  
                                                          

        out_score=np.reshape(out_score,[-1])
        

        xmin_list=videoDataTest[videoName]["xmin"]
        xmax_list=videoDataTest[videoName]["xmax"]
        xmin_score_list=videoDataTest[videoName]["xmin_score"]
        xmax_score_list=videoDataTest[videoName]["xmax_score"]
        match_iou = videoDataTest[videoName]["match_iou"]
        match_ioa = videoDataTest[videoName]["match_ioa"]
        match_xmin_list = videoDataTest[videoName]["match_xmin"]
        match_xmax_list = videoDataTest[videoName]["match_xmax"]

        latentDf=pd.DataFrame()
        latentDf["xmin"]=xmin_list
        latentDf["xmax"]=xmax_list
        latentDf["xmin_score"]=xmin_score_list
        latentDf["xmax_score"]=xmax_score_list
        latentDf["iou_score"]=out_score
        latentDf["match_iou"]=match_iou
        latentDf["match_ioa"]=match_ioa
        latentDf["match_xmin"]=match_xmin_list
        latentDf["match_xmax"]=match_xmax_list
           
        latentDf.to_csv("./output/PEM_results/"+videoName+".csv",index=False)