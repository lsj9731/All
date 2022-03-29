# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 20:25:55 2017

@author: wzmsltw
"""

import tensorflow as tf
import numpy as np
import pandas as pd

import PEM_load_data
import matplotlib.pyplot as plt


def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


def pem_loss(anchors_iou,match_iou,config):



    # iou regressor
    u_hmask=tf.cast(match_iou>0.7,dtype=tf.float32)
    u_mmask=tf.cast(tf.logical_and(match_iou<=0.7,match_iou>0.3),dtype=tf.float32)
    u_lmask=tf.cast(match_iou<0.3,dtype=tf.float32)
    
    num_h=tf.reduce_sum(u_hmask)
    num_m=tf.reduce_sum(u_mmask)
    num_l=tf.reduce_sum(u_lmask)

    r_m= config.u_ratio_m * num_h/(num_m)
    r_m=tf.minimum(r_m,1)
    u_smmask=tf.random_uniform([tf.shape(u_hmask)[0]],dtype=tf.float32)
    u_smmask=u_smmask*u_mmask
    u_smmask=tf.cast(u_smmask > (1. - r_m), dtype=tf.float32)    
    
    r_l= config.u_ratio_l * num_h/(num_l)
    r_l=tf.minimum(r_l,1)
    u_slmask=tf.random_uniform([tf.shape(u_hmask)[0]],dtype=tf.float32)
    u_slmask=u_slmask*u_lmask
    u_slmask=tf.cast(u_slmask > (1. - r_l), dtype=tf.float32)  
    
    iou_weights=u_hmask+u_smmask+u_slmask
    iou_loss=abs_smooth(match_iou-anchors_iou)
    iou_loss=tf.losses.compute_weighted_loss(iou_loss,iou_weights)
    
    num_iou=[tf.reduce_sum(u_hmask),tf.reduce_sum(u_smmask),tf.reduce_sum(u_slmask)]

    loss={'iou_loss':iou_loss,'num_iou':num_iou}    

    return loss

def pem_train(X_feature_action,X_feature_start,X_feature_end,Y_iou,LR,config):
    
    X=tf.concat((X_feature_action,X_feature_start,X_feature_end),axis=1)
    net=0.1*tf.matmul(X, config.W["iou_0"]) + config.biases["iou_0"]
    net=tf.nn.relu(net)
    net=0.1*tf.matmul(net, config.W["iou_1"]) + config.biases["iou_1"]
    net=tf.nn.sigmoid(net)
    
    anchors_iou=tf.reshape(net,[-1])
    
    loss=pem_loss(anchors_iou,Y_iou,config)

    Latent_Net_trainable_variables=tf.trainable_variables()

    latent_l2 = 0.000025 * sum(tf.nn.l2_loss(tf_var) for tf_var in Latent_Net_trainable_variables)
    latent_cost=10*loss["iou_loss"]+latent_l2
    latent_optimizer=tf.train.AdamOptimizer(learning_rate=LR).minimize(latent_cost,var_list=Latent_Net_trainable_variables)
    loss["l2"]=latent_l2

    return latent_optimizer,loss,anchors_iou
    
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
      
        self.num_random=10
        self.batch_size=400
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


def plotInfo(axs,info,color):
    axs["iou_loss"].set_title("iou_loss")
    axs["l2"].set_title("l2")
    axs["iou_loss"].plot(info["iou_loss"],color)
    axs["l2"].plot(info["l2"],color)
    plt.pause(0.001)

if __name__ == "__main__":
    config = Config()
    
    X_feature_action = tf.placeholder(tf.float32, [None,16])
    X_feature_start = tf.placeholder(tf.float32, [None,16])
    X_feature_end = tf.placeholder(tf.float32, [None,16])
    Y_iou=tf.placeholder(tf.float32,[None])
    LR= tf.placeholder(tf.float32)
    
    
    latent_optimizer,loss,prop_score=pem_train(X_feature_action,X_feature_start,X_feature_end,Y_iou,LR,config)
    
    model_saver=tf.train.Saver(var_list=tf.trainable_variables(),max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement =True
    sess=tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()  


    fullDataTrain=PEM_load_data.getFullData("Val") 
    fullDataTest=PEM_load_data.getFullData("Test") 
    numPropsTrain=len(fullDataTrain["iou_list"])
    numPropsTest=len(fullDataTest["iou_list"])
    

    train_info={"iou_loss":[],"l2":[]}
    val_info={"iou_loss":[],"l2":[]}

    fig1 = plt.figure("train")
    ax1 = fig1.add_subplot(121) 
    ax2 = fig1.add_subplot(122)
    ax1.grid(True, linestyle = "-.", color = "g", linewidth = "1")  
    ax2.grid(True, linestyle = "-.", color = "g", linewidth = "1")  
    axs={"iou_loss":ax1,"l2":ax2}
    
    for epoch in range(0,config.training_epochs):
    ## TRAIN ##
        batch_prop_list=PEM_load_data.getBatchList(numPropsTrain,config.batch_size,shuffle=True)
        mini_info={"iou_loss":[],"l2":[]}
        for batch_props in batch_prop_list:
            batch_feature_action,batch_feature_start,batch_feature_end,batch_iou_list,batch_ioa_list=PEM_load_data.getBatchData(fullDataTrain,batch_props)
            _,out_loss,out_score,out_alpha=sess.run([latent_optimizer,loss,prop_score,config.alpha],feed_dict={X_feature_action:batch_feature_action,
                                                              X_feature_start:batch_feature_start,
                                                              X_feature_end:batch_feature_end,
                                                              Y_iou:batch_iou_list,
                                                              LR:config.learning_rates[epoch]})  
            mini_info["iou_loss"].append(out_loss["iou_loss"])
            mini_info["l2"].append(out_loss["l2"])

        train_info["iou_loss"].append(np.mean(mini_info["iou_loss"]))
        train_info["l2"].append(np.mean(mini_info["l2"]))
        plotInfo(axs,train_info,'r')
        
        model_saver.save(sess,"models/PEM/pem_model_epoch",global_step=epoch)
        
        batch_prop_list=PEM_load_data.getBatchList(numPropsTest,config.batch_size,shuffle=False)
        mini_info={"iou_loss":[],"l2":[]}
        for batch_props in batch_prop_list:
            batch_feature_action,batch_feature_start,batch_feature_end,batch_iou_list,batch_ioa_list=PEM_load_data.getBatchData(fullDataTest,batch_props)
            out_loss=sess.run(loss,feed_dict={X_feature_action:batch_feature_action,
                                                              X_feature_start:batch_feature_start,
                                                              X_feature_end:batch_feature_end,
                                                              Y_iou:batch_iou_list,
                                                              LR:config.learning_rates[epoch]})  
            mini_info["iou_loss"].append(out_loss["iou_loss"])
            mini_info["l2"].append(out_loss["l2"])


        val_info["iou_loss"].append(np.mean(mini_info["iou_loss"]))
        val_info["l2"].append(np.mean(mini_info["l2"]))
        plotInfo(axs,val_info,'b')