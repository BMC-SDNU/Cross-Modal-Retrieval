#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 22:10:05 2017

@author: zhan
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tcl
import random
import scipy.io as sci
from sklearn import preprocessing
from tensorflow.contrib.layers.python.layers import initializers
from deepnet.fxeval import *
import os,sys
import datetime, time

def computer_av_precision(test, database, label, R):
    m, n = np.shape(database)
    m1, n1 = np.shape(test)
    av_precision = np.zeros([m1, 1])
    for i in range(m1):
        distance = np.sum(np.square(test[i] - database), 1)
        sort = np.argsort(distance)
        cumulate = 0.0
        tp_counter = 0.0
        for j in range(R):
            if np.sum(np.abs(label[sort[j]] - label[i])) == 0:
                tp_counter += 1.0
                cumulate = cumulate + (float(tp_counter)/ float(j+1))
        
        if tp_counter !=0:
            av_precision[i] = cumulate/float(tp_counter)
    mean_precision = np.mean(av_precision)
    return mean_precision
def computer_av_precision3(test, database, label, R):
    m, n = np.shape(database)
    m1, n1 = np.shape(test)
    av_precision = np.zeros([m1, 1])
    for i in range(m1):
        h1 = np.multiply(test[i,:], np.ones([len(database), 1]))
        distance = 1.0 - np.sum(np.multiply(h1, database),1)/ \
                          np.sqrt(np.sum(np.square(h1),1))/ \
                          np.sqrt(np.sum(np.square(database),1))
        sort = np.argsort(distance)
        cumulate = 0.0
        tp_counter = 0.0
        for j in range(R):
            if np.sum(np.abs(label[sort[j]] - label[i])) == 0:
                tp_counter += 1.0
                cumulate = cumulate + (float(tp_counter)/ float(j+1))
        
        if tp_counter !=0:
            av_precision[i] = cumulate/float(tp_counter)
    mean_precision = np.mean(av_precision)
    return mean_precision
def Label_transfor(y, size):
    m= len(y)
    Label = np.zeros([m, size])
    for i in range(m):
        Label[i, y[i]-1] = 1
    return Label       
def distance(x, y, name):
    if name == 'COS':
        dis =  1.0 - np.sum(np.multiply(x, y),1)/ \
                          np.sqrt(np.sum(np.square(x),1))/ \
                          np.sqrt(np.sum(np.square(y),1))
    else:
        dis = np.sum(np.square(np.subtract(x, y)),1)
    return dis     
class celebA():
    def __init__(self, Csize = 512):
        data1 = np.load('data/wiki_MM_10k.npz')
        self.Tsize = len(data1['T_tr'][0])
        self.Isize = len(data1['I_tr'][0])
        self.Csize = Csize
        self.chooseI = np.arange(0, self.Isize, 1)
        self.chooseT = np.arange(0, self.Tsize, 1)


        scaler = preprocessing.StandardScaler().fit(data1['I_tr'])

        self.Tr_I = scaler.transform(data1['I_tr'])#/ np.sqrt(self.Isize) 

        self.Tr_T = data1['T_tr'] #/scale
        self.Tr_L = data1['L_tr']


        self.Vl_I = scaler.transform(data1['I_vl'])#/ np.sqrt(self.Isize)
        self.Vl_T =  data1['T_vl']#/scale 
        self.Vl_L = data1['L_vl']

        self.Te_I = scaler.transform(data1['I_te'])#/ np.sqrt(self.Isize)

        self.Te_T =  data1['T_te']#/scale 
        self.Te_L = data1['L_te']
          

        self.batch_count = 0
        self.numberT = np.arange(0, len(self.Tr_T), 1)
        self.numberI = np.arange(0, len(self.Tr_I), 1)
        
    def __call__(self,batch_size = 2, sI1 = 0.5, sT1 = 0.5):
        batch_number = len(self.Tr_I)/batch_size
        if self.batch_count < batch_number-1:
            self.batch_count += 1
        else:
            self.batch_count = 0
            random.shuffle(self.numberT)
            random.shuffle(self.numberI)
        num = self.numberT[self.batch_count*batch_size:(self.batch_count+1)*batch_size]    
        num1 = self.numberI[self.batch_count*batch_size:(self.batch_count+1)*batch_size]    

        random.shuffle(self.chooseI)
        random.shuffle(self.chooseT)
        
        p = int(sT1*self.Tsize)
        p1 = int(sI1*self.Isize)
        
        epi = np.random.uniform(-1e-9, 1e-9)
        epi1 = 0
        T = self.Tr_T[num,:] +epi
        T_n = np.array(T)
        T_n[:,self.chooseT[:p]] = epi1
        I = self.Tr_I[num,:]+epi
        I_n = np.array(I)
        I_n[:,self.chooseI[:p1]] = epi1
                    
        T1 = self.Tr_T[num1,:]+epi
        T1_n = np.array(T1)
        T1_n[:,self.chooseT[:p]] = epi1
        I1 = self.Tr_I[num1,:]+epi
        I1_n = np.array(I1)
        I1_n[:,self.chooseI[:p1]] = epi1      
        name = 'COS'
        DI = np.reshape(distance(I, I1, name),  [ batch_size, 1])
        DT = np.reshape(distance(T, T1, name),  [ batch_size, 1])

        D = np.sqrt(DI* DT + 0.00001)
       # D = D/np.mean(D)
        return I, I_n, I1, I1_n, T, T_n, T1, T1_n, D, D
def cos_dis(x, y):
    dis = 1.0 - tf.reduce_sum(tf.multiply(x, y),1)/ \
                          tf.sqrt(tf.reduce_sum(np.square(x),1))/ \
                          tf.sqrt(tf.reduce_sum(np.square(y),1))
    return dis     
def edu_dis(x, y):
    dis = tf.reduce_sum(tf.square(x - y),1)
    return dis 
def auto_loss(x, y):
    loss = tf.reduce_mean(edu_dis(x, y))#/tf.reduce_sum(tf.square(y),1)) 
    return loss 
def euc_loss(x, y):
    loss = tf.reduce_mean(edu_dis(x, y)) 
    return loss
def cos_loss(x, y):
    loss = tf.reduce_mean(cos_dis(x, y)) 
    return loss 
if __name__ == '__main__':
    batch = 64
    Csize = 1024
    weight_decay = 0.5
    lr = 1e-4
    data = celebA(Csize)
    with tf.name_scope('inputs'):
        Image = tf.placeholder(tf.float32, shape=[None, data.Isize], name = "Image") 
        Image_n = tf.placeholder(tf.float32, shape=[None, data.Isize], name = "Image_n") 
        Image1 = tf.placeholder(tf.float32, shape=[None, data.Isize], name = "Image1") 
        Image1_n = tf.placeholder(tf.float32, shape=[None, data.Isize], name = "Image_n") 
       
        Text = tf.placeholder(tf.float32, shape=[None, data.Tsize], name ="Text")
        Text_n = tf.placeholder(tf.float32, shape=[None, data.Tsize], name ="Text_n")
        Text1 = tf.placeholder(tf.float32, shape=[None, data.Tsize], name ="Text1")
        Text1_n = tf.placeholder(tf.float32, shape=[None, data.Tsize], name ="Text1_n")
        
        DI = tf.placeholder(tf.float32, shape = [None, 1], name ="DI")
        DT = tf.placeholder(tf.float32, shape = [None, 1], name ="DT")
    with tf.name_scope('encoder1'):   
        
        wi1 = tf.get_variable(name='weight_encoder_i1', shape=[data.Isize, data.Csize], 
                regularizer=tcl.l2_regularizer(weight_decay),
                initializer = initializers.xavier_initializer(),
                trainable=True)
        bi1 = tf.get_variable(name='bias_encoder_i1', shape=[1, data.Csize],
                initializer = tf.zeros_initializer(),
                trainable=True)
        wt1 = tf.get_variable(name='weight_encoder_t1', shape=[data.Tsize, data.Csize], 
                regularizer=tcl.l2_regularizer(weight_decay),
                initializer = initializers.xavier_initializer(),
                trainable=True)
        bt1 = tf.get_variable(name='bias_encoder_t1', shape=[1, data.Csize],
                initializer = tf.zeros_initializer(),
                trainable=True)
        
        i =  tf.matmul(Image_n, wi1) + bi1 
        i_n = tf.nn.tanh(i)
        t= tf.matmul(Text_n, wt1) + bt1 
        t_n = tf.nn.tanh(t)
        
        i1 =  tf.matmul(Image1_n, wi1) + bi1 
        i1_n = tf.nn.tanh( i1 )
        t1 = tf.matmul(Text1_n, wt1) + bt1 
        t1_n = tf.nn.tanh(t1)             
        
    filename = 'wikiMM_net11/best.ckpt'
    saver = tf.train.Saver({"weight_encoder_i1":wi1, "bias_encoder_i1": bi1,
                            'weight_encoder_t1': wt1, "bias_encoder_t1": bt1,   
                           })
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    save = saver.restore(sess, filename)
    begin = time.time()
    I_te= sess.run(
                   i_n, 
                   feed_dict={Image_n: data.Te_I})

    T_te= sess.run(
                   t_n, 
                   feed_dict={Text_n: data.Te_T})
    I_vl= sess.run(
                   i_n, 
                   feed_dict={Image_n: data.Vl_I})

    T_vl= sess.run(
                   t_n, 
                   feed_dict={Text_n: data.Vl_T})
    I_tr= sess.run(
                   i_n, 
                   feed_dict={Image_n: data.Tr_I})

    T_tr= sess.run(
                   t_n, 
                   feed_dict={Text_n: data.Tr_T})
    sci.savemat('best_data', 
                {'I_tr': I_tr, 'T_tr':T_tr, 'L_tr':data.Tr_L, 'I_te': I_te, 'T_te':T_te, 'L_te':data.Te_L,'I_vl': I_vl, 'T_vl':T_vl, 'L_vl':data.Vl_L})
    end = time.time()
    print (end-begin)