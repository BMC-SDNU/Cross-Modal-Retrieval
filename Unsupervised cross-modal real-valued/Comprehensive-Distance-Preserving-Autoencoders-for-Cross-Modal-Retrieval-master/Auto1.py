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


###################################################################
# I_tr:features of training set for image data
# I_te:features of testing set for image data
# T_te:features of training set for text data
# T_te:features of testing set for text data
# L_tr:category label of training set
# L_te:category label of testing set
# Model_Paramets():data preprocessing;
#                  set part of the input elements as zeros;
#
###############################################################


# calculation the MAP
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

# calculation the MAP
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




    
def distance(x, y, name):
    if name == 'COS':
        dis =  1.0 - np.sum(np.multiply(x, y),1)/ \
                          np.sqrt(np.sum(np.square(x),1))/ \
                          np.sqrt(np.sum(np.square(y),1))
    else:
        dis = np.sum(np.square(np.subtract(x, y)),1)
    return dis 


    
class Model_Paramets():
    def __init__(self, Csize = 512):
        data1 = np.load('data/wiki_MM_10k.npz')
        self.Tsize = len(data1['T_tr'][0])
        self.Isize = len(data1['I_tr'][0])
        self.Csize = Csize
        self.chooseI = np.arange(0, self.Isize, 1)
        self.chooseT = np.arange(0, self.Tsize, 1)


        scaler = preprocessing.StandardScaler().fit(data1['I_tr'])

        self.Tr_I = scaler.transform(data1['I_tr']) 
        self.Tr_T = data1['T_tr'] 
        self.Tr_L = data1['L_tr']


        self.Vl_I = scaler.transform(data1['I_vl'])
        self.Vl_T =  data1['T_vl']
        self.Vl_L = data1['L_vl']

        self.Te_I = scaler.transform(data1['I_te'])

        self.Te_T =  data1['T_te']
        self.Te_L = data1['L_te']
          

        self.batch_count = 0
        self.numberT = np.arange(0, len(self.Tr_T), 1)
        self.numberI = np.arange(0, len(self.Tr_I), 1)
        
    def __call__(self,batch_size = 2, sI1 = 0.5, sT1 = 0.5):
        batch_size = batch_size * 2
        batch_number = len(self.Tr_I)/batch_size
        if self.batch_count < batch_number-1:
            self.batch_count += 1
        else:
            self.batch_count = 0
            random.shuffle(self.numberT)
            random.shuffle(self.numberI)
        num2 = self.numberT[self.batch_count*batch_size:(self.batch_count+1)*batch_size]    
        num = num2[:len(num2)/2]
        num1 = num2[len(num2)/2:]
        
        batch_size = batch_size/2

        random.shuffle(self.chooseI)
        random.shuffle(self.chooseT)
        
        p = int(sT1*self.Tsize)
        p1 = int(sI1*self.Isize)
        
        epi = np.random.uniform(-1e-9, 1e-9)
        epi1 = 0
        T = self.Tr_T[num,:] +epi
        T_n = np.array(T)
        T_n[:,self.chooseT[:p]] = epi1
        I = self.Tr_I[num,:]
        I_n = np.array(I)
        I_n[:,self.chooseI[:p1]] = epi1
        
        
        T1 = self.Tr_T[num1,:] +epi
        T1_n = np.array(T1)
        T1_n[:,self.chooseT[:p]] = epi1
        I1 = self.Tr_I[num1,:]
        I1_n = np.array(I1)
        I1_n[:,self.chooseI[:p1]] = epi1       
        name = 'COS'
        DI = np.reshape(distance(I, I1, name),  [ batch_size, 1])
        DT = np.reshape(distance(T, T1, name),  [ batch_size, 1])
        D = np.sqrt(DI * DT + 0.000001)      
        return I, I_n, I1, I1_n, T, T_n, T1, T1_n, D, D



def cos_dis(x, y):
    dis = 1.0 - tf.reduce_sum(tf.multiply(x, y),1)/ \
                          tf.sqrt(tf.reduce_sum(tf.square(x),1))/ \
                          tf.sqrt(tf.reduce_sum(tf.square(y),1))
    return dis   

  
def edu_dis(x, y):
    dis = tf.sqrt(tf.reduce_sum(tf.square(x - y),1))
    return dis 


def auto_loss(x, y):
    loss = tf.reduce_mean(edu_dis(x, y)) 
    return loss 


def cos_loss(x, y):
    loss = tf.reduce_mean(cos_dis(x, y)) 
    return loss 


if __name__ == '__main__':
    batch = 64
    Csize = 1024#int(sys.argv[1])
    sI = 0.5#float(sys.argv[2])
    sT = 0.7#float(sys.argv[3])
    w1 = 0.3#float(sys.argv[4])
    w2 = 0.001#float(sys.argv[5])
    weight_decay = 0.5
    lr = 1e-4
    data = Model_Paramets(Csize)
    
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
        
          
    with tf.name_scope('decoder1'):   
       
        wyi1 = tf.get_variable(name='weight_decoder_i1', shape=[data.Csize, data.Isize], 
                regularizer=tcl.l2_regularizer(weight_decay),
                initializer = initializers.xavier_initializer())
        byi1 = tf.get_variable(name='bias_decoder_i1', shape=[1, data.Isize],
                initializer = tf.zeros_initializer())      
        wyt1 = tf.get_variable(name='weight_decoder_t1', shape=[data.Csize, data.Tsize], 
                regularizer=tcl.l2_regularizer(weight_decay),
                initializer = initializers.xavier_initializer())
        byt1 = tf.get_variable(name='bias_decoder_t1', shape=[1, data.Tsize],
                initializer = tf.zeros_initializer())
        
        yi1 = tf.matmul(i_n, wyi1) + byi1
        
        yi1_1 = tf.matmul(i1_n, wyi1) + byi1
             
        yt2 = tf.matmul(t_n, wyt1) + byt1
        
        yt2_1 = tf.matmul(t1_n, wyt1) + byt1
        
    with tf.device('/CPU:0'): 
        with tf.name_scope('loss'):
            loss1 = (auto_loss(yi1,Image) \
                +auto_loss(yt2,Text)  \
                +auto_loss(yi1_1,Image1) \
                +auto_loss(yt2_1,Text1))
            
            D = tf.reshape(tf.sqrt(tf.multiply(cos_dis(Image, Image1), 
                                               cos_dis(Text, Text1))+0.00001), [batch, 1])
            
            loss2 = cos_loss(i_n, t_n)+cos_loss(i1_n, t1_n)
                 
            I1 =  tf.reshape(cos_dis(i_n, i1_n), [batch, 1])
            T1 =  tf.reshape(cos_dis(t_n, t1_n), [batch, 1])
        
            I2 =  tf.reshape(cos_dis(i_n, t1_n), [batch, 1])
            T2 =  tf.reshape(cos_dis(t_n, i1_n), [batch, 1])
        
            lo2 = tf.abs(I2 - D) + tf.abs(T2 - D) 
            lo3 = tf.abs(I1 - D) +  tf.abs(T1 - D)
        
            loss3 = tf.reduce_mean(lo2)
            loss4 = tf.reduce_mean(lo3)    
        
            loss = loss2  + (loss3 + loss4) * w1 + loss1 * w2 
 
    solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
     
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    if os.path.exists('lastbest.npy'):
        s = 0#np.load('lastbest.npy')
        save_score = s
    else:
        save_score = 0
    score = 0
    best_t = np.zeros([3])
    best_v = np.zeros([3])
    filename = 'wikiMM_net11/best.ckpt'
    filename1 = str(Csize) +'_'+ str(sI) +'_'+ str(sT) +'_' + \
                str(w1) +'_'+ str(w2) 
    batch_number = len(data.Tr_I)/batch * 5
    last = 0
    begin = time.time()
    for epoch in range(batch_number*60):
        if epoch - last > batch_number*3:
            break    
        Im, Im_n, Im1, Im1_n, Tx, Tx_n, Tx1, Tx1_n, DI1, DT1 = data(batch_size =batch, sI1 = sI, sT1 = sT)
        sess.run(
            solver,
            feed_dict={Image: Im, Image_n: Im_n, Image1: Im1, Image1_n: Im1_n, 
                       Text:  Tx, Text_n:  Tx_n, Text1:  Tx1, Text1_n:  Tx1_n,
                       DI:DI1, DT:DT1}
            )   
        if epoch %batch_number == 0 :
            t_Tx, t_Im= sess.run(
                    [t_n, i_n], 
                    feed_dict={Text_n: data.Vl_T, Image_n: data.Vl_I})
            map_te_TtoI = computer_av_precision3(t_Tx, t_Im, data.Vl_L, 50)
            map_te_ItoT = computer_av_precision3(t_Im, t_Tx, data.Vl_L, 50)	    
            if epoch % 50 == 0:
                print('Iter: {} for vali; ItoT: {: .4}; TtoI: {: .4}'.format(epoch, map_te_ItoT, map_te_TtoI))
            if map_te_TtoI + map_te_ItoT > score:
                if score > save_score:
                	save = saver.save(sess, filename)
                save_score = score
                best_v[0] = map_te_TtoI
                best_v[1] = map_te_ItoT
                best_v[2] = (map_te_TtoI + map_te_ItoT)/2.0
                score = map_te_TtoI + map_te_ItoT                 
                t_Tx, t_Im= sess.run(
                    [t_n, i_n], 
                    feed_dict={Text_n: data.Te_T, Image_n: data.Te_I})
                map_te_TtoI = computer_av_precision3(t_Tx, t_Im, data.Te_L, 50)
                map_te_ItoT = computer_av_precision3(t_Im, t_Tx, data.Te_L, 50)	    
                print('Iter: {} for test; ItoT: {: .4}; TtoI: {: .4}'.format(epoch, map_te_ItoT, map_te_TtoI))
                best_t[0] = map_te_TtoI
                best_t[1] = map_te_ItoT
                best_t[2] = (map_te_TtoI + map_te_ItoT)/2.0
                last = epoch
    end = time.time()
    print (begin-end)
    f1 = open('result_validation.txt', "a")
    f1.write('Iter: ')
    f1.write(str(last))
    f1.write('\t')    
    f1.write(filename1)
    f1.write('\t')
    f1.write('T2I: ')
    f1.write(str(best_v[0]))
    f1.write('\t')
    f1.write('I2T: ')
    f1.write(str(best_v[1]))
    f1.write('\t')
    f1.write('AVG: ')
    f1.write(str(best_v[2]))
    f1.write('\n')
    f1.close() 
    f1 = open('result_test.txt', "a")
    f1.write('Iter: ')
    f1.write(str(last))
    f1.write('\t')    
    f1.write(filename1)
    f1.write('\t')
    f1.write('T2I: ')
    f1.write(str(best_t[0]))
    f1.write('\t')
    f1.write('I2T: ')
    f1.write(str(best_t[1]))
    f1.write('\t')
    f1.write('AVG: ')
    f1.write(str(best_t[2]))
    f1.write('\n')
    f1.close() 
    np.save('lastbest', save_score)


