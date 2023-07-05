#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:42:47 2018

@author: zhan
"""
from scipy.spatial.distance import pdist, squareform, cdist
import numpy as np
import scipy.io as sci
import os,sys
import datetime

###################################################################
# I_tr:features of training set for image data
# I_te:features of testing set for image data
# T_te:features of training set for text data
# T_te:features of testing set for text data
# L_tr:category label of training set
# L_te:category label of testing set

###############################################################


def unifyKnnKernel(Z,tr_n_I, te_n_I, tr_n_T, te_n_T,k):
    x1 = np.concatenate([range(tr_n_I,tr_n_I+te_n_I),
                         range(tr_n_I+te_n_I+tr_n_T,tr_n_I+te_n_I+tr_n_T+te_n_T)]);
    x2 = np.concatenate([range(0,tr_n_I),
                         range(tr_n_I+te_n_I,tr_n_I+te_n_I+tr_n_T)]);
    y1 = np.concatenate([range(0,tr_n_I), range(tr_n_I+te_n_I,tr_n_I+te_n_I+tr_n_T)]);
    W = Z[x1,:];
    W = W[:,y1];
    W = W;
    Y = Z[x2,:];
    Y = Y[:,y1];
    Y = Y;
    KN = -np.sort(-W);
    I = np.argsort(-W);
    for i in range(0,te_n_I + te_n_T):
        k1 = np.reshape(KN[i,0:k], [1, k]);
        knn = np.concatenate([k1, np.zeros([1,tr_n_I + tr_n_T-k])],1);
        W[i,I[i,:]] = knn;
    WI = W[0:te_n_I, :];
    WT = W[te_n_I:te_n_I+te_n_T, :];

    WI_s = np.reshape(np.sum(WI, 1), [len(WI),1]);
    WT_s = np.reshape(np.sum(WT, 1), [len(WI),1]);
    WI = WI/np.tile(WI_s, [1, tr_n_I+tr_n_T]);
    WT = WT/np.tile(WT_s, [1, tr_n_T+tr_n_I]);

    #W = np.concatenate([WI,WT]);
    m = np.reshape(range(tr_n_I), [tr_n_I,1]);
    m1 = np.tile(np.concatenate([m, m]),[1,(tr_n_I+tr_n_T)]);
    Y0 = (m1 == m1.T);  
    Y1 = np.multiply(Y,(1.-Y0))+Y0;
    h = Y1;
    W_IT = np.matmul(np.matmul(WI,h), WT.T);
    
    return W_IT

def computer_av(distance, label):
    m, n = np.shape(distance)
    av_precision = np.zeros([m, 1])
    sort = np.argsort(-distance)
    for i in range(m):
        cumulate = 0.0
        tp_counter = 0.0
        for j in range(50):
            if np.sum(np.abs(label[sort[i,j]] - label[i])) == 0:
                tp_counter += 1.0
                cumulate = cumulate + (float(tp_counter)/ float(j+1))
        
        if tp_counter !=0:
            av_precision[i] = cumulate/float(tp_counter)
    mean_precision = np.mean(av_precision)
    return mean_precision  

 
if __name__ == '__main__':
    data1 = sci.loadmat('best_data.mat')  
    begin = datetime.datetime.now()
    D1 = pdist(np.concatenate([data1['I_tr'], data1['I_te'], 
                              data1['T_tr'], data1['T_te']]),'cosine');
    Z1 = 1.0-squareform(D1)/2.0;
    h = []
    p = []
    for k in range(10, 1000, 10):       
        distance = unifyKnnKernel(Z1,
                                  len(data1['I_tr']),len(data1['I_te']),
                                  len(data1['T_tr']),len(data1['T_te']),
                                  k)
        end = datetime.datetime.now()
        
        
        re1 = computer_av(distance,data1['L_te'].T)
        re2 = computer_av(distance.T, data1['L_te'].T)
        avg = (re1 + re2)/2.0
        print k
        print('The KNN test result:ItoT:{: .4}; TtoI: {: .4}; avg: {: .4}'.format(re1, re2, avg))
        

        f1 = open('knn_test.txt', "a")
        f1.write('k: ')
        f1.write(str(k))
        f1.write('\t')
        f1.write('T2I: ')
        f1.write(str(re1))
        f1.write('\t')
        f1.write('I2T: ')
        f1.write(str(re2))
        f1.write('\t')
        f1.write('AVG: ')
        f1.write(str(avg))
        f1.write('\n')
        f1.close()



