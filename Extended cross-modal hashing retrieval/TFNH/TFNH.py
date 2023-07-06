from setting import *
from ops import *
import scipy.io as sio
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import matplotlib.pyplot as plt
from calc_hammingranking import calc_map

class TFNH(object):
    def __init__(self, sess):
        self.I_tr = I_tr
        self.T_tr = T_tr
        self.L_tr = L_tr
        self.I_te = I_te
        self.T_te = T_te
        self.L_te = L_te
        
        self.pair = pair
        self.pair_batch_size = pair_batch_size
        self.img_batch_size = img_batch_size
        self.txt_batch_size = txt_batch_size
        self.BATCH_NUM = BATCH_NUM
        self.batch_size = batch_size
        self.TOTAL_EPOCH = TOTAL_EPOCH
        self.all_num = all_num
        
        self.hid_dim = hid_dim
        self.hash_dim = hash_dim
        self.dis_dim = dis_dim
        self.dis_out_dim = dis_out_dim        
        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.lab_dim = lab_dim
        self.fusion_dim = fusion_dim
        
        self.lr_fn = lr_fn
        self.lr_dn = lr_dn
        self.kp = kp
        
        self.fusion_net = fusion_net
        self.classification_net = classification_net
        self.discriminative_net1 = discriminative_net1
        self.discriminative_net2 = discriminative_net2
        
        self.build_model()        
        self.sess = sess
         
    def build_model(self):
        self.ph = {}
        self.ph['fusion_input1'] = tf.placeholder(tf.float32, [None, self.fusion_dim])
        self.ph['fusion_input2'] = tf.placeholder(tf.float32, [None, self.fusion_dim])
        self.ph['fusion_input3'] = tf.placeholder(tf.float32, [None, self.fusion_dim])
        self.ph['lab1'] = tf.placeholder(tf.float32, [None, self.lab_dim])
        self.ph['lab2'] = tf.placeholder(tf.float32, [None, self.lab_dim])
        self.ph['lab3'] = tf.placeholder(tf.float32, [None, self.lab_dim])
        self.ph['kp'] = tf.placeholder(tf.float32)
        
        # fusion network
        self.f_feat = self.fusion_net(self.ph['fusion_input1'], self.hid_dim, self.hash_dim, self.ph['kp'])
        self.i_feat = self.fusion_net(self.ph['fusion_input2'], self.hid_dim, self.hash_dim, self.ph['kp'], reuse=True)
        self.t_feat = self.fusion_net(self.ph['fusion_input3'], self.hid_dim, self.hash_dim, self.ph['kp'], reuse=True)
        
        # classification network
        self.class_net1 = self.classification_net(self.f_feat, self.hash_dim, self.lab_dim)
        self.class_net2 = self.classification_net(self.i_feat, self.hash_dim, self.lab_dim, reuse=True)
        self.class_net3 = self.classification_net(self.t_feat, self.hash_dim, self.lab_dim, reuse=True)
        
        # feature
        self.dis_feat1 = tf.concat((self.f_feat, self.i_feat), 0)
        self.dis_feat2 = tf.concat((self.f_feat, self.t_feat), 0)
        
        # discriminator
        self.dis_net1 = self.discriminative_net1(self.dis_feat1, self.dis_dim, self.dis_out_dim)
        self.dis_net2 = self.discriminative_net2(self.dis_feat2, self.dis_dim, self.dis_out_dim)
        
        # gragh loss
        W1 = tf.matmul(self.ph['lab1'], tf.transpose(self.ph['lab1']))
        D1 = tf.matrix_diag(tf.reduce_sum(W1, 1))
        L1 = tf.add(W1, -D1)
        self.graph_loss1 = beta[0] * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.f_feat), L1), self.f_feat)) / (self.pair_batch_size * self.pair_batch_size)
        
        d11 = tf.reduce_sum(tf.square(self.ph['fusion_input2']), 1)
        d12 = tf.matmul(self.ph['fusion_input2'], tf.transpose(self.ph['fusion_input2']))
        dist1 = tf.transpose(-2 * d12 + d11) + d11
        wt1 = tf.cast((yita[0] * dist1) < tf.reduce_max(dist1), tf.float32)
        W2 = tf.matmul(wt1, tf.transpose(wt1))
        D2 = tf.matrix_diag(tf.reduce_sum(W2, 1))
        L2 = tf.add(W2, -D2)
        self.graph_loss2 = beta[1] * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.i_feat), L2), self.i_feat)) / (self.img_batch_size * self.img_batch_size)
        
        d21 = tf.reduce_sum(tf.square(self.ph['fusion_input3']), 1)
        d22 = tf.matmul(self.ph['fusion_input3'], tf.transpose(self.ph['fusion_input3']))
        dist2 = tf.transpose(-2 * d22 + d21) + d21
        wt2 = tf.cast((yita[1] * dist2) < tf.reduce_max(dist2), tf.float32)
        W3 = tf.matmul(wt2, tf.transpose(wt2))
        D3 = tf.matrix_diag(tf.reduce_sum(W3, 1))
        L3 = tf.add(W3, -D3)
        self.graph_loss3 = beta[2] * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.t_feat), L3), self.t_feat)) / (self.txt_batch_size * self.txt_batch_size)
        self.graph_loss = (self.graph_loss1 + self.graph_loss2 + self.graph_loss3) / 3.0
        
        # classification loss
        self.class_loss1 = tf.reduce_mean(tf.nn.l2_loss(self.ph['lab1']-self.class_net1))
        self.class_loss2 = tf.reduce_mean(tf.nn.l2_loss(self.ph['lab2']-self.class_net2))
        self.class_loss3 = tf.reduce_mean(tf.nn.l2_loss(self.ph['lab3']-self.class_net3))
        self.class_loss = alpha * (self.class_loss1 + self.class_loss2 + self.class_loss3) / 3.0
        
        self.acc1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.ph['lab1'], 1), tf.argmax(self.class_net1, 1)), tf.float32))
        self.acc2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.ph['lab2'], 1), tf.argmax(self.class_net2, 1)), tf.float32))
        self.acc3 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.ph['lab3'], 1), tf.argmax(self.class_net3, 1)), tf.float32))
        
        # discrimative loss
        pair_domain = tf.concat([tf.ones([self.pair_batch_size, 1]), tf.zeros([self.pair_batch_size, 1])], 1)
        img_domain = tf.concat([tf.zeros([self.img_batch_size, 1]), tf.ones([self.img_batch_size, 1])], 1)
        txt_domain = tf.concat([tf.zeros([self.txt_batch_size, 1]), tf.ones([self.txt_batch_size, 1])], 1)
        domain_label1 = tf.concat([pair_domain, img_domain], 0)
        domain_label2 = tf.concat([pair_domain, txt_domain], 0)
        domain_loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.dis_net1, labels=domain_label1)
        domain_loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.dis_net2, labels=domain_label2)
        self.domain_loss1 = mu * tf.reduce_mean(domain_loss1)
        self.domain_loss2 = mu * tf.reduce_mean(domain_loss2)
        
        self.total_loss = self.graph_loss + self.class_loss - self.domain_loss1 - self.domain_loss2
        
        # get variables
        self.all_vars = tf.trainable_variables()
        self.fn_vars = [v for v in self.all_vars if 'fn_' in v.name]
        self.cl_vars = [v for v in self.all_vars if 'cl_' in v.name]
        self.dn1_vars = [v for v in self.all_vars if 'dis1_' in v.name]
        self.dn2_vars = [v for v in self.all_vars if 'dis2_' in v.name]
        
    def train_model(self):
        train_fn = tf.train.AdamOptimizer(learning_rate=self.lr_fn).minimize(self.total_loss, var_list=self.fn_vars+self.cl_vars)
        train_dn1 = tf.train.AdamOptimizer(learning_rate=self.lr_dn).minimize(self.domain_loss1, var_list=self.dn1_vars)
        train_dn2 = tf.train.AdamOptimizer(learning_rate=self.lr_dn).minimize(self.domain_loss2, var_list=self.dn2_vars)
        
        init = tf.global_variables_initializer()
        
        self.sess.run(init)   
        for epoch in range(self.TOTAL_EPOCH):
            for batch in range(self.BATCH_NUM):
                pair_batch = np.hstack((self.I_tr[batch*self.pair_batch_size: (batch+1)*self.pair_batch_size,], self.T_tr[batch*self.pair_batch_size: (batch+1)*self.pair_batch_size,]))
                img_batch = np.hstack((self.I_tr[batch*self.img_batch_size: (batch+1)*self.img_batch_size,], np.zeros([self.img_batch_size, self.txt_dim])))
                txt_batch = np.hstack((np.zeros([self.txt_batch_size, self.img_dim]), self.T_tr[batch*self.txt_batch_size: (batch+1)*self.txt_batch_size,]))
            
                pair_y_batch = self.L_tr[batch*self.pair_batch_size: (batch+1)*self.pair_batch_size,]
                img_y_batch = self.L_tr[batch*self.img_batch_size: (batch+1)*self.img_batch_size,]
                txt_y_batch = self.L_tr[batch*self.txt_batch_size: (batch+1)*self.txt_batch_size,]

                self.sess.run([train_fn, train_dn1, train_dn2], 
                             feed_dict={self.ph['fusion_input1']: pair_batch,
                                        self.ph['fusion_input2']: img_batch,
                                        self.ph['fusion_input3']: txt_batch,
                                        self.ph['lab1']: pair_y_batch,
                                        self.ph['lab2']: img_y_batch,
                                        self.ph['lab3']: txt_y_batch,
                                        self.ph['kp']: self.kp})
            
        if self.pair:
            self.test_paired_model()
        else:
            self.test_unpaired_model()
            
    def test_paired_model(self):
        tr_input = np.hstack((self.I_tr, self.T_tr))
        te_img_input = np.hstack((self.I_te, np.zeros(self.T_te.shape)))
        te_txt_input = np.hstack((np.zeros(self.I_te.shape), self.T_te))
    
        final_feature = self.sess.run(self.f_feat, feed_dict={self.ph['fusion_input1']: tr_input, self.ph['kp']: 1.0})
        txt_feature = self.sess.run(self.i_feat, feed_dict={self.ph['fusion_input2']: te_txt_input, self.ph['kp']: 1.0})
        img_feature = self.sess.run(self.t_feat, feed_dict={self.ph['fusion_input3']: te_img_input, self.ph['kp']: 1.0})
        
        Hash = np.sign(final_feature)
        H_txt = np.sign(txt_feature)
        H_img = np.sign(img_feature)
        
        map_t2i = calc_map(H_txt, Hash, self.L_te, self.L_tr)
        map_i2t = calc_map(H_img, Hash, self.L_te, self.L_tr)
    
        print('paired--------------------------------')
        print('hash_dim = ' + str(self.hash_dim) + ', alpha = ' + str(alpha) + ', beta = ' + str(beta) + ', mu = ' + str(mu))
        print('mapi2t = ' + str(map_i2t) + ', mapt2i = ' + str(map_t2i))
    
    def test_unpaired_model(self):
        num_img = self.img_batch_size * self.BATCH_NUM
        num_txt = self.txt_batch_size * self.BATCH_NUM
        num_fusion = self.pair_batch_size * self.BATCH_NUM
        
        if num_img == 0:
            num_img = self.all_num
        if num_txt == 0:
            num_txt = self.all_num
            
        new_I = np.hstack((self.I_tr[: num_img], np.zeros([num_img, self.txt_dim])))
        new_T = np.hstack((np.zeros([num_txt, self.img_dim]), self.T_tr[: num_txt]))
        tr_input = np.hstack((self.I_tr, self.T_tr))
        te_img_input = np.hstack((self.I_te, np.zeros(self.T_te.shape)))
        te_txt_input = np.hstack((np.zeros(self.I_te.shape), self.T_te))
        
        I_L = self.L_tr[: num_img]
        T_L = self.L_tr[: num_txt]
        new_I_retrieval = self.sess.run(self.i_feat, feed_dict={self.ph['fusion_input2']: new_I, self.ph['kp']: 1.0})
        new_T_retrieval = self.sess.run(self.t_feat, feed_dict={self.ph['fusion_input3']: new_T, self.ph['kp']: 1.0})
        final_feature = self.sess.run(self.f_feat, feed_dict={self.ph['fusion_input1']: tr_input, self.ph['kp']: 1.0})
        txt_feature = self.sess.run(self.i_feat, feed_dict={self.ph['fusion_input2']: te_txt_input, self.ph['kp']: 1.0})
        img_feature = self.sess.run(self.t_feat, feed_dict={self.ph['fusion_input3']: te_img_input, self.ph['kp']: 1.0})
        
        Hash = np.sign(final_feature)
        H_txt = np.sign(txt_feature)
        H_img = np.sign(img_feature)
        new_R_I = np.sign(new_I_retrieval)
        new_R_T = np.sign(new_T_retrieval)
        new_R_I[:num_fusion,] = Hash[:num_fusion,]
        new_R_T[:num_fusion,] = Hash[:num_fusion,]
    
        new_map_i2t = calc_map(H_img, new_R_T, self.L_te, T_L)
        new_map_t2i = calc_map(H_txt, new_R_I, self.L_te, I_L)
    
        print('unpaired--------------------------------')
        print('hash_dim = ' + str(self.hash_dim) + ', alpha = ' + str(alpha) + ', beta = ' + str(beta) + ', mu = ' + str(mu))
        print('mapi2t = ' + str(new_map_i2t) + ', mapt2i = ' + str(new_map_t2i))
        
        
        