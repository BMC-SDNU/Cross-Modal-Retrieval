########################################################################################
# 2016                                                                                 #
# Collective Deep Quantization for Cross-Modal Retrieval                               #
# Details:                                                                             #
#                                                                                      #
#                                                                                      #
# Model from https://github.com/guerzh/tf_weights                                      #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

from numpy import *

import os
import tensorflow as tf
import numpy as np
import sys
from scipy.misc import imread, imresize
import time
from datetime import datetime
from multiprocessing import Pool
from sklearn.cluster import MiniBatchKMeans
import random

def get_allrel(args):
    return _get_allrel(*args)

def _get_allrel(dis, database_codes, query_codes, offset):
    print 'getting relations of: ', offset
    all_rel = np.zeros([query_codes.shape[0], database_codes.shape[0]])
    for i in xrange(query_codes.shape[0]):
        for j in xrange(database_codes.shape[0]):
            A1 = np.where(query_codes[i] == 1)[0]
            A2 = np.where(database_codes[j] == 1)[0]
            all_rel[i, j] = sum([dis[x][y] for x in A1 for y in A2])
        if i % 100 == 0:
            print offset, ' reaching: ', i
    print "allrel part ", offset
    print all_rel
    print "query codes wrong:"
    print np.sum(np.sum(query_codes, 1) != 4)
    print "database codes wrong:"
    print np.sum(np.sum(database_codes, 1) != 4)
    return all_rel

class MAPs:
    def __init__(self, C, subspace_num, subcenter_num, R):
        self.dis = []
        for i in xrange(subcenter_num * subspace_num):
            self.dis.append([])
            for j in xrange(subcenter_num * subspace_num):
                self.dis[i].append(self.distance(C[i, :], C[j, :]))
        self.dis = np.array(self.dis)
        print "all dis:"
        print self.dis
        self.C = C
        self.subspace_num = subspace_num
        self.subcenter_num = subcenter_num
        self.R = R

    ### Use dot product as the distance metric
    def distance(self, a, b):
        return np.dot(a, b)

    def get_allrel(self, database_codes, query_codes):
        print "#calc mAPs# getting all relations"
        self.all_rel = np.zeros([query_codes.shape[0], database_codes.shape[0]])
        p = 10
        batch_size = database_codes.shape[0] / p
        print 'batch size of: ', batch_size
        P = Pool(p)
        self.all_rel = np.hstack(P.map(get_allrel,
              zip([self.dis] * p,
                  [database_codes[i * batch_size: (i + 1) * batch_size, :] for i in xrange(p)],
                  [query_codes] * p,
                  [i * batch_size for i in xrange(p)])))
        P.close()
        print self.all_rel.shape
        print self.all_rel

    ### Use Symmetric Quantizer Distance (SQD) for evaluation
    def get_mAPs_SQD(self, database, query):
        self.all_rel = np.dot(np.dot(query.codes, self.C), np.dot(database.codes, self.C).T)
        ids = np.argsort(-self.all_rel, 1)
        APx = []
        query_labels = query.get_labels()
        database_labels = database.get_labels()
        print "#calc mAPs# calculating mAPs"
        for i in xrange(self.all_rel.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                APx.append(np.sum(Px * imatch) / rel)
            if i % 100 == 0:
                print "step: ", i
        print "mAPs: ", np.mean(np.array(APx))
        return np.mean(np.array(APx))
    
    ### Use Symmetric Quantizer Distance (SQD) for evaluation
    def get_mAPs_AQD(self, database, query):
        self.all_rel = np.dot(query.output, np.dot(database.codes, self.C).T)
        ids = np.argsort(-self.all_rel, 1)
        APx = []
        query_labels = query.get_labels()
        database_labels = database.get_labels()
        print "#calc mAPs# calculating mAPs"
        for i in xrange(self.all_rel.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                APx.append(np.sum(Px * imatch) / rel)
            if i % 100 == 0:
                print "step: ", i
        print "mAPs: ", np.mean(np.array(APx))
        return np.mean(np.array(APx))

    ### Directly use deep feature for evaluation, which can be regarded as the upper bound of performance.
    def get_mAPs_by_feature(self, database, query):
        self.all_rel = np.dot(query.output, database.output.T)
        ids = np.argsort(-self.all_rel, 1)
        APx = []
        query_labels = query.get_labels()
        database_labels = database.get_labels()
        print "#calc mAPs# calculating mAPs"
        for i in xrange(self.all_rel.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                APx.append(np.sum(Px * imatch) / rel)
            if i % 100 == 0:
                print "step: ", i
        print "mAPs: ", np.mean(np.array(APx))
        return np.mean(np.array(APx))

class cdq:
    def __init__(self, config):
        ### initialize the hyper-parameters
        self.device = config['device']
        self.centers_device = config['centers_device']
        self.txt_dim = config['txt_dim']
        self.label_dim = config['label_dim']
        self.output_dim = config['output_dim']
        self.subspace_num = config['n_subspace']
        self.subcenter_num = config['n_subcenter']
        self.shuffle = config['shuffle']
        self.batch_size = config['batch_size']
        self.code_batch_size = config['code_batch_size']
        self.n_train = config['n_train']
        self.n_database = config['n_database']
        self.n_query = config['n_query']
        self.cq_lambda = config['cq_lambda']

        self.max_iter = config['max_iter']
        self.max_epoch = config['training_epoch']
        self.img_model = config['img_model']
        self.txt_model = config['txt_model']
        self.train_stage = config.get('train', True)

        self.max_iter_update_Cb = config['max_iter_update_Cb']
        self.max_iter_update_b = config['max_iter_update_b']
        self.moving_average_decay = config['moving_average_decay']# The decay to use for the moving average. 
        self.num_epochs_per_decay = config['num_epochs_per_decay']# Epochs after which learning rate decays.
        self.learning_rate_decay_factor = config['learning_rate_decay_factor']# Learning rate decay factor.
        self.initial_learning_rate_img = config['initial_learning_rate_img']# Initial learning rate for image layer.
        self.initial_learning_rate_txt = config['initial_learning_rate_txt']# Initial learning rate for text layer.

        self.alpha = config['alpha']

        self.save_dir = config['save_dir'] + 'lr_' + str(self.initial_learning_rate_img) + '_' + str(self.initial_learning_rate_txt) + '_cq_lambda_'+ str(self.cq_lambda) + '_subspace_' + str(self.subspace_num) + '_updateB_' + str(self.max_iter_update_b)  + '_' + self.img_model + '_' + self.txt_model +'_epoch_' + str(self.max_epoch) + '_' + str(self.output_dim) + '_alpha_' + str(self.alpha) + '.npz'

        self.config = config

        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config = configProt)

        ### initialize Centers
        for d in [self.device, self.centers_device]:
            with tf.device(d):
                self.C = tf.Variable(tf.random_uniform([self.subspace_num * self.subcenter_num, self.output_dim],
                                            minval = -1, maxval = 1, dtype = tf.float32, name = 'centers'))

        with tf.device(self.device):
            self.imgs = tf.placeholder(tf.float32, [None, 256, 256, 3])
            self.txts = tf.placeholder(tf.float32, [None, config['txt_dim']])
            self.img_label = tf.placeholder(tf.float32, [None, config['label_dim']])
            self.txt_label = tf.placeholder(tf.float32, [None, config['label_dim']])

            self.img_last_layer, self.img_output = self.img_layers()
            self.txt_last_layer, self.txt_output = self.txt_layers()
            
            ### Centers shared in different modalities (image & text)
            ### Binary codes for different modalities (image & text)
            self.b_img = tf.placeholder(tf.float32, [None, self.subspace_num * self.subcenter_num])
            self.b_txt = tf.placeholder(tf.float32, [None, self.subspace_num * self.subcenter_num])
            self.ICM_m = tf.placeholder(tf.int32, [])
            self.ICM_b_m = tf.placeholder(tf.float32, [None, self.subcenter_num])
            self.ICM_b_all = tf.placeholder(tf.float32, [None, self.subcenter_num * self.subspace_num])
            self.ICM_X = tf.placeholder(tf.float32, [self.code_batch_size, self.output_dim])
            self.ICM_C_m = tf.slice(self.C, [self.ICM_m * self.subcenter_num, 0], [self.subcenter_num, self.output_dim])
            self.ICM_X_residual = tf.add(tf.sub(self.ICM_X, tf.matmul(self.ICM_b_all, self.C)), tf.matmul(self.ICM_b_m, self.ICM_C_m))
            ICM_X_expand = tf.expand_dims(self.ICM_X_residual, 2)
            ICM_C_m_expand = tf.expand_dims(tf.transpose(self.ICM_C_m), 0)
            ICM_sum_squares = tf.reduce_sum(tf.square(tf.squeeze(tf.sub(ICM_X_expand, ICM_C_m_expand))), reduction_indices = 1)
            ICM_best_centers = tf.argmin(ICM_sum_squares, 1)
            self.ICM_best_centers_one_hot = tf.one_hot(ICM_best_centers, self.subcenter_num, dtype = tf.float32)
            
        self.loss_functions()

        with tf.device(self.centers_device):
            self.img_output_all = tf.placeholder(tf.float32, [None, self.output_dim])
            self.txt_output_all = tf.placeholder(tf.float32, [None, self.output_dim])
            
            self.img_b_all = tf.placeholder(tf.float32, [None, self.subspace_num * self.subcenter_num])
            self.txt_b_all = tf.placeholder(tf.float32, [None, self.subspace_num * self.subcenter_num])
        
        self.all_parameters = self.deep_parameters_img + self.deep_parameters_img_lastlayer + self.deep_parameters_txt + self.deep_parameters_txt_lastlayer + [self.C]

        # validation procedure
        if 'model_weights' in config.keys():
            self.load_model(config['model_weights'])

    def loss_functions(self):
        with tf.device(self.device):
            ### Loss Function
            ### O = L + \lambda (Q^x + Q^y)
            ### L = sum_{ij} (log (1 + exp(alpha * <u_i,v_j>)) - alpha * s_ij * <u_i, v_j>)
            ### Q^x = || u - C * b_x ||
            ### Q^y = || v - C * b_y ||
            ### InnerProduct Value \in [-15, 15]
            InnerProduct = tf.clip_by_value(tf.mul(self.alpha, tf.matmul(self.img_last_layer, tf.transpose(self.txt_last_layer))), -1.5e1, 1.5e1)
            Sim = tf.clip_by_value(tf.matmul(self.img_label, tf.transpose(self.txt_label)), 0.0, 1.0)
            t_ones = tf.ones([tf.shape(self.img_last_layer)[0], tf.shape(self.txt_last_layer)[0]])

            self.cross_entropy_loss = tf.reduce_mean(tf.sub(tf.log(tf.add(t_ones, tf.exp(InnerProduct))), tf.mul(Sim, InnerProduct)))
            
            self.cq_loss_img = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(self.img_last_layer, tf.matmul(self.b_img, self.C))), 1))
            self.cq_loss_txt = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(self.txt_last_layer, tf.matmul(self.b_txt, self.C))), 1))
            self.q_lambda = tf.Variable(self.cq_lambda, name='lambda')
            self.cq_loss = tf.mul(self.q_lambda, tf.add(self.cq_loss_img, self.cq_loss_txt))
            self.total_loss = tf.add(self.cross_entropy_loss, self.cq_loss)
            

    def img_layers(self):
        if self.img_model == 'alexnet':
            if self.train_stage:
                return self.img_alexnet_layers()
            else:
                print "############## validation #############"
                return self.img_alexnet_layers_validation()
        

    def img_alexnet_layers(self):
        self.deep_parameters_img = []
        self.deep_parameters_img_lastlayer = []

        self.deep_parameters_img_ip = []
        self.deep_parameters_img_bias = []
        self.deep_parameters_img_lastlayer_ip = []
        self.deep_parameters_img_lastlayer_bias = []

        net_data = np.load(self.config['weights']).item()
    
        # Image processing for training the network. Note the many random
        # distortions applied to the image.
        reshaped_image = tf.cast(self.imgs, tf.float32)

        if self.img_model == 'alexnet':
            IMAGE_SIZE = 227

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(reshaped_image, [self.batch_size, height, width, 3])

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                   max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                 lower=0.2, upper=1.8)

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            distorted_image = distorted_image-mean

        # conv1
        # output 96, kernel 11, stride 4
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(net_data['conv1'][0], name='weights')
            conv = tf.nn.conv2d(distorted_image, kernel, [1, 4, 4, 1], padding='VALID')
            biases = tf.Variable(net_data['conv1'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = tf.nn.relu(out, name=scope)
            self.deep_parameters_img += [kernel, biases]
            self.deep_parameters_img_ip += [kernel]
            self.deep_parameters_img_bias += [biases]
        
        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool1')

        # lrn1
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn1 = tf.nn.local_response_normalization(self.pool1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
        
        # conv2
        # output 256, pad 2, kernel 5, group 2
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(net_data['conv2'][0], name='weights')

            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.lrn1)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(3, output_groups)

            biases = tf.Variable(net_data['conv2'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out, name=scope)
            self.deep_parameters_img += [kernel, biases]
            self.deep_parameters_img_ip += [kernel]
            self.deep_parameters_img_bias += [biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool2')

        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn2 = tf.nn.local_response_normalization(self.pool2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

        # conv3
        # output 384, pad 1, kernel 3
        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(net_data['conv3'][0], name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv3'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = tf.nn.relu(out, name=scope)
            self.deep_parameters_img += [kernel, biases]
            self.deep_parameters_img_ip += [kernel]
            self.deep_parameters_img_bias += [biases]

        # conv4
        # output 384, pad 1, kernel 3, group 2
        with tf.name_scope('conv4') as scope:
            kernel = tf.Variable(net_data['conv4'][0], name='weights')

            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.conv3)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(3, output_groups)

            biases = tf.Variable(net_data['conv4'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4 = tf.nn.relu(out, name=scope)
            self.deep_parameters_img += [kernel, biases]
            self.deep_parameters_img_ip += [kernel]
            self.deep_parameters_img_bias += [biases]

        # conv5
        # output 256, pad 1, kernel 3, group 2
        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(net_data['conv5'][0], name='weights')

            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.conv4)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(3, output_groups)

            biases = tf.Variable(net_data['conv5'][1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5 = tf.nn.relu(out, name=scope)
            self.deep_parameters_img += [kernel, biases]
            self.deep_parameters_img_ip += [kernel]
            self.deep_parameters_img_bias += [biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool5')

        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(net_data['fc6'][0], name='weights')
            fc1b = tf.Variable(net_data['fc6'][1],
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)

            self.fc1 = tf.nn.dropout(tf.nn.relu(fc1l), 0.5)
            self.fc1o = tf.nn.relu(fc1l)

            self.deep_parameters_img += [fc1w, fc1b]
            self.deep_parameters_img_ip += [fc1w]
            self.deep_parameters_img_bias += [fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(net_data['fc7'][0], name='weights')
            fc2b = tf.Variable(net_data['fc7'][1],
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.dropout(tf.nn.relu(fc2l), 0.5)

            fc2lo = tf.nn.bias_add(tf.matmul(self.fc1o, fc2w), fc2b)
            self.fc2o = tf.nn.relu(fc2lo)

            self.deep_parameters_img += [fc2w, fc2b]
            self.deep_parameters_img_ip += [fc2w]
            self.deep_parameters_img_bias += [fc2b]

        # fc3
        with tf.name_scope('fc8') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, self.output_dim],
                                                         dtype=tf.float32,
                                                         stddev=1e-2), name='weights')
            fc3b = tf.Variable(tf.constant(0.0, shape=[self.output_dim], dtype=tf.float32), trainable=True, name='biases')
            fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.fc3 = tf.nn.tanh(fc3l, name='image_last_layer')

            fc3lo = tf.nn.bias_add(tf.matmul(self.fc2o, fc3w), fc3b)
            self.fc3o = tf.nn.tanh(fc3lo, name='image_output')
            self.deep_parameters_img_lastlayer += [fc3w, fc3b]
            self.deep_parameters_img_lastlayer_ip += [fc3w]
            self.deep_parameters_img_lastlayer_bias += [fc3b]

        # return the output of image layer
        return self.fc3, self.fc3o


    def img_alexnet_layers_validation(self):
        self.deep_parameters_img = []
        self.deep_parameters_img_lastlayer = []

        net_data = np.load(self.config['model_weights'])['model']
    
        # Image processing for training the network. Note the many random
        # distortions applied to the image.
        
        reshaped_image = tf.cast(self.imgs, tf.float32)

        if self.img_model == 'alexnet':
            IMAGE_SIZE = 227

        height = IMAGE_SIZE
        width = IMAGE_SIZE
        
        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(reshaped_image, [self.batch_size, height, width, 3])
        
        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                   max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                 lower=0.2, upper=1.8)

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            distorted_image = distorted_image-mean

        # conv1
        # output 96, kernel 11, stride 4
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(net_data[0], name='weights')
            conv = tf.nn.conv2d(distorted_image, kernel, [1, 4, 4, 1], padding='VALID')
            biases = tf.Variable(net_data[1],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = tf.nn.relu(out, name=scope)
            self.deep_parameters_img += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool1')
        
        # lrn1
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn1 = tf.nn.local_response_normalization(self.pool1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
        
        # conv2
        # output 256, pad 2, kernel 5, group 2
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(net_data[2], name='weights')

            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.lrn1)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(3, output_groups)

            biases = tf.Variable(net_data[3],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out, name=scope)
            self.deep_parameters_img += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool2')

        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn2 = tf.nn.local_response_normalization(self.pool2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

        # conv3
        # output 384, pad 1, kernel 3
        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(net_data[4], name='weights')
            conv = tf.nn.conv2d(self.lrn2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data[5],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = tf.nn.relu(out, name=scope)
            self.deep_parameters_img += [kernel, biases]

        # conv4
        # output 384, pad 1, kernel 3, group 2
        with tf.name_scope('conv4') as scope:
            kernel = tf.Variable(net_data[6], name='weights')

            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.conv3)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(3, output_groups)

            biases = tf.Variable(net_data[7],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4 = tf.nn.relu(out, name=scope)
            self.deep_parameters_img += [kernel, biases]

        # conv5
        # output 256, pad 1, kernel 3, group 2
        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(net_data[8], name='weights')

            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.conv4)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(3, output_groups)

            biases = tf.Variable(net_data[9],
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5 = tf.nn.relu(out, name=scope)
            self.deep_parameters_img += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool5')

        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(net_data[10], name='weights')
            fc1b = tf.Variable(net_data[11],
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)

            self.fc1 = tf.nn.relu(fc1l)
            self.fc1o = tf.nn.dropout(tf.nn.relu(fc1l), 0.5)
            
            self.deep_parameters_img += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(net_data[12], name='weights')
            fc2b = tf.Variable(net_data[13],
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)

            fc2lo = tf.nn.bias_add(tf.matmul(self.fc1o, fc2w), fc2b)
            self.fc2o = tf.nn.dropout(tf.nn.relu(fc2lo), 0.5)

            self.deep_parameters_img += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc8') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, self.output_dim],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[self.output_dim], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.fc3 = tf.nn.tanh(fc3l, name='image_last_layer')

            fc3lo = tf.nn.bias_add(tf.matmul(self.fc2o, fc3w), fc3b)
            self.fc3o = tf.nn.tanh(fc3lo, name='image_output')
            self.deep_parameters_img_lastlayer += [fc3w, fc3b]

        # return the output of image layer
        return self.fc3, self.fc3o

    def txt_layers(self):
        return self.txt_mlp_layers()

    def txt_mlp_layers(self):
        # txt_fc1
        self.deep_parameters_txt_ip = []
        self.deep_parameters_txt_bias = []
        self.deep_parameters_txt_lastlayer_ip = []
        self.deep_parameters_txt_lastlayer_bias = []
        with tf.name_scope('txt_fc1') as scope:
            self.deep_parameters_txt = []
            self.deep_parameters_txt_lastlayer = []
            txt_fc1w = tf.Variable(tf.truncated_normal([self.txt_dim, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-2), name='weights')
            txt_fc1b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            txt_fc1l = tf.nn.bias_add(tf.matmul(self.txts, txt_fc1w), txt_fc1b)
            self.txt_fc1 = tf.nn.dropout(tf.nn.relu(txt_fc1l), 0.5)
            
            txt_fc1lo = tf.nn.bias_add(tf.matmul(self.txts, txt_fc1w), txt_fc1b)
            self.txt_fc1o = tf.nn.relu(txt_fc1lo)
            
            self.deep_parameters_txt += [txt_fc1w, txt_fc1b]
            self.deep_parameters_txt_ip += [txt_fc1w]
            self.deep_parameters_txt_bias += [txt_fc1b]

        # txt_fc2
        with tf.name_scope('txt_fc2') as scope:
            txt_fc2w = tf.Variable(tf.truncated_normal([4096, self.output_dim],
                                                         dtype=tf.float32,
                                                         stddev=1e-2), name='weights')
            txt_fc2b = tf.Variable(tf.constant(0.0, shape=[self.output_dim], dtype=tf.float32),
                                 trainable=True, name='biases')

            txt_fc2l = tf.nn.bias_add(tf.matmul(self.txt_fc1, txt_fc2w), txt_fc2b)
            self.txt_fc2 = tf.nn.tanh(txt_fc2l, name='text_output')

            txt_fc2lo = tf.nn.bias_add(tf.matmul(self.txt_fc1o, txt_fc2w), txt_fc2b)
        
            self.txt_fc2o = tf.nn.tanh(txt_fc2lo, name='text_output')
            
            self.deep_parameters_txt_lastlayer += [txt_fc2w, txt_fc2b]
            self.deep_parameters_txt_lastlayer_ip += [txt_fc2w]
            self.deep_parameters_txt_lastlayer_bias += [txt_fc2b]
        
        # return the output of text layer
        return self.txt_fc2, self.txt_fc2o

    def load_weights_img(self, weight_file):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            try:
                self.sess.run(self.deep_parameters_img[i].assign(weights[k]))
            except:
                print "shape mismatch: " + keys[i]

    def load_model(self, model_file):
        '''
        load model (deep parameters, centers) from model_file
        '''
        weights = np.load(model_file)['model']
        i = 0
        for item in weights:
            try:
                self.sess.run(self.all_parameters[i].assign(item))
            except:
                print "shape mismatch: " + str(i)
            i += 1
            
    def save_model(self, model_file):
        '''
        save model (deep parameters, centers) to model_file
        '''
        save_param = []
        for item in self.all_parameters:
            save_param += [self.sess.run(item)]
        
        np.savez(model_file, model=save_param)

    def initial_centers(self, img_output, txt_output):
        C_init = np.zeros([self.subspace_num * self.subcenter_num, self.output_dim])
        print "#cdq train# initilizing Centers"
        all_output = np.vstack([img_output, txt_output])
        for i in xrange(self.subspace_num):
            kmeans = MiniBatchKMeans(n_clusters=self.subcenter_num).fit(all_output[:, i * self.output_dim / self.subspace_num: (i + 1) * self.output_dim / self.subspace_num])
            C_init[i * self.subcenter_num: (i + 1) * self.subcenter_num, i * self.output_dim / self.subspace_num: (i + 1) * self.output_dim / self.subspace_num] = kmeans.cluster_centers_
            print "step: ", i, " finish"
        return C_init

    def update_centers(self, img_dataset, txt_dataset):
        '''
        Optimize:
            self.C = (U * hu^T + V * hv^T) (hu * hu^T + hv * hv^T)^{-1}
            self.C^T = (hu * hu^T + hv * hv^T)^{-1} (hu * U^T + hv * V^T)
            but all the C need to be replace with C^T :
            self.C = (hu * hu^T + hv * hv^T)^{-1} (hu^T * U + hv^T * V)
        '''
        old_C_value = self.sess.run(self.C)
        
        h = tf.concat(0, [self.img_b_all, self.txt_b_all])
        U = tf.concat(0, [self.img_output_all, self.txt_output_all])
        smallResidual = tf.constant(np.eye(self.subcenter_num * self.subspace_num, dtype = np.float32) * 0.001)
        Uh = tf.matmul(tf.transpose(h), U)
        hh = tf.add(tf.matmul(tf.transpose(h), h), smallResidual)
        compute_centers = tf.matmul(tf.matrix_inverse(hh), Uh)
        
        update_C = self.C.assign(compute_centers)
        C_value = self.sess.run(update_C, feed_dict = {
            self.img_output_all: img_dataset.full_output_without_shuffle(), 
            self.txt_output_all: txt_dataset.full_output_without_shuffle(),
            self.img_b_all: img_dataset.full_codes_without_shuffle(),
            self.txt_b_all: txt_dataset.full_codes_without_shuffle(),
            })
        
        print 'updated C is:'
        print C_value
        print "non zeros:"
        print len(np.where(np.sum(C_value, 1) != 0)[0])

    def update_codes_ICM(self, output, code):
        '''
        Optimize:
            min || output - self.C * codes ||
            min || output - codes * self.C ||
        args:
            output: [n_train, n_output]
            self.C: [n_subspace * n_subcenter, n_output]
                [C_1, C_2, ... C_M]
            codes: [n_train, n_subspace * n_subcenter]
        '''

        code = np.zeros(code.shape)
        
        for iterate in xrange(self.max_iter_update_b):
            start = time.time()
            time_init = 0.0
            time_compute_centers = 0.0
            time_append = 0.0
            
            sub_list = [i for i in range(self.subspace_num)]
            random.shuffle(sub_list)
            for m in sub_list:
                # update the code in subspace m
                # dim: [subcenter * subspace, subcenter * subspace]
                
                best_centers_one_hot_val = self.sess.run(self.ICM_best_centers_one_hot, feed_dict = {
                    self.ICM_b_m: code[:, m * self.subcenter_num: (m + 1) * self.subcenter_num],
                    self.ICM_b_all: code,
                    self.ICM_m: m,
                    self.ICM_X: output,
                })

                code[:, m * self.subcenter_num: (m + 1) * self.subcenter_num] = best_centers_one_hot_val

        return code


    def update_codes_batch(self, dataset, batch_size):
        '''
        update codes in batch size
        '''
        total_batch = int(dataset.n() / batch_size)
        print "start update codes in batch size ", batch_size

        dataset.finish_epoch()
        
        for i in xrange(total_batch):
            print "Iter ", i, "of ", total_batch
            output_val, code_val = dataset.next_batch_output_codes(batch_size)
            print output_val, code_val
            codes_val = self.update_codes_ICM(output_val, code_val)
            print np.sum(np.sum(codes_val, 0) != 0)
            dataset.copy_batch_codes(codes_val, batch_size)

        print "update_code wrong:"
        print np.sum(np.sum(dataset.codes, 1) != 4)
        
        print "######### update codes done ##########"
        
    def train_deep_networks(self, global_step):
        
        # Variables that affect learning rate.
        num_batches_per_epoch = self.n_train / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        self.img_lr = tf.train.exponential_decay(self.initial_learning_rate_img, global_step, decay_steps,
                                    self.learning_rate_decay_factor, staircase=True)
        self.img_lr_last = tf.train.exponential_decay(self.initial_learning_rate_img*10, global_step, decay_steps,
                                    self.learning_rate_decay_factor, staircase=True)
        
        self.txt_lr = tf.train.exponential_decay(self.initial_learning_rate_txt, global_step, decay_steps,
                                    self.learning_rate_decay_factor, staircase=True)
        self.txt_lr_last = tf.train.exponential_decay(self.initial_learning_rate_txt*10, global_step, decay_steps,
                                    self.learning_rate_decay_factor, staircase=True)

        # Compute gradients of deep neural networks, 
        # without Centers and Binary Codes.
        apply_gradient_op_img = tf.train.MomentumOptimizer(learning_rate=self.img_lr, momentum=0.9).minimize(self.total_loss, var_list=self.deep_parameters_img, global_step=global_step)
        apply_gradient_op_img_last = tf.train.MomentumOptimizer(learning_rate=self.img_lr*10, momentum=0.9).minimize(self.total_loss, var_list=self.deep_parameters_img_lastlayer, global_step=global_step)
        apply_gradient_op_txt = tf.train.MomentumOptimizer(learning_rate=self.txt_lr, momentum=0.9).minimize(self.total_loss, var_list=self.deep_parameters_txt+self.deep_parameters_txt_lastlayer, global_step=global_step)
        apply_gradient_op = tf.group(apply_gradient_op_img, apply_gradient_op_img_last, apply_gradient_op_txt)
        
        return apply_gradient_op

    def train(self, img_dataset, txt_dataset):
        with tf.device(self.device):
            global_step = tf.Variable(0, trainable=False)
            self.global_step = global_step

            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            train_op_deep_networks = self.train_deep_networks(global_step)

            # Build an initialization operation to run below.
            init = tf.initialize_all_variables()

            # Start running operations on the Graph.
            self.sess.run(init)
            
            print "#CDQ Train# initialize variables done!!!"

        # Start the queue runners.
        global_step = 0
        
        for epoch in xrange(self.max_epoch):
            avg_cost = 0
            step = 0
            total_batch = int(self.n_train / self.batch_size)
            
            print "#CDQ Train# Start Epoch: " + str(epoch)
            print "#CDQ Train# training total batch is " + str(total_batch)

            # update Centers and Binary Codes for each epochs
            # first epoch do not optimize C and b (*)
            
            if epoch > 0:
                with tf.device(self.centers_device):
                    for i in xrange(self.max_iter_update_Cb):
                        print "#CDQ Train# update codes and centers in ", i, " iter"
                        
                        # compute codes if epoch == 1
                        if epoch == 1:
                            self.sess.run(self.C.assign(self.initial_centers(img_dataset.full_output_without_shuffle(), txt_dataset.full_output_without_shuffle())))

                        self.update_codes_batch(img_dataset, self.code_batch_size)
                        self.update_codes_batch(txt_dataset, self.code_batch_size)
                            
                        # update Centers
                        self.update_centers(img_dataset, txt_dataset)
                        
                    print "#CDQ Train# update centers and codes done!!!"
            
            with tf.device(self.device):            
                for i in xrange(total_batch):
                    images, image_labels, image_codes = img_dataset.next_batch(self.batch_size)
                    texts, text_labels, text_codes = txt_dataset.next_batch(self.batch_size)
                    
                    if epoch > 0:
                        assign_lambda = self.q_lambda.assign(self.cq_lambda)
                    else:
                        assign_lambda = self.q_lambda.assign(0.0)

                    self.sess.run([assign_lambda])

                    # update deep conv nets
                    start_time = time.time()
                    _, cross_entropy_l, cq_l, batch_img_output, batch_txt_output, img_last_layer, txt_last_layer = self.sess.run([train_op_deep_networks, self.cross_entropy_loss, self.cq_loss, self.img_output, self.txt_output, self.img_last_layer, self.txt_last_layer], feed_dict = {self.imgs: images, self.img_label: image_labels, self.b_img: image_codes, self.txts: texts, self.txt_label: text_labels, self.b_txt: text_codes})
                    
                    duration = time.time() - start_time
                    
                    # copy output batch
                    img_dataset.copy_batch_output(batch_img_output, self.batch_size)
                    txt_dataset.copy_batch_output(batch_txt_output, self.batch_size)

                    assert not np.isnan(cross_entropy_l), 'Model diverged with cross-entropy loss = NaN'
                    assert not np.isnan(cq_l), 'Model diverged with composite quantization loss = NaN'
                    
                    if step % 10 == 0:
                        num_examples_per_step = self.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        format_str = ('%s: step %4d, cross-entropy loss = %.4f, cq loss = %.4f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print (format_str % (datetime.now(), global_step, cross_entropy_l, cq_l,
                                             examples_per_sec, sec_per_batch))

                        print self.sess.run([self.img_lr, self.txt_lr, self.global_step])
                        bC_img, bC_txt = self.sess.run([tf.matmul(self.b_img, self.C), tf.matmul(self.b_txt, self.C)], feed_dict = {self.imgs: images, self.img_label: image_labels, self.b_img: image_codes, self.txts: texts, self.txt_label: text_labels, self.b_txt: text_codes})
                        print "img original:"
                        print img_last_layer
                        print "img reconstr:"
                        print bC_img
                        print "txt original:"
                        print txt_last_layer
                        print "txt reconstr:"
                        print bC_txt

                    avg_cost += (cross_entropy_l + cq_l) / self.n_train * self.batch_size
                    step += 1
                    global_step += 1
                print "#CDQ Train# Epoch: ", "%04d" % (epoch + 1), "cost = ", "{:.9f}".format(avg_cost)
        # save current point after training
        self.save_model(self.save_dir)

    def validation(self, database_img, database_txt, query_img, query_txt):
        with tf.device(self.device):
            C_tmp = self.sess.run(self.C)
            # get database codes
            total_batch = int(self.n_database / self.batch_size)
            print "#CDQ Validation# Database total batch: " + str(total_batch)
            for i in xrange(total_batch):
                images, image_labels, image_codes = database_img.next_batch(self.batch_size)
                texts, text_labels, text_codes = database_txt.next_batch(self.batch_size)
                start_time = time.time()
                batch_img_output, batch_txt_output, loss = self.sess.run([self.img_last_layer, self.txt_output, self.cross_entropy_loss], feed_dict = {self.imgs: images, self.txts: texts, self.img_label: image_labels, self.txt_label: text_labels})

                print batch_img_output, batch_txt_output

                database_img.copy_batch_output(batch_img_output, self.batch_size)
                database_txt.copy_batch_output(batch_txt_output, self.batch_size)
                duration = time.time() - start_time
                print str(i) + " / " + str(total_batch) + " batch, time = " + str(duration)
                print "loss: ", loss

            self.update_codes_batch(database_img, self.code_batch_size)
            self.update_codes_batch(database_txt, self.code_batch_size)
                
            # get query codes
            total_batch = int(self.n_query / self.batch_size)
            print "#CDQ Validation# Query total batch: " + str(total_batch)
            for i in xrange(total_batch):
                images, image_labels, image_codes = query_img.next_batch(self.batch_size)
                texts, text_labels, text_codes = query_txt.next_batch(self.batch_size)
                start_time = time.time()
                batch_img_output, batch_txt_output, loss = self.sess.run([self.img_last_layer, self.txt_output, self.cross_entropy_loss], feed_dict = {self.imgs: images, self.txts: texts, self.img_label: image_labels, self.txt_label: text_labels})
                query_img.copy_batch_output(batch_img_output, self.batch_size)
                query_txt.copy_batch_output(batch_txt_output, self.batch_size)
                duration = time.time() - start_time
                print str(i) + " / " + str(total_batch) + " batch, time = " + str(duration)
                print "loss: ", loss

            self.update_codes_batch(query_img, self.code_batch_size)
            self.update_codes_batch(query_txt, self.code_batch_size)

            print "#CDQ validation# Codes computed"

            mAPs = MAPs(self.sess.run(self.C), self.subspace_num, self.subcenter_num, self.config["R"])

            return {
                'i2t_AQD': mAPs.get_mAPs_AQD(database_txt, query_img),
                't2i_AQD': mAPs.get_mAPs_AQD(database_img, query_txt),
                'i2t_nocq': mAPs.get_mAPs_by_feature(database_txt, query_img),
                't2i_nocq': mAPs.get_mAPs_by_feature(database_img, query_txt),
                }

class Dataset(object):
    def __init__(self, dataset, code_dim, output_dim, config):
        """
        Args:
          config:
            'device': '/gpu:1'
          dataset:
            .data: [n_samples, n_input] 
            .label: [n_samples, n_label] label is [0, 1]
        """
        print "Initalizing Dataset"
        self.dataset = dataset
        self.get_labels = self.dataset.get_labels

        self.codes = np.zeros((self.dataset.n_samples, code_dim))
        self.output = np.zeros((self.dataset.n_samples, output_dim), dtype=np.float32)

        self.n_samples = self.dataset.n_samples

        self._perm = np.arange(self.n_samples)
        self._index_in_epoch = 0
        self._epochs_complete = 0
        print "Dataset already"
        return

    def next_batch(self, batch_size):
        """
        Args:
          batch_size
        Returns:
          (
            [batch_size, n_input]: next batch of images
            [batch_size, n_z]: next batch of centers it belongs
          )
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.n_samples:
            self._epochs_complete += 1
            # Shuffle the data
            np.random.shuffle(self._perm)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        data, label = self.dataset.data(self._perm[start: end])

        return (data, label,
                self.codes[self._perm[start: end], :])

    def next_batch_data(self, batch_size):
        """
        Args:
          batch_size
        Returns:
          (
            [batch_size, n_input]: next batch of images
            [batch_size, n_z]: next batch of centers it belongs
          )
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.n_samples:
            self._epochs_complete += 1
            # Shuffle the data
            np.random.shuffle(self._perm)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        data, label = self.dataset.data(self._perm[start: end])

        return (data, label)

    def finish_epoch(self):
        start = 0
        np.random.shuffle(self._perm)
    
    def next_batch_output_codes(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.n_samples:
            # Shuffle the data
            np.random.shuffle(self._perm)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        return (
            self.output[self._perm[start: end], :],
            self.codes[self._perm[start: end], :],
        )

    def full_dataset_without_shuffle(self):
        return self.dataset.all_data()
    
    def full_output_without_shuffle(self):
        return self.output

    def full_codes_without_shuffle(self):
        return self.codes

    def copy_codes(self, codes):
        self.codes = codes
        return

    def copy_batch_codes(self, codes, batch_size):
        start = self._index_in_epoch - batch_size
        end = self._index_in_epoch
        self.codes[self._perm[start: end], :] = codes
        return 

    def copy_batch_output(self, output, batch_size):
        start = self._index_in_epoch - batch_size
        end = self._index_in_epoch
        self.output[self._perm[start: end], :] = output
        return 

    def n(self):
        return self.n_samples
    
def train(train_img, train_txt, config):
    model = cdq(config)

    img_dataset = Dataset(train_img, config['n_subspace'] * config['n_subcenter'], config['output_dim'], config)
    txt_dataset = Dataset(train_txt, config['n_subspace'] * config['n_subcenter'], config['output_dim'], config)
    
    model.train(img_dataset, txt_dataset)
    
    return model

def validation(database_img, database_txt, query_img, query_txt, model, config):
    database_img = Dataset(database_img, config['n_subspace'] * config['n_subcenter'], config['output_dim'], config)
    database_txt = Dataset(database_txt, config['n_subspace'] * config['n_subcenter'], config['output_dim'], config)
    query_img = Dataset(query_img, config['n_subspace'] * config['n_subcenter'], config['output_dim'], config)
    query_txt = Dataset(query_txt, config['n_subspace'] * config['n_subcenter'], config['output_dim'], config)
    
    ret = model.validation(database_img, database_txt, query_img, query_txt)
    print ret
    
