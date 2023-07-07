#coding:utf-8
import tensorflow as tf
import scipy.misc
import numpy as np
import scipy.io
LAYER1_NODE = 8192

def txt_net_strucuture(text_input, dimy, bit,num_class):

	W_fc8 = tf.random_normal([1, dimy, 1, LAYER1_NODE], stddev=1.0) * 0.01
	b_fc8 = tf.random_normal([1, LAYER1_NODE], stddev=1.0) * 0.01
	fc1W = tf.Variable(W_fc8)
	fc1b = tf.Variable(b_fc8)

	conv1 = tf.nn.conv2d(text_input, fc1W, strides=[1, 1, 1, 1], padding='VALID')
	layer1 = tf.nn.relu(tf.nn.bias_add(conv1, tf.squeeze(fc1b)))
	#hash1
	W_fc2 = tf.random_normal([1, 1, LAYER1_NODE, bit], stddev=1.0) * 0.01
	b_fc2 = tf.random_normal([1, bit], stddev=1.0) * 0.01
	fc2W = tf.Variable(W_fc2)
	fc2b = tf.Variable(b_fc2)
	#hash2
	W_fc3 = tf.random_normal([1, 1, LAYER1_NODE, bit], stddev=1.0) * 0.01
	b_fc3 = tf.random_normal([1, bit], stddev=1.0) * 0.01
	fc3W = tf.Variable(W_fc3)
	fc3b = tf.Variable(b_fc3)

	num_class1 = 8
	num_class2 = 27
	# classify weight
	W_classify_t1 = tf.random_normal([bit, num_class1], stddev=1.0) * 0.01
	b_classify_t1 = tf.random_normal([num_class1], stddev=1.0) * 0.01
	w_c_t1 = tf.Variable(W_classify_t1, name='w' + str(22))
	b_c_t1 = tf.Variable(b_classify_t1, name='bias' + str(22))
	# classify weight
	W_classify_t2 = tf.random_normal([bit, num_class2], stddev=1.0) * 0.01
	b_classify_t2 = tf.random_normal([num_class2], stddev=1.0) * 0.01
	w_c_t2 = tf.Variable(W_classify_t2, name='w' + str(23))
	b_c_t2 = tf.Variable(b_classify_t2, name='bias' + str(23))
	conv2_1 = tf.nn.conv2d(layer1, fc2W, strides=[1, 1, 1, 1], padding='VALID')

	output_g1 = tf.transpose(tf.squeeze(tf.nn.bias_add(conv2_1, tf.squeeze(fc2b)), [1, 2]))  #从张量形状中移除大小为1的维度
	conv2_2 = tf.tanh(tf.nn.conv2d(layer1, fc3W, strides=[1, 1, 1, 1], padding='VALID'))
	output_g2 = tf.tanh(tf.transpose(tf.squeeze(tf.nn.bias_add(conv2_2, tf.squeeze(fc3b)), [1, 2])) ) #从张量形状中移除大小为1的维度
	classify_t1 = tf.matmul(tf.transpose(output_g1), w_c_t1) + b_c_t1
	classify_t2 = tf.matmul(tf.transpose(output_g2), w_c_t2) + b_c_t2

	print("now is classidy of text:",classify_t1.shape,classify_t2.shape)

	return output_g1,output_g2,classify_t1,classify_t2
