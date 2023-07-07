#coding:utf-8
import tensorflow as tf
import scipy.misc
import numpy as np
import scipy.io

MODEL_DIR = '/home/share/sunchangchang/data/imagenet-vgg-f.mat'
def img_net_strucuture(input_image, bit,num_class):
    print("start")
    data = scipy.io.loadmat(MODEL_DIR)
    layers = (
    	'conv1', 'relu1', 'norm1', 'pool1','conv2', 'relu2', 'norm2', 'pool2','conv3', 'relu3', 'conv4', 'relu4', 'conv5',
    	'relu5', 'pool5','fc6', 'relu6', 'fc7', 'relu7','fc8')
    weights = data['layers'][0]
    mean = data['meta']['normalization'][0][0][0]
    net = {}
    ops = []
    current = tf.convert_to_tensor(input_image,dtype='float32')
    for i, name in enumerate(layers[:-1]):
        if name.startswith('conv'):
            print(i,name)
            kernels, bias = weights[i]['weights'][0][0][0]

            bias = bias.reshape(-1)
            # pad = weights[i][0][0][1]
            # stride = weights[i][0][0][4]
            pad = weights[i]['pad'][0][0]
            stride = weights[i]['stride'][0][0]
            current = _conv_layer(current, kernels, bias, pad, stride, i, ops, net)
        elif name.startswith('relu'):
            print(i, name)
            current = tf.nn.relu(current)
        elif name.startswith('pool'):
            print(i, name)
            pad = weights[i]['pad'][0][0]
            stride = weights[i]['stride'][0][0]
            #area = weights[i][0][0][5]
            area=weights[i]['pool'][0][0]
            current = _pool_layer(current, stride, pad, area)
        elif name.startswith('fc'):
            print(i, name)
            kernels, bias = weights[i]['weights'][0][0][0]

            bias = bias.reshape(-1)
            current = _full_conv(current, kernels, bias, i, ops, net)
        elif name.startswith('norm'):
            current = tf.nn.local_response_normalization(current, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
        net[name] = current
    #
    W_fc8 = tf.random_normal([4096, bit], stddev=1.0) * 0.01
    b_fc8 = tf.random_normal([bit],stddev = 1.0) * 0.01
    w1 = tf.Variable(W_fc8, name='w' + str(20))
    b1 = tf.Variable(b_fc8, name='bias' + str(20))
    #
    W_fc9 = tf.random_normal([4096, bit], stddev=1.0) * 0.01
    b_fc9 = tf.random_normal([bit], stddev=1.0) * 0.01
    w2 = tf.Variable(W_fc9, name='w' + str(21))
    b2 = tf.Variable(b_fc9, name='bias' + str(21))

    num_class1 = 8
    num_class2 = 27
    # classify weight
    W_classify1 = tf.random_normal([bit, num_class1], stddev=1.0) * 0.01
    b_classify1 = tf.random_normal([num_class1], stddev=1.0) * 0.01
    w_c1 = tf.Variable(W_classify1, name='w' + str(22))  # w22
    b_c1 = tf.Variable(b_classify1, name='bias' + str(22))  # bias22
    # classify weight
    W_classify2 = tf.random_normal([bit, num_class2], stddev=1.0) * 0.01
    b_classify2 = tf.random_normal([num_class2], stddev=1.0) * 0.01
    w_c2 = tf.Variable(W_classify2, name='w' + str(23))  # w22
    b_c2 = tf.Variable(b_classify2, name='bias' + str(23))  # bias22

    ops.append(w1)
    ops.append(b1)
    ops.append(w2)
    ops.append(b2)
    # print(current)
    # print(tf.squeeze(current))
    # print(current.shape)
    # print(w.shape)
    fc8 = tf.tanh(tf.matmul(tf.squeeze(current),w1) + b1)   #(64,?)
    fc9 = tf.tanh(tf.matmul(tf.squeeze(current), w2) + b2)  # (64,?)
    classify_i1 = tf.matmul(fc8,w_c1) + b_c1
    classify_i2 = tf.matmul(fc9,w_c2) + b_c2
    return net, mean,classify_i1,classify_i2

def _conv_layer(input, weights, bias,pad,stride,i,ops,net):
    pad = pad[0]
    stride= stride[0]
    input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
    w = tf.Variable(weights,name='w'+str(i),dtype='float32')
    b = tf.Variable(bias,name='bias'+str(i),dtype='float32')
    ops.append(w)
    ops.append(b)
    net['weights' + str(i)] = w
    net['b' + str(i)] = b
    conv = tf.nn.conv2d(input, w, strides=[1,stride[0],stride[1],1],padding='VALID',name='conv'+str(i))
    return tf.nn.bias_add(conv, b,name='add'+str(i))

def _full_conv(input, weights, bias,i,ops,net):
    w = tf.Variable(weights, name='w' + str(i),dtype='float32')
    b = tf.Variable(bias, name='bias' + str(i),dtype='float32')
    ops.append(w)
    ops.append(b)
    net['weights' + str(i)] = w
    net['b' + str(i)] = b
    conv = tf.nn.conv2d(input, w,strides=[1,1,1,1],padding='VALID',name='fc'+str(i))
    return tf.nn.bias_add(conv, b,name='add'+str(i))

def _pool_layer(input,stride,pad,area):
    pad = pad[0]
    area = area[0]
    stride = stride[0]
    input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
    return tf.nn.max_pool(input, ksize=[1, area[0], area[1], 1], strides=[1,stride[0],stride[1],1],padding='VALID')

def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel

def get_meanpix(data_path):
    data = scipy.io.loadmat(data_path)
    mean = data['meta']['normalization'][0][0][0]
    return mean
