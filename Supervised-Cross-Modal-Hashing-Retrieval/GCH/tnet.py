import scipy.misc
import tensorflow as tf
import scipy.io
from ops import *
from setting import *
# import graph as lg


def lab_net(input_label, bit, numClass, reuse=False, name='Lab_Network'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        LAYER1_NODE = 512
        OUT_NODE = 512
        labnet = {}


        labnet['conv1'] = lrelu(batch_norm(conv2d(input=input_label, kernel=[1, numClass, 1, LAYER1_NODE], strides=[1, 1, 1, 1],
                               padding='VALID', init_rate=1.0, name='lab_conv1'), name='lab_conv1_BN'))

        labnet['feature'] = lrelu(batch_norm(conv2d(input=labnet['conv1'], kernel=[1, 1, LAYER1_NODE, OUT_NODE], strides=[1, 1, 1, 1],
                               padding='VALID', init_rate=1.0, name='lab_feat'), name='lab_feat_BN'))

        labnet['hash'] = tanh(conv2d(input=labnet['feature'], kernel=[1, 1, OUT_NODE, bit], strides=[1, 1, 1, 1],
                               padding='VALID', init_rate=1.0, name='lab_hash'))

        labnet['label'] = sigmoid(conv2d(input=labnet['hash'], kernel=[1, 1, bit, numClass], strides=[1, 1, 1, 1],
                               padding='VALID', init_rate=1.0, name='lab_label'))

        return tf.squeeze(labnet['hash']), labnet['feature'], tf.squeeze(labnet['label'])


def img_net_itpair(inputs, bit, numclass, reuse=False, name='Img_Network'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        data = scipy.io.loadmat(MODEL_DIR)
        layers = (
            'conv1', 'relu1', 'norm1', 'pool1', 'conv2', 'relu2', 'norm2', 'pool2', 'conv3', 'relu3', 'conv4', 'relu4',
            'conv5', 'relu5', 'pool5', 'fc6', 'relu6', 'fc7', 'relu7')
        weights = data['layers'][0]

        imgnet = {}
        current = tf.convert_to_tensor(inputs, dtype='float32')
        for i, name in enumerate(layers):
            if name.startswith('conv'):
                kernels, bias = weights[i][0][0][0][0]
                bias = bias.reshape(-1)
                pad = weights[i][0][0][1]
                stride = weights[i][0][0][4]
                current = conv_layer(current, kernels, bias, pad, stride, i, imgnet)
            elif name.startswith('relu'):
                current = tf.nn.relu(current)
            elif name.startswith('pool'):
                stride = weights[i][0][0][1]
                pad = weights[i][0][0][2]
                area = weights[i][0][0][5]
                current = pool_layer(current, stride, pad, area)
            elif name.startswith('fc'):
                kernels, bias = weights[i][0][0][0][0]
                bias = bias.reshape(-1)
                current = full_conv(current, kernels, bias, i, imgnet)
            elif name.startswith('norm'):
                current = tf.nn.local_response_normalization(current, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
            imgnet[name] = current
        # mask = attentionLayer(current, 'softmax')
        # current = tf.multiply(current, mask)
        imgnet['feature'] = lrelu(batch_norm(
            conv2d(input=current, kernel=[1, 1, 4096, SEMANTIC_EMBED], strides=[1, 1, 1, 1],
                                       padding='VALID', init_rate=1.0, name='img_feat'), name='img_feat_BN'))

        imgnet['hash'] = tanh(batch_norm(conv2d(input=imgnet['feature'], kernel=[1, 1, SEMANTIC_EMBED, bit], strides=[1, 1, 1, 1],
                                       padding='VALID', init_rate=1.0, name='img_hash'), name='img_hash_BN'))

        imgnet['label'] = sigmoid(conv2d(input=imgnet['feature'], kernel=[1, 1, SEMANTIC_EMBED, numclass], strides=[1, 1, 1, 1],
                                       padding='VALID', init_rate=1.0, name='img_label'))

        return tf.squeeze(imgnet['hash']), imgnet['feature'], tf.squeeze(imgnet['label'])

def txt_net_itpair(text_input, bit, dimy, numclass, reuse=False, name='Txt_Network'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        txtnet={}
        LAYER1_NODE = 512

        txtnet['conv1'] = lrelu(batch_norm(
                            conv2d(input=text_input, kernel=[1, dimy, 1, LAYER1_NODE], strides=[1, 1, 1, 1], padding='VALID',
                                            init_rate=1.0, name='txt_ext_conv1'), name='txt_conv1_BN'))

        txtnet['conv2'] = lrelu(batch_norm(
                            conv2d(input=txtnet['conv1'], kernel=[1, 1, LAYER1_NODE, LAYER1_NODE], strides=[1, 1, 1, 1], padding='VALID',
                                            init_rate=1.0, name='txt_ext_conv2'), name='txt_conv2_BN'))

        txtnet['feature'] = lrelu(batch_norm(
                                conv2d(input=txtnet['conv2'], kernel=[1, 1, LAYER1_NODE, SEMANTIC_EMBED], strides=[1, 1, 1, 1], padding='VALID',
                                            init_rate=1.0, name='txt_feat'), name='txt_feat_BN'))

        txtnet['hash'] = tanh(conv2d(input=txtnet['feature'], kernel=[1, 1, SEMANTIC_EMBED, bit], strides=[1, 1, 1, 1], padding='VALID',
                                            init_rate=1.0, name='txt_hash'))

        txtnet['label'] = sigmoid(conv2d(input=txtnet['feature'], kernel=[1, 1, SEMANTIC_EMBED, numClass], strides=[1, 1, 1, 1], padding='VALID',
                                            init_rate=1.0, name='txt_label'))

        return tf.squeeze(txtnet['hash']), txtnet['feature'], tf.squeeze(txtnet['label'])


def GCN_stack(x, indices, data, name):
    with tf.variable_scope(name):
        fc1 = GraphConvLayer(input_dim=SEMANTIC_EMBED,
                             output_dim=512,
                             name="fc1",
                             # activation=tf.maximum,
                             activation=tf.nn.relu,
                             indices=indices, data=data, x=x, sparse=False)

        fc2 = GraphConvLayer(input_dim=512,
                             output_dim=bit,
                             name="fc2",
                             activation=tf.nn.tanh,
                             indices=indices, data=data, x=fc1)

        fc2_exp = tf.expand_dims(tf.expand_dims(fc2, 0), 1)
        psu_lab = sigmoid(conv2d(input=fc2_exp, kernel=[1, 1, bit, dimLab], strides=[1, 1, 1, 1], padding='VALID',
                                                    init_rate=1.0, name='psu_lab'))


    return fc2, tf.squeeze(psu_lab)

def full_conv_stack(x, adj_norm):
    with tf.variable_scope("graph_Layer"):
        weight_1 = tf.get_variable(name='weight_1', shape=[1, 1, SEMANTIC_EMBED, bit],
                                   # initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
                                   initializer=tf.glorot_uniform_initializer())

        weight_2 = tf.get_variable(name='weight_2', shape=[1, 1, 1024, bit],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))

        weight_3 = tf.get_variable(name='weight_3', shape=[1, 1, bit, dimLab],
                                   initializer=tf.glorot_uniform_initializer())

        fc_1 = full_conv_no_bias(input=x,
                                 weights=weight_1,
                                 adj=adj_norm,
                                 # activation=tf.nn.relu)
                                 activation=tf.maximum)

        psu_lab = conv2d(input=tf.expand_dims(fc_1, 1), kernel=[1, 1, bit, dimLab], strides=[1, 1, 1, 1],
                                       padding='VALID', init_rate=1.0, name='psu_lab')
    return fc_1, tf.squeeze(psu_lab, axis=1)

def sum_to_vec_1(x):
    with tf.variable_scope("mlp_1"):
        weight_1 = tf.get_variable(name='vec_weight_1', shape=[1, 1, dimLab*batch_size, 1024],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        weight_2 = tf.get_variable(name='vec_weight_2', shape=[1, 1, 1024, batch_size],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        vec_1 = lrelu(batch_norm(
            tf.nn.conv2d(x, weight_1, strides=[1, 1, 1, 1], padding='VALID', name='vec_1'), name='vec_1_BN'))

        vec_2 = tanh(batch_norm(
            tf.nn.conv2d(vec_1, weight_2, strides=[1, 1, 1, 1], padding='VALID', name='vec_2'), name='vec_2_BN'))
    return vec_2

def sum_to_vec_2(x):
    with tf.variable_scope("mlp_2"):
        weight_1 = tf.get_variable(name='vec_weight_1', shape=[1, 1, dimLab*batch_size, 1024],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        weight_2 = tf.get_variable(name='vec_weight_2', shape=[1, 1, 1024, batch_size],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        vec_1 = lrelu(batch_norm(
            tf.nn.conv2d(x, weight_1, strides=[1, 1, 1, 1], padding='VALID', name='vec_1'), name='vec_1_BN'))

        vec_2 = sigmoid(batch_norm(
            tf.nn.conv2d(vec_1, weight_2, strides=[1, 1, 1, 1], padding='VALID', name='vec_2'), name='vec_2_BN'))
    return vec_2


def ablation(x):
    with tf.variable_scope("ablation"):
        weight_1 = tf.get_variable(name='ablation_w_1', shape=[1, 1, SEMANTIC_EMBED, bit],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))

        weight_2 = tf.get_variable(name='ablation_w_2', shape=[1, 1, bit, dimLab],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))

        fc_1 = tf.nn.conv2d(x, weight_1, strides=[1, 1, 1, 1], padding='VALID', name='fc_1')

        psu_lab = tf.nn.conv2d(fc_1, weight_2, strides=[1, 1, 1, 1], padding='VALID', name='psu_lab')

        return tf.squeeze(fc_1), tf.squeeze(psu_lab)

def Att_pooling_code(x):
    with tf.variable_scope("att_pooling_code"):
        ws1 = tf.get_variable(name='w_1', shape=[256, bit],
                              initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        ws2 = tf.get_variable(name='w_2', shape=[batch_size, 256],
                              initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))

        aux = tanh(tf.matmul(ws1, tf.transpose(x)))
        A = tf.nn.softmax(tf.matmul(ws2, aux))

        out = tf.matmul(A, x)

    return out


def Att_pooling_logit(x):
    with tf.variable_scope("att_pooling_logit"):
        ws1 = tf.get_variable(name='w_1', shape=[256, dimLab],
                              initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        ws2 = tf.get_variable(name='w_2', shape=[batch_size, 256],
                              initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))

        aux = tanh(tf.matmul(ws1, tf.transpose(x)))
        A = tf.nn.softmax(tf.matmul(ws2, aux))

        out = tf.matmul(A, x)

    return out

def _get_tensor_shape(x):
    s = x.get_shape().as_list()
    return [i if i is not None else -1 for i in s]

def spatial_softmax(fm):
    fm_shape = _get_tensor_shape(fm)
    n_grids = fm_shape[1] ** 2
    # transpose feature map
    fm = tf.transpose(fm, perm=[0, 3, 1, 2])
    t_fm_shape = _get_tensor_shape(fm)
    fm = tf.reshape(fm, shape=[-1, n_grids])
    # apply softmax
    prob = tf.nn.softmax(fm)
    # reshape back
    prob = tf.reshape(prob, shape=t_fm_shape)
    prob = tf.transpose(prob, perm=[0, 2, 3, 1])
    return prob

def attentionLayer(x, pool_method='sigmoid'):
    with tf.variable_scope("attentionLayer"):
        weight_1 = tf.get_variable(name='attentionLayer_w_1', shape=[1, 1, SEMANTIC_EMBED, 256],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))

        weight_2 = tf.get_variable(name='attentionLayer_w_2', shape=[1, 1, 256, 1],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))

        fc_1 = lrelu(batch_norm(tf.nn.conv2d(x, weight_1, strides=[1, 1, 1, 1], padding='VALID', name='fc_1')))
        if pool_method == 'sigmoid':
            net = tf.nn.sigmoid(batch_norm(tf.nn.conv2d(fc_1, weight_2, strides=[1, 1, 1, 1], padding='VALID', name='fc_2')))
        else:
            net = spatial_softmax(batch_norm(tf.nn.conv2d(fc_1, weight_2, strides=[1, 1, 1, 1], padding='VALID', name='fc_2')))

    return net