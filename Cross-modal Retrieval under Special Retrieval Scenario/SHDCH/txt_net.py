import tensorflow as tf

LAYER1_NODE = 8192


def txt_net_strucuture(text_input, dimy, bit):
    W_fc8 = tf.random_normal([1, dimy, 1, LAYER1_NODE], stddev=1.0) * 0.01
    b_fc8 = tf.random_normal([1, LAYER1_NODE], stddev=1.0) * 0.01
    fc1W = tf.Variable(W_fc8)
    fc1b = tf.Variable(b_fc8)

    conv1 = tf.nn.conv2d(text_input, fc1W, strides=[1, 1, 1, 1], padding='VALID')

    layer1 = tf.nn.relu(tf.nn.bias_add(conv1, tf.squeeze(fc1b)))

    W_fc2 = tf.random_normal([1, 1, LAYER1_NODE, bit], stddev=1.0) * 0.01
    b_fc2 = tf.random_normal([1, bit], stddev=1.0) * 0.01
    fc2W = tf.Variable(W_fc2)
    fc2b = tf.Variable(b_fc2)

    conv2 = tf.nn.conv2d(layer1, fc2W, strides=[1, 1, 1, 1], padding='VALID')
    output_g = tf.tanh(tf.transpose(tf.squeeze(tf.nn.bias_add(conv2, tf.squeeze(fc2b)), [1, 2])))
    return output_g
