import tensorflow as tf
import tensorflow.contrib.slim as slim
    
def fusion_net(fusion_input, hid_dim, hash_dim, kp, reuse=False):
    net = slim.fully_connected(fusion_input, hid_dim, reuse=reuse, scope='fn_fc_0')
    net = slim.dropout(net, keep_prob=kp)
    net = slim.fully_connected(net, int(hid_dim/2), reuse=reuse, scope='fn_fc_1')
    net = slim.dropout(net, keep_prob=kp)
    feat = slim.fully_connected(net, hash_dim, activation_fn=tf.nn.tanh, reuse=reuse, scope='fn_fc_2')
    return feat
    
def classification_net(feat, hash_dim, lab_dim, reuse=False):
    class_net = slim.fully_connected(feat, int(hash_dim/2), reuse=reuse, scope='cl_fc_0')
    class_net = slim.fully_connected(class_net, lab_dim, reuse=reuse, scope='cl_fc_1')
    return class_net
    
def discriminative_net1(dis_feat, dis_dim, dis_out_dim):
    dis_net = slim.fully_connected(dis_feat, dis_dim, scope='dis1_fc_0')
    dis_net = slim.fully_connected(dis_net, int(dis_dim/2), scope='dis1_fc_1')
    dis_net = slim.fully_connected(dis_net, dis_out_dim, scope='dis1_fc_2')
    return dis_net
     
def discriminative_net2(dis_feat, dis_dim, dis_out_dim):
    dis_net = slim.fully_connected(dis_feat, dis_dim, scope='dis2_fc_0')
    dis_net = slim.fully_connected(dis_net, int(dis_dim/2), scope='dis2_fc_1')
    dis_net = slim.fully_connected(dis_net, dis_out_dim, scope='dis2_fc_2')
    return dis_net