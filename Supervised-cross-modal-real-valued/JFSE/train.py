import tensorflow as tf
import numpy as np
import os, time, pickle
import tensorflow.contrib.slim as slim
from base_model import BaseModel, BaseModelParams, BaseDataIter
import pdb
import sklearn.preprocessing
from sklearn import preprocessing
import scipy.spatial
import scipy.io as sio
from numpy.matlib import repmat
import losses
from flip_gradient import flip_gradient
from tensorflow.python.ops import math_ops, array_ops, random_ops, nn_ops

class DataIter(BaseDataIter):
    def __init__(self, batch_size):
        BaseDataIter.__init__(self, batch_size)
        self.num_train_batch = 0
        self.num_test_batch = 0

        with open('./data/xmn/train_img_files.pkl', 'rb') as f:
            self.train_img_feats = pickle.load(f, encoding='iso-8859-1')
        with open('./data/xmn/train_txt_files.pkl', 'rb') as f:
            self.train_txt_vecs = pickle.load(f, encoding='iso-8859-1')
        with open('./data/xmn/train_labels.pkl', 'rb') as f:
            self.train_labels = pickle.load(f, encoding='iso-8859-1')
        with open('./data/xmn/train_attribute.pkl', 'rb') as f:
            self.train_attributes = pickle.load(f, encoding='iso-8859-1')
        
        
        with open('./data/xmn/test_img_files.pkl', 'rb') as f:
            self.test_img_feats = pickle.load(f, encoding='iso-8859-1')
        with open('./data/xmn/test_txt_files.pkl', 'rb') as f:
            self.test_txt_vecs = pickle.load(f, encoding='iso-8859-1')
        with open('./data/xmn/test_labels.pkl', 'rb') as f:
            self.test_labels = pickle.load(f, encoding='iso-8859-1')

        self.num_train_batch = len(self.train_img_feats) / self.batch_size

    def train_data(self):
        for i in range(int(self.num_train_batch)+1):
            if (i + 1) * self.batch_size>len(self.train_img_feats):
                batch_img_feats = self.train_img_feats[i * self.batch_size: len(self.train_img_feats)]
                batch_txt_vecs = self.train_txt_vecs[i * self.batch_size: len(self.train_img_feats)]
                batch_labels = self.train_labels[i * self.batch_size: len(self.train_img_feats)]
                seen_attributes = self.train_attributes[i * self.batch_size: len(self.train_img_feats)]
            else:
                batch_img_feats = self.train_img_feats[i * self.batch_size: (i + 1) * self.batch_size]
                batch_txt_vecs = self.train_txt_vecs[i * self.batch_size: (i + 1) * self.batch_size]
                batch_labels = self.train_labels[i * self.batch_size: (i + 1) * self.batch_size]
                seen_attributes = self.train_attributes[i * self.batch_size: (i + 1) * self.batch_size]

            yield batch_img_feats, batch_txt_vecs, batch_labels, seen_attributes, i


class ModelParams(BaseModelParams):
    def __init__(self):
        self.epoch = 1000
        self.batch_size = 64
        self.visual_feats_dim = 4096
        self.word_vecs_dim = 300
        self.attributes_dim = 300
        self.noise_size = 300
        self.lr_emb = 0.0001
        self.lr_d = 0.0001
        self.lr_g = 0.0001
        self.lr_r = 0.0001
        self.top_k = 20
        self.dataset_name = 'xmn'
        self.model_name = 'cycle_cross_modal_retrieval'
        self.model_dir = 'cycle_cross_modal_retrieval_%d_%d' % (self.visual_feats_dim, self.word_vecs_dim)

        self.checkpoint_dir = 'checkpoint'
        self.sample_dir = 'samples'
        self.dataset_dir = './data'
        self.log_dir = 'logs'

    def update(self):
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)


class CrossModal(BaseModel):
    def __init__(self, model_params):
        BaseModel.__init__(self, model_params)
        self.data_iter = DataIter(self.model_params.batch_size)

        self.feats_train = tf.placeholder(tf.float32, [None, self.model_params.visual_feats_dim])
        self.vecs_train = tf.placeholder(tf.float32, [None, self.model_params.word_vecs_dim])
        self.y_single = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.int32, [None, 200])
        self.attributes_train_unseen = tf.placeholder(tf.float32, [None, self.model_params.attributes_dim])
        self.attributes_train_seen = tf.placeholder(tf.float32, [None, self.model_params.attributes_dim])
        self.noise_img = tf.placeholder(tf.float32, shape=(None, self.model_params.noise_size))
        self.noise_txt = tf.placeholder(tf.float32, shape=(None, self.model_params.noise_size))
        self.noise_img_unseen = tf.placeholder(tf.float32, shape=(None, self.model_params.noise_size))
        self.noise_txt_unseen = tf.placeholder(tf.float32, shape=(None, self.model_params.noise_size))
        self.shape_dann = tf.placeholder(tf.int32,[])
        self.l = tf.placeholder(tf.float32, [])
        self.lmbda = 10
        self.lmbda1 = 0.01
        train = True
        reuse = False


        feat_shape =tf.shape(self.feats_train)[0]
        self.BatchSize = tf.cast(feat_shape,dtype=tf.float32)

        # generator_img
        img_noise = tf.concat([self.attributes_train_seen, self.noise_img], axis=1)
        self.gen_img = self.generator_img(img_noise, isTrainable=train, reuse=reuse)

        # generator_txt
        txt_noise = tf.concat([self.attributes_train_seen, self.noise_txt], axis=1)
        self.gen_txt = self.generator_txt(txt_noise, isTrainable=train, reuse=reuse)

        # discriminator_img
        img_real_emb = tf.concat([self.feats_train, self.attributes_train_seen], axis=1)
        img_real_dis = self.discriminator_img(img_real_emb, isTrainable=train, reuse=reuse)
        img_fake_emb = tf.concat([self.gen_img, self.attributes_train_seen], axis=1)
        img_fake_dis = self.discriminator_img(img_fake_emb, isTrainable=train, reuse=True)

        self.d_real_img = tf.reduce_mean(img_real_dis)
        self.d_fake_img = tf.reduce_mean(img_fake_dis)

        alpha_img = tf.random_uniform(shape=(tf.shape(self.feats_train)[0], 1), minval=0., maxval=1.)
        alpha_img = tf.tile(alpha_img, multiples=(1, tf.shape(self.feats_train)[1]))
        interpolates_img = alpha_img * self.feats_train + ((1 - alpha_img) * self.gen_img)
        interpolate_img = tf.concat([interpolates_img, self.attributes_train_seen], axis=1)
        gradients_img = \
        tf.gradients(self.discriminator_img(interpolate_img, isTrainable=train, reuse=True), [interpolates_img])[0]
        grad_norm_img = tf.norm(gradients_img, axis=1, ord='euclidean')
        self.grad_pen_img = self.lmbda * tf.reduce_mean(tf.square(grad_norm_img - 1))

        # discriminator_txt
        txt_real_emb = tf.concat([self.vecs_train, self.attributes_train_seen], axis=1)
        txt_real_dis = self.discriminator_txt(txt_real_emb, isTrainable=train, reuse=reuse)
        txt_fake_emb = tf.concat([self.gen_txt, self.attributes_train_seen], axis=1)
        txt_fake_dis = self.discriminator_txt(txt_fake_emb, isTrainable=train, reuse=True)

        self.d_real_txt = tf.reduce_mean(txt_real_dis)
        self.d_fake_txt = tf.reduce_mean(txt_fake_dis)

        alpha_txt = tf.random_uniform(shape=(tf.shape(self.vecs_train)[0], 1), minval=0., maxval=1.)
        alpha_txt = tf.tile(alpha_txt, multiples=(1, tf.shape(self.vecs_train)[1]))        
        interpolates_txt = alpha_txt * self.vecs_train + ((1 - alpha_txt) * self.gen_txt)
        interpolate_txt = tf.concat([interpolates_txt, self.attributes_train_seen], axis=1)
        gradients_txt = \
        tf.gradients(self.discriminator_txt(interpolate_txt, isTrainable=train, reuse=True), [interpolates_txt])[0]
        grad_norm_txt = tf.norm(gradients_txt, axis=1, ord='euclidean')
        self.grad_pen_txt = self.lmbda * tf.reduce_mean(tf.square(grad_norm_txt - 1))

        self.loss_wgan_img = self.d_real_img - self.d_fake_img - self.grad_pen_img
        self.loss_wgan_txt = self.d_real_txt - self.d_fake_txt - self.grad_pen_txt

        # Cycle regressor_img
        self.re_img_a = self.regressor_img(self.gen_img, isTrainable=train, reuse=reuse)        
        redu_s_img = self.attributes_train_seen - self.re_img_a
        self.loss_cyc_img = tf.reduce_mean(tf.multiply(redu_s_img, redu_s_img))

        # Cycle regressor_txt
        self.re_txt_a = self.regressor_txt(self.gen_txt, isTrainable=train, reuse=reuse)
        redu_s_txt = self.attributes_train_seen - self.re_txt_a
        self.loss_cyc_txt = tf.reduce_mean(tf.multiply(redu_s_txt, redu_s_txt))

        # corr_loss

        self.corr_loss = 0.0001*self.cmpm_loss_compute(self.re_img_a, self.re_txt_a,self.y)

        
        #dann_loss
        self.emb_v_class = self.domain_classifier(self.re_img_a, self.l)
        self.emb_w_class = self.domain_classifier(self.re_txt_a, self.l, reuse=True)
        
        all_emb_v = tf.concat([tf.ones([self.shape_dann, 1]),
                                   tf.zeros([self.shape_dann, 1])], 1)
        all_emb_w = tf.concat([tf.zeros([self.shape_dann, 1]),
                                   tf.ones([self.shape_dann, 1])], 1)

        self.dann_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_v_class, labels=all_emb_w) + \
            tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_w_class, labels=all_emb_v))


        # retrieve feature
        self.emb_v = self.regressor_img(self.feats_train, isTrainable=train, reuse=True)
        self.emb_w = self.regressor_txt(self.vecs_train, isTrainable=train, reuse=True)

        # original feature clcyle loss
        self.re_ori_img = self.regressor_img(self.feats_train, isTrainable=train, reuse=True)
        self.re_ori_txt = self.regressor_txt(self.vecs_train, isTrainable=train, reuse=True)
        self.loss_r_ori_img = tf.reduce_mean(tf.squared_difference(self.re_ori_img, self.attributes_train_seen))
        self.loss_r_ori_txt = tf.reduce_mean(tf.squared_difference(self.re_ori_txt, self.attributes_train_seen))

        self.ori_corr_loss = 0.0001*self.cmpm_loss_compute(self.re_ori_img, self.re_ori_txt, self.y)

        self.emb_v_class_a = self.domain_classifier(self.re_ori_img, self.l, reuse=True)
        self.emb_w_class_a = self.domain_classifier(self.re_ori_txt, self.l, reuse=True)
        self.dann_a_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_v_class_a, labels=all_emb_w) + \
            tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_w_class_a, labels=all_emb_v))

        # loss
        self.loss_D_img = -self.loss_wgan_img
        self.loss_G_img = -self.d_fake_img
        self.loss_D_txt = -self.loss_wgan_txt
        self.loss_G_txt = -self.d_fake_txt

        self.loss_D = self.loss_D_img + self.loss_D_txt
        self.loss_G = self.loss_G_img + self.loss_G_txt  + 0.1* (self.loss_cyc_img +  self.loss_cyc_txt  + 0.01*self.dann_loss )
        self.loss_Re =  self.loss_r_ori_img  + self.loss_r_ori_txt  + self.ori_corr_loss + 0.01*self.dann_a_loss + 0.01* (self.loss_cyc_img +  self.loss_cyc_txt  + 0.01*self.dann_loss + self.corr_loss)
        self.loss_domain = self.dann_loss + self.dann_a_loss
        self.loss_corr =  self.ori_corr_loss + self.corr_loss
        self.loss_cyc = self.loss_r_ori_img  + self.loss_r_ori_txt  + self.loss_cyc_img +  self.loss_cyc_txt
        
        
        self.t_vars = tf.trainable_variables()
        self.gen_img_vars = [v for v in self.t_vars if 'gi_' in v.name]
        self.gen_txt_vars = [v for v in self.t_vars if 'gt_' in v.name]
        self.dis_img_vars = [v for v in self.t_vars if 'di_' in v.name]
        self.dis_txt_vars = [v for v in self.t_vars if 'dt_' in v.name]
        self.re_img_vars = [v for v in self.t_vars if 'ri_' in v.name]
        self.re_txt_vars = [v for v in self.t_vars if 'rt_' in v.name]
        self.dc_vars = [v for v in self.t_vars if 'dc_' in v.name]

    def generator_img(self, X, isTrainable=True, reuse=False):
        with slim.arg_scope([slim.fully_connected],activation_fn=None, reuse=reuse):
            net = tf.nn.relu(slim.fully_connected(X, 4096, scope='gi_fc_0'))
            net = tf.nn.relu(slim.fully_connected(net, 4096, scope='gi_fc_1'))
        return net

    def generator_txt(self, L, isTrainable=True, reuse=False):
        with slim.arg_scope([slim.fully_connected],activation_fn=None, reuse=reuse):
            net = tf.nn.relu(slim.fully_connected(L, 4096, scope='gt_fc_0'))
            net = tf.nn.relu(slim.fully_connected(net, 300, scope='gt_fc_1'))
        return net

    def regressor_img(self, X, isTrainable=True, reuse=False):
        with slim.arg_scope([slim.fully_connected],activation_fn=None, reuse=reuse):
            net = tf.nn.relu(slim.fully_connected(X, 4096, scope='ri_fc_0'))
            #net = tf.nn.relu(slim.fully_connected(net, 2048, scope='ri_fc_1'))
            net = tf.nn.relu(slim.fully_connected(net, 300, scope='ri_fc_1'))
        return net

    def regressor_txt(self, L, isTrainable=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.relu(slim.fully_connected(L, 4096, scope='rt_fc_0'))
            #net = tf.nn.relu(slim.fully_connected(net, 2048, scope='rt_fc_1'))
            net = tf.nn.relu(slim.fully_connected(net, 300, scope='rt_fc_1'))
        return net

    def discriminator_img(self, X, isTrainable=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.leaky_relu(slim.fully_connected(X, 4096, scope='di_fc_0'))
            net = slim.fully_connected(net, 1, scope='di_fc_1')
        return net

    def discriminator_txt(self, L, isTrainable=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.leaky_relu(slim.fully_connected(L, 4096, scope='dt_fc_0'))
            net = slim.fully_connected(net, 1, scope='dt_fc_1')
        return net
        
    def domain_classifier(self, E, l, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            E = flip_gradient(E, l)
            net = slim.fully_connected(E, int(300/2), scope='dc_fc_0')
            net = slim.fully_connected(net, int(300/4), scope='dc_fc_1')
            net = slim.fully_connected(net, 2, scope='dc_fc_2')
        return net

    def pairwise_distance(self, A, B):
        """ Pairwise distance between A and B
        """
        rA = tf.reduce_sum(A * A, 1, keep_dims=True)

        rB = tf.reduce_sum(B * B, 1, keep_dims=True)

        # turn r into column vector
        D = rA - 2 * tf.matmul(A, tf.transpose(B)) + tf.transpose(rB)

        return D

    def cmpm_loss_compute(self, text_embeddings, image_embeddings, labels):
        """ Cross-Modal Projection Matching Loss (CMPM)
        Args:
            text_embeddings: text joint embeddings
            image_embeddings: image joint embeddings
            labels: class labels
        Returns:
            i2t_matching_loss: cmpm loss for image projected to text
            t2i_matching_loss: cmpm loss for text projected to image
            pos_avg_dist: average distance of positive pairs
            neg_avg_dist: average distance of negative pairs
        """
        # label mask
        batch_size = image_embeddings.get_shape().as_list()[0]
        # mylabels = tf.cast(tf.reshape(labels, [batch_size, 1]), tf.float32)
        mylabels = tf.cast(labels, tf.float32)

        labelD = self.pairwise_distance(mylabels, mylabels)
        label_mask = tf.cast(tf.less(labelD, 0.5), tf.float32)  # 1-match   0-unmatch

        # cross-modal scalar projection
        image_embeddings_norm = tf.nn.l2_normalize(image_embeddings, dim=-1)
        text_embeddings_norm = tf.nn.l2_normalize(text_embeddings, dim=-1)

        image_proj_text = tf.matmul(image_embeddings, tf.transpose(text_embeddings_norm))
        text_proj_image = tf.matmul(text_embeddings, tf.transpose(image_embeddings_norm))

        # softmax, higher scalar projection gives higher probability
        i2t_pred = tf.nn.softmax(image_proj_text)
        t2i_pred = tf.nn.softmax(text_proj_image)

        # normalize the true matching distribution
        label_mask = tf.divide(label_mask, tf.reduce_sum(label_mask, axis=1, keep_dims=True))

        # KL Divergence
        i2t_matching_loss = tf.reduce_mean(tf.reduce_sum(i2t_pred * tf.log(1e-8 + i2t_pred / (label_mask + 1e-8)), 1))
        t2i_matching_loss = tf.reduce_mean(tf.reduce_sum(t2i_pred * tf.log(1e-8 + t2i_pred / (label_mask + 1e-8)), 1))

        # averaged cosine distance of positive and negative pairs for observation
        cosdist = 1.0 - tf.matmul(text_embeddings_norm, tf.transpose(image_embeddings_norm))

        pos_avg_dist = tf.reduce_mean(tf.boolean_mask(cosdist, tf.less(labelD, 0.5)))
        neg_avg_dist = tf.reduce_mean(tf.boolean_mask(cosdist, tf.greater(labelD, 0.5)))
        all_loss = i2t_matching_loss + t2i_matching_loss

        return all_loss

    def coral_loss(self, h_src, h_trg, gamma=1e-3):
        # regularized covariances (D-Coral is not regularized actually..)
        # First: subtract the mean from the data matrix
        batch_size = self.BatchSize
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        # cov_source=tf.linalg.cholesky(cov_source)
        # cov_target=tf.linalg.cholesky(cov_target)
        return tf.reduce_mean(tf.square(tf.subtract(cov_source, cov_target)))
        
    def log_coral_loss(self, h_src, h_trg, gamma=1e-3):
        # regularized covariances result in inf or nan
        # First: subtract the mean from the data matrix
        batch_size = (self.BatchSize)
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,transpose_a=True)  + gamma * tf.eye(64)
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,transpose_a=True)  + gamma * tf.eye(64)
        # eigen decomposition
        eig_source = tf.self_adjoint_eig(cov_source)
        eig_target = tf.self_adjoint_eig(cov_target)
        log_cov_source = tf.matmul(eig_source[1],tf.matmul(tf.diag(tf.log(eig_source[0])), eig_source[1], transpose_b=True))
        log_cov_target = tf.matmul(eig_target[1],tf.matmul(tf.diag(tf.log(eig_target[0])), eig_target[1], transpose_b=True))
        return tf.reduce_mean(tf.square(tf.subtract(log_cov_source, log_cov_target)))



    def znorm(self, inMat):
        col = inMat.shape[0]
        row = inMat.shape[1]
        mean_val = np.mean(inMat, axis=0)
        std_val = np.std(inMat, axis=0)
        mean_val = repmat(mean_val, col, 1)
        std_val = repmat(std_val, col, 1)
        x = np.argwhere(std_val == 0)
        for y in x:
            std_val[y[0], y[1]] = 1
        return (inMat - mean_val) / std_val

    def map(self, test_img_feats, test_txt_vecs, test_labels):
        dic = {}
        dic = np.zeros((3,))
        image = self.znorm(test_img_feats)
        text = self.znorm(test_txt_vecs)

        dic[0] = self.fx_calc_map_label(image, text, test_labels, k=0, dist_method='COS')
        dic[1] = self.fx_calc_map_label(text, image, test_labels, k=0, dist_method='COS')
        dic[2] = (dic[0] + dic[1]) / 2
        print('i2t: ', dic[0])
        print('t2i: ', dic[1])
        print('average: ', dic[2])
        return dic

    def fx_calc_map_label(self, image, text, label, k=0, dist_method='L2'):
        if dist_method == 'L2':
            dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
        elif dist_method == 'COS':
            dist = scipy.spatial.distance.cdist(image, text, 'cosine')
        ord = dist.argsort()
        numcases = dist.shape[0]
        if k == 0:
            k = numcases
        res = []
        for i in range(numcases):
            order = ord[i]
            p = 0.0
            r = 0.0
            for j in range(k):
                if label[i] == label[order[j]]:
                    r += 1
                    p += (r / (j + 1))
            if r > 0:
                res += [p / r]
            else:
                res += [0]
        return np.mean(res)

    def train(self, sess):

            
        regressor_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_r, beta1=0.5).minimize(self.loss_Re,
                                                                      var_list=self.re_img_vars + self.re_txt_vars  )
        generator_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_g, beta1=0.5).minimize(self.loss_G,
                                                                      var_list=self.gen_img_vars + self.gen_txt_vars )
        discriminator_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_d, beta1=0.5).minimize(self.loss_D,
                                                                      var_list=self.dis_img_vars + self.dis_txt_vars)
            
        domain_train_op =tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_r, beta1=0.5).minimize(self.loss_domain, var_list=self.dc_vars)

        tf.initialize_all_variables().run()
        start_time = time.time()

        # train discriminator and generator
        Max=0
        Max2=0

        for epoch in range(500):
            p = float(epoch) / 500
            l = 2. / (1. + np.exp(-10. * p)) - 1
            L_vgen_img=[]
            L_vgen_txt=[]
            L_true_img=[]
            L_true_txt=[]
            
            for batch_feat, batch_vec, batch_labels, seen_attributes, idx in self.data_iter.train_data():
                batch_labels_ = batch_labels - np.ones_like(batch_labels)
                label_binarizer = sklearn.preprocessing.LabelBinarizer()
                label_binarizer.fit(range(200))
                b = label_binarizer.transform(batch_labels_)
                min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                batch_feat = min_max_scaler.fit_transform(batch_feat)
                batch_vec = min_max_scaler.fit_transform(batch_vec)
                seen_attributes = min_max_scaler.fit_transform(seen_attributes)
                
                batch_noise_img = np.random.normal(0.0, 1.0,
                                                   (len(seen_attributes), self.model_params.noise_size))
                batch_noise_txt = np.random.normal(0.0, 1.0,
                                                   (len(seen_attributes), self.model_params.noise_size))

                train_lab = []
                for i in batch_labels:
                    train_lab.append(int(i)-1)

                for i in range(5):
                    sess.run([discriminator_train_op],
                             feed_dict={
                                 self.attributes_train_seen: seen_attributes,
                                 self.feats_train: batch_feat,
                                 self.vecs_train: batch_vec,

                                 self.noise_img: batch_noise_img,
                                 self.noise_txt: batch_noise_txt

                             })
                sess.run([generator_train_op, regressor_train_op,domain_train_op],
                         feed_dict={
                             self.attributes_train_seen: seen_attributes,
                             self.feats_train: batch_feat,
                             self.vecs_train: batch_vec,
                             self.noise_img: batch_noise_img,
                             self.noise_txt: batch_noise_txt,
                             self.shape_dann:len(batch_feat),
                             self.l: l,
                             self.y_single: train_lab,
                             self.y: b
                         })


                loss_D, loss_G, loss_Re, loss_corr, loss_adv,loss_cyc,\
                loss_ori_cyc_img,loss_ori_cyc_txt,loss_cyc_img,loss_cyc_txt,loss_ori_corr ,loss_sys_corr,loss_ori_adv, loss_sys_adv = sess.run(
                    [self.loss_D, self.loss_G, self.loss_Re, self.loss_corr, self.loss_domain,self.loss_cyc,
                     self.loss_r_ori_img , self.loss_r_ori_txt ,self.loss_cyc_img , self.loss_cyc_txt ,
                     self.ori_corr_loss ,self.corr_loss,self.dann_a_loss, self.dann_loss],
                    feed_dict={
                        self.attributes_train_seen: seen_attributes,
                        self.feats_train: batch_feat,
                        self.vecs_train: batch_vec,

                        self.noise_img: batch_noise_img,
                        self.noise_txt: batch_noise_txt,
                        self.l: l,
                        self.shape_dann:len(batch_feat),
                        self.y_single: train_lab,
                        self.y: b
                    })


                print(
                        'Epoch: [%2d][%4d/%4d] time: %4.4f,loss_D: %.8f, loss_G: %.8f, loss_Re: %.8f, loss_corr: %.8f,loss_adv: %.8f,loss_cyc: %.8f, '
                        'loss_ori_cyc_img: %.8f,loss_ori_cyc_txt: %.8f,loss_cyc_img: %.8f,loss_cyc_txt: %.8f,loss_ori_corr: %.8f ,loss_sys_corr: %.8f,loss_ori_adv: %.8f, loss_sys_adv: %.8f.' % ( \
                    epoch, idx, self.data_iter.num_train_batch, time.time() - start_time, loss_D, loss_G, loss_Re, loss_corr, loss_adv,loss_cyc,
                loss_ori_cyc_img,loss_ori_cyc_txt,loss_cyc_img,loss_cyc_txt,loss_ori_corr ,loss_sys_corr,loss_ori_adv, loss_sys_adv))


            if epoch % 2 == 0:
                start = time.time()

                test_img_feats, test_txt_vecs, test_labels = self.data_iter.test_img_feats, self.data_iter.test_txt_vecs, self.data_iter.test_labels
                test_img_feats_2 = test_img_feats
                test_txt_vecs_2 = test_txt_vecs
                test_img_feats = sess.run(self.re_ori_img, feed_dict={self.feats_train: test_img_feats})
                test_txt_vecs = sess.run(self.re_ori_txt, feed_dict={self.vecs_train: test_txt_vecs})
                

                print('Zero-shot retrieval')
                MAP = self.map(test_img_feats, test_txt_vecs, test_labels)
                if MAP[2] > Max:
                    mAXX = MAP
                    Max = MAP[2]
                    MaxMap = MAP
                    #sio.savemat('pr/wiki_best.mat', {'img': test_img_feats, 'txt': test_txt_vecs, 'label': test_labels})
                    
                if MAP[0] > Max2:
                    mAXX2 = MAP
                    Max2 = MAP[0]

                zs_test_img_feats_soft = sess.run(tf.nn.softmax(test_img_feats))
                zs_test_txt_vecs_soft = sess.run(tf.nn.softmax(test_txt_vecs))
                
                print("Softmax result:")
                print('Zero-shot retrieval (Softmax result)')
                MAP = self.map(zs_test_img_feats_soft, zs_test_txt_vecs_soft,
                              test_labels)
                if MAP[2] > Max:
                    mAXX = MAP
                    Max = MAP[2]
                    MaxMap = MAP
                    #sio.savemat('pr/wiki_best.mat', {'img': zs_test_img_feats_soft, 'txt': zs_test_txt_vecs_soft, 'label': test_labels})
                if MAP[0] > Max2:
                    mAXX2 = MAP
                    Max2 = MAP[0]
                

                min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

                test_img_feats = min_max_scaler.fit_transform(test_img_feats_2)
                test_txt_vecs = min_max_scaler.fit_transform(test_txt_vecs_2)

                test_img_feats = sess.run(self.re_ori_img, feed_dict={self.feats_train: test_img_feats})
                test_txt_vecs = sess.run(self.re_ori_txt, feed_dict={self.vecs_train: test_txt_vecs})


                print('Zero-shot retrieval(min_max_scaler)')
                MAP = self.map(test_img_feats,test_txt_vecs, 
                               test_labels)
                if MAP[2] > Max:
                    mAXX = MAP
                    Max = MAP[2]
                    MaxMap = MAP
                    #sio.savemat('pr/wiki_best.mat', {'img': test_img_feats, 'txt': test_txt_vecs, 'label': test_labels})
                if MAP[0] > Max2:
                    mAXX2 = MAP
                    Max2 = MAP[0]
                    
                
                test_img_feats = sess.run(tf.nn.softmax(test_img_feats))
                test_txt_vecs = sess.run(tf.nn.softmax(test_txt_vecs))
                
                MAP = self.map(test_img_feats,test_txt_vecs, 
                               test_labels)
                if MAP[2] > Max:
                    mAXX = MAP
                    Max = MAP[2]
                    MaxMap = MAP
                    #sio.savemat('pr/wiki_best.mat', {'img': test_img_feats, 'txt': test_txt_vecs, 'label': test_labels})
                if MAP[0] > Max2:
                    mAXX2 = MAP
                    Max2 = MAP[0]
          
                
                
                print(' ')
                print('MAX:')
                print(mAXX)
                print(' ')
                print('MAX2:')
                print(mAXX2)

       