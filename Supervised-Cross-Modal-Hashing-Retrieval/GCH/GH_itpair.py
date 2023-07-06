import tensorflow as tf
import os
import cPickle as pickle
from tqdm import tqdm
import scipy.io as sio
from setting import *
from tnet import *
from utils.calc_hammingranking import calc_map

class GH(object):
    def __init__(
            self, sess):
        self.batch_size = batch_size
        self.num_proposal = num_proposal
        self.var = {}

        self.train_L = train_L
        self.train_X = train_x
        self.train_Y = train_y
        # self.train_s = Smt_train

        self.query_L = query_L
        self.query_X = query_x
        self.query_Y = query_y

        self.retrieval_L = retrieval_L
        self.retrieval_X = retrieval_x
        self.retrieval_Y = retrieval_y

        self.Lr_img = lr_img
        self.Lr_txt = lr_txt
        self.Lr_lab = lr_lab
        self.Lr_gph = lr_gph
        self.Sim = Sim

        self.meanpix = meanpix
        self.img_net_itpair = img_net_itpair
        self.txt_net_itpair = txt_net_itpair
        self.lab_net = lab_net
        self.GCN_stack = GCN_stack
        self.full_conv_stack = full_conv_stack
        self.sum_to_vec_1 = sum_to_vec_1
        self.sum_to_vec_2 = sum_to_vec_2
        self.Att_pooling_logit = Att_pooling_logit
        self.Att_pooling_code = Att_pooling_code


        self.mse_loss = mse_criterion
        self.sce_loss = sce_criterion

        self.image_size = image_size
        self.numClass = numClass
        self.dimText = dimText
        self.dimLab = dimLab
        self.phase = phase
        self.checkpoint_dir = checkpoint_dir
        self.dataset_dir = dataset_dir
        self.bit = bit
        self.num_train = num_train
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.SEMANTIC_EMBED = SEMANTIC_EMBED
        self.decay = decay
        self.decay_steps = decay_steps
        self._build_model()
        self.saver = tf.train.Saver()
        self.sess = sess

    def _build_model(self):
        self.ph = {}
        # self.ph['label_input'] = tf.placeholder(tf.float32, [None, self.numClass], name='label_input')
        self.ph['label_input'] = tf.placeholder(tf.float32, [None, 1, self.numClass, 1], name='label_input')
        self.ph['image_input'] = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='image_input')
        self.ph['text_input'] = tf.placeholder(tf.float32, [None, 1, self.dimText, 1], name='text_input')
        self.ph['lr_txt'] = tf.placeholder(tf.float32, None, name='lr_txt')
        self.ph['lr_img'] = tf.placeholder(tf.float32, None, name='lr_img')
        self.ph['learn_rate'] = tf.placeholder(tf.float32, None, name='learn_rate')
        self.ph['keep_prob'] = tf.placeholder(tf.float32, None, name='keep_prob')
        self.ph['Adj'] = tf.placeholder('float32', [None, None], name='Adj')
        self.ph['indices'] = tf.placeholder(tf.int64, name='indices')
        self.ph['data'] = tf.placeholder(tf.float32, name='data')
        self.ph['GCN_in'] = tf.placeholder('float32', [None, None], name='GCN_in')
        # self.ph['graph_logit'] = tf.placeholder('float32', [None, self.numClass], name='graph_logit')
        # self.ph['graph_code'] = tf.placeholder('float32', [None, self.bit], name='graph_code')
        self.ph['I'] = tf.placeholder('float32', [None, self.bit], name='I')
        self.ph['T'] = tf.placeholder('float32', [None, self.bit], name='T')
        self.ph['B'] = tf.placeholder('float32', [None, self.bit], name='B')
        self.ph['H'] = tf.placeholder('float32', [None, self.bit], name='H')
        self.ph['Sim'] = tf.placeholder('float32', [None, self.batch_size], name='Sim')
        self.ph['I_feat'] = tf.placeholder('float32', [None, 1, 1, self.SEMANTIC_EMBED], name='I_feat')
        self.ph['T_feat'] = tf.placeholder('float32', [None, 1, 1, self.SEMANTIC_EMBED], name='T_feat')
        self.ph['epoch'] = tf.placeholder(tf.float32, None, name='Pepoch')
        self.ph['L_batch'] = tf.placeholder('float32', [None, self.numClass], name='L_batch')

        # construct image network
        self.Hsh_I, self.Fea_I, self.Lab_I = self.img_net_itpair(self.ph['image_input'], self.bit, self.numClass, reuse=False,
                                                          name="Img_Network")

        # construct text network
        self.Hsh_T, self.Fea_T, self.Lab_T = self.txt_net_itpair(self.ph['text_input'], self.bit, self.dimText, self.numClass,
                                                          reuse=False, name="Txt_Network")

        # construct label network
        self.Hsh_L, self.Fea_L, self.Lab_L = self.lab_net(self.ph['label_input'], self.bit, self.dimLab, reuse=False,
                                                          name="Lab_Network")

        self.Fea_I = tf.squeeze(self.Fea_I, axis=[1, 2])
        self.Fea_T = tf.squeeze(self.Fea_T, axis=[1, 2])
        self.Fea_I_slfatt = tf.matmul(self.Fea_I, tf.transpose(self.Fea_T))
        self.Fea_T_slfatt = tf.matmul(self.Fea_T, tf.transpose(self.Fea_I))


        self.Fea_I_w = matrix_norm(self.Fea_I_slfatt)
        self.Fea_T_w = matrix_norm(self.Fea_T_slfatt)

        # self.Fea_I_norm = tf.matmul(tf.transpose(self.Fea_I_w), self.Fea_I)
        # self.Fea_T_norm = tf.matmul(tf.transpose(self.Fea_T_w), self.Fea_T)
        self.Fea_I_norm_ori = tf.matmul(self.Fea_T_w, self.Fea_I)
        self.Fea_T_norm_ori = tf.matmul(self.Fea_I_w, self.Fea_T)

        # self.Fea_I_norm = tf.reshape(self.Fea_I_norm_ori, [batch_size*dimLab, SEMANTIC_EMBED])
        # self.Fea_T_norm = tf.reshape(self.Fea_T_norm_ori, [batch_size*dimLab, SEMANTIC_EMBED])

        # self.Fea_I_1d_norm = tf.sqrt(tf.reduce_sum(tf.square(self.Fea_I_norm), axis=2))
        # self.Fea_T_1d_norm = tf.sqrt(tf.reduce_sum(tf.square(self.Fea_T_norm), axis=2))

        # self.adj = tf.div(tf.matmul(tf.expand_dims(self.Fea_I_1d_norm, 2), tf.transpose(tf.expand_dims(self.Fea_I_1d_norm, 2), [0, 2, 1])),
        #                   tf.matmul(self.Fea_I_norm, tf.transpose(self.Fea_T_norm, [0, 2, 1]))) #TODO: use cosine distance instead
        # self.adj = tf.expand_dims(self.adj, 1)

        # construct graph conv net
        # self.GCN_in = tf.add(0.5*self.Fea_I, 0.5*self.Fea_T)
        self.GCN_in = tf.add(0.5*self.Fea_I_norm_ori, 0.5*self.Fea_T_norm_ori)
        # self.GCN_in = tf.concat([self.Fea_I_norm_ori, self.Fea_T_norm_ori], 1)
        self.graph_code, self.graph_logit = self.GCN_stack(self.GCN_in, self.ph['indices'], self.ph['data'], name= "graph_Layer")
        # self.graph_code, self.graph_logit = self.GCN_stack(self.ph['I_feat'], self.ph['T_feat'], name="graph_Layer")
        item_g = tf.matmul(self.graph_code, tf.transpose(self.graph_code))
        self.Loss_pair_g = self.mse_loss(tf.multiply(self.ph['Sim'], item_g), tf.log(1.0 + tf.exp(item_g)))
        self.Loss_G_lab = self.sce_loss(self.graph_logit, self.ph['L_batch']) # TODO: Something represent relationships between pair and other data in a vector form

        self.Loss_G = self.Loss_G_lab + self.Loss_pair_g

        # theta_L_2 = tf.matmul(self.ph['H'], tf.transpose(self.Hsh_L))  #
        # self.Loss_pair_Hsh_L = self.mse_loss(tf.multiply(self.ph['Sim'], theta_L_2), tf.log(1.0 + tf.exp(theta_L_2)))
        # self.Loss_quant_L = self.mse_loss(self.Hsh_L, self.ph['B'])
        # self.Loss_label_L = self.sce_loss(self.ph['L_batch'], self.Lab_L)
        # self.Loss_l = gamma * self.Loss_pair_Hsh_L + eta * self.Loss_label_L#alpha * self.Loss_pair_Fea_L + beta * self.Loss_quant_L + +

        item_l = tf.matmul(self.ph['H'], tf.transpose(self.Hsh_L))
        self.Loss_l = self.mse_loss(tf.multiply(self.ph['Sim'], item_l), tf.log(1.0 + tf.exp(item_l))) + \
                      self.mse_loss(self.Lab_L, self.ph['L_batch'])

        # Update graph network
        # item_ig = tf.matmul(self.Hsh_L, tf.transpose(self.graph_code))
        # self.Loss_I_G = self.mse_loss(self.ph['B'], self.graph_code) + \
        #                 self.mse_loss(tf.multiply(self.ph['Sim'], item_ig), tf.log(1.0 + tf.exp(item_ig)))
        # item_tg = tf.matmul(self.Hsh_L, tf.transpose(self.graph_code))
        # self.Loss_T_G = self.mse_loss(self.ph['B'], self.graph_code) + \
        #                 self.mse_loss(tf.multiply(self.ph['Sim'], item_tg), tf.log(1.0 + tf.exp(item_tg)))

        # Update image network
        self.Loss_I_lab = self.mse_loss(self.Hsh_I, self.ph['B']) \
                        + self.mse_loss(self.Lab_I, self.ph['L_batch'])  # J3
        self.Loss_I_g = self.mse_loss(self.graph_code, self.Hsh_I) + self.mse_loss(self.Hsh_I, self.ph['B'])  # J2

        item_i = tf.matmul(self.ph['H'], tf.transpose(self.Hsh_I))
        self.Loss_pair_i = self.mse_loss(tf.multiply(self.ph['Sim'], item_i), tf.log(1.0 + tf.exp(item_i)))
        self.Loss_I_hsh = self.Loss_pair_i + self.mse_loss(self.Hsh_I, self.ph['B'])  # J1

        # Update text network
        self.Loss_T_lab = self.mse_loss(self.Hsh_T, self.ph['B']) \
                        + self.mse_loss(self.Lab_T, self.ph['L_batch'])
        self.Loss_T_g = self.mse_loss(self.graph_code, self.Hsh_T) + self.mse_loss(self.Hsh_T, self.ph['B'])

        item_t = tf.matmul(self.ph['H'], tf.transpose(self.Hsh_T))
        self.Loss_pair_t = self.mse_loss(tf.multiply(self.ph['Sim'], item_t), tf.log(1.0 + tf.exp(item_t)))
        self.Loss_T_hsh = self.Loss_pair_t + self.mse_loss(self.Hsh_T, self.ph['B'])

        # train variable
        all_vars = tf.trainable_variables()
        self.g_lab_vars = [var for var in all_vars if 'Lab_Network' in var.name]
        self.g_img_vars = [var for var in all_vars if 'Img_Network' in var.name]
        self.g_txt_vars = [var for var in all_vars if 'Txt_Network' in var.name]
        self.graph_vars = [var for var in all_vars if 'graph_Layer' in var.name]

        # for i ,t in enumerate(self.graph_Layer):
        #     print i, t.name
        # #
        # print 'all'

        # Learning rate
        self.lr_lab = tf.train.exponential_decay(
            learning_rate=self.Lr_lab, global_step=self.ph['epoch'], decay_steps=1, decay_rate=decay, staircase=True)
        self.lr_img = tf.train.exponential_decay(
            learning_rate=self.Lr_img, global_step=self.ph['epoch'], decay_steps=1, decay_rate=decay, staircase=True)
        self.lr_txt = tf.train.exponential_decay(
            learning_rate=self.Lr_txt, global_step=self.ph['epoch'], decay_steps=1, decay_rate=decay, staircase=True)
        self.lr_gph = tf.train.exponential_decay(
            learning_rate=self.Lr_gph, global_step=self.ph['epoch'], decay_steps=1, decay_rate=decay, staircase=True)

        opt_l = tf.train.AdamOptimizer(self.ph['learn_rate'])
        gradient_l = opt_l.compute_gradients(self.Loss_l, var_list=self.g_lab_vars)
        self.train_l = opt_l.apply_gradients(gradient_l)

        opt_i_l = tf.train.AdamOptimizer(self.ph['learn_rate'])
        gradient_i_l = opt_i_l.compute_gradients(self.Loss_I_lab, var_list=self.g_img_vars)
        self.train_i_l = opt_i_l.apply_gradients(gradient_i_l)

        opt_i_h = tf.train.AdamOptimizer(self.ph['learn_rate'])
        gradient_i_h = opt_i_h.compute_gradients(self.Loss_I_hsh, var_list=self.g_img_vars)
        self.train_i_h = opt_i_h.apply_gradients(gradient_i_h)

        opt_i_g = tf.train.AdamOptimizer(self.ph['learn_rate'])
        gradient_i_g = opt_i_g.compute_gradients(self.Loss_I_g, var_list=self.g_img_vars)
        self.train_i_g = opt_i_g.apply_gradients(gradient_i_g)

        opt_t_l = tf.train.AdamOptimizer(self.ph['learn_rate'])
        gradient_t_l = opt_t_l.compute_gradients(self.Loss_T_lab, var_list=self.g_txt_vars)
        self.train_t_l = opt_t_l.apply_gradients(gradient_t_l)

        opt_t_h = tf.train.AdamOptimizer(self.ph['learn_rate'])
        gradient_t_h = opt_t_h.compute_gradients(self.Loss_T_hsh, var_list=self.g_txt_vars)
        self.train_t_h = opt_t_h.apply_gradients(gradient_t_h)

        opt_t_g = tf.train.AdamOptimizer(self.ph['learn_rate'])
        gradient_t_g = opt_t_g.compute_gradients(self.Loss_T_g, var_list=self.g_txt_vars)
        self.train_t_g = opt_t_g.apply_gradients(gradient_t_g)


        opt_g = tf.train.AdamOptimizer(self.ph['learn_rate'])
        gradient_all = opt_g.compute_gradients(self.Loss_G, var_list=self.graph_vars) #TODO: no var list ( var_list=self.graph_vars )
        self.train_all = opt_g.apply_gradients(gradient_all)

    def Train(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.var['F'] = np.random.randn(self.num_train, self.bit) #Img
        self.var['B_F'] = np.sign(self.var['F'])  # Img
        self.var['Fea_F'] = np.random.randn(self.num_train, SEMANTIC_EMBED) #Img
        self.var['G'] = np.random.randn(self.num_train, self.bit) #Txt
        self.var['B_G'] = np.sign(self.var['G']) #Txt
        self.var['Fea_G'] = np.random.randn(self.num_train, SEMANTIC_EMBED) #Txt
        self.var['H'] = np.random.randn(self.num_train, self.bit) #Label
        self.var['B_H'] = np.sign(self.var['H']) #Label
        self.var['M'] = np.random.randn(self.num_train, self.bit) #Graph
        self.var['B_M'] = np.sign(self.var['M']) #Graph
        self.var['Fea_M'] = np.random.randn(self.num_train, SEMANTIC_EMBED) #Graph
        self.var['L_H'] = np.random.randn(self.num_train, self.numClass)
        self.var['L_M'] = np.random.randn(self.num_train, self.numClass)
        # var['LABEL_L'] = np.random.randn(self.num_train, self.numClass)
        # var['LABEL_I'] = np.random.randn(self.num_train, self.numClass)
        # var['LABEL_T'] = np.random.randn(self.num_train, self.numClass)
        # var['feat_I'] = np.random.randn(self.num_train, self.SEMANTIC_EMBED)
        # var['feat_T'] = np.random.randn(self.num_train, self.SEMANTIC_EMBED)
        # var['feat_L'] = np.random.randn(self.num_train, self.SEMANTIC_EMBED)

        # Iterations
        for epoch in range(Epoch):
            results = {}
            results['loss_labNet'] = []
            results['loss_imgNet'] = []
            results['loss_txtNet'] = []
            results['Loss_D'] = []
            results['mapl2l'] = []
            results['mapi2i'] = []
            results['mapt2t'] = []

            if epoch % 1 == 0:
                print '++++++++Start train++++++++'
                if epoch <= MAX_ITER:
                    for idx in range(10):
                        # Train
                        learn_rate = self.sess.run(self.lr_lab, feed_dict={self.ph['epoch']: epoch})
                        self.train_lab(learn_rate)

                    qBL = self.generate_code(self.query_L, self.bit, "label")
                    rBL = self.generate_code(self.retrieval_L, self.bit, "label")

                    mapl2l = calc_map(qBL, rBL, self.query_L, self.retrieval_L)
                    print '=================================================='
                    print '...test map: map(l->l): {0}'.format(mapl2l)
                    print '=================================================='

                    for idx in range(5):
                        # Train
                        learn_rate = self.sess.run(self.lr_txt, feed_dict={self.ph['epoch']: epoch})
                        self.train_txt(learn_rate, update='label')  # original effective loss
                        # self.train_txt(learn_rate, update='self')
                        # self.train_txt(learn_rate, update='g')

                    qBY = self.generate_code(self.query_Y, self.bit, "text")
                    rBL = self.generate_code(self.retrieval_L, self.bit, "label")
                    qBL = self.generate_code(self.query_L, self.bit, "label")
                    rBY = self.generate_code(self.retrieval_Y, self.bit, "text")

                    mapy2l = calc_map(qBY, rBL, self.query_L, self.retrieval_L)
                    mapl2y = calc_map(qBL, rBY, self.query_L, self.retrieval_L)
                    print '=================================================='
                    print '...test map: map(y->l): {0}'.format(mapy2l)
                    print '...test map: map(l->y): {0}'.format(mapl2y)
                    print '=================================================='

                    for idx in range(5):
                        # Train
                        learn_rate = self.sess.run(self.lr_img, feed_dict={self.ph['epoch']: epoch})
                        self.train_img(learn_rate, update='label')  # original effective loss
                        # self.train_img(learn_rate, update='self')
                        # self.train_img(learn_rate, update='g')

                    qBX = self.generate_code(self.query_X, self.bit, "image")
                    rBL = self.generate_code(self.retrieval_L, self.bit, "label")
                    qBL = self.generate_code(self.query_L, self.bit, "label")
                    rBX = self.generate_code(self.retrieval_X, self.bit, "image")

                    mapx2l = calc_map(qBX, rBL, self.query_L, self.retrieval_L)
                    mapl2x = calc_map(qBL, rBX, self.query_L, self.retrieval_L)
                    print '=================================================='
                    print '...test map: map(x->l): {0}'.format(mapx2l)
                    print '...test map: map(l->x): {0}'.format(mapl2x)
                    print '=================================================='

                    for idx in range(5):
                        # Train
                        learn_rate = self.sess.run(self.lr_gph, feed_dict={self.ph['epoch']: epoch})
                        self.train_whole(learn_rate)

                    for idx in range(5):
                        # Train
                        learn_rate = self.sess.run(self.lr_txt, feed_dict={self.ph['epoch']: epoch})
                        self.train_txt(learn_rate, update='g')

                    for idx in range(5):
                        # Train
                        learn_rate = self.sess.run(self.lr_img, feed_dict={self.ph['epoch']: epoch})
                        self.train_img(learn_rate, update='g')

                    print "********test************"
                    self.test(self.phase)

    def test(self, phase):
        test = {}
        print '=========================================================='
        print '  ====                 Test map in all              ===='
        print '=========================================================='

        if phase == 'test' and self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        test['qBX'] = self.generate_code(self.query_X, self.bit, "image")
        test['qBY'] = self.generate_code(self.query_Y, self.bit, "text")
        test['rBX'] = self.generate_code(self.retrieval_X, self.bit, "image")
        test['rBY'] = self.generate_code(self.retrieval_Y, self.bit, "text")

        test['mapi2t'] = calc_map(test['qBX'], test['rBY'], self.query_L, self.retrieval_L)
        test['mapt2i'] = calc_map(test['qBY'], test['rBX'], self.query_L, self.retrieval_L)
        test['mapi2i'] = calc_map(test['qBX'], test['rBX'], self.query_L, self.retrieval_L)
        test['mapt2t'] = calc_map(test['qBY'], test['rBY'], self.query_L, self.retrieval_L)
        print '=================================================='
        print '...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (test['mapi2t'], test['mapt2i'])
        print '...test map: map(t->t): %3.3f, map(i->i): %3.3f' % (test['mapt2t'], test['mapi2i'])
        print '=================================================='

        # Save hash code
        datasetStr = DATA_DIR.split('/')[-1]
        dataset_bit_net = datasetStr + str(bit)
        mat_name = dataset_dir + '_' + str(bit) + '.mat'
        savePath = os.path.join(os.getcwd(), mat_name)
        if os.path.exists(savePath):
            os.remove(savePath)
        sio.savemat(dataset_bit_net, {'Qi': test['qBX'], 'Qt': test['qBY'],
                                      'Di': test['rBX'], 'Dt': test['rBY'],
                                      'retrieval_L': retrieval_L, 'query_L': query_L})

    def train_img(self, learn_rate, update):
        print 'update image net'+' using ' + str(update)
        for iter in tqdm(xrange(num_train / self.batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: self.batch_size]
            image = self.train_X[ind, :, :, :]
            image = image - np.repeat(self.meanpix[np.newaxis, :, :, :], self.batch_size, axis=0)
            text = self.train_Y[ind, :]
            text = text.reshape([text.shape[0], 1, text.shape[1], 1])
            label = train_L[ind, :]
            Label = label[:,np.newaxis,:,np.newaxis]
            S = self.Sim[:,ind][ind, :]
            indices, data, adj = self.s_adj(label)
            if update == 'label':
                self.train_i_l.run(feed_dict={self.ph['image_input']: image,
                                              self.ph['text_input']: text,
                                              self.ph['L_batch']: label,
                                              self.ph['label_input']: Label,
                                              self.ph['learn_rate']: learn_rate,
                                              self.ph['Sim']: S,
                                              self.ph['H']: self.var['H'][ind, :],
                                              self.ph['B']: self.var['B_H'][ind, :]})

            elif update == 'g':
                self.train_i_g.run(feed_dict={self.ph['image_input']: image,
                                              self.ph['text_input']: text,
                                              self.ph['L_batch']: label,
                                              self.ph['learn_rate']: learn_rate,
                                              self.ph['Sim']: S,
                                              self.ph['indices']: indices,
                                              self.ph['data']: data,
                                              self.ph['H']: self.var['H'][ind, :],
                                              self.ph['B']: self.var['B_H'][ind, :]})
            else:
                self.train_i_h.run(feed_dict={self.ph['image_input']: image,
                                              self.ph['text_input']: text,
                                              self.ph['L_batch']: label,
                                              self.ph['learn_rate']: learn_rate,
                                              self.ph['Sim']: S,
                                              self.ph['H']: self.var['M'][ind, :],
                                              self.ph['B']: self.var['B_M'][ind, :]})

            Hsh_I, Fea_I, Loss_I = self.sess.run([self.Hsh_I, self.Fea_I, self.Loss_I_hsh],
                                                 feed_dict={self.ph['image_input']: image,
                                                            self.ph['text_input']: text,
                                                            self.ph['L_batch']: self.var['L_H'][ind, :], #label,
                                                            self.ph['Sim']: S,
                                                            self.ph['H']: self.var['M'][ind, :],
                                                            self.ph['B']: self.var['B_M'][ind, :]})
            self.var['Fea_F'][ind, :] = np.squeeze(Fea_I)
            self.var['F'][ind, :] = Hsh_I
            self.var['B_F'][ind, :] = np.sign(Hsh_I)
            if iter % 10 == 0:
                print '...Loss_I_'+str(update)+':%3.3f' % (Loss_I)
        # return Loss_I

    def train_lab(self, learn_rate):
        print 'update lab net'
        for iter in tqdm(xrange(100)):#num_train / self.batch_size
            index = np.random.permutation(num_train)
            ind = index[0: self.batch_size]
            # image = self.train_X[ind]
            # image = image - np.repeat(self.meanpix[np.newaxis, :, :, :], self.batch_size, axis=0)
            # text = self.train_Y[ind, :]
            # text = text.reshape([text.shape[0], 1, text.shape[1], 1])
            label = train_L[ind, :]
            Label = label[:, np.newaxis, :, np.newaxis]
            S = self.Sim[:, ind]
            self.sess.run(self.train_l, feed_dict={
                                        # self.ph['image_input']: image,
                                        # self.ph['text_input']: text,
                                        self.ph['label_input']: Label,
                                        self.ph['learn_rate']: learn_rate,
                                        self.ph['Sim']: S,
                                        self.ph['B']: self.var['B_H'][ind, :],
                                        self.ph['L_batch']: label,
                                        self.ph['H']: self.var['H']})

            Hsh_L, Loss_L, Lab_L = self.sess.run([self.Hsh_L, self.Lab_L, self.Loss_l],
                                          feed_dict={
                                              # self.ph['image_input']: image,
                                              # self.ph['text_input']: text,
                                              self.ph['label_input']: Label,
                                              self.ph['Sim']: S,
                                              self.ph['B']:self.var['B_H'][ind, :],
                                              self.ph['L_batch']: label,
                                              self.ph['H']: self.var['H']})
            self.var['H'][ind, :] = Hsh_L
            self.var['B_H'][ind, :] = np.sign(Hsh_L)
            self.var['L_H'][ind, :] = Lab_L
        # return Loss_L

    def train_txt(self, learn_rate, update):
        print 'update text net'+' using ' + str(update)
        for iter in tqdm(xrange(num_train / self.batch_size)):#num_train / self.batch_size
            index = np.random.permutation(num_train)
            ind = index[0: self.batch_size]
            image = self.train_X[ind, :, :, :]
            image = image - np.repeat(self.meanpix[np.newaxis, :, :, :], self.batch_size, axis=0)
            text = self.train_Y[ind, :]
            text = text[:, np.newaxis, :, np.newaxis]
            label = train_L[ind, :]
            Label = label[:, np.newaxis, :, np.newaxis]
            S = self.Sim[:, ind][ind, :]
            indices, data, adj = self.s_adj(label)
            if update == "label":
                self.train_t_l.run(feed_dict={self.ph['image_input']: image,
                                              self.ph['text_input']: text,
                                              self.ph['L_batch']: self.var['L_H'][ind, :], #label,
                                              self.ph['label_input']: Label,
                                              self.ph['learn_rate']: learn_rate,
                                              self.ph['Sim']: S,
                                              self.ph['H']: self.var['H'][ind, :],
                                              self.ph['B']: self.var['B_H'][ind, :]})

            elif update == 'g':
                self.train_t_g.run(feed_dict={self.ph['image_input']: image,
                                              self.ph['text_input']: text,
                                              # self.ph['L_batch']: self.var['L_H'][ind, :],
                                              self.ph['L_batch']:  label,
                                              # self.ph['label_input']: Label,
                                              self.ph['learn_rate']: learn_rate,
                                              self.ph['Sim']: S,
                                              self.ph['indices']: indices,
                                              self.ph['data']: data,
                                              self.ph['H']: self.var['H'][ind, :],
                                              self.ph['B']: self.var['B_H'][ind, :]})
            else:
                self.train_t_h.run(feed_dict={self.ph['image_input']: image,
                                              self.ph['text_input']: text,
                                              self.ph['L_batch']: label,
                                              self.ph['learn_rate']: learn_rate,
                                              self.ph['Sim']: S,
                                              self.ph['H']: self.var['M'][ind, :],
                                              self.ph['B']: self.var['B_M'][ind,:]})

            Hsh_T, Fea_T, Loss_T = self.sess.run([self.Hsh_T, self.Fea_T, self.Loss_T_hsh],
                                                 feed_dict={self.ph['image_input']: image,
                                                            self.ph['text_input']: text,
                                                            self.ph['L_batch']: label,
                                                            self.ph['Sim']: S,
                                                            self.ph['H']: self.var['M'][ind, :],
                                                            self.ph['B']: self.var['B_M'][ind,:]})
            self.var['Fea_G'][ind, :] = np.squeeze(Fea_T)
            self.var['G'][ind, :] = Hsh_T
            self.var['B_G'][ind, :] = np.sign(Hsh_T)

            if iter % 10 == 0:
                print '...Loss_T_'+str(update)+':%3.3f' % (Loss_T)

        # return Loss_T


    def train_whole(self, learn_rate):
        print 'update whole net'
        for iter in tqdm(xrange(100)):#num_train / self.batch_size
            index = np.random.permutation(num_train)
            ind = index[0: self.batch_size]
            image = self.train_X[ind]
            image = image - np.repeat(self.meanpix[np.newaxis, :, :, :], self.batch_size, axis=0)
            text = self.train_Y[ind, :]
            text = text.reshape([text.shape[0], 1, text.shape[1], 1])
            label = train_L[ind, :]
            # Label = label[:, np.newaxis, :, np.newaxis]
            # I_feat = self.var['Fea_F'][ind, :]
            # T_feat = self.var['Fea_G'][ind, :]
            # I_feat = I_feat[:, np.newaxis, np.newaxis, :]
            # T_feat = T_feat[:, np.newaxis, np.newaxis, :]
            S = self.Sim[:, ind][ind, :]
            # semantics = Smt_train[ind, :]
            # indices, data, adj = self.semantic_adj(semantics)
            #
            # lab_batch = self.Fea_L.eval(feed_dict={self.ph['label_input']: Label})
            # adj = self.hash_l_adj(lab_batch)
            indices, data, adj = self.s_adj(label)
            # img_feat, txt_feat = self.sess.run([self.Fea_I_norm, self.Fea_I_norm],
            #                                    feed_dict={
            #                                        self.ph['image_input']: image,
            #                                        self.ph['text_input']: text,
            #                                    })
            # adj = self.nmlz_adj_itpair(image, text)
            # adj = self.semantic_adj(semantics)
            # graph_code = self.code_pooling(graph_code_ori)
            # graph_logit = self.logit_pooling(graph_logit_ori)

            self.train_all.run(feed_dict={self.ph['image_input']: image,
                                          self.ph['text_input']: text,
                                          self.ph['L_batch']: label,
                                          self.ph['learn_rate']: learn_rate,
                                          self.ph['Sim']: S,
                                          self.ph['indices']: indices,
                                          self.ph['data']: data,
                                          })
            # self.train_all.run(feed_dict={self.ph['image_input']: image,
            #                               self.ph['text_input']: text,
            #                               self.ph['label_input']: Label,
            #                               self.ph['learn_rate']: learn_rate,
            #                               self.ph['Sim']: S,
            #                               self.ph['Adj']: adj,
            #                               self.ph['H']: self.var['H'][ind, :],
            #                               self.ph['B']: self.var['B'][ind, :]})

            # graph_code, Loss_all, Loss_i, Loss_t, Loss_xe = self.sess.run([self.graph_code, self.Loss_G, self.Loss_I_G, self.Loss_T_G, self.Loss_xe],
            graph_code, Lab_G, Loss_all = self.sess.run([self.graph_code, self. graph_logit, self.Loss_G],
                                                              feed_dict={self.ph['image_input']: image,
                                                                         self.ph['text_input']: text,
                                                                         self.ph['L_batch']: label,
                                                                         self.ph['Sim']: S,
                                                                         self.ph['indices']: indices,
                                                                         self.ph['data']: data,
                                                                         })
            self.var['M'][ind, :] = graph_code
            self.var['B_M'][ind, :] = np.sign(graph_code)
            self.var['L_M'][ind, :] = Lab_G
            # graph_feat, graph_logit = self.sess.run([self.graph_code_ori, self.graph_logit_ori],
            #               feed_dict={self.ph['image_input']: image,
            #                          self.ph['text_input']: text,
            #                          self.ph['Sim']: S,
            #                          self.ph['Adj']: adj,
            #                          self.ph['GCN_in']: gcn_in
            #                          })
            # print "graph code:{0}".format(graph_feat)
            # print "graph logit:{0}".format(graph_logit)

        # return Loss_all#, Loss_i, Loss_t, Loss_xe


    def generate_code(self, Modal, bit, generate):
        batch_size = 128
        if generate == "label":
            pass
            num_data = Modal.shape[0]
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            B = np.zeros([num_data, bit], dtype=np.float32)
            for iter in tqdm(xrange(num_data / batch_size + 1)):
                ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
                label = Modal[ind, :]
                Label = label[:,np.newaxis,:,np.newaxis]
                Hsh_L = self.Hsh_L.eval(feed_dict={self.ph['label_input']: Label})
                B[ind, :] = Hsh_L
        elif generate == "image":
            num_data = len(Modal)
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            B = np.zeros([num_data, bit], dtype=np.float32)
            for iter in tqdm(xrange(num_data / batch_size + 1)):
                ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
                mean_pixel = np.repeat(self.meanpix[:, :, :, np.newaxis], len(ind), axis=3)
                image = Modal[ind, :, :, :].astype(np.float32)
                image = image - mean_pixel.astype(np.float32).transpose(3, 0, 1, 2)
                Hsh_I = self.Hsh_I.eval(feed_dict={self.ph['image_input']: image})
                B[ind, :] = Hsh_I
        else:
            num_data = Modal.shape[0]
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            B = np.zeros([num_data, bit], dtype=np.float32)
            for iter in tqdm(xrange(num_data / batch_size + 1)):
                ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
                text = Modal[ind, :].astype(np.float32)
                text = text.reshape([text.shape[0], 1, text.shape[1], 1])
                Hsh_T = self.Hsh_T.eval(feed_dict={self.ph['text_input']: text})
                B[ind, :] = Hsh_T
        B = np.sign(B)
        return B

    def calc_labnet_loss(self, H, label_, feature, SIM):
        term1 = np.sum(np.power((label_ - self.train_L), 2))

        theta_2 = np.matmul(H, np.transpose(H)) / 2
        term2 = np.sum(np.log(1 + np.exp(theta_2)) - SIM * theta_2)
        theta_3 = np.matmul(feature, np.transpose(feature)) / 2
        term3 = np.sum(np.log(1 + np.exp(theta_3)) - SIM * theta_3)

        loss = alpha * term1 + gamma * term2 + beta * term3  # + gama4 * term4 + gama5 * term5
        print 'label:', term1
        print 'pairwise_hash:', term2
        print 'pairwise_feat:', term3
        return loss

    def calc_loss(self, B, F, G, H, Sim, label_, label, alpha, beta, gamma, eta):
        theta = np.matmul(F, np.transpose(G)) / 2
        term1 = np.sum(np.log(1 + np.exp(theta)) - Sim * theta)

        term2 = np.sum(np.power(B - F, 2) + np.power(B - G, 2))
        term3 = np.sum(np.power(H - F, 2) + np.power(H - G, 2))
        term4 = np.sum(np.power((label_ - label), 2))

        loss = alpha * term1 + beta * term2 + gamma * term3 + eta * term4
        print 'pairwise:', term1
        print 'quantization:', term2
        print 'hash_feature:', term3
        print 'labe_predict:', term4
        return loss

    def calc_isfrom_acc(self, train_isfrom_, Train_ISFROM):
        erro = Train_ISFROM.shape[0] - np.sum(
            np.equal(np.sign(train_isfrom_ - 0.5), np.sign(Train_ISFROM - 0.5)).astype(int))
        acc = np.divide(np.sum(np.equal(np.sign(train_isfrom_ - 0.5), np.sign(Train_ISFROM - 0.5)).astype('float32')),
                        Train_ISFROM.shape[0])
        return erro, acc

    def save(self, checkpoint_dir, step):
        model_name = "GH"
        model_dir = "%s_%s" % (self.dataset_dir, self.bit)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.bit)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        print checkpoint_dir

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def nmlz_adj_batch(self, img_data, txt_data, batchsize=batch_size):
        n = self.numClass
        Adj = np.zeros([n * batchsize, n * batchsize], dtype=np.float32)
        score = np.zeros([n * batchsize, n * batchsize], dtype=np.float32)
        img_feat = self.Fea_I_norm_ori.eval(
            feed_dict={self.ph['image_input']: img_data, self.ph['text_input']: txt_data})  # 128*24*512
        txt_feat = self.Fea_T_norm_ori.eval(
            feed_dict={self.ph['text_input']: txt_data, self.ph['image_input']: img_data})  # 128*24*512
        for batch in range(batch_size):
            for num_n in range(n):
                for i in xrange(n):
                    for j in xrange(n):
                        adj = (np.dot(img_feat[batch, i, :], txt_feat[batch, j, :])) \
                              / (np.linalg.norm(img_feat[batch, i, :]) * np.linalg.norm(txt_feat[batch, j, :]))
                        score[num_n*batch_size + i, num_n*batch_size + j] = adj
        # di, q = np.linalg.eig(Adj)
        # n = np.where(di > 0, di, 0)
        # assert (np.max(n - di) == 0)
        idx = []
        for line in range(n * batchsize):
            idx.append(np.argsort(score[line, :])[0:2])
        # Adj = np.where(Adj > np.mean(Adj), 1, 0)
        for ind in range(n * batchsize):
            if ind != idx[ind][0]:
                Adj[ind, idx[ind][0]] = 1
            else:
                Adj[ind, idx[ind][1]] = 1
        Adj = Adj + np.eye(n * batchsize, n * batchsize)
        # Adj = Adj + eye
        # for k in range(batch_size * n):
        #     if Adj[k, k] == 0:
        #         Adj[k, k] = 1
        d = np.sum(Adj, axis=1)
        D = np.diag(np.power(d, -0.5))
        # term_1 = np.dot(self.neg_matpower(D, -0.5), Adj)
        term_1 = np.dot(D, Adj)
        Adj_out = np.dot(term_1, D)

        return Adj_out

    def nmlz_adj_itpair(self, img_data, txt_data, batchsize=batch_size):
        n = self.numClass
        Adj = np.zeros([batchsize, batchsize], dtype=np.float32)
        score = np.zeros([batchsize, batchsize], dtype=np.float32)
        img_feat = self.Fea_I.eval(
            feed_dict={self.ph['image_input']: img_data, self.ph['text_input']: txt_data})  # 128*512
        txt_feat = self.Fea_T.eval(
            feed_dict={self.ph['text_input']: txt_data, self.ph['image_input']: img_data})  # 128*512
        img_feat = normalize(img_feat)
        txt_feat = normalize(txt_feat)
        for i in range(batchsize):
            for j in range(batchsize):
                adj = (np.dot(img_feat[i, :], txt_feat[j, :].transpose())) \
                      / (np.linalg.norm(img_feat[i, :]) * np.linalg.norm(txt_feat[j, :]))
                score[i, j] = adj
        # di, q = np.linalg.eig(Adj)
        # n = np.where(di > 0, di, 0)
        # assert (np.max(n - di) == 0)
        idx = []
        for line in range(batchsize):
            idx.append(np.argsort(score[line, :])[0:2])
        # Adj = np.where(Adj > np.mean(Adj), 1, 0)
        for ind in range(batchsize):
            if ind != idx[ind][0]:
                Adj[ind, idx[ind][0]] = 1
            else:
                Adj[ind, idx[ind][1]] = 1
        Adj = Adj + np.eye(batchsize, batchsize)
        # Adj = Adj + eye
        # for k in range(batch_size * n):
        #     if Adj[k, k] == 0:
        #         Adj[k, k] = 1
        d = np.sum(Adj, axis=1)
        D = np.diag(np.power(d, -0.5))
        # term_1 = np.dot(self.neg_matpower(D, -0.5), Adj)
        term_1 = np.dot(D, Adj)
        Adj_out = np.dot(term_1, D)

        return Adj_out


    def code_pooling(self, code_ori):
        code_out = []
        code_temp = np.empty([dimLab, self.bit])
        for i in range(batch_size):
            for j in range(self.bit):
                code_temp = np.max(code_ori[dimLab*i:dimLab*(i+1), j])
                code_out.append(code_temp)
        code_out = np.reshape(code_out, [batch_size, self.bit])
        return code_out

    def logit_pooling(self, logit_ori):
        logit_out = []
        logit_temp = np.empty([dimLab, dimLab])
        for i in range(batch_size):
            for j in range(dimLab):
                logit_temp = np.max(logit_ori[dimLab*i:dimLab*(i+1), j])
                logit_out.append(logit_temp)
        logit_out = np.reshape(logit_out, [batch_size, dimLab])
        return logit_out

    def semantic_adj(self, semantics, batchsize=batch_size):
        # embedding_matrix = pickle.load(file('/home/xrq/Dataset/IJCAI19/Flickr/FlickrDict.pkl', 'rb')) # 24*300
        # semantics =  np.dot(label, embedding_matrix) # batch * 300
        adj_tmp = np.zeros([batch_size, batch_size], dtype=np.float32)
        score = []
        for i in range(batch_size):
            for j in range(batch_size):
                score.append(calc_similarity(semantics[i, :], semantics[j, :]))
        score = np.reshape(score, [batch_size, batch_size])
        score = score - np.eye(batch_size, batch_size)
        Adj = np.where(score > np.mean(score), 1, 0)
        # sio.savemat('score.mat', {'score': score})
        # idx = score.argmax(axis=1)
        # tmp = score
        # for line in range(batchsize):
        #     mean_tmp = np.mean(score[line, :])
        #     tmp[line, :] = np.where(score[line, :] > mean_tmp, 1, 0)

        # for k in range(len(idx)):
        #     adj_tmp[k, idx[k]] = 1
        # Adj = adj_tmp # use max value in adj
        # Adj = adj_tmp  # use all indices in adj_m
        Adj = Adj + np.eye(batch_size, batch_size)
        d = np.sum(Adj, axis=1)
        D = np.diag(np.power(d, -0.5))
        term_1 = np.dot(D, Adj)
        Adj_out = np.dot(term_1, D)

        row = []
        col = []
        data = []
        for i in range(batchsize):
            for j in range(batchsize):
                if Adj_out[i, j] != 0:
                    row.append(i)
                    col.append(j)
                    data.append(Adj_out[i, j])
        # Adj = csr_matrix((data, (row, col)), shape=(batch_size, batch_size))
        indices = []
        for i in range(len(data)):
            tmp = []
            tmp.append(row[i])
            tmp.append(col[i])
            indices.append(tmp)

        return indices, data, Adj_out


    def hash_l_adj(self, lab_batch, batchsize=batch_size):
        lab_batch = np.squeeze(lab_batch)
        adj_tmp = np.dot(lab_batch, lab_batch.transpose())
        adj_tmp = adj_tmp / np.max(adj_tmp)
        adj_tmp = adj_tmp - np.eye(batchsize, batchsize)
        idx = adj_tmp.argmax(axis=1)

        for k in range(len(idx)):
            adj_tmp[k, idx[k]] = 1
        Adj = adj_tmp + np.eye(batch_size, batch_size)
        d = np.sum(Adj, axis=1)
        D = np.diag(np.power(d, -0.5))
        # term_1 = np.dot(self.neg_matpower(D, -0.5), Adj)
        term_1 = np.dot(D, Adj)
        Adj_out = np.dot(term_1, D)

        return Adj_out

    def s_adj(self, label, batchsize=batch_size):
        adj_tmp = np.dot(label, label.transpose())
        Adj = adj_tmp #+ np.eye(batch_size, batch_size)
        d = np.sum(Adj, axis=1)
        D = np.diag(np.power(d, -0.5))
        term_1 = np.dot(D, Adj)
        Adj_out = np.dot(term_1, D)

        row = []
        col = []
        data = []
        for i in range(batchsize):
            for j in range(batchsize):
                if Adj_out[i, j] != 0:
                    row.append(i)
                    col.append(j)
                    data.append(Adj_out[i, j])
        # Adj = csr_matrix((data, (row, col)), shape=(batch_size, batch_size))
        indices = []
        for i in range(len(data)):
            tmp = []
            tmp.append(row[i])
            tmp.append(col[i])
            indices.append(tmp)

        return indices, data, Adj_out
