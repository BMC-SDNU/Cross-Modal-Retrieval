import os
import numpy as np
import tensorflow as tf
import pickle

from load_data import loading_data
from img_net import img_net_strucuture
from txt_net import txt_net_strucuture
from utils.calc_hammingranking import calc_map

from datetime import datetime

# environmental setting: setting the following parameters based on your experimental environment.
select_gpu = '3'
per_process_gpu_memory_fraction = 0.9

# data parameters
DATA_DIR = './DataSet/FashionVC/' 
TRAINING_SIZE = 16862
QUERY_SIZE = 3000
DATABASE_SIZE = 16862
# DATA_DIR = './DataSet/Ssense/'
# TRAINING_SIZE = 13696
# QUERY_SIZE = 2000
# DATABASE_SIZE = 13696

# hyper-parameters
MAX_ITER = 30
num_class1 = 8
num_class2 = 27
# num_class1 = 4
# num_class2 = 28
num_class = num_class1 + num_class2
alpha = 0.3
beta = 0.7
gamma = 10
eta = 100
bit = 128

filename = 'log/result_' + datetime.now().strftime("%d-%h-%m-%s") + '_' + str(bit) + 'bits_FashionVC.pkl'


def train_img_net(image_input, cur_f_batch, var, ph, train_x, train_L, lr, train_step_x, mean_pixel_):
    F = var['F']
    batch_size = var['batch_size']
    num_train = train_x.shape[0]
    for iter in range((int)(num_train / batch_size)):
        index = np.random.permutation(num_train)
        ind = index[0: batch_size]
        sample_L = train_L[ind, :]
        image = train_x[ind, :, :, :].astype(np.float64)
        image = image - mean_pixel_.astype(np.float64)

        cur_f = cur_f_batch.eval(feed_dict={image_input: image})
        F[:, ind] = cur_f

        train_step_x.run(
            feed_dict={ph['L1']: sample_L[:, 0:num_class1], ph['L2']: sample_L[:, num_class1:num_class],
                       ph['b_batch']: var['B'][:, ind],
                       ph['y1']: var['Y1'], ph['y2']: var['Y2'], image_input: image})

    return F


def train_txt_net(text_input, cur_g_batch, var, ph, train_y, train_L, lr, train_step_y):
    G = var['G']
    batch_size = var['batch_size']
    num_train = train_x.shape[0]
    for iter in range((int)(num_train / batch_size)):
        index = np.random.permutation(num_train)
        ind = index[0: batch_size]
        sample_L = train_L[ind, :]
        text = train_y[ind, :].astype(np.float32)
        text = text.reshape([text.shape[0], 1, text.shape[1], 1])

        cur_g = cur_g_batch.eval(feed_dict={text_input: text})
        G[:, ind] = cur_g

        train_step_y.run(
            feed_dict={ph['L1']: sample_L[:, 0:num_class1], ph['L2']: sample_L[:, num_class1:num_class],
                       ph['b_batch']: var['B'][:, ind],
                       ph['y1']: var['Y1'], ph['y2']: var['Y2'], text_input: text})
    return G


def split_data(images, tags, labels):
    X = {}
    X['query'] = images[0:QUERY_SIZE, :, :, :]
    X['train'] = images[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE, :, :, :]
    X['retrieval'] = images[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE, :, :, :]

    Y = {}
    Y['query'] = tags[0:QUERY_SIZE, :]
    Y['train'] = tags[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE, :]
    Y['retrieval'] = tags[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE, :]

    L = {}
    L['query'] = labels[0:QUERY_SIZE, :]
    L['train'] = labels[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE, :]
    L['retrieval'] = labels[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE, :]

    return X, Y, L


def calc_loss(B, F, G, bc1, bc2, Sim12, L1, L2, alpha, beta, gamma, eta):
    term1 = alpha * np.sum(np.power((bit * L1 - np.matmul(np.transpose(F), bc1)), 2) + np.power(
        (bit * L1 - np.matmul(np.transpose(G), bc1)), 2))
    term2 = beta * np.sum(np.power((bit * L2 - np.matmul(np.transpose(F), bc2)), 2) + np.power(
        (bit * L2 - np.matmul(np.transpose(G), bc2)), 2))
    term3 = eta * np.sum(np.power((bit * Sim12 - np.matmul(np.transpose(bc1), bc2)), 2))
    term4 = gamma * np.sum(np.power((B - F), 2) + np.power((B - G), 2))

    print("term1:  ", term1, "term2:  ", term2, "term3:  ", term3, "term4:  ", term4)

    loss = term1 + term2 + term3 + term4
    return loss


def generate_image_code(image_input, cur_f_batch, X, bit, mean_pixel):
    batch_size = 128
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter in range((int)(num_data / batch_size) + 1):
        ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
        mean_pixel_ = np.repeat(mean_pixel[:, :, :, np.newaxis], len(ind), axis=3)
        image = X[ind, :, :, :].astype(np.float32) - mean_pixel_.astype(np.float32).transpose(3, 0, 1, 2)

        cur_f = cur_f_batch.eval(feed_dict={image_input: image})
        B[ind, :] = cur_f.transpose()
    B = np.sign(B)
    return B


def generate_text_code(text_input, cur_g_batch, Y, bit):
    batch_size = 128
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter in range((int)(num_data / batch_size) + 1):
        ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
        text = Y[ind, :].astype(np.float32)
        text = text.reshape([text.shape[0], 1, text.shape[1], 1])
        cur_g = cur_g_batch.eval(feed_dict={text_input: text})
        B[ind, :] = cur_g.transpose()
    B = np.sign(B)
    return B



if __name__ == '__main__':
    images, tags, labels = loading_data(DATA_DIR)
    S12 = np.zeros([num_class1, num_class2])
    index1 = 0
    index2 = 0
    for i in range(0, len(labels)):
        for j in range(0, len(labels[0])):
            if (labels[i][j] == 1 and j < num_class1):
                index1 = j
            elif labels[i][j] == 1 and j >= num_class1:
                index2 = j
        S12[index1][index2 - num_class1] = 1

    ydim = tags.shape[1]

    X, Y, L = split_data(images, tags, labels)
    print('...loading and splitting data finish')
    gpuconfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction))
    os.environ["CUDA_VISIBLE_DEVICES"] = select_gpu
    batch_size = 128
    with tf.Graph().as_default(), tf.Session(config=gpuconfig) as sess:
        # construct image network
        image_input = tf.placeholder(tf.float32, (None,) + (224, 224, 3))
        net, _meanpix = img_net_strucuture(image_input, bit)
        mean_pixel_ = np.repeat(_meanpix[:, :, :, np.newaxis], batch_size, axis=3).transpose(3, 0, 1, 2)
        print(mean_pixel_.shape)
        cur_f_batch = tf.transpose(net['fc8'])
        # construct text network
        text_input = tf.placeholder(tf.float32, (None,) + (1, ydim, 1))
        cur_g_batch = txt_net_strucuture(text_input, ydim, bit)

        # training algorithm
        train_L = L['train']
        train_L1 = train_L[:, 0:num_class1]
        train_L2 = train_L[:, num_class1:num_class]
        print(train_L.shape)
        train_x = X['train']
        train_y = Y['train']

        query_L = L['query']
        query_x = X['query']
        query_y = Y['query']

        retrieval_L = L['retrieval']
        retrieval_x = X['retrieval']
        retrieval_y = Y['retrieval']
        num_train = train_x.shape[0]


        var = {}
        lr = 0.0001

        var['lr'] = lr
        var['batch_size'] = batch_size

        var['F'] = np.random.randn(bit, num_train)
        var['G'] = np.random.randn(bit, num_train)
        var['Y1'] = np.sign(np.random.randn(bit, num_class1))
        var['Y2'] = np.sign(np.random.randn(bit, num_class2))
        var['B'] = np.sign(var['F'] + var['G'])

        ph = {}
        ph['lr'] = tf.placeholder('float32', (), name='lr')

        ph['L1'] = tf.placeholder('float32', [batch_size, num_class1], name='L1')
        ph['L2'] = tf.placeholder('float32', [batch_size, num_class2], name='L2')

        ph['b_batch'] = tf.placeholder('float32', [bit, batch_size], name='b_batch')
        ph['y1'] = tf.placeholder('float32', [bit, num_class1], name='y1')
        ph['y2'] = tf.placeholder('float32', [bit, num_class2], name='y2')

        l1_x = tf.reduce_sum(tf.pow(bit * ph['L1'] - tf.matmul(tf.transpose(cur_f_batch), ph['y1']), 2))
        l2_x = tf.reduce_sum(tf.pow(bit * ph['L2'] - tf.matmul(tf.transpose(cur_f_batch), ph['y2']), 2))
        quantization_x = tf.reduce_sum(tf.pow((ph['b_batch'] - cur_f_batch), 2))

        loss_x = tf.div(alpha * l1_x + gamma * quantization_x +beta * l2_x, float(num_train * batch_size))

        l1_y = tf.reduce_sum(tf.pow(bit * ph['L1'] - tf.matmul(tf.transpose(cur_g_batch), ph['y1']), 2))
        l2_y = tf.reduce_sum(tf.pow(bit * ph['L2'] - tf.matmul(tf.transpose(cur_g_batch), ph['y2']), 2))
        quantization_y = tf.reduce_sum(tf.pow((ph['b_batch'] - cur_g_batch), 2))

        loss_y = tf.div(alpha * l1_y + gamma * quantization_y + beta * l2_y, float(num_train * batch_size))

        optimizer = tf.train.AdamOptimizer(0.0001)

        gradient_x = optimizer.compute_gradients(loss_x)
        gradient_y = optimizer.compute_gradients(loss_y)
        train_step_x = optimizer.apply_gradients(gradient_x)
        train_step_y = optimizer.apply_gradients(gradient_y)
        sess.run(tf.global_variables_initializer())
        loss_ = calc_loss(var['B'], var['F'], var['G'], var['Y1'], var['Y2'], S12, train_L1, train_L2, alpha, beta,
                          gamma, eta)
        print('...epoch: %3d, loss: %3.3f' % (0, loss_))
        result = {}
        result['loss'] = []
        result['imapi2t'] = []
        result['imapt2i'] = []
        result['alpha'] = alpha
        result['beta'] = beta
        result['gamma'] = gamma
        result['eta'] = eta
        result['bit'] = bit

        print('...training procedure starts')

        for epoch in range(MAX_ITER):
            lr = var['lr']
            # update F
            var['F'] = train_img_net(image_input, cur_f_batch, var, ph, train_x, train_L, lr, train_step_x, mean_pixel_)

            # update G
            var['G'] = train_txt_net(text_input, cur_g_batch, var, ph, train_y, train_L, lr, train_step_y)

            # update B
            var['B'] = np.sign(gamma * (var['F'] + var['G']))

            # update Y
            Q1 = bit * (alpha * (np.matmul(var['F'], train_L1) + np.matmul(var['G'], train_L1)) +
                        eta * np.matmul(var['Y2'], np.transpose(S12)))
            Q2 = bit * (beta * (np.matmul(var['F'], train_L2) + np.matmul(var['G'], train_L2)) +
                        eta * np.matmul(var['Y1'], S12))



            for i in range(3):
                F = var['F']
                G = var['G']
                Y1 = var['Y1']
                Y2 = var['Y2']
                for k in range(bit):
                    sel_ind = np.setdiff1d([ii for ii in range(bit)], k)
                    Y1_ = Y1[sel_ind, :]
                    y1k = np.transpose(Y1[k, :])
                    Y2_ = Y2[sel_ind, :]
                    y2k = np.transpose(Y2[k, :])
                    Fk = np.transpose(F[k, :])
                    F_ = F[sel_ind, :]
                    Gk = np.transpose(G[k, :])
                    G_ = G[sel_ind, :]

                    y1 = np.sign(np.transpose(Q1[k, :]) - eta * Y1_.transpose().dot(Y2_.dot(y2k))
                                  - alpha * (Y1_.transpose().dot(F_.dot(Fk)) + Y1_.transpose().dot(G_.dot(Gk))))

                    var['Y1'][k, :] = np.transpose(y1)
                if np.linalg.norm(var['Y1']-Y1) < 1e-6 * np.linalg.norm(Y1):
                    break
            for i in range(3):
                F = var['F']
                G = var['G']
                Y2 = var['Y2']
                Y1 = var['Y1']
                for k in range(bit):
                    sel_ind = np.setdiff1d([ii for ii in range(bit)], k)
                    Y1_ = Y1[sel_ind, :]
                    y1k = np.transpose(Y1[k, :])
                    Y2_ = Y2[sel_ind, :]
                    y2k = np.transpose(Y2[k, :])
                    Fk = np.transpose(F[k, :])
                    F_ = F[sel_ind, :]
                    Gk = np.transpose(G[k, :])
                    G_ = G[sel_ind, :]
                    q1 = np.transpose(Q1[k, :])
                    q2 = np.transpose(Q2[k, :])

                    y2 = np.sign(np.transpose(Q2[k, :]) - eta * Y2_.transpose().dot(Y1_.dot(y1k))
                                  - beta * (Y2_.transpose().dot(F_.dot(Fk)) + Y2_.transpose().dot(G_.dot(Gk))))
                    var['Y2'][k, :] = np.transpose(y2)
                if np.linalg.norm(var['Y2'] - Y2) < 1e-6 * np.linalg.norm(Y2):
                    break




            # calculate loss
            loss_ = calc_loss(var['B'], var['F'], var['G'], var['Y1'], var['Y2'], S12, train_L1, train_L2, alpha,
                              beta, gamma, eta)
            print("{}".format(datetime.now()))
            print('...epoch: %3d, loss: %3.3f, comment: update B' % (epoch + 1, loss_))

            result['loss'].append(loss_)

            if epoch % 5 == 0:
                qBX = generate_image_code(image_input, cur_f_batch, query_x, bit, _meanpix)
                qBY = generate_text_code(text_input, cur_g_batch, query_y, bit)
                rBX = generate_image_code(image_input, cur_f_batch, retrieval_x, bit, _meanpix)
                rBY = generate_text_code(text_input, cur_g_batch, retrieval_y, bit)


                mapi2t = calc_map(qBX, rBY, query_L, retrieval_L)
                mapt2i = calc_map(qBY, rBX, query_L, retrieval_L)

                result['imapi2t'].append(mapi2t)
                result['imapt2i'].append(mapt2i)

                print("{}".format(datetime.now()))
                print('...test map: map(i->t): \033[1;32;40m%3.3f\033[0m, map(t->i): \033[1;32;40m%3.3f\033[0m' % (
                    mapi2t, mapt2i))

                fp = open(filename, 'wb')
                pickle.dump(result, fp)
                fp.close()
        print('...training procedure finish')
        qBX = generate_image_code(image_input, cur_f_batch, query_x, bit, _meanpix)
        qBY = generate_text_code(text_input, cur_g_batch, query_y, bit)
        rBX = generate_image_code(image_input, cur_f_batch, retrieval_x, bit, _meanpix)
        rBY = generate_text_code(text_input, cur_g_batch, retrieval_y, bit)

        mapi2t = calc_map(qBX, rBY, query_L, retrieval_L)
        mapt2i = calc_map(qBY, rBX, query_L, retrieval_L)
        print('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (mapi2t, mapt2i))

        result['mapi2t'] = mapi2t
        result['mapt2i'] = mapt2i
        result['lr'] = lr

        fp = open(filename, 'wb')
        pickle.dump(result, fp)

        fp.close()