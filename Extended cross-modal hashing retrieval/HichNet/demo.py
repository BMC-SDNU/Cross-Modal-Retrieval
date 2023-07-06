#coding:utf-8
import os
import numpy as np
import tensorflow as tf
import pickle
import scipy.io
from load_data import loading_cloth_data
from net_structure_img import img_net_strucuture
from net_structure_txt import txt_net_strucuture
from utils.calc_hammingranking import calc_map
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# environmental setting: setting the following parameters based on your experimental environment.
select_gpu = '0'
per_process_gpu_memory_fraction = 0.3

# data parameters
DATA_DIR = '/home/share/sunchangchang/data/data2/'
TRAINING_SIZE = 16862
QUERY_SIZE = 3000
DATABASE_SIZE = 16862

# hyper-parameters
MAX_ITER = 500
afa=1
gamma = 1
eta = 1

first=1
ita=1
u1 = 1
u2 = 1
h1=0.2
h2=0.8
v1=0.2
v2=0.8
bit = 32
num_class=35

filename = 'log/result_' + datetime.now().strftime("%d-%H-%M-%S") + '_' + str(bit) + 'bits_MIRFLICKR-25K.pkl'
def train_img_net(image_input, cur_f_batch1,cur_f_batch2, var, ph, train_x, train_L, lr, train_step_x, mean_pixel_, Sim1,Sim2):
    F1 = var['F1']
    F2 = var['F2']
    batch_size = var['batch_size']
    num_train = train_x.shape[0]
    for iter in range((int)(round((num_train / batch_size)))):
        index = np.random.permutation(num_train)
        ind = index[0: batch_size]
        unupdated_ind = np.setdiff1d(range(num_train), ind)
        sample_L = train_L[ind, :]
        image = train_x[ind, :, :, :].astype(np.float64)
        image = image - mean_pixel_.astype(np.float64)  #对图片的像素做了一个处理，减去了一个均值
        image=image.transpose(0,2 ,3,1)  #把像素通道换到最后一维
        S1,S2 = calc_neighbor(sample_L, train_L)
        cur_f1 = cur_f_batch1.eval(feed_dict={image_input: image})
        cur_f2 = cur_f_batch2.eval(feed_dict={image_input: image})
        cur_classify = train_L[ind, :]
        F1[:, ind] = cur_f1
        F2[:, ind] = cur_f2
        train_step_x.run(feed_dict={ph['S_x1']: S1,ph['S_x2']: S2, ph['G1']: var['G1'],ph['G2']: var['G2'],
                                    ph['b_batch1']: var['B1'][:, ind],
                                    ph['b_batch2']: var['B2'][:, ind],
                                    ph['F1_']: F1[:, unupdated_ind],ph['F2_']: F2[:, unupdated_ind], ph['lr']: lr,
                                    image_input: image,
                                    ph['classify_score']: cur_classify
                                    })
    return F1,F2

def train_txt_net(text_input, cur_g_batch1,cur_g_batch2, var, ph, train_y, train_L, lr, train_step_y, Sim1,Sim2):
    G1 = var['G1']
    G2 = var['G2']
    batch_size = var['batch_size']
    num_train = train_x.shape[0]
    for iter in range((int)(round((num_train / batch_size)))):
        index = np.random.permutation(num_train)
        ind = index[0: batch_size]
        unupdated_ind = np.setdiff1d(range(num_train), ind)
        sample_L = train_L[ind, :]
        text = train_y[ind, :].astype(np.float32)
        text = text.reshape([text.shape[0], 1, text.shape[1], 1])
        S1,S2 = calc_neighbor(train_L, sample_L)
        cur_g1 = cur_g_batch1.eval(feed_dict={text_input: text})
        cur_g2 = cur_g_batch2.eval(feed_dict={text_input: text})
        G1[:, ind] = cur_g1
        G2[:, ind] = cur_g2
        text_class = train_L[ind, :]
        train_step_y.run(feed_dict={ph['S_y1']: S1,ph['S_y2']: S2, 
                                    ph['F1']: var['F1'],ph['F2']: var['F2'],
                                    ph['b_batch1']: var['B1'][:, ind],ph['b_batch2']: var['B2'][:, ind],
                                    ph['G1_']: G1[:, unupdated_ind],  ph['G2_']: G2[:, unupdated_ind],
                                    ph['lr']: lr, text_input: text,
                                    ph['classify_t_score']: text_class})
    return G1,G2

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

def calc_neighbor(label_1, label_2):
    label_1=np.array(label_1)
    label_2=np.array(label_2)
    la=label_1[:,0:8]
    laa=label_1[:,8:35]
    lb=label_2[:,0:8]
    lbb=label_2[:,8:35]
    sim1=(np.dot(la,lb.transpose())>0).astype(int)
    sim2=(np.dot(laa,lbb.transpose())>0).astype(int)
    Sim1=sim1
    Sim2=sim2
    return Sim1,Sim2

def calc_loss(B1,B2, F1,F2, G1,G2, Sim1,Sim2, gamma, eta):
    theta1 = np.matmul(np.transpose(F1), G1) / 2
    theta2 = np.matmul(np.transpose(F2), G2) / 2
    term1 = np.sum(np.log(1 + np.exp(theta1)) - Sim1 * theta1)+np.sum(np.log(1 + np.exp(theta2)) - Sim2 * theta2)
    term2 = np.sum(np.power((B1 - F1), 2) + np.power(B1 - G1, 2))+np.sum(np.power((B2 - F2), 2) + np.power(B2 - G2, 2))
    term3 = np.sum(np.power(np.matmul(F1, np.ones((F1.shape[1], 1))), 2))+np.sum(np.power(np.matmul(F2, np.ones((F2.shape[1], 1))), 2)) + np.sum(
        np.power(np.matmul(G1, np.ones((F1.shape[1], 1))), 2))+ np.sum(
        np.power(np.matmul(G2, np.ones((F1.shape[1], 1))), 2))
    print("term1:  ", term1, "term2:  ", term2, "term3:  ", term3)
    loss = afa * term1 + gamma * term2 + eta * term3
    return loss

def generate_image_code(image_input, cur_f_batch1,cur_f_batch2, X, bit, mean_pixel):
    batch_size = 128
    num_data = X.shape[0]   #测试集的个数
    index = np.linspace(0, num_data - 1, num_data).astype(int)  #在指定的间隔内返回均匀间隔的数字
    B1= np.zeros([num_data, bit], dtype=np.float32)    #用来存储得到的hash code
    B2 = np.zeros([num_data, bit], dtype=np.float32)
    for iter in range(num_data / batch_size + 1):
        ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
        m = mean_pixel['averageImage'][0]
        mean_pixel_ = np.repeat(m[:, :, :, np.newaxis], len(ind), axis=3).transpose(3, 2, 1, 0)
        image = X[ind, :, :, :].astype(np.float32) - mean_pixel_.astype(np.float32)
        image=image.transpose(0,2,3,1)
        cur_f1 = cur_f_batch1.eval(feed_dict={image_input: image})
        cur_f2 = cur_f_batch2.eval(feed_dict={image_input: image})
        B1[ind, :] = cur_f1.transpose()
        B2[ind, :] = cur_f2.transpose()
    B1 = np.sign(B1)
    B2 = np.sign(B2)
    return B1,B2

def generate_text_code(text_input, cur_g_batch1,cur_g_batch2, Y, bit):
    batch_size = 128
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B1 = np.zeros([num_data, bit], dtype=np.float32)
    B2 = np.zeros([num_data, bit], dtype=np.float32)
    for iter in range(num_data / batch_size + 1):
        ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
        text = Y[ind, :].astype(np.float32)
        text = text.reshape([text.shape[0], 1, text.shape[1], 1])
        cur_g1 = cur_g_batch1.eval(feed_dict={text_input: text})
        cur_g2 = cur_g_batch2.eval(feed_dict={text_input: text})
        B1[ind, :] = cur_g1.transpose()
        B2[ind, :] = cur_g2.transpose()
    B1 = np.sign(B1)
    B2 = np.sign(B2)
    return B1,B2

def test_validation(B, query_L, train_L, qBX, qBY):
    mapi2t = calc_map(qBX, B, query_L, train_L)
    mapt2i = calc_map(qBY, B, query_L, train_L)
    return mapi2t, mapt2i

def isNaN(num):
    return num != num

if __name__ == '__main__':
    images, tags, labels = loading_cloth_data(DATA_DIR)
    ydim = tags.shape[1]

    X, Y, L = split_data(images, tags, labels)

    print('...loading and splitting data finish')
    gpuconfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction))
    os.environ["CUDA_VISIBLE_DEVICES"] = select_gpu
    batch_size = 128
    file1=open("record/loss_"+  datetime.now().strftime("%d-%H-%M-%S")+".txt",'w')
    file2 = open("record/map_"  + datetime.now().strftime("%d-%H-%M-%S") + ".txt", 'w')
    with tf.Session(config=gpuconfig) as sess:
        # construct image network
        init = tf.global_variables_initializer()
        sess.run(init)
        image_input = tf.placeholder(tf.float32, (None,) + (224, 224, 3),name="image_input")
        net, _meanpix ,classify_i_batch1,classify_i_batch2 = img_net_strucuture(image_input, bit,num_class)  #64
        m = _meanpix['averageImage'][0]
        mean = np.repeat(m[:, :, :, np.newaxis], batch_size, axis=3)
        mean_pixel_ = mean.transpose(3, 2, 1, 0)
        cur_f_batch1 = tf.transpose(net['fc8'])
        cur_f_batch2 = tf.transpose(net['fc9'])
        tf.add_to_collection('pre_network',cur_f_batch1)
        tf.add_to_collection('pre_network', cur_f_batch2)

        text_input = tf.placeholder(tf.float32, (None,) + (1, ydim, 1),name="text_input")
        cur_g_batch1,cur_g_batch2,classify_t_batch1,classify_t_batch2 = txt_net_strucuture(text_input, ydim, bit,num_class)                    ###################
        tf.add_to_collection('pre_network',cur_g_batch1)
        tf.add_to_collection('pre_network', cur_g_batch2)
        tf.add_to_collection('pre_network', cur_g_batch1)
        tf.add_to_collection('pre_network', cur_g_batch2)

        # training HiCHNet algorithm
        train_L = L['train']
        train_x = X['train']
        train_y = Y['train']

        query_L = L['query']
        query_x = X['query']
        query_y = Y['query']

        retrieval_L = L['retrieval']
        retrieval_x = X['retrieval']
        retrieval_y = Y['retrieval']
        num_train = train_x.shape[0]

        Sim1,Sim2 = calc_neighbor(train_L, train_L)
        var = {}
        lr = np.linspace(np.power(10, -1.), np.power(10, -5.), MAX_ITER)    #设置在不同轮时的学习率

        var['lr'] = lr
        var['batch_size'] = batch_size
        var['F1'] = np.random.randn(bit, num_train)
        var['G1'] = np.random.randn(bit, num_train)
        var['B1'] = np.sign(var['F1'] + var['G1'])

        var['F2'] = np.random.randn(bit, num_train)
        var['G2'] = np.random.randn(bit, num_train)
        var['B2'] = np.sign(var['F2'] + var['G2'])

        unupdated_size = num_train - batch_size
        var['unupdated_size'] = unupdated_size

        ph = {}
        ph['lr'] = tf.placeholder('float32', (), name='lr')
        ph['S_x1'] = tf.placeholder('float32', [batch_size, num_train], name='pS_x1')
        ph['S_y1'] = tf.placeholder('float32', [num_train, batch_size], name='pS_y1')
        ph['S_x2'] = tf.placeholder('float32', [batch_size, num_train], name='pS_x2')
        ph['S_y2'] = tf.placeholder('float32', [num_train, batch_size], name='pS_y2')

        ph['F1'] = tf.placeholder('float32', [bit, num_train], name='pF1')
        ph['G1'] = tf.placeholder('float32', [bit, num_train], name='pG1')
        ph['F2'] = tf.placeholder('float32', [bit, num_train], name='pF2')
        ph['G2'] = tf.placeholder('float32', [bit, num_train], name='pG2')

        ph['F1_'] = tf.placeholder('float32', [bit, unupdated_size], name='unupdated_F1')
        ph['G1_'] = tf.placeholder('float32', [bit, unupdated_size], name='unupdated_G1')
        ph['F2_'] = tf.placeholder('float32', [bit, unupdated_size], name='unupdated_F2')
        ph['G2_'] = tf.placeholder('float32', [bit, unupdated_size], name='unupdated_G2')

        ph['b_batch1'] = tf.placeholder('float32', [bit, batch_size], name='b_batch')
        ph['b_batch2'] = tf.placeholder('float32', [bit, batch_size], name='b_batch')
        ph['ones_'] = tf.constant(np.ones([unupdated_size, 1], 'float32'))
        ph['ones_batch'] = tf.constant(np.ones([batch_size, 1], 'float32'))

        ph['classify_score'] = tf.placeholder('float32', [batch_size, num_class], name='classify_score')
        ph['classify_t_score'] = tf.placeholder('float32', [batch_size, num_class], name='classify_t_score')

        theta_x1 = 1.0 / 2 * tf.matmul(tf.transpose(cur_f_batch1), ph['G1'])
        theta_y1 = 1.0 / 2 * tf.matmul(tf.transpose(ph['F1']), cur_g_batch1)
        theta_x2 = 1.0 / 2 * tf.matmul(tf.transpose(cur_f_batch2),ph['G2'])
        theta_y2 = 1.0 / 2 * tf.matmul(tf.transpose(ph['F2']), cur_g_batch2)

        loss_i_classify = v2 * tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(logits=classify_i_batch2,
                                                    labels=ph['classify_score'][:, 8:35])) + v1 * tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(logits=classify_i_batch1,
                                                    labels=ph['classify_score'][:, 0:8]))

        logloss_x1 = -tf.reduce_sum(tf.multiply(ph['S_x1'], theta_x1) - tf.log(1.0 + tf.exp(theta_x1)))
        logloss_x2 = -tf.reduce_sum(tf.multiply(ph['S_x2'], theta_x2) - tf.log(1.0 + tf.exp(theta_x2)))
        quantization_x1 = tf.reduce_sum(tf.pow((ph['b_batch1'] - cur_f_batch1), 2))  # （64，128）
        quantization_x2 = tf.reduce_sum(tf.pow((ph['b_batch2'] - cur_f_batch2), 2))  # （64，128）   （#    （64，？）
        balance_x1 = tf.reduce_sum(
            tf.pow(tf.matmul(cur_f_batch1, ph['ones_batch']) + tf.matmul(ph['F1_'], ph['ones_']), 2))
        balance_x2 = tf.reduce_sum(
            tf.pow(tf.matmul(cur_f_batch2, ph['ones_batch']) + tf.matmul(ph['F2_'], ph['ones_']), 2))

        loss_x = tf.div(first*((h1*logloss_x1+h2*logloss_x2 )+ (quantization_x1+quantization_x2) +(balance_x1+balance_x2))
                        +ita*loss_i_classify, float(num_train * batch_size))

        loss_t_classify = v2*tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=classify_t_batch2,
                                                                                labels=ph['classify_t_score'][:,8:35]))+v1*tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=classify_t_batch1,
                                                                                labels=ph['classify_t_score'][:,0:8]))
        logloss_y1 = -tf.reduce_sum(tf.multiply(ph['S_y1'], theta_y1) - tf.log(1.0 + tf.exp(theta_y1)))
        logloss_y2 = -tf.reduce_sum(tf.multiply(ph['S_y2'], theta_y2) - tf.log(1.0 + tf.exp(theta_y2)))
        quantization_y1 = tf.reduce_sum(tf.pow((ph['b_batch1'] - cur_g_batch1), 2))
        quantization_y2 = tf.reduce_sum(tf.pow((ph['b_batch2'] - cur_g_batch2), 2))
        balance_y1 = tf.reduce_sum(
            tf.pow(tf.matmul(cur_g_batch1, ph['ones_batch']) + tf.matmul(ph['G1_'], ph['ones_']), 2))
        balance_y2 = tf.reduce_sum(tf.pow(tf.matmul(cur_g_batch2, ph['ones_batch']) + tf.matmul(ph['G2_'], ph['ones_']), 2))
        loss_y = tf.div(first*((h1*logloss_y1+h2*logloss_y2)+  (quantization_y1+quantization_y2) + (balance_y1+balance_y2))+ita*loss_t_classify, float(num_train * batch_size))
        optimizer = tf.train.GradientDescentOptimizer(ph['lr'])
        gradient_x = optimizer.compute_gradients(loss_x)
        gradient_y = optimizer.compute_gradients(loss_y)
        train_step_x = optimizer.apply_gradients(gradient_x)
        train_step_y = optimizer.apply_gradients(gradient_y)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        loss_ = calc_loss(var['B1'],var['B2'], var['F1'],var['F2'], var['G1'],var['G2'], Sim1,Sim2, gamma, eta)
        print('...epoch: %3d, loss: %3.3f' % (0, loss_))
        result = {}
        result['loss'] = []
        result['loss_i'] = []
        result['imapi2t'] = []
        result['imapt2i'] = []
        result['imapi2i'] = []
        result['imapt2t'] = []
        print('...training procedure starts')
        for epoch in range(0,MAX_ITER):
            lr = var['lr'][epoch]
            # update F
            var['F1'],var['F2']  = train_img_net(image_input, cur_f_batch1,cur_f_batch2, var, ph, train_x, train_L, lr, train_step_x, mean_pixel_,
                                     Sim1,Sim2)
            # update G
            var['G1'],var['G2'] = train_txt_net(text_input, cur_g_batch1,cur_g_batch2, var, ph, train_y, train_L, lr, train_step_y, Sim1,Sim2)
            # update B
            var['B1'] = np.sign(gamma * (var['F1'] + var['G1']))
            var['B2'] = np.sign(gamma * (var['F2'] + var['G2']))

            # calculate loss
            loss_ = calc_loss(var['B1'],var['B2'], var['F1'], var['F2'], var['G1'],var['G2'], Sim1,Sim2, gamma, eta)
            if (isNaN(loss_)):
                break
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),epoch)
            print('...epoch: %3d, loss: %3.3f, comment: update B' % (epoch + 1, loss_))
            file1.write(str(epoch)+" "+str(loss_)+"\n")
            result['loss'].append(loss_)
            if(epoch%5==0):
                qBX1,qBX2 = generate_image_code(image_input, cur_f_batch1,cur_f_batch2, query_x, bit, _meanpix)
                qBY1,qBY2 = generate_text_code(text_input, cur_g_batch1,cur_g_batch2, query_y, bit)
                rBX1,rBX2 = generate_image_code(image_input, cur_f_batch1,cur_f_batch2, retrieval_x, bit, _meanpix)
                rBY1,rBY2 = generate_text_code(text_input, cur_g_batch1, cur_g_batch2,retrieval_y, bit)

                mapi2t = calc_map(qBX1,qBX2, rBY1,rBY2, query_L, retrieval_L)
                # mapi2i = calc_map(qBX1,qBX2, rBX1,rBX2, query_L, retrieval_L)
                mapt2i = calc_map(qBY1,qBY2, rBX1,rBX2, query_L, retrieval_L)
                # mapt2t = calc_map(qBY1,qBY2, rBY1,rBY2, query_L, retrieval_L)
                mapi2i=0
                mapt2t=0
                print("{}".format(datetime.now()))
                print('...test map: map(i->t): \033[1;32;40m%3.3f\033[0m, map(t->i): \033[1;32;40m%3.3f\033[0m ,map(i->i): \033[1;32;40m%3.3f\033[0m, map(t->t): \033[1;32;40m%3.3f\033[0m' % (
                mapi2t, mapt2i, mapi2i, mapt2t))
                if (epoch % 10 == 0):
                    scipy.io.savemat('/home/share/sunchangchang/data/women/model_64_1_1_0.2_0.8_0.2_0.8/model' + str(epoch) + '.mat', {'F1': var['F1'], 'F2': var['F2'],'G1': var['G1'],'G2': var['G2'], 'B1': var['B1'],'B2': var['B2']})
                    saver.save(sess, "/home/share/sunchangchang/data/women/clothmodel_64_1_1_0.2_0.8_0.2_0.8/mode" + str(
                        epoch + 1) + ".ckpt")
                # file2.write('...test map: map(i->t): %3.3f, map(t->i): %3.3f ,map(i->i): %3.3f, map(t->t): %3.3f \n' % (
                # mapi2t, mapt2i, mapi2i, mapt2t))
                result['mapi2t'] = mapi2t
                result['mapt2i'] = mapt2i
                result['lr'] = lr
                result['imapi2t'].append(mapi2t)
                result['imapt2i'].append(mapt2i)
                result['imapi2i'].append(mapi2i)
                result['imapt2t'].append(mapt2t)
        file1.close()
        file2.close()
        print('...training procedure finish')
        plt.plot(result['imapi2t'], 'b--', label='imgi2t')
        plt.plot(result['imapt2i'], 'r--', label='imgt2i')
        plt.plot(result['imapi2i'], 'y-', label='imgi2i')
        plt.plot(result['imapt2t'], 'k-', label='imgt2t')
        plt.title('test map')
        plt.legend()
        plt.show()

        fp = open(filename, 'wb')
        pickle.dump(result, fp)

        fp.close()
