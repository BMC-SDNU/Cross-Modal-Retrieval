import cPickle, random, pdb
import scipy.io as sio
import tensorflow as tf
import numpy as np
import utils as ut
from map import *
from dis_model_nn import DIS
from gen_model_nn import GEN

IMAGE_DIM = 4096
TEXT_DIM = 1000
OUTPUT_DIM = 64
HIDDEN_DIM = 1024
CLASS_DIM = 10
WEIGHT_DECAY = 0.01
D_LEARNING_RATE = 0.001
G_LEARNING_RATE = 0.001
TEMPERATURE = 0.2
BETA = 1
GAMMA = 0.1
REWARD_BASE = 10.0

WORKDIR = '../nus/'
DIS_MODEL_BEST_FILE = './model/dis_best_nn.model'

database_img = open(WORKDIR + 'list/database_img.txt', 'r').read().split('\r\n')
database_txt = open(WORKDIR + 'list/database_txt.txt', 'r').read().split('\r\n')
test_img = open(WORKDIR + 'list/test_img.txt', 'r').read().split('\r\n')
test_txt = open(WORKDIR + 'list/test_txt.txt', 'r').read().split('\r\n')

feature_dict = ut.load_all_feature_for_test(WORKDIR + 'list/', WORKDIR + 'feature/')

def extract_feature(sess, model, list):
	list_size = len(list) - 1
	result = np.zeros((list_size, OUTPUT_DIM))
	
	for i in range(list_size):
		item = list[i]
		input_data = np.asarray(feature_dict[item])
		input_data_dim = input_data.shape[0]
		input_data = input_data.reshape(1, input_data_dim)
		
		if item.split('.')[-1] == 'jpg':
			output_hash = sess.run(model.image_hash, feed_dict={model.image_data: input_data})
		elif item.split('.')[-1] == 'txt':
			output_hash = sess.run(model.text_hash, feed_dict={model.text_data: input_data})
			
		result[i] = output_hash
		
	return result

def test():	
	discriminator_param = cPickle.load(open(DIS_MODEL_BEST_FILE))
	discriminator = DIS(IMAGE_DIM, TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM, WEIGHT_DECAY, D_LEARNING_RATE, BETA, GAMMA, REWARD_BASE, loss ='svm', param=discriminator_param)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	sess.run(tf.initialize_all_variables())
	
	I_db = extract_feature(sess, discriminator, database_img)
	T_db = extract_feature(sess, discriminator, database_txt)
	I_te = extract_feature(sess, discriminator, test_img)
	T_te = extract_feature(sess, discriminator, test_txt)
	sio.savemat('./result/DIS_nus_' + str(OUTPUT_DIM) + '.mat', {'B_I_db': I_db, 'B_T_db': T_db, 'B_I_te': I_te, 'B_T_te': T_te})
	sess.close()

if __name__ == '__main__':
	test()
	