import cPickle, random, pdb, time
import tensorflow as tf
import numpy as np
import utils as ut
from map import *
from dis_model_nn import DIS
from gen_model_nn import GEN

GPU_ID = 0
OUTPUT_DIM = 64

SELECTNUM = 2
SAMPLERATIO = 50

WHOLE_EPOCH = 30
D_EPOCH = 1
G_EPOCH = 2
GS_EPOCH = 30
D_DISPLAY = 1
G_DISPLAY = 10

IMAGE_DIM = 4096
TEXT_DIM = 1000
HIDDEN_DIM = 8192
CLASS_DIM = 10
BATCH_SIZE = 1
WEIGHT_DECAY = 0.01
D_LEARNING_RATE = 0.001
G_LEARNING_RATE = 0.01
BETA = OUTPUT_DIM / 8.0
GAMMA = 0.1

WORKDIR = '../nus/'
GEN_MODEL_BEST_FILE = './model/gan_best_nn_' + str(OUTPUT_DIM) + '.model'
DIS_MODEL_BEST_FILE = './model/dis_best_nn_' + str(OUTPUT_DIM) + '.model'

GEN_MODEL_NEWEST_FILE = './model/gan_newest_nn_' + str(OUTPUT_DIM) + '.model'
DIS_MODEL_NEWEST_FILE = './model/dis_newest_nn_' + str(OUTPUT_DIM) + '.model'

DIS_MODEL_PRETRAIN_FILE = './model/dis_pretrain_nn_' + str(OUTPUT_DIM) + '.model'

train_i2t, train_i2t_pos, train_i2t_neg, train_t2i, train_t2i_pos, train_t2i_neg, test_i2t, test_i2t_pos, test_t2i, test_t2i_pos = ut.load_all_query_url(WORKDIR + 'list/', CLASS_DIM)

feature_dict = ut.load_all_feature(WORKDIR + 'list/', WORKDIR + 'feature/')
label_dict = ut.load_all_label(WORKDIR + 'list/')

record_file = open('record_' + str(OUTPUT_DIM) + '.txt', 'w')
record_file.close()

def generate_samples(sess, generator, train_list, train_pos, train_neg, flag):
	data = []
	for query in train_pos:
		pos_list = train_pos[query]
		candidate_neg_list = train_neg[query]
		candidate_list = train_list[query]
		
		random.shuffle(pos_list)
		random.shuffle(candidate_neg_list)
		random.shuffle(candidate_list)
		sample_size = int(len(candidate_list) / SAMPLERATIO)
		candidate_list = candidate_list[0 : sample_size]
		
		if flag == 'i2t':
			query_data = np.asarray(feature_dict[query]).reshape(1, IMAGE_DIM)
			candidate_data = np.asarray([feature_dict[url] for url in candidate_list])
			candidate_score = sess.run(generator.pred_score,
										feed_dict={generator.image_data: query_data,
													generator.text_data: candidate_data})
		elif flag == 't2i':
			query_data = np.asarray(feature_dict[query]).reshape(1, TEXT_DIM)
			candidate_data = np.asarray([feature_dict[url] for url in candidate_list])
			candidate_score = sess.run(generator.pred_score,
										feed_dict={generator.text_data: query_data,
													generator.image_data: candidate_data})
		
		exp_rating = np.exp(candidate_score)
		prob = exp_rating / np.sum(exp_rating)
		
		# neg_list = np.random.choice(candidate_list, size=[SELECTNUM], p = prob)
		neg_list = []
		for i in range(SELECTNUM):
		  while True:
			 neg = np.random.choice(candidate_list, p = prob)
			 if neg not in pos_list:
				neg_list.append(neg)
				break
				
		# for i in range(SELECTNUM):
			# neg_list.append(candidate_neg_list[i])
					
		# for i in range(SELECTNUM * 2):
			# data.append((query, pos_list[i], neg_list[i]))
			
		for i in range(SELECTNUM):
			data.append((query, pos_list[i], neg_list[i]))
	
	count = 0.0
	for item in data:
		query, pos, neg = item
		query_label = label_dict[query]
		neg_label = label_dict[neg]
		for i in range(CLASS_DIM):
			if query_label[i] == 1.0 and query_label[i] == neg_label[i]:
				count += 1.0
				break
	
	print('Pos sample ratio: %.4f' % (count / len(data)))
	with open('record.txt', 'a') as record_file:
		record_file.write('Pos sample ratio: %.4f\n' % (count / len(data)))
	
	random.shuffle(data)
	return data

def train_discriminator(sess, discriminator, dis_train_list, flag):
	train_size = len(dis_train_list)
	index = 1
	while index < train_size:
		input_query = []
		input_pos = []
		input_neg = []
		pos_pair_label = []
		neg_pair_label = []
		
		if index + BATCH_SIZE <= train_size:
			for i in range(index, index + BATCH_SIZE):
				query, pos, neg = dis_train_list[i]
				input_query.append(feature_dict[query])
				input_pos.append(feature_dict[pos])
				input_neg.append(feature_dict[neg])
		else:
			for i in range(index, train_size):
				query, pos, neg = dis_train_list[i]
				input_query.append(feature_dict[query])
				input_pos.append(feature_dict[pos])
				input_neg.append(feature_dict[neg])
					
		index += BATCH_SIZE
		
		query_data = np.asarray(input_query)
		input_pos = np.asarray(input_pos)
		input_neg = np.asarray(input_neg)
		
		if flag == 'i2t':
			d_loss = sess.run(discriminator.i2t_loss,
						 feed_dict={discriminator.image_data: query_data,
									discriminator.text_data: input_pos,
									discriminator.text_neg_data: input_neg})
			_ = sess.run(discriminator.i2t_updates,
						 feed_dict={discriminator.image_data: query_data,
									discriminator.text_data: input_pos,
									discriminator.text_neg_data: input_neg})
		elif flag == 't2i':
			d_loss = sess.run(discriminator.t2i_loss,
						 feed_dict={discriminator.text_data: query_data,
									discriminator.image_data: input_pos,
									discriminator.image_neg_data: input_neg})
			_ = sess.run(discriminator.t2i_updates,
						 feed_dict={discriminator.text_data: query_data,
									discriminator.image_data: input_pos,
									discriminator.image_neg_data: input_neg})
	
	print('D_Loss: %.4f' % d_loss)
	return discriminator
	
def train_generator(sess, generator, discriminator, train_list, train_pos, flag):
	for query in train_pos.keys():
		pos_list = train_pos[query]
		candidate_list = train_list[query]
		
		random.shuffle(candidate_list)
		sample_size = int(len(candidate_list) / SAMPLERATIO)
		candidate_list = candidate_list[0 : sample_size]
		
		random.shuffle(pos_list)
		pos_list = pos_list[0:SELECTNUM]
		
		candidate_data = np.asarray([feature_dict[url] for url in candidate_list])
		
		if flag == 'i2t':
			query_data = np.asarray(feature_dict[query]).reshape(1, IMAGE_DIM)
			candidate_score = sess.run(generator.pred_score,
								feed_dict={generator.image_data: query_data,
											generator.text_data: candidate_data})
		elif flag == 't2i':
			query_data = np.asarray(feature_dict[query]).reshape(1, TEXT_DIM)
			candidate_score = sess.run(generator.pred_score,
								feed_dict={generator.image_data: candidate_data,
											generator.text_data: query_data})

		exp_rating = np.exp(candidate_score)
		prob = exp_rating / np.sum(exp_rating)
		neg_index = np.random.choice(np.arange(len(candidate_list)), size = [SELECTNUM], p = prob)
		neg_list = np.array(candidate_list)[neg_index]
		neg_index = np.asarray(neg_index)
		
		input_pos = np.asarray([feature_dict[url] for url in pos_list])
		input_neg = np.asarray([feature_dict[url] for url in neg_list])
		
		# pdb.set_trace()
		
		if flag == 'i2t':
			neg_reward = sess.run(discriminator.i2t_reward,
								  feed_dict={discriminator.image_data: query_data,
											discriminator.text_data: input_pos,
											discriminator.text_neg_data: input_neg})
			g_loss = sess.run(generator.gen_loss,
					    feed_dict={generator.image_data: query_data,
										generator.text_data: input_neg,
										generator.reward: neg_reward})
			_ = sess.run(generator.gen_updates,
					    feed_dict={generator.image_data: query_data,
										generator.text_data: input_neg,
										generator.reward: neg_reward})
			
		elif flag == 't2i':
			neg_reward = sess.run(discriminator.t2i_reward,
								  feed_dict={discriminator.text_data: query_data,
											discriminator.image_data: input_pos,
											discriminator.image_neg_data: input_neg})
			g_loss = sess.run(generator.gen_loss,
					    feed_dict={generator.text_data: query_data,
										generator.image_data: input_neg,
										generator.reward: neg_reward})
			_ = sess.run(generator.gen_updates,
					    feed_dict={generator.text_data: query_data,
										generator.image_data: input_neg,
										generator.reward: neg_reward})
			
	print('G_Loss: %.4f' % g_loss)
	return generator

def main():
	with tf.device('/gpu:' + str(GPU_ID)):
		dis_param = cPickle.load(open(DIS_MODEL_PRETRAIN_FILE))
		# gen_param = cPickle.load(open(GEN_MODEL_PRETRAIN_FILE))
		discriminator = DIS(IMAGE_DIM, TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM, WEIGHT_DECAY, D_LEARNING_RATE, BETA, GAMMA, param = dis_param)
		# generator = GEN(IMAGE_DIM, TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM, CLASS_DIM, WEIGHT_DECAY, G_LEARNING_RATE, param = gen_param)
		# discriminator = DIS(IMAGE_DIM, TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM, WEIGHT_DECAY, D_LEARNING_RATE, BETA, GAMMA, param = None)
		generator = GEN(IMAGE_DIM, TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM, CLASS_DIM, WEIGHT_DECAY, G_LEARNING_RATE, param = None)
		
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		sess.run(tf.initialize_all_variables())

		print('start adversarial training')
		map_best_val_gen = 0.0
		map_best_val_dis = 0.0

		for epoch in range(WHOLE_EPOCH):
			print('Training D ...')
			for d_epoch in range(D_EPOCH):
				print('d_epoch: ' + str(d_epoch))
				if d_epoch % GS_EPOCH == 0:
					print('negative text sampling for d using g ...')
					dis_train_i2t_list = generate_samples(sess, generator, train_i2t, train_i2t_pos, train_i2t_neg, 'i2t')
					print('negative image sampling for d using g ...')
					dis_train_t2i_list = generate_samples(sess, generator, train_t2i, train_t2i_pos, train_t2i_neg, 't2i')
				
				discriminator = train_discriminator(sess, discriminator, dis_train_i2t_list, 'i2t')
				discriminator = train_discriminator(sess, discriminator, dis_train_t2i_list, 't2i')
				
				if (d_epoch + 1) % (D_DISPLAY) == 0:
					i2t_test_map = MAP(sess, discriminator, test_i2t_pos, test_i2t, feature_dict, 'i2t')
					print('E%d D%d I2T_Test_MAP: %.4f' % (epoch, d_epoch, i2t_test_map))
					t2i_test_map = MAP(sess, discriminator, test_t2i_pos, test_t2i, feature_dict, 't2i')
					print('E%d D%d T2I_Test_MAP: %.4f' % (epoch, d_epoch, t2i_test_map))
					
					with open('record.txt', 'a') as record_file:
						record_file.write('E%d D%d I2T_Test_MAP: %.4f\n' % (epoch, d_epoch, i2t_test_map))
						record_file.write('E%d D%d T2I_Test_MAP: %.4f\n' % (epoch, d_epoch, t2i_test_map))
					
					average_map = 0.5 * (i2t_test_map + t2i_test_map)
					if average_map > map_best_val_dis:
						map_best_val_dis = average_map
						discriminator.save_model(sess, DIS_MODEL_BEST_FILE)
				discriminator.save_model(sess, DIS_MODEL_NEWEST_FILE)
				
			print('Training G ...')		
			for g_epoch in range(G_EPOCH):
				print('g_epoch: ' + str(g_epoch))
				generator = train_generator(sess, generator, discriminator, train_i2t, train_i2t_pos, 'i2t')
				generator = train_generator(sess, generator, discriminator, train_t2i, train_t2i_pos, 't2i')
				
				if (g_epoch + 1) % (G_DISPLAY) == 0:
					i2t_test_map = MAP(sess, generator, test_i2t_pos, test_i2t, feature_dict, 'i2t')
					print('E%d G%d I2T_Test_MAP: %.4f' % (epoch, g_epoch, i2t_test_map))
					t2i_test_map = MAP(sess, generator, test_t2i_pos, test_t2i, feature_dict, 't2i')
					print('E%d G%d T2I_Test_MAP: %.4f' % (epoch, g_epoch, t2i_test_map))
					
					with open('record.txt', 'a') as record_file:
						record_file.write('E%d G%d I2T_Test_MAP: %.4f\n' % (epoch, g_epoch, i2t_test_map))
						record_file.write('E%d G%d T2I_Test_MAP: %.4f\n' % (epoch, g_epoch, t2i_test_map))
					
					average_map = 0.5 * (i2t_test_map + t2i_test_map)
					if average_map > map_best_val_gen:
						map_best_val_gen = average_map
						generator.save_model(sess, GEN_MODEL_BEST_FILE)
				generator.save_model(sess, GEN_MODEL_NEWEST_FILE)
					
		sess.close()
if __name__ == '__main__':
	main()