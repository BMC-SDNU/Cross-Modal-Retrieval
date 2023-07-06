import tensorflow as tf
import cPickle, pdb


class GEN:
	def __init__(self, image_dim, text_dim, hidden_dim, output_dim, class_dim, weight_decay, learning_rate, param=None):
		self.image_dim = image_dim
		self.text_dim = text_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.class_dim = class_dim
		self.weight_decay = weight_decay
		self.learning_rate = learning_rate
		self.params = []

		self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
		self.image_data = tf.placeholder(tf.float32, shape=[None, self.image_dim], name="image_data")
		self.image_label = tf.placeholder(tf.float32, shape=[None, self.class_dim], name="image_label")
		self.text_data = tf.placeholder(tf.float32, shape=[None, self.text_dim], name="text_data")
		self.text_label = tf.placeholder(tf.float32, shape=[None, self.class_dim], name="text_label")

		with tf.variable_scope('generator'):
			if param == None:
				self.Wq_1 = tf.get_variable('Wq_1', [self.image_dim, self.hidden_dim],
										 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Wq_2 = tf.get_variable('Wq_2', [self.hidden_dim, self.output_dim],
										 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Wq_3 = tf.get_variable('Wq_3', [self.hidden_dim, self.class_dim],
										 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Bq_1 = tf.get_variable('Bq_1', [self.hidden_dim], initializer=tf.constant_initializer(0.0))
				self.Bq_2 = tf.get_variable('Bq_2', [self.output_dim], initializer=tf.constant_initializer(0.0))
				self.Bq_3 = tf.get_variable('Bq_3', [self.class_dim], initializer=tf.constant_initializer(0.0))
				
				self.Wc_1 = tf.get_variable('Wc_1', [self.text_dim, self.hidden_dim],
										 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Wc_2 = tf.get_variable('Wc_2', [self.hidden_dim, self.output_dim],
										 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Wc_3 = tf.get_variable('Wc_3', [self.hidden_dim, self.class_dim],
										 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
				self.Bc_1 = tf.get_variable('Bc_1', [self.hidden_dim], initializer=tf.constant_initializer(0.0))
				self.Bc_2 = tf.get_variable('Bc_2', [self.output_dim], initializer=tf.constant_initializer(0.0))
				self.Bc_3 = tf.get_variable('Bc_3', [self.class_dim], initializer=tf.constant_initializer(0.0))
			else:
				self.Wq_1 = tf.Variable(param[0])
				self.Wq_2 = tf.Variable(param[1])
				self.Wq_3 = tf.Variable(param[2])
				self.Bq_1 = tf.Variable(param[3])
				self.Bq_2 = tf.Variable(param[4])
				self.Bq_3 = tf.Variable(param[5])
				
				self.Wc_1 = tf.Variable(param[6])
				self.Wc_2 = tf.Variable(param[7])
				self.Wc_3 = tf.Variable(param[8])
				self.Bc_1 = tf.Variable(param[9])
				self.Bc_2 = tf.Variable(param[10])
				self.Bc_3 = tf.Variable(param[11])
				
			self.params.append(self.Wq_1)
			self.params.append(self.Wq_2)
			self.params.append(self.Wq_3)
			self.params.append(self.Bq_1)
			self.params.append(self.Bq_2)
			self.params.append(self.Bq_3)
			
			self.params.append(self.Wc_1)
			self.params.append(self.Wc_2)
			self.params.append(self.Wc_3)
			self.params.append(self.Bc_1)
			self.params.append(self.Bc_2)
			self.params.append(self.Bc_3)

		# Given batch query-url pairs, calculate the matching score
		# For all urls of one query	
		self.image_rep1 = tf.nn.tanh(tf.nn.xw_plus_b(self.image_data, self.Wq_1, self.Bq_1))
		self.image_rep2 = tf.nn.xw_plus_b(self.image_rep1, self.Wq_2, self.Bq_2)
		self.image_class_pred = tf.nn.xw_plus_b(self.image_rep1, self.Wq_3, self.Bq_3)
		self.image_sig = tf.sigmoid(self.image_rep2)
		self.image_hash = tf.cast(self.image_sig + 0.5, tf.int32)
		
		self.text_rep1 = tf.nn.tanh(tf.nn.xw_plus_b(self.text_data, self.Wc_1, self.Bc_1))
		self.text_rep2 = tf.nn.xw_plus_b(self.text_rep1, self.Wc_2, self.Bc_2)
		self.text_class_pred = tf.nn.xw_plus_b(self.text_rep1, self.Wc_3, self.Bc_3)
		self.text_sig = tf.sigmoid(self.text_rep2)
		self.text_hash = tf.cast(self.text_sig + 0.5, tf.int32)
			
		self.pred_score = -tf.reduce_sum(tf.square(self.image_sig - self.text_sig), 1)
		self.hash_score = tf.reduce_sum(tf.cast(tf.equal(self.image_hash, self.text_hash), tf.float32), 1)
		
		
		self.gen_prob = tf.reshape(tf.nn.softmax(tf.reshape(self.pred_score, [1, -1])), [-1]) + 0.00000000000000000001
											   
		self.gen_loss = tf.reduce_mean(tf.abs(self.gen_prob - self.reward)) \
						+ tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.image_class_pred, labels = self.image_label)) \
						+ tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.text_class_pred, labels = self.text_label)) \
						+ self.weight_decay * (tf.nn.l2_loss(self.Wq_1) + tf.nn.l2_loss(self.Wq_2) + tf.nn.l2_loss(self.Wq_3)
											   + tf.nn.l2_loss(self.Bq_1) + tf.nn.l2_loss(self.Bq_2) + tf.nn.l2_loss(self.Bq_3)
											   + tf.nn.l2_loss(self.Wc_1) + tf.nn.l2_loss(self.Wc_2) + tf.nn.l2_loss(self.Wc_3)
											   + tf.nn.l2_loss(self.Bc_1) + tf.nn.l2_loss(self.Bc_2) + tf.nn.l2_loss(self.Bc_3))
		
		self.global_step = tf.Variable(0, trainable=False)
		self.lr_step = tf.train.exponential_decay(self.learning_rate, self.global_step, 20000, 0.96, staircase=True)
		self.optimizer = tf.train.GradientDescentOptimizer(self.lr_step)
		self.gen_updates = self.optimizer.minimize(self.gen_loss, var_list=self.params)

	def save_model(self, sess, filename):
		param = sess.run(self.params)
		cPickle.dump(param, open(filename, 'w'))