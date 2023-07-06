import linecache, pdb

def push_query(query, url, dict):
	if query in dict:
		dict[query].append(url)
	else:
		dict[query] = [url]
	return dict
	
def make_train_dict(query_list, url_list, semi_list, query_label, url_label, label_dim):
	query_url = {}
	query_pos = {}
	query_neg = {}
	query_pair = {}
	query_num = len(query_list) - 1
	url_num = len(url_list) - 1
	semi_num = len(semi_list) - 1
	
	for i in range(query_num):
		query = query_list[i]
		for j in range(url_num):
			url = url_list[j]
			if i == j:
				push_query(query, url, query_url)
				push_query(query, url, query_pos)
				push_query(query, url, query_pair)
			elif query_label[i] == url_label[j]:
				push_query(query, url, query_url)
				push_query(query, url, query_pos)
			else:
				push_query(query, url, query_url)
				push_query(query, url, query_neg)
		for j in range(semi_num):
			url = semi_list[j]
			push_query(query, url, query_url)
			
	return query_url, query_pos, query_neg, query_pair

def make_test_dict(query_list, url_list, query_label, url_label, label_dim):
	query_url = {}
	query_pos = {}
	query_num = len(query_list) - 1
	url_num = len(url_list) - 1
	
	for i in range(query_num):
		query = query_list[i]
		for j in range(url_num):
			url = url_list[j]				
			if query_label[i] == url_label[j]:
				push_query(query, url, query_url)
				push_query(query, url, query_pos)
			else:
				push_query(query, url, query_url)
	return query_url, query_pos	

def load_all_query_url(list_dir, label_dim):
	train_img = open(list_dir + 'train_img.txt', 'r').read().split('\r\n')
	test_img = open(list_dir + 'test_img.txt', 'r').read().split('\r\n')
	semi_img = open(list_dir + 'test_img.txt', 'r').read().split('\r\n')

	train_txt = open(list_dir + 'train_txt.txt', 'r').read().split('\r\n')
	test_txt = open(list_dir + 'test_txt.txt', 'r').read().split('\r\n')
	semi_txt = open(list_dir + 'test_txt.txt', 'r').read().split('\r\n')

	train_label = open(list_dir + 'train_label.txt', 'r').read().split('\r\n')
	test_label = open(list_dir + 'test_label.txt', 'r').read().split('\r\n')
	
	train_i2t, train_i2t_pos, train_i2t_neg, train_i2t_pair = make_train_dict(train_img, train_txt, semi_txt, train_label, train_label, label_dim)
	train_t2i, train_t2i_pos, train_t2i_neg, train_t2i_pair = make_train_dict(train_txt, train_img, semi_img, train_label, train_label, label_dim)
	
	test_i2t, test_i2t_pos = make_test_dict(test_img, train_txt, test_label, train_label, label_dim)
	test_t2i, test_t2i_pos = make_test_dict(test_txt, train_img, test_label, train_label, label_dim)
	
	return train_i2t, train_i2t_pos, train_i2t_neg, train_i2t_pair, train_t2i, train_t2i_pos, train_t2i_neg, train_t2i_pair, test_i2t, test_i2t_pos, test_t2i, test_t2i_pos


def load_all_feature(list_dir, feature_dir):
	feature_dict = {}
	for dataset in ['train', 'test']:
		for modal in ['img', 'txt']:
			list = open(list_dir + dataset + '_' + modal + '.txt', 'r').read().split('\r\n')
			feature = open(feature_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
			
			for i in range(len(list) - 1):
				item = list[i]
				feature_string = feature[i].split()
				feature_float_list = []
				for j in range(len(feature_string)):
					feature_float_list.append(float(feature_string[j]))
				feature_dict[item] = feature_float_list
	return feature_dict
	
def load_all_feature_for_test(list_dir, feature_dir):
	feature_dict = {}
	for dataset in ['train', 'test']:
		for modal in ['img', 'txt']:
			list = open(list_dir + dataset + '_' + modal + '.txt', 'r').read().split('\r\n')
			feature = open(feature_dir + dataset + '_' + modal + '.txt', 'r').read().split('\n')
			
			for i in range(len(list) - 1):
				item = list[i]
				feature_string = feature[i].split()
				feature_float_list = []
				for j in range(len(feature_string)):
					feature_float_list.append(float(feature_string[j]))
				feature_dict[item] = feature_float_list
	return feature_dict

def load_all_label(list_dir):
	label_dict = {}
	for dataset in ['train', 'test']:
		for modal in ['img', 'txt']:
			list = open(list_dir + dataset + '_' + modal + '.txt', 'r').read().split('\r\n')
			label = open(list_dir + dataset + '_label.txt', 'r').read().split('\r\n')
			for i in range(len(list) - 1):
				label_dict[list[i]] = int(label[i])
	return label_dict
	
	
def get_query_pos(file, semi_flag):
	query_pos = {}
	with open(file) as fin:
		for line in fin:
			cols = line.split()
			rank = float(cols[0])
			query = cols[1]
			url = cols[2]
			if rank > semi_flag:
				if query in query_pos:
					query_pos[query].append(url)
				else:
					query_pos[query] = [url]
	return query_pos


def file_len(fname):
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1


# Get batch data from training set
def get_batch_data(file, index, size):
	pos = []
	neg = []
	for i in range(index, index + size):
		line = linecache.getline(file, i)
		line = line.strip().split()
		pos.append([float(x) for x in line[0].split(',')])
		neg.append([float(x) for x in line[1].split(',')])
	return pos, neg