import numpy as np
import pdb

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)
	
def MAP(sess, model, query_pos_test, query_index_url_test, feature_dict, flag):
	rs = []
	hash_dict = {}
	for item in feature_dict:
		input_data = np.asarray(feature_dict[item])
		input_data_dim = input_data.shape[0]
		input_data = input_data.reshape(1, input_data_dim)
		
		if item.split('.')[-1] == 'jpg':
			output_hash = sess.run(model.image_hash, feed_dict={model.image_data: input_data})
		elif item.split('.')[-1] == 'txt':
			output_hash = sess.run(model.text_hash, feed_dict={model.text_data: input_data})
		hash_dict[item] = output_hash
	
	for query in query_pos_test.keys():
		pos_set = set(query_pos_test[query])
		pred_list = query_index_url_test[query]
		
		pred_list_score = []
		query_hash = hash_dict[query]
		for candidate in pred_list:
			score = 0
			candidate_hash = hash_dict[candidate]
			for i in range(query_hash.shape[1]):
				if query_hash[0][i] == candidate_hash[0][i]:
					score += 1
			pred_list_score.append(score)
			
			# pdb.set_trace()
		
		pred_url_score = zip(pred_list, pred_list_score)
		pred_url_score = sorted(pred_url_score, key=lambda x: x[1], reverse=True)

		r = [0.0] * len(pred_list_score)
		for i in range(0, len(pred_list_score)):
			(url, score) = pred_url_score[i]
			if url in pos_set:
				r[i] = 1.0
		rs.append(r)
			
	return np.mean([average_precision(r) for r in rs])