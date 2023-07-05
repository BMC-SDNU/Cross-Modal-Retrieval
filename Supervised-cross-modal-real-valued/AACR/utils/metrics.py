import numpy as np
import math
from scipy.spatial.distance import cdist


def ap_k(y_true, s_predict, k):
    order = np.argsort(s_predict)[::-1][:k]  # descend
    y_predict = y_true[order]

    num_hits = 0
    ap = 0.0
    for i in range(k):
        if y_predict[i]:
            num_hits += 1
            ap += float(num_hits) / (i + 1)
    if num_hits != 0:
        ap /= num_hits
    return ap


def ap_k_multipos(y_true, s_predict, k):
    # k (n,) is a numpy array which defines multiple position
    order = np.argsort(s_predict)[::-1][:k[-1]]  # descend
    y_predict = y_true[order]
    ap = np.zeros((k.shape[0],))
    k = np.insert(k, 0, 0)
    num_hits = 0
    for j in range(k.shape[0]-1):
        for i in range(k[j], k[j+1]):
            if y_predict[i]:
                num_hits += 1
                ap[j] += float(num_hits) / (i + 1)
        if num_hits != 0:
            if j != k.shape[0]-2:
                ap[j+1] = ap[j]
            ap[j] /= num_hits
    return ap


def evaluate_unilabel(image_representation, text_representation, label, metric='cos'):
    if metric == 'cos':
        score = 1-cdist(image_representation, text_representation, 'cosine')
    elif metric == 'ip':
        score = np.dot(image_representation, text_representation.T)
    k = np.array([10, 50, 100, label.shape[0]])
    map_i2t = np.zeros((k.shape[0],))
    for i in range(label.shape[0]):
        y_true = (label == label[i]).flatten()
        ap = ap_k_multipos(y_true, score[i, :], k)
        map_i2t += ap
    map_i2t /= label.shape[0]
    for i in range(map_i2t.shape[0]):
        print('%f ' % map_i2t[i], end='')
    print(' ')

    map_t2i = np.zeros((k.shape[0],))
    for i in range(label.shape[0]):
        y_true = (label == label[i]).flatten()
        ap = ap_k_multipos(y_true, score[:, i].T, k)
        map_t2i += ap
    map_t2i /= label.shape[0]
    for i in range(map_t2i.shape[0]):
        print('%f ' % map_t2i[i], end='')
    print(' ')
    return map_i2t, map_t2i


def evaluate_multilabel(image_representation, text_representation, label, metric='cos'):
    if metric == 'cos':
        score = 1-cdist(image_representation, text_representation, 'cosine')
    elif metric == 'ip':
        score = np.dot(image_representation, text_representation.T)
    k = np.array([10, 50, 100, label.shape[0]])
    map_i2t = np.zeros((k.shape[0],))
    for i in range(label.shape[0]):
        y_true = np.sum(label[:, label[i]>0], 1)
        ap = ap_k_multipos(y_true, score[i, :], k)
        map_i2t += ap
    map_i2t /= label.shape[0]
    for i in range(map_i2t.shape[0]):
        print('%f ' % map_i2t[i], end='')
    print(' ')

    map_t2i = np.zeros((k.shape[0],))
    for i in range(label.shape[0]):
        y_true = np.sum(label[:, label[i]>0], 1)
        ap = ap_k_multipos(y_true, score[:, i].T, k)
        map_t2i += ap
    map_t2i /= label.shape[0]
    for i in range(map_t2i.shape[0]):
        print('%f ' % map_t2i[i], end='')
    print(' ')
    return map_i2t, map_t2i


def evaluate_multilabel_large(query, doc, label, metric='cos'):
    batch_size = 5000
    batch_num = math.floor(query.shape[0] / batch_size)
    k = np.array([50, label.shape[0]])
    map = np.zeros((k.shape[0],))
    for batch_iter in range(batch_num):
        if metric == 'cos':
            score = 1 - cdist(query[batch_iter*batch_size:(batch_iter+1)*batch_size], doc, 'cosine')
        elif metric == 'ip':
            score = np.dot(query[batch_iter*batch_size:(batch_iter+1)*batch_size], doc.T)
        for i in range(batch_size):
            y_true = np.sum(label[:, label[batch_iter*batch_size+i] > 0], 1)
            ap = ap_k_multipos(y_true, score[i, :], k)
            map += ap
    if batch_num * batch_size < query.shape[0]:
        if metric == 'cos':
            score = 1 - cdist(query[batch_num*batch_size:], doc, 'cosine')
        elif metric == 'ip':
            score = np.dot(query[batch_num*batch_size:], doc.T)
        for i in range(query.shape[0]-batch_num*batch_size):
            y_true = np.sum(label[:, label[batch_num*batch_size+i] > 0], 1)
            ap = ap_k_multipos(y_true, score[i, :], k)
            map += ap
    map /= query.shape[0]
    for i in range(map.shape[0]):
        print('%f' % map[i], end='')
    print(' ')


def evaluate(image_representation, text_representation, label, metric='cos'):
    if len(label.shape) == 1 or label.shape[1] == 1:
        return evaluate_unilabel(image_representation, text_representation, label, metric)
    else:
        if image_representation.shape[0] > 10000:
            evaluate_multilabel_large(image_representation, text_representation, label, metric)
            evaluate_multilabel_large(text_representation, image_representation, label, metric)
        else:
            return evaluate_multilabel(image_representation, text_representation, label, metric)


def evaluate_s_unilabel(score, label):
    k = np.array([50, label.shape[0]])
    map_i2t = np.zeros((k.shape[0],))
    for i in range(label.shape[0]):
        y_true = (label == label[i]).flatten()
        ap = ap_k_multipos(y_true, score[i, :], k)
        map_i2t += ap
    map_i2t /= label.shape[0]
    for i in range(map_i2t.shape[0]):
        print('%f ' % map_i2t[i], end='')

    map_t2i = np.zeros((k.shape[0],))
    for i in range(label.shape[0]):
        y_true = (label == label[i]).flatten()
        ap = ap_k_multipos(y_true, score[:, i].T, k)
        map_t2i += ap
    map_t2i /= label.shape[0]
    for i in range(map_t2i.shape[0]):
        print('%f ' % map_t2i[i], end='')
    print(' ')


def evaluate_s_multilabel(score, label):
    k = np.array([50, label.shape[0]])
    map_i2t = np.zeros((k.shape[0],))
    for i in range(label.shape[0]):
        y_true = np.sum(label[:, label[i]>0], 1)
        ap = ap_k_multipos(y_true, score[i, :], k)
        map_i2t += ap
    map_i2t /= label.shape[0]
    for i in range(map_i2t.shape[0]):
        print('%f ' % map_i2t[i], end='')

    map_t2i = np.zeros((k.shape[0],))
    for i in range(label.shape[0]):
        y_true = np.sum(label[:, label[i]>0], 1)
        ap = ap_k_multipos(y_true, score[:, i].T, k)
        map_t2i += ap
    map_t2i /= label.shape[0]
    for i in range(map_t2i.shape[0]):
        print('%f ' % map_t2i[i], end='')
    print(' ')


def evaluate_s(score, label):
    if len(label.shape) == 1 or label.shape[1] == 1:
        evaluate_s_unilabel(score, label)
    else:
        evaluate_s_multilabel(score, label)