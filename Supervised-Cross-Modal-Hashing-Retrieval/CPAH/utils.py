import torch
import numpy as np
import visdom
import pickle
from tqdm import tqdm
from PIL import Image


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def image_from_numpy(x):
    if x.max() > 1.0:
        x = x / 255
    if type(x) != np.ndarray:
        x = x.numpy()
    im = Image.fromarray(np.uint8(x * 255))
    im.show()


def pr_curve(qB, rB, query_label, retrieval_label, tqdm_label=''):
    if tqdm_label != '':
        tqdm_label = 'PR-curve ' + tqdm_label

    num_query = qB.shape[0]  # length of query (each sample from query compared to retrieval samples)
    num_bit = qB.shape[1]  # length of hash code
    P = torch.zeros(num_query, num_bit + 1)  # precisions (for each sample)
    R = torch.zeros(num_query, num_bit + 1)  # recalls (for each sample)

    # for each sample from query calculate precision and recall
    for i in tqdm(range(num_query), desc=tqdm_label):
        # gnd[j] == 1 if same class, otherwise 0, ground truth
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # tsum (TP + FN): total number of samples of the same class
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)  # hamming distances from qB[i, :] (current query sample) to retrieval samples
        # tmp[k,j] == 1 if hamming distance to retrieval sample j is less or equal to k (distance), 0 otherwise
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        # total (TP + FP): total[k] is count of distances less or equal to k (from current query sample to retrieval samples)
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.0001  # replace zeros with 0.1 to avoid division by zero
        # select only same class samples from tmp (ground truth masking, only rows where gnd == 1 proceed further)
        t = gnd * tmp
        # count (TP): number of true (correctly selected) samples of the same class for any given distance k
        count = t.sum(dim=-1)
        p = count / total  # TP / (TP + FP)
        r = count / tsum  # TP / (TP + FN)
        P[i] = p
        R[i] = r
    # mask to calculate P mean value (among all query samples)
    #mask = (P > 0).float().sum(dim=0)
    #mask = mask + (mask == 0).float() * 0.001
    #P = P.sum(dim=0) / mask
    # mask to calculate R mean value (among all query samples)
    #mask = (R > 0).float().sum(dim=0)
    #mask = mask + (mask == 0).float() * 0.001
    #R = R.sum(dim=0) / mask
    P = P.mean(dim=0)
    R = R.mean(dim=0)
    return P, R


def p_top_k(qB, rB, query_label, retrieval_label, K, tqdm_label=''):
    if tqdm_label != '':
        tqdm_label = 'AP@K ' + tqdm_label

    num_query = qB.shape[0]
    PK = torch.zeros(len(K)).to(qB.device)

    for i in tqdm(range(num_query), desc=tqdm_label):
        # ground_truth[j] == 1 if same class (if at least 1 same label), otherwise 0, ground truth
        ground_truth = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # count of samples, that shall be retrieved
        tp_fn = ground_truth.sum()
        if tp_fn == 0:
            continue

        hamm_dist = calc_hamming_dist(qB[i, :], rB).squeeze()

        # for each k in K
        for j, k in enumerate(K):
            k = min(k, retrieval_label.shape[0])
            _, sorted_indexes = torch.sort(hamm_dist)
            retrieved_indexes = sorted_indexes[:k]
            retrieved_samples = ground_truth[retrieved_indexes]
            PK[j] += retrieved_samples.sum() / k

    PK = PK / num_query

    """
    import matplotlib.pyplot as plt
    plt.semilogx(K, PK)
    plt.savefig('/home/george/Downloads/_fig.png')
    """

    return PK


def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


class Visualizer(object):

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        self.index = {}

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def __getattr__(self, name):
        return getattr(self.vis, name)


def pr_curve2(qB, rB, query_label, retrieval_label, tqdm_label=''):
    if tqdm_label != '':
        tqdm_label = 'PR-curve ' + tqdm_label

    num_query = qB.shape[0]
    num_bit = qB.shape[1] + 1  # range(0, qB.shape[1])

    P = torch.zeros(num_query, num_bit)  # precisions (for all samples, for each radius)
    R = torch.zeros(num_query, num_bit)  # recalls (for all samples, for each radius)

    # i - current sample num
    for i in tqdm(range(num_query), desc=tqdm_label):

        ground_truth = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # count of samples, that shall be retrieved
        tp_fn = ground_truth.sum()
        if tp_fn == 0:
            continue

        # hamming distances from current sample to all db samples
        hamm_dist = calc_hamming_dist(qB[i, :], rB)

        """replace from this point for alternative calculation"""

        # 1 if sample is retrieved for each hamming radius in range(0, num_bit), 0 otherwise
        retrieved_for_each_radius = hamm_dist <= torch.arange(0, num_bit).reshape(-1, 1).float().to(hamm_dist.device)
        retrieved_for_each_radius = retrieved_for_each_radius.float()

        # count of retrieved samples for each hamming radius in range(0, num_bit)
        tp_fp = retrieved_for_each_radius.sum(dim=-1)
        tp_fp = tp_fp + (tp_fp == 0).float() * 0.0001  # replace zeros with 0.1 to avoid division by zero

        # intersection of retrieved and ground truth for each hamming radius in range(0, num_bit)
        retrieved_true = ground_truth * retrieved_for_each_radius
        # count of TP samples for each hamming radius in range(0, num_bit)
        tp = retrieved_true.sum(dim=-1)

        Pi = tp / tp_fp  # TP / (TP + FP), precisions (for current sample, for each hamming radius)
        Ri = tp / tp_fn  # TP / (TP + FN), recalls (for current sample, for each hamming radius)

        P[i] = Pi
        R[i] = Ri

        """
        # alternative calculation (slower)
        
        Pi = torch.zeros(num_bit)  # precisions (for current sample, for each hamming radius)
        Ri = torch.zeros(num_bit)  # recalls (for current sample, for each hamming radius)
        
        # k - current hamming radius
        for k in range(num_bit):
            retrieved_samples = (hamm_dist <= k).float().squeeze()  # sample is retrieved if hamming dist < k (current hamming radius)
            tp_fp = retrieved_samples.sum()

            retrieved_positive_samples = retrieved_samples * ground_truth

            tp = retrieved_positive_samples.sum()

            Pik = precision(tp, tp_fp)  # precision (for current sample, for current radius)
            Rik = recall(tp, tp_fn)  # recall (for current sample, for current radius)

            Pi[k] = Pik
            Ri[k] = Rik
        
        P[i] = Pi
        R[i] = Ri
        """

    # get mean value for precision and recall for each hamming radius in range(0, num_bit)
    P = P.mean(dim=0)
    R = R.mean(dim=0)

    """
    import matplotlib.pyplot as plt
    plt.plot(R, P)
    plt.savefig('/home/george/Downloads/_fig.png')
    """

    return P, R


def recall(TP, TP_plus_FN):
    # (relevant_samples in retrieved_samples) / relevant_samples
    # TP / (TP + FN)
    return TP / (TP_plus_FN + 0.0001)


def precision(TP, TP_plus_FP):
    # (relevant_samples in retrieved_samples) / retrieved_samples
    # TP / (TP + FP)
    return TP / (TP_plus_FP + 0.0001)


########################################################################################################################
########################################################################################################################
########################################################################################################################


import json
import os
import h5py
import random


def read_json(file_name, suppress_console_info=False):
    """
    Read JSON

    :param file_name: input JSON path
    :param suppress_console_info: toggle console printing
    :return: dictionary from JSON
    """
    with open(file_name, 'r') as f:
        data = json.load(f)
        if not suppress_console_info:
            print("Read from:", file_name)
    return data


def get_image_file_names(data, suppress_console_info=False):
    """
    Get list of image file names

    :param data: original data from JSON
    :param suppress_console_info: toggle console printing
    :return: list of strings (file names)
    """

    file_names = []
    for img in data['images']:
        file_names.append(img["filename"])
    if not suppress_console_info:
        print("Total number of files:", len(file_names))
    return file_names


def get_labels(data, suppress_console_info=False):
    """
    Get list of labels

    :param data: original data from JSON
    :param suppress_console_info: toggle console printing
    :return: list ints (labels)
    """

    labels = []
    for img in data['images']:
        labels.append(img["classcode"])
    if not suppress_console_info:
        print("Total number of labels:", len(labels))
    return labels


def get_captions(data, suppress_console_info=False):
    """
    Get list of formatted captions

    :param data: original data from JSON
    :return: list of strings (captions)
    """

    def format_caption(string):
        return string.replace('.', '').replace(',', '').replace('!', '').replace('?', '').lower()

    captions = []
    augmented_captions_rb = []
    augmented_captions_bt_prob = []
    augmented_captions_bt_chain = []
    for img in data['images']:
        for sent in img['sentences']:
            captions.append(format_caption(sent['raw']))
            try:
                augmented_captions_rb.append(format_caption(sent['aug_rb']))
            except:
                pass
            try:
                augmented_captions_bt_prob.append(format_caption(sent['aug_bt_prob']))
            except:
                pass
            try:
                augmented_captions_bt_chain.append(format_caption(sent['aug_bt_chain']))
            except:
                pass
    if not suppress_console_info:
        print("Total number of captions:", len(captions))
        print("Total number of augmented captions RB:", len(augmented_captions_rb))
        print("Total number of augmented captions BT (prob):", len(augmented_captions_bt_prob))
        print("Total number of augmented captions BT (chain):", len(augmented_captions_bt_chain))
    return captions, augmented_captions_rb, augmented_captions_bt_prob, augmented_captions_bt_chain


def add_prefix_to_filename(file_name, prefix):
    """
    Adds prefix to the file name

    :param file_name: file name string
    :param prefix: prefix
    :return: file name string with prefix
    """
    bn = os.path.basename(file_name)
    dn = os.path.dirname(file_name)
    return os.path.join(dn, prefix + bn)


def write_json(file_name, data):
    """
    Write dictionary to JSON file

    :param file_name: output path
    :param data: dictionary
    :return: None
    """
    bn = os.path.basename(file_name)
    dn = os.path.dirname(file_name)
    name, ext = os.path.splitext(bn)
    file_name = os.path.join(dn, name + '.json')
    with open(file_name, 'w') as f:
        f.write(json.dumps(data, indent='\t'))
    print("Written to:", file_name)


def print_parsed_args(parsed):
    """
    Print parsed arguments

    :param parsed: Namespace of parsed arguments
    :return: None
    """
    vs = vars(parsed)
    print("Parsed arguments: ")
    for k, v in vs.items():
        print("\t" + str(k) + ": " + str(v))


def write_hdf5(out_file, data, dataset_name):
    """
    Write to h5 file

    :param out_file: file name
    :param data: data to write
    :return:
    """
    bn = os.path.basename(out_file)
    dn = os.path.dirname(out_file)
    name, ext = os.path.splitext(bn)
    out_file = os.path.join(dn, name + '.h5')
    with h5py.File(out_file, 'w') as hf:
        print("Saved as '.h5' file to", out_file)
        hf.create_dataset(dataset_name, data=data)


def read_hdf5(file_name, dataset_name, normalize=False):
    """
    Read from h5 file

    :param file_name: file name
    :param dataset_name: dataset name
    :param normalize: normalize loaded values
    :return:
    """
    with h5py.File(file_name, 'r') as hf:
        print("Read from:", file_name)
        data = hf[dataset_name][:]
        if normalize:
            data = (data - data.mean()) / data.std()
        return data


def write_pickle(path, data):
    """
    Write pickle

    :param path: path
    :param data: data to write
    :return:
    """
    dn = os.path.dirname(path)
    if not os.path.exists(dn):
        os.makedirs(dn)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def select_idxs(seq_length, n_to_select, n_from_select, seed=42):
    """
    Select n_to_select indexes from each consequent n_from_select indexes from range with length seq_length, split
    selected indexes to separate arrays

    Example:

    seq_length = 20
    n_from_select = 5
    n_to_select = 2

    input, range of length seq_length:
    range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    sequences of length n_from_select:
    sequences = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]

    selected n_to_select elements from each sequence
    selected = [[0, 4], [7, 9], [13, 14], [16, 18]]

    output, n_to_select lists of length seq_length / n_from_select:
    output = [[0, 7, 13, 16], [4, 9, 14, 18]]

    :param seq_length: length of sequence, say 10
    :param n_to_select: number of elements to select
    :param n_from_select: number of consequent elements
    :return:
    """
    random.seed(seed)
    idxs = [[] for _ in range(n_to_select)]
    for i in range(seq_length // n_from_select):
        ints = random.sample(range(n_from_select), n_to_select)
        for j in range(n_to_select):
            idxs[j].append(i * n_from_select + ints[j])
    return idxs


def calc_map_k2(qB, rB, query_label, retrieval_label, k=None):
    """
    calculate MAPs

    :param qB: query binary codes
    :param rB: response binary codes
    :param query_label: labels of query
    :param retrieval_label: labels of response
    :param k: k
    :return:
    """
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_map_rad2(qB, rB, query_label, retrieval_label):
    """
    calculate MAPs, in regard to hamming radius

    :param qB: query binary codes
    :param rB: response binary codes
    :param query_label: labels of query
    :param retrieval_label: labels of response
    :return:
    """

    num_query = qB.shape[0]  # length of query (each sample from query compared to retrieval samples)
    num_bit = qB.shape[1]  # length of hash code
    P = torch.zeros(num_query, num_bit + 1)  # precisions (for each sample)

    # for each sample from query calculate precision and recall
    for i in range(num_query):
        # gnd[j] == 1 if same class, otherwise 0, ground truth
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # tsum (TP + FN): total number of samples of the same class
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)  # hamming distances from qB[i, :] (current sample) to retrieval samples
        # tmp[k,j] == 1 if hamming distance to retrieval sample j is less or equal to k (distance), 0 otherwise
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        # total (TP + FP): total[k] is count of distances less or equal to k (from query sample to retrieval samples)
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.0001  # replace zeros with 0.1 to avoid division by zero
        # select only same class samples from tmp (ground truth masking, only rows where gnd == 1 proceed further)
        t = gnd * tmp
        # count (TP): number of true (correctly selected) samples of the same class for any given distance k
        count = t.sum(dim=-1)
        p = count / total  # TP / (TP + FP)
        P[i] = p
    P = P.mean(dim=0)
    return P


def pr_curve3(qB, rB, query_label, retrieval_label, tqdm_label=''):
    """
    Calculate PR curve, each point - hamming radius

    :param qB: query hash code
    :param rB: retrieval hash codes
    :param query_label: query label
    :param retrieval_label: retrieval label
    :param tqdm_label: label for tqdm's output
    :return:
    """
    if tqdm_label != '':
        tqdm_label = 'PR-curve ' + tqdm_label

    num_query = qB.shape[0]  # length of query (each sample from query compared to retrieval samples)
    num_bit = qB.shape[1]  # length of hash code
    P = torch.zeros(num_query, num_bit + 1)  # precisions (for each sample)
    R = torch.zeros(num_query, num_bit + 1)  # recalls (for each sample)

    # for each sample from query calculate precision and recall
    for i in tqdm(range(num_query), desc=tqdm_label):
        # gnd[j] == 1 if same class, otherwise 0, ground truth
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # tsum (TP + FN): total number of samples of the same class
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)  # hamming distances from qB[i, :] (current sample) to retrieval samples
        # tmp[k,j] == 1 if hamming distance to retrieval sample j is less or equal to k (distance), 0 otherwise
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        # total (TP + FP): total[k] is count of distances less or equal to k (from query sample to retrieval samples)
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.0001  # replace zeros with 0.1 to avoid division by zero
        # select only same class samples from tmp (ground truth masking, only rows where gnd == 1 proceed further)
        t = gnd * tmp
        # count (TP): number of true (correctly selected) samples of the same class for any given distance k
        count = t.sum(dim=-1)
        p = count / total  # TP / (TP + FP)
        r = count / tsum  # TP / (TP + FN)
        P[i] = p
        R[i] = r
    # mask to calculate P mean value (among all query samples)
    # mask = (P > 0).float().sum(dim=0)
    # mask = mask + (mask == 0).float() * 0.001
    # P = P.sum(dim=0) / mask
    # mask to calculate R mean value (among all query samples)
    # mask = (R > 0).float().sum(dim=0)
    # mask = mask + (mask == 0).float() * 0.001
    # R = R.sum(dim=0) / mask
    P = P.mean(dim=0)
    R = R.mean(dim=0)
    return P.cpu().numpy().tolist(), R.cpu().numpy().tolist()


def p_top_k2(qB, rB, query_label, retrieval_label, K, tqdm_label=''):
    """
    P@K curve

    :param qB: query hash code
    :param rB: retrieval hash codes
    :param query_label: query label
    :param retrieval_label: retrieval label
    :param K: K's for curve
    :param tqdm_label: label for tqdm's output
    :return:
    """
    if tqdm_label != '':
        tqdm_label = 'AP@K ' + tqdm_label

    num_query = qB.shape[0]
    PK = torch.zeros(len(K)).to(qB.device)

    for i in tqdm(range(num_query), desc=tqdm_label):
        # ground_truth[j] == 1 if same class (if at least 1 same label), otherwise 0, ground truth
        ground_truth = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # count of samples, that shall be retrieved
        tp_fn = ground_truth.sum()
        if tp_fn == 0:
            continue

        hamm_dist = calc_hamming_dist(qB[i, :], rB).squeeze()

        # for each k in K
        for j, k in enumerate(K):
            k = min(k, retrieval_label.shape[0])
            _, sorted_indexes = torch.sort(hamm_dist)
            retrieved_indexes = sorted_indexes[:k]
            retrieved_samples = ground_truth[retrieved_indexes]
            PK[j] += retrieved_samples.sum() / k

    PK = PK / num_query

    return PK.cpu().numpy().tolist()

