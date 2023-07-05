import numpy as np
import scipy.io
import os
from coco.args import opt

COMMON_PATH = '/media/ling/DATA/'


class TrainDataSet(object):
    def __init__(self, img, txt, lab, tri):
        """Construct a DataSet.
        """
        self._img = img
        self._txt = txt
        self._lab = (lab > 0).astype(np.float32)
        self._pos = 0
        self._num_sample = tri.shape[0]
        self._tri = tri

    def next_batch(self, batch_size):
        if self._pos + batch_size >= self._num_sample:
            idx1 = np.arange(self._pos, self._num_sample)
            self._pos = self._pos + batch_size - self._num_sample
            idx2 = np.arange(0, self._pos)
            idx = np.hstack((idx1, idx2))
        else:
            idx = np.arange(self._pos, self._pos + batch_size)
            self._pos += batch_size

        img1 = self._img[self._tri[idx, 0], :]
        img2 = self._img[self._tri[idx, 1], :]
        img3 = self._img[self._tri[idx, 2], :]
        txt1 = self._txt[self._tri[idx, 0], :]
        txt2 = self._txt[self._tri[idx, 1], :]
        txt3 = self._txt[self._tri[idx, 2], :]
        lab1 = self._lab[self._tri[idx, 0], :]
        lab2 = self._lab[self._tri[idx, 1], :]
        lab3 = self._lab[self._tri[idx, 2], :]

        img = np.vstack((img1, img2, img3))
        txt = np.vstack((txt1, txt2, txt3))
        lab = np.vstack((lab1, lab2, lab3))

        return img, txt, lab


# for testing
class TestDataSet(object):
    def __init__(self, img, txt, label):
        """construct a dataset.
        """
        self._img = img
        self._txt = txt
        self._label = label
        self._idx_in_epoch = 0
        self._num_examples = img.shape[0]

    def next_batch(self):
        """Return the next `batch_size` examples from this data set."""
        return self._img, self._txt, self._label


def get_data(tt='nop'):
    if tt == 'nop':
        data = np.load('datasets/mscoco/data_split.npz')
        data1 = np.load('datasets/mscoco/tfidf_split.npz')
    elif tt == 'cca':
        data = scipy.io.loadmat('datasets/mscoco/cca_icptlog_coco_' + str(opt.txt_dim) + '.mat')
    triplets = np.load('sample_tri_coco.npy')
    return TrainDataSet(data['img_tr'], data['txt_tr'], data['label_tr'], triplets), TestDataSet(data['img_te'], data['txt_te'], data['label_te'])


def gen_tri_mc():
    data = np.load('datasets/mscoco/data_split.npz')
    label = data['label_tr']
    #
    num_sample = int(1e5)
    triplets = np.zeros((num_sample, 3), dtype=np.int32)
    for i in range(num_sample):
        c_idx = np.random.randint(0, label.shape[0])
        while 1:
            p_idx = np.random.randint(0, label.shape[0])
            if np.any(np.logical_and(label[c_idx], label[p_idx])):
                break
        while 1:
            n_idx = np.random.randint(0, label.shape[0])
            if not np.any(np.logical_and(label[c_idx], label[n_idx])):
                break
        triplets[i, 0] = c_idx
        triplets[i, 1] = p_idx
        triplets[i, 2] = n_idx
    np.save('sample_tri_coco', triplets)


if __name__ == "__main__":
    gen_tri_mc()