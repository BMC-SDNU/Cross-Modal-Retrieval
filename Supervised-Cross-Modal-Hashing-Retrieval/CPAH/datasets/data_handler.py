import numpy as np
import h5py
import scipy.io as scio


def load_data(path, type='ucm'):
    print('Loading', type)
    if type == 'ucm' or type == 'rsicd':
        return load_ucm(path)
    elif type == 'flickr25k':
        return load_flickr25k(path)
    else:
        return load_nus_wide(path)


def load_flickr25k(path):
    data_file = scio.loadmat(path)
    images = data_file['images'][:]
    mean, std = images.mean(), images.std()
    images = (images - mean) / std
    tags = data_file['YAll'][:]
    labels = data_file['LAll'][:]
    return images, tags, labels


def load_nus_wide(path_dir):
    data_file = scio.loadmat(path_dir)
    images = data_file['image'][:]
    images = (images - images.mean()) / images.std()
    tags = data_file['text'][:]
    labels = data_file['label'][:]
    return images, tags, labels


def load_ucm(path):
    with h5py.File(path, "r") as hf:
        np.random.seed(42)
        images = hf['images'][:]
        images = (images - images.mean()) / images.std()
        labels = hf['labels'][:]
        tags = hf['bow'][:]  # embeddings
        tags = (tags - tags.mean()) / tags.std()
        images = duplicate_data(images, 5)  # 5 times more captions than images
        labels = duplicate_data(labels, 5)  # 5 times more captions than labels
        perm = np.random.permutation(len(images))
        images = images[perm]
        labels = labels[perm]
        tags = tags[perm]
    return images, tags, labels


def duplicate_data(data, n):
    """
    Duplicates each value of 0-dim n times
        for n = 3: (1, 2, 3) -> (1, 1, 1, 2, 2, 2, 3, 3, 3)

    :param data: original data
    :param n: number of duplications
    :return:
    """
    new_data = np.zeros((data.shape[0] * n, data.shape[1]), dtype=data.dtype)
    idx = 0
    for d in data:
        for i in range(n):
            new_data[idx] = d
            idx += 1
    return new_data


