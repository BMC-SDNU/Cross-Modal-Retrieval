import os
import numpy as np
import h5py
import scipy.io as scio


def load_data(path, type='flickr25k'):
    if type == 'flickr25k':
        return load_flickr25k(path)
    else:
        return load_nus_wide(path)


def load_flickr25k(path):
    data_file = scio.loadmat(path)
    images = data_file['images'][:]
    images = (images - images.mean()) / images.std()
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


def load_pretrain_model(path):
    return scio.loadmat(path)

