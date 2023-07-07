#coding:utf-8
import h5py
import scipy.io as sio
import numpy as np
def loading_data(path):
    image_path = path + "NImage.mat"
    label_path = path + "Label.mat"
    tag_path = path + "Tags.mat"
    images = sio.loadmat(image_path)['img']
    images = images.transpose(0, 3, 1, 2)
    tags = sio.loadmat(tag_path)['tags']
    labels = sio.loadmat(label_path)["label"]
    return images, tags, labels

def loading_cloth_data(path):
    image_path = path + "image.mat"
    label_path = path + "label.mat"
    tag_path = path + "tag.mat"
    images=sio.loadmat(image_path)['Image']
    images=images.transpose(0,3,1,2)
    tags = sio.loadmat(tag_path)['Tag']
    labels = sio.loadmat(label_path)["Label"]
    return images, tags, labels
if __name__=='__main__':
    path = '/home/share/sunchangchang/data/data1/'
    images, tags, labels = loading_cloth_data(path)
    print (images.shape)
    print (tags.shape)
    print (labels.shape)