# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
from pycocotools.coco import COCO


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, transform, data_path, data_split, vocab):
        self.vocab = vocab
        self.data_path = data_path
        loc = data_path + '/'
        self.data_split = data_split
        self.transform = transform
        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        if self.data_path.endswith('coco_precomp'):
            self.imid = []
            with open(loc+'%s_ids.txt' % data_split, 'rb') as f:
                for line in f:
                    self.imid.append(line.strip())

        elif self.data_path.endswith('f30k_precomp'):
            imgpath = '/data3/zhangyf/cross_modal_retrieval/Multimodal_Retrieval/data/f30k'
            self.imgdir = os.path.join(imgpath, 'images')
            cap = os.path.join(imgpath, 'dataset_flickr30k.json')
            self.dataset = jsonmod.load(open(cap, 'r'))['images']
            self.ids = []
            if data_split == 'dev':
                for i, d in enumerate(self.dataset):
                    if d['split'] == 'val':
                        self.ids += [(i, x) for x in range(len(d['sentences']))]
            else:
                for i, d in enumerate(self.dataset):
                    if d['split'] == data_split:
                        self.ids += [(i, x) for x in range(len(d['sentences']))]
        
        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index/self.im_div
        #print 'start'
        #print img_id

        if self.data_path.endswith('coco_precomp'):
            if self.data_split == 'train':
                if img_id<82783:
                    s = self.imid[img_id].zfill(12)
                    path = '/data3/zhangyf/cross_modal_retrieval/Multimodal_Retrieval/data/coco/images/train2014/' + 'COCO_train2014_' + s + '.jpg'
                else:
                    s = self.imid[img_id].zfill(12)
                    path = '/data3/zhangyf/cross_modal_retrieval/Multimodal_Retrieval/data/coco/images/val2014/' + 'COCO_val2014_' + s + '.jpg'
            else:
                s = self.imid[img_id].zfill(12)
                path = '/data3/zhangyf/cross_modal_retrieval/Multimodal_Retrieval/data/coco/images/val2014/' + 'COCO_val2014_' + s + '.jpg'

            oimage = Image.open(path).convert('RGB')

        elif self.data_path.endswith('f30k_precomp'):
            ann_id = self.ids[img_id]
            f30k_img_id = ann_id[0]
            path = self.imgdir + '/' + self.dataset[f30k_img_id]['filename']

            oimage = Image.open(path).convert('RGB')

        if self.transform is not None:
            oimage = self.transform(oimage)                

        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        #print path
        #print caption
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return oimage, image, target, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[2]), reverse=True)
    oimages, images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    oimages = torch.stack(oimages, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return oimages, images, targets, lengths, ids

def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomSizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Scale(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Scale(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform

def get_precomp_loader(transform, data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(transform, data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    transform = get_transform(data_name, 'train', opt)
    train_loader = get_precomp_loader(transform, dpath, 'train', vocab, opt,
                                      batch_size, True, workers)

    transform = get_transform(data_name, 'val', opt)
    val_loader = get_precomp_loader(transform, dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    transform = get_transform(data_name, 'test', opt)
    test_loader = get_precomp_loader(transform, dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader




