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
import codecs

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'
        self.data_split = data_split

        # Captions
        self.captions = []
        with codecs.open(loc+'%s_caps.txt' % data_split, 'r', encoding='utf-8') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc+'%s_feat.npy' % data_split)
        self.length = len(self.captions)
        #self.postag = jsonmod.load(open(loc+'postag_%s.json' % data_split, 'r'))
        #assert len(self.postag) == self.length

        print("split: %s, total images: %d, total captions: %d" % (data_split, self.images.shape[0], self.length))
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

        #self.valid_pos = ["NN", "NNS", "NNP", "NNPS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS"]
        #self.pos_dict = {"no": 0, "NN": 1, "RB": 2, "VB":3, "JJ":4}

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index/self.im_div)
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())

        #caption_pos = [0]
        #for pos in nltk.pos_tag(tokens):
        #    if pos[1] not in self.valid_pos:
        #        caption_pos.append(self.pos_dict["no"])
        #    else:
        #        caption_pos.append(self.pos_dict[pos[1][:2]])
        #caption_pos.append(0)
        #caption_pos = self.postag[index]
        target_pos = None#torch.Tensor(caption_pos)

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        #if len(caption) != len(caption_pos):
        #    print(tokens)
        #    print(len(caption), len(caption_pos))

        #ssert len(caption) == len(caption_pos)

        target = torch.Tensor(caption)
        return image, target, target_pos, index, img_id

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
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, captions_pos, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = torch.LongTensor([len(cap) for cap in captions])
    targets = torch.zeros(len(captions), max(lengths)).long()
    targets_pos = torch.zeros(len(captions), max(lengths)).long()
    masks = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
        #targets_pos[i, :end] = captions_pos[i][:end]
        masks[i, :end] = 1

    return images, targets, targets_pos, lengths, masks, ids, None

def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
