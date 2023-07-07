"""Dataloader"""

import csv
import torch
import torch.utils.data as data
import os
import nltk
import numpy as np


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp, cc152k_precomp
    """

    def __init__(self, data_path, data_split, vocab, opt=None):
        self.vocab = vocab
        loc = data_path + '/'

        # load the raw captions
        self.captions = []

        if 'cc152k' in opt.data_name:
            with open(loc + '%s_caps.tsv' % data_split) as f:
                tsvreader = csv.reader(f, delimiter='\t')
                for line in tsvreader:
                    self.captions.append(line[1].strip())
        else:
            with open(loc + '%s_caps.txt' % data_split, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    self.captions.append(line.strip())

        # load the image features
        self.images = np.load(loc + '%s_ims.npy' % data_split)
        img_len = self.images.shape[0]

        self.noisy_inx = np.arange(img_len)
        if data_split == 'train' and opt.noise_ratio > 0.0:
            noise_file = opt.noise_file
            if os.path.exists(noise_file):
                print('=> load noisy index from {}'.format(noise_file))
                self.noisy_inx = np.load(noise_file)
            else:
                noise_ratio = opt.noise_ratio
                inx = np.arange(img_len)
                np.random.shuffle(inx)
                noisy_inx = inx[0: int(noise_ratio * img_len)]
                shuffle_noisy_inx = np.array(noisy_inx)
                np.random.shuffle(shuffle_noisy_inx)
                self.noisy_inx[noisy_inx] = shuffle_noisy_inx
                print('Noisy ratio: %g' % noise_ratio)
        self.length = len(self.captions)

        # rkiros data has redundancy in images, we divide by 5
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000
            if 'cc152k' in opt.data_name:
                self.length = 1000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = self.noisy_inx[int(index / self.im_div)]
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        if img_id == list(self.noisy_inx).index(img_id):
            label = 1
        else:
            label = 0
        return image, target, index, img_id, label

    def __len__(self):
        return self.length


def collate_fn(data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, label = zip(*data)

    # Merge images (convert tuple of 2D tensor to 3D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, list(ids), list(label)


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split, vocab, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the train_loader
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    # get the val_loader
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    # get the test_loader
    test_loader = get_precomp_loader(dpath, 'test', vocab, opt,
                                     batch_size, False, workers)
    return train_loader, val_loader, test_loader


def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the test_loader
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
