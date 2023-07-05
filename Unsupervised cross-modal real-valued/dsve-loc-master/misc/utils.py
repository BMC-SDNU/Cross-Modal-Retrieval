"""
****************** COPYRIGHT AND CONFIDENTIALITY INFORMATION ******************
Copyright (c) 2018 [Thomson Licensing]
All Rights Reserved
This program contains proprietary information which is a trade secret/business \
secret of [Thomson Licensing] and is protected, even if unpublished, under \
applicable Copyright laws (including French droit d'auteur) and/or may be \
subject to one or more patent(s).
Recipient is to retain this program in confidence and is not permitted to use \
or make copies thereof other than as permitted in a written agreement with \
[Thomson Licensing] unless otherwise expressly allowed by applicable laws or \
by [Thomson Licensing] under express agreement.
Thomson Licensing is a company of the group TECHNICOLOR
*******************************************************************************
This scripts permits one to reproduce training and experiments of:
    Engilberge, M., Chevallier, L., Pérez, P., & Cord, M. (2018, April).
    Finding beans in burgers: Deep semantic-visual embedding with localization.
    In Proceedings of CVPR (pp. 3984-3993)

Author: Martin Engilberge
"""

import os

import nltk
import pickle
import torch

from nltk.tokenize import word_tokenize
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Namespace:
    """ Namespace class to manually instantiate joint_embedding model """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_dictionary(dir_st):
    path_dico = os.path.join(dir_st, 'dictionary.txt')
    if not os.path.exists(path_dico):
        print("Invalid path no dictionary found")
    with open(path_dico, 'r') as handle:
        dico_list = handle.readlines()
    dico = {word.strip(): idx for idx, word in enumerate(dico_list)}
    return dico


def preprocess(text):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(text)
    result = list()
    for s in sents:
        tokens = word_tokenize(s)
        result.append(tokens)

    return result


def flatten(l):
    return [item for sublist in l for item in sublist]


def encode_sentences(sents, embed, dico):
    sents_list = list()
    for sent in sents:
        sent_tok = preprocess(sent)[0]
        sent_in = Variable(torch.FloatTensor(1, len(sent_tok), 620))
        for i, w in enumerate(sent_tok):
            try:
                sent_in.data[0, i] = torch.from_numpy(embed[dico[w]])
            except KeyError:
                sent_in.data[0, i] = torch.from_numpy(embed[dico["UNK"]])

        sents_list.append(sent_in)
    return sents_list


def encode_sentence(sent, embed, dico, tokenize=True):
    if tokenize:
        sent_tok = preprocess(sent)[0]
    else:
        sent_tok = sent

    sent_in = torch.FloatTensor(len(sent_tok), 620)

    for i, w in enumerate(sent_tok):
        try:
            sent_in[i, :620] = torch.from_numpy(embed[dico[w]])
        except KeyError:
            sent_in[i, :620] = torch.from_numpy(embed[dico["UNK"]])

    return sent_in


def save_checkpoint(state, is_best, model_name, epoch):
    if is_best:
        torch.save(state, './weights/best_' + model_name + ".pth.tar")


def log_epoch(logger, epoch, train_loss, val_loss, lr, batch_train, batch_val, data_train, data_val, recall):
    logger.add_scalar('Loss/Train', train_loss, epoch)
    logger.add_scalar('Loss/Val', val_loss, epoch)
    logger.add_scalar('Learning/Rate', lr, epoch)
    logger.add_scalar('Learning/Overfitting', val_loss / train_loss, epoch)
    logger.add_scalar('Time/Train/Batch Processing', batch_train, epoch)
    logger.add_scalar('Time/Val/Batch Processing', batch_val, epoch)
    logger.add_scalar('Time/Train/Data loading', data_train, epoch)
    logger.add_scalar('Time/Val/Data loading', data_val, epoch)
    logger.add_scalar('Recall/Val/CapRet/R@1', recall[0][0], epoch)
    logger.add_scalar('Recall/Val/CapRet/R@5', recall[0][1], epoch)
    logger.add_scalar('Recall/Val/CapRet/R@10', recall[0][2], epoch)
    logger.add_scalar('Recall/Val/CapRet/MedR', recall[2], epoch)
    logger.add_scalar('Recall/Val/ImgRet/R@1', recall[1][0], epoch)
    logger.add_scalar('Recall/Val/ImgRet/R@5', recall[1][1], epoch)
    logger.add_scalar('Recall/Val/ImgRet/R@10', recall[1][2], epoch)
    logger.add_scalar('Recall/Val/ImgRet/MedR', recall[3], epoch)


def collate_fn_padded(data):
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = pad_sequence(captions, batch_first=True)

    return images, targets, lengths


def collate_fn_cap_padded(data):
    captions = data

    lengths = [len(cap) for cap in captions]
    targets = pad_sequence(captions, batch_first=True)

    return targets, lengths


def collate_fn_semseg(data):
    images, size, targets = zip(*data)
    images = torch.stack(images, 0)

    return images, size, targets


def collate_fn_img_padded(data):
    images = data
    images = torch.stack(images, 0)

    return images


def load_obj(path):
    with open(os.path.normpath(path + '.pkl'), 'rb') as f:
        return pickle.load(f)


def save_obj(obj, path):
    with open(os.path.normpath(path + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
