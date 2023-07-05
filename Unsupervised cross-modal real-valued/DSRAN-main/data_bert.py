# -----------------------------------------------------------
# Dual Semantic Relations Attention Network (DSRAN) implementation based on
# "VSE++: Improving Visual-Semantic Embeddings with Hard Negatives"
# "Learning Dual Semantic Relations with Graph Attention for Image-Text Matching"
# Keyu Wen, Xiaodong Gu, and Qingrong Cheng
# IEEE Transactions on Circuits and Systems for Video Technology, 2020
# Writen by Keyu Wen & Linyang Li, 2020
# ------------------------------------------------------------

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod
from collections import OrderedDict
import copy
from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch.nn.functional as F
import h5py


def get_paths(path, name='coco'):
    roots = {}
    ids = {}
    if 'coco' == name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy')))

        roots['train'] = roots['trainrestval']
        ids['train'] = ids['trainrestval']
    elif 'f30k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids


class CocoDataset(data.Dataset):

    def __init__(self, root, json, tokenizer, feature_path=None, region_bbox_file=None, max_seq_len=32, transform=None, ids=None):
        self.root = root
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
        if ids is None:
            self.ids = list(self.coco.anns.keys())
        else:
            self.ids = ids
        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.region_bbox_file = region_bbox_file
        self.region_det_file_prefix = feature_path

    def __getitem__(self, index):
        root, caption, img_id, path, image, img_rcnn, img_pe = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)

        target = self.get_text_input(caption)
        return img_rcnn, img_pe, target, index, img_id, image

    def get_raw_item(self, index):
        if index < self.bp:
            coco = self.coco[0]
            root = self.root[0]
        else:
            coco = self.coco[1]
            root = self.root[1]
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')
        img_rcnn, img_pe = self.get_rcnn(path)
        return root, caption, img_id, path, image, img_rcnn, img_pe

    def __len__(self):
        return len(self.ids)

    def get_text_input(self, caption):
        caption_tokens = self.tokenizer.tokenize(caption)
        caption_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
        caption_ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)
        if len(caption_ids) >= self.max_seq_len:
            caption_ids = caption_ids[:self.max_seq_len]
        else:
            caption_ids = caption_ids + [0] * (self.max_seq_len - len(caption_ids))
        caption = torch.tensor(caption_ids)
        return caption

    def get_rcnn(self, path):
        img_id = path.split('/')[-1].split('.')[0]
        with h5py.File(self.region_det_file_prefix + '_feat' + img_id[-3:] + '.h5', 'r') as region_feat_f, \
                h5py.File(self.region_det_file_prefix + '_cls' + img_id[-3:] + '.h5', 'r') as region_cls_f, \
                h5py.File(self.region_bbox_file, 'r') as region_bbox_f:

            img = torch.from_numpy(region_feat_f[img_id][:]).float()
            cls_label = torch.from_numpy(region_cls_f[img_id][:]).float()
            vis_pe = torch.from_numpy(region_bbox_f[img_id][:])

        # lazy normalization of the coordinates...

        w_est = torch.max(vis_pe[:, [0, 2]]) * 1. + 1e-5
        h_est = torch.max(vis_pe[:, [1, 3]]) * 1. + 1e-5
        vis_pe[:, [0, 2]] /= w_est
        vis_pe[:, [1, 3]] /= h_est
        rel_area = (vis_pe[:, 3] - vis_pe[:, 1]) * (vis_pe[:, 2] - vis_pe[:, 0])
        rel_area.clamp_(0)

        vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), vis_pe[:, 5:]), -1)  # confident score
        normalized_coord = F.normalize(vis_pe.data[:, :5] - 0.5, dim=-1)
        vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), \
                            F.layer_norm(cls_label, [1601])), dim=-1)  # 1601 hard coded...

        return img, vis_pe


class FlickrDataset(data.Dataset):

    def __init__(self, root, json, split, tokenizer, feature_path=None, region_bbox_file=None, max_seq_len=32,
                 transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]
        self.region_bbox_file = region_bbox_file
        self.feature_path = feature_path

    def __getitem__(self, index):
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']
        path_orig = copy.deepcopy(path)
        path = path.replace('.jpg', '.npy')
        feature_path = self.feature_path
        # orig image
        image_orig = Image.open(os.path.join(root, path_orig)).convert('RGB')
        if self.transform is not None:
            image_orig = self.transform(image_orig)
        target = self.get_text_input(caption)
        image, img_pos = self.get_rcnn(os.path.join(feature_path, path))  # return img-feature 100 2048 & pos-feature

        return image, img_pos, target, index, img_id, image_orig

    def get_rcnn(self, img_path):
        if os.path.exists(img_path) and os.path.exists(img_path.replace('.npy', '_cls_prob.npy')):
            img = torch.from_numpy(np.load(img_path))
            img_id = img_path.split('/')[-1].split('.')[0]
            cls_label = torch.from_numpy(np.load(img_path.replace('.npy', '_cls_prob.npy')))
            with h5py.File(self.region_bbox_file, 'r') as region_bbox_f:
                vis_pe = torch.from_numpy(region_bbox_f[img_id][:])

            # lazy normalization of the coordinates...

            w_est = torch.max(vis_pe[:, [0, 2]]) * 1. + 1e-5
            h_est = torch.max(vis_pe[:, [1, 3]]) * 1. + 1e-5
            vis_pe[:, [0, 2]] /= w_est
            vis_pe[:, [1, 3]] /= h_est
            rel_area = (vis_pe[:, 3] - vis_pe[:, 1]) * (vis_pe[:, 2] - vis_pe[:, 0])
            rel_area.clamp_(0)

            vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), vis_pe[:, 5:]), -1)  # confident score
            normalized_coord = F.normalize(vis_pe.data[:, :5] - 0.5, dim=-1)
            vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), \
                                F.layer_norm(cls_label, [1601])), dim=-1)  # 1601 hard coded...
        else:
            img = torch.randn(100, 2048)
            vis_pe = torch.randn(100, 1601 + 6)
        return img, vis_pe

    def get_text_input(self, caption):
        caption_tokens = self.tokenizer.tokenize(caption)
        caption_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
        caption_ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)
        if len(caption_ids) >= self.max_seq_len:
            caption_ids = caption_ids[:self.max_seq_len]
        else:
            caption_ids = caption_ids + [0] * (self.max_seq_len - len(caption_ids))
        caption = torch.tensor(caption_ids)
        return caption

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    images, img_pos, captions, ids, img_ids, image_orig = zip(*data)
    images = torch.stack(images, 0)
    img_pos = torch.stack(img_pos, 0)
    captions = torch.stack(captions, 0)
    images_orig = torch.stack(image_orig, 0)
    return images, images_orig, img_pos, captions, ids


def get_tokenizer(bert_path):
    tokenizer = BertTokenizer(bert_path + 'vocab.txt')
    return tokenizer


def get_loader_single(data_name, split, root, json, transform,
                      batch_size=128, shuffle=True,
                      num_workers=10, ids=None, collate_fn=collate_fn,
                      feature_path=None,
                      region_bbox_file=None,
                      bert_path=None
                      ):
    if 'coco' in data_name:
        dataset = CocoDataset(root=root, json=json,
                              feature_path=feature_path,
                              region_bbox_file=region_bbox_file,
                              tokenizer=get_tokenizer(bert_path),
                              max_seq_len=32, transform=transform, ids=ids)
    elif 'f30k' in data_name:
        dataset = FlickrDataset(root=root, split=split, json=json,
                                feature_path=feature_path,
                                region_bbox_file=region_bbox_file,
                                tokenizer=get_tokenizer(bert_path),
                                max_seq_len=32, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
        # t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_loaders(data_name, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    roots, ids = get_paths(dpath, data_name)

    transform = get_transform(data_name, 'train', opt)
    train_loader = get_loader_single(opt.data_name, 'train',
                                     roots['train']['img'],
                                     roots['train']['cap'],
                                     transform, ids=ids['train'],
                                     batch_size=batch_size, shuffle=True,
                                     num_workers=workers,
                                     collate_fn=collate_fn,
                                     feature_path=opt.feature_path,
                                     region_bbox_file=opt.region_bbox_file,
                                     bert_path=opt.bert_path
                                     )

    transform = get_transform(data_name, 'val', opt)

    val_loader = get_loader_single(opt.data_name, 'val',
                                   roots['val']['img'],
                                   roots['val']['cap'],
                                   transform, ids=ids['val'],
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=workers,
                                   collate_fn=collate_fn,
                                   feature_path=opt.feature_path,
                                   region_bbox_file=opt.region_bbox_file,
                                   bert_path=opt.bert_path
                                   )

    return train_loader, val_loader


def get_test_loader(split_name, data_name, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)

    roots, ids = get_paths(dpath, data_name)

    transform = get_transform(data_name, split_name, opt)
    test_loader = get_loader_single(opt.data_name, split_name,
                                    roots[split_name]['img'],
                                    roots[split_name]['cap'],
                                    transform, ids=ids[split_name],
                                    batch_size=batch_size, shuffle=False,
                                    num_workers=workers,
                                    collate_fn=collate_fn,
                                    feature_path=opt.feature_path,
                                    region_bbox_file=opt.region_bbox_file,
                                    bert_path=opt.bert_path
                                    )

    return test_loader


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        if self.count == 0:
            return str(self.val)
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    def __init__(self):
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)
