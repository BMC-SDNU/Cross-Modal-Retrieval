import json
import os
import numpy as np
import pickle
from PIL import Image
from transformers import BertTokenizer

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import mindspore.common.dtype as mstype


class FlickrDataset:

    def __init__(self, anno_file, image_path, split, seq_len, bert_pretrain, img_size, use_raw_img):
        super().__init__()
        self.image_path = image_path
        self.seq_len = seq_len
        self.img_size = img_size
        raw_dataset = json.load(open(anno_file, 'r'))['images']
        self.ids = []
        self.img_ids = []
        self.img_dataset = []
        self.txt_dataset = []
        self.labels = []
        cnt_train = 0
        for i, d in enumerate(raw_dataset):
            if split == 'train':
                if cnt_train < 10000 and d['split'] == 'train':
                    self.ids.append(i)
                    self.img_ids.append(d['imgid'])
                    if use_raw_img:
                        self.img_dataset.append(Image.open(os.path.join(self.image_path, d['filename'])).convert('RGB').resize((self.img_size, self.img_size)))
                    self.txt_dataset.append(d['sentences'])
                    self.labels.append(d['label'])
                    cnt_train += 1
            elif split == 'query':
                if d['split'] == 'test' or d['split'] == 'val':
                    self.ids.append(i)
                    self.img_ids.append(d['imgid'])
                    if use_raw_img:
                        self.img_dataset.append(Image.open(os.path.join(self.image_path, d['filename'])).convert('RGB').resize((self.img_size, self.img_size)))
                    self.txt_dataset.append(d['sentences'])
                    self.labels.append(d['label'])
            elif split == 'db':
                if d['split'] == 'train':
                    self.ids.append(i)
                    self.img_ids.append(d['imgid'])
                    if use_raw_img:
                        self.img_dataset.append(Image.open(os.path.join(self.image_path, d['filename'])).convert('RGB').resize((self.img_size, self.img_size)))
                    self.txt_dataset.append(d['sentences'])
                    self.labels.append(d['label'])
        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrain)
        merge_txt = [' '.join(c) for c in self.txt_dataset]
        self.txt_dataset = [(self.tokenizer.encode(c, max_length=self.seq_len, padding='max_length', truncation=True)) for c in merge_txt]
        self.txt_masks = [[1 if txt_id != 0 else 0 for txt_id in txt_ids] for txt_ids in self.txt_dataset]
        self.dataset_size = len(self.labels)

        size_file = os.path.join(image_path, 'sizes.pkl')
        if os.path.isfile(size_file):
            with open(size_file, 'rb') as f:
                self.sizes = pickle.load(f)
        else:
            sizes = []
            for im in raw_dataset:
                path = im['filename']
                image = Image.open(os.path.join(image_path, path))
                sizes.append(image.size)

            with open(size_file, 'wb') as f:
                pickle.dump(sizes, f)
            self.sizes = sizes

    def __getitem__(self, index):
        img = self.img_dataset[index]
        txt = self.txt_dataset[index]
        txt_mask = self.txt_masks[index]
        label = self.labels[index]
        
        return img, txt, txt_mask, label

    def get_data(self):
        return self.img_ids, self.txt_dataset, self.txt_masks, self.labels, self.sizes, self.ids

    def __len__(self):
        return self.dataset_size


class BottomUpFeaturesDataset:
    def __init__(self, underlying_dataset, buf_feat, buf_box):
        self.img_ids, self.txt_dataset, self.txt_masks, self.labels, self.sizes, self.ids = underlying_dataset.get_data()
        self.buf_feat = buf_feat
        self.buf_box = buf_box
        self.img_feat = []
        self.img_boxes = []
        for img_id, id in zip(self.img_ids, self.ids):
            img_feat_path = os.path.join(self.buf_feat, '{}.npz'.format(img_id))
            img_box_path = os.path.join(self.buf_box, '{}.npy'.format(img_id))

            self.img_feat.append(np.load(img_feat_path)['feat'])
            self.img_boxes.append(np.load(img_box_path) / np.tile(self.sizes[id], 2))
        
    def __getitem__(self, index):
        img_feat = self.img_feat[index]
        img_boxes = self.img_boxes[index]
        txt = self.txt_dataset[index]
        txt_mask = self.txt_masks[index]
        label = self.labels[index]

        return img_feat, img_boxes, txt, txt_mask, label

    def __len__(self):
        return len(self.img_ids)


def get_dataset(config, split, is_distributed=False, group_size=1, rank=0):
    transforms_list = transforms.Compose([vision.ToTensor(), vision.RandomHorizontalFlip(0.5)])
    
    dataset = FlickrDataset(config.anno_file, config.image_path, split, config.seq_len, config.bert_pretrain, config.img_size, config.use_raw_img)
    shuffle = split == 'train'
    if config.use_raw_img:
        if is_distributed:
            dataset = ds.GeneratorDataset(dataset, ['img', 'txt', 'txt_mask', 'label'], shuffle=shuffle,
                                        num_shards=group_size, shard_id=rank)
        else:
            dataset = ds.GeneratorDataset(dataset, ['img', 'txt', 'txt_mask', 'label'], shuffle=shuffle, num_parallel_workers=2)

        dataset = dataset.map(operations=transforms_list, input_columns="img")
    else:
        dataset = BottomUpFeaturesDataset(dataset, config.buf_feat, config.buf_box)
        if is_distributed:
            dataset = ds.GeneratorDataset(dataset, ['img_feat', 'img_box', 'txt', 'txt_mask', 'label'], shuffle=shuffle,
                                        num_shards=group_size, shard_id=rank)
        else:
            dataset = ds.GeneratorDataset(dataset, ['img_feat', 'img_box', 'txt', 'txt_mask', 'label'], shuffle=shuffle, num_parallel_workers=2)
    dataset = dataset.map(operations=transforms.TypeCast(mstype.float32), input_columns="label")

    return dataset.batch(config.bs, drop_remainder=True)