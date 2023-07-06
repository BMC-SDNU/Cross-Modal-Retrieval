import torch
from datasets.data_handler import data_enhance
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, images, tags, labels, test=None):
        self.test = test
        if test is None:
            train_images = images[opt.query_size: opt.query_size + opt.training_size]
            train_tags = tags[opt.query_size: opt.query_size + opt.training_size]
            train_labels = labels[opt.query_size: opt.query_size + opt.training_size]
            # train_labels = train_labels * 0.99 + (1 - train_labels) * 0.01
            if opt.data_enhance:
                self.images, self.tags, self.labels = data_enhance(train_images, train_tags, train_labels)
            else:
                self.images, self.tags, self.labels = train_images, train_tags, train_labels
        else:
            self.query_labels = labels[0: opt.query_size]
            self.db_labels = labels[opt.query_size: opt.query_size + opt.db_size]
            if test == 'image.query':
                self.images = images[0: opt.query_size]
            elif test == 'image.db':
                self.images = images[opt.query_size: opt.query_size + opt.db_size]
            elif test == 'text.query':
                self.tags = tags[0: opt.query_size]
            elif test == 'text.db':
                self.tags = tags[opt.query_size: opt.query_size + opt.db_size]
        # if hasattr(self, 'images'):
        #     self.images = preprocess(self.images, mean=(0.336, 0.324, 0.293), std=(0.182, 0.182, 0.190))

    def __getitem__(self, index):
        if self.test is None:
            return (
                index,
                torch.from_numpy(self.images[index].astype('float32')),
                torch.from_numpy(self.tags[index].astype('float32')),
                torch.from_numpy(self.labels[index].astype('float32'))
            )
        elif self.test.startswith('image'):
            return torch.from_numpy(self.images[index].astype('float32'))
        elif self.test.startswith('text'):
            return torch.from_numpy(self.tags[index].astype('float32'))

    def __len__(self):
        if self.test is None:
            return len(self.images)
        elif self.test.startswith('image'):
            return len(self.images)
        elif self.test.startswith('text'):
            return len(self.tags)

    def get_labels(self):
        if self.test is None:
            return torch.from_numpy(self.labels.astype('float32'))
        else:
            return (
                torch.from_numpy(self.query_labels.astype('float32')),
                torch.from_numpy(self.db_labels.astype('float32'))
            )

