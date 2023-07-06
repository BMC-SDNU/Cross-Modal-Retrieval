import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import random

class CubDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform = None):
        super(CubDataset, self).__init__()
        self.input_transform = input_transform

        name_list = []
        label_list = []
        
        with open(list_path, 'r') as f:
            for line in f.readlines(): 
                imagename, class_label = line.split()
                name_list.append(imagename)
                label_list.append(int(class_label))

        self.image_filenames = [os.path.join(image_dir, x) for x in name_list]
        self.label_list = label_list

    def __getitem__(self, index):
        imagename = self.image_filenames[index]

        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input) 

        class_label = self.label_list[index]

        return input, class_label

    def __len__(self):
        return len(self.image_filenames)

class CubTextDataset(data.Dataset):
    def __init__(self, image_dir, list_path, split):
        super(CubTextDataset, self).__init__()
        self.split = split
        self.vocabulary = list("abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
        self.max_length = 448

        texts, labels = [], []
        with open(list_path, 'r') as f:
            for line in f.readlines():
                path = line.split()[0]#text path
                label = int(line.split()[-1])
                for line in open(os.path.join(image_dir, path)):
                    text = line.split("\n")[0]
                texts.append(text)
                labels.append(label)

        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = []
        if(self.split == 'train'):
            num = random.randrange(0, 10, 2)#0, 2, 4, 8
            data += [0]*num
            data += [self.vocabulary.index(i) + 1 for i in list(raw_text) if i in self.vocabulary]
        else:
            data = [self.vocabulary.index(i) + 1 for i in list(raw_text) if i in self.vocabulary]
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif len(data) < self.max_length:
            data += [0] * (self.max_length - len(data))
        input = np.array(data, dtype=np.int64)
        class_label = self.labels[index]

        return input, class_label

    def __len__(self):
       return len(self.labels)
