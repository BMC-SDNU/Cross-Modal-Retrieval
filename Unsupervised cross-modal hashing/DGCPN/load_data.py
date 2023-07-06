from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn import preprocessing
import pdb
import torch

torch.multiprocessing.set_sharing_strategy('file_system')


def split_data(images, tags, labels, q_num, v_num, t_num, d_num):
    X = {}
    X['query'] = images[q_num]
    X['val'] = images[v_num]
    X['train'] = images[t_num]
    X['retrieval'] = images[d_num]

    Y = {}
    Y['query'] = tags[q_num] 
    Y['val'] = tags[v_num]  
    Y['train'] = tags[t_num] 
    Y['retrieval'] = tags[d_num]

    L = {}
    L['query'] = labels[q_num]
    L['val'] = labels[v_num]
    L['train'] = labels[t_num]
    L['retrieval'] = labels[d_num]
    return X, Y, L


class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        
        label = self.labels[index]
        return img, text, label, index

    def __len__(self):
        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count


def get_loader_wiki(p, batch_size):
    path = '/media/zhouhao/02A04EDFA04ED935/dataset/wiki/'

    train_x = np.load(path + 'train_feature_vgg19.npy', allow_pickle=True)
    train_y = np.load(path + 'train_txt.npy', allow_pickle=True)
    train_L = np.load(path + 'train_label.npy', allow_pickle=True)
    validation_L = np.load(path + 'test_label.npy', allow_pickle=True)
    validation_x = np.load(path + 'test_feature_vgg19.npy', allow_pickle=True)
    validation_y = np.load(path + 'test_txt.npy', allow_pickle=True)

    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    train_L = np.concatenate(train_L)
    validation_L = np.concatenate(validation_L)
    validation_x = np.concatenate(validation_x)
    validation_y = np.concatenate(validation_y)

    train_L = train_L.tolist()
    train_L_one = np.zeros([np.shape(train_L)[0], 10])
    for var in range(np.shape(train_L)[0]):
        train_L_one[var, train_L[var]] = 1.
    train_L = train_L_one
    validation_L = validation_L.tolist()

    validation_L_one = np.zeros([np.shape(validation_L)[0], 10])
    for var in range(np.shape(validation_L)[0]):
        validation_L_one[var, validation_L[var]] = 1.

    validation_L = validation_L_one

    query_L = validation_L[231:]
    query_x = validation_x[231:]
    query_y = validation_y[231:]
    validation_L = validation_L[:231]
    validation_x = validation_x[:231]
    validation_y = validation_y[:231]

    retrieval_L = retrieval_vL = train_L
    retrieval_x = retrieval_vx = train_x
    retrieval_y = retrieval_vy = train_y

    imgs = {'train': train_x, 'query': query_x, 'database': retrieval_x, 'databasev': retrieval_vx, 'validation': validation_x}
    texts = {'train': train_y, 'query': query_y, 'database': retrieval_y, 'databasev': retrieval_vy, 'validation': validation_y}
    labels = {'train': train_L, 'query': query_L, 'database': retrieval_L, 'databasev': retrieval_vL, 'validation': validation_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database', 'databasev', 'validation']}

    shuffle = {'query': False, 'train': True, 'database': False, 'validation': False, 'databasev': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=4) for x in ['query', 'train', 'database', 'databasev', 'validation']}

    return dataloader, (train_x, train_y, train_L)


def get_loader(data_name, batch_size):
    path = '/media/zhouhao/02A04EDFA04ED935/dataset/' + data_name + '/'

    images = np.load(path + data_name + '_vgg19.npy')
    tags = np.load(path + 'tags.npy')
    label = np.load(path + 'label.npy')

    q_num = np.load(path + 'test_num.npy')
    v_num = np.load(path + 'val_num.npy')
    t_num = np.load(path + 'train_num.npy')
    d_num = np.load(path + 'database_num.npy')

    images = images.astype(np.float32)
    tags = tags.astype(np.float32)
    label = label.astype(int)

    X, Y, L = split_data(images, tags, label, q_num, v_num, t_num, d_num)

    # x: images   y:tags   L:labels
    train_L = L['train']
    train_x = X['train']
    train_y = Y['train']

    validation_L = L['val']
    validation_x = X['val']
    validation_y = Y['val']

    query_L = L['query']
    query_x = X['query']
    query_y = Y['query']

    retrieval_L = L['retrieval']
    retrieval_x = X['retrieval']
    retrieval_y = Y['retrieval']
    # pdb.set_trace()
    if data_name == 'nus':
        retrieval_vL = L['retrieval'][:10000]
        retrieval_vx = X['retrieval'][:10000]
        retrieval_vy = Y['retrieval'][:10000]
                
    else:
        retrieval_vL = L['retrieval']
        retrieval_vx = X['retrieval']
        retrieval_vy = Y['retrieval']

    imgs = {'train': train_x, 'query': query_x, 'database': retrieval_x, 'databasev': retrieval_vx, 'validation': validation_x}
    texts = {'train': train_y, 'query': query_y, 'database': retrieval_y, 'databasev': retrieval_vy, 'validation': validation_y}
    labels = {'train': train_L, 'query': query_L, 'database': retrieval_L, 'databasev': retrieval_vL, 'validation': validation_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database', 'databasev', 'validation']}

    shuffle = {'query': False, 'train': True, 'database': False, 'validation': False, 'databasev': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=4) for x in ['query', 'train', 'database', 'databasev', 'validation']}
    # pdb.set_trace()
    return dataloader, (train_x, train_y, train_L)

