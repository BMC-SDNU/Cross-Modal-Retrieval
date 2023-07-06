import torch
import numpy as np


def sample_ucm_query(select_samples_per_class=50, num_classes=21, elements_in_class=500, seed=42):
    """
    Selects 'samples_per_class' elements from each of 'num_classes' classes.
    
    :param select_samples_per_class: how many samples to select from class
    :param num_classes: number of classes
    :param elements_in_class: elements in each class (balanced case)
    :param seed: sampling seed
    :return: indices of selected samples
    """
    np.random.seed(seed)
    selected = []
    # 21 class, 500 samples each
    for i in range(num_classes):
        # select 50 random samples from each class
        c = i * elements_in_class + np.random.choice(elements_in_class, select_samples_per_class, replace=False)
        selected.append(c)
    return np.sort(np.array(selected).reshape(-1))  # reshape to 1d array


def sample_ucm_train(db_index, select_samples_per_class=250, num_classes=21, elements_in_class=450, seed=42):
    np.random.seed(seed)
    train_idx = []
    # 21 class, 500 samples each
    for i in range(num_classes):
        # select 50 random samples from each class
        c = i * elements_in_class + np.random.choice(elements_in_class, select_samples_per_class, replace=False)
        train_idx.append(c)
    train_idx = np.sort(np.array(train_idx).reshape(-1))
    return list(np.array(db_index)[train_idx])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, images, tags, labels, test=None):
        self.test = test
        all_index = np.arange(tags.shape[0])
        if opt.flag == 'ucm_':
            query_index = sample_ucm_query(seed=42)  # select 50 out of 500 elements for each of 21 classes
            db_index = list(set(range(len(images))) - set(query_index))
            training_index = sample_ucm_train(db_index, seed=42)
        elif opt.flag == 'ucm' or opt.flag == 'rsicd':
            query_index = all_index[opt.db_size:]
            training_index = all_index[:opt.training_size]
            db_index = all_index[:opt.db_size]
        elif opt.flag == 'mir':
            query_index = all_index[opt.db_size:]
            training_index = all_index[:opt.training_size]
            db_index = all_index[:opt.db_size]
        else:
            query_index = all_index[:opt.query_size]
            training_index = all_index[opt.query_size: opt.query_size + opt.training_size]
            db_index = all_index[opt.query_size:]

        if test is None:
            train_images = images[training_index]
            train_tags = tags[training_index]
            train_labels = labels[training_index]
            self.images, self.tags, self.labels = train_images, train_tags, train_labels
        else:
            self.query_labels = labels[query_index]
            self.db_labels = labels[db_index]
            if test == 'image.query':
                self.images = images[query_index]
            elif test == 'image.db':
                self.images = images[db_index]
            elif test == 'text.query':
                self.tags = tags[query_index]
            elif test == 'text.db':
                self.tags = tags[db_index]

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
