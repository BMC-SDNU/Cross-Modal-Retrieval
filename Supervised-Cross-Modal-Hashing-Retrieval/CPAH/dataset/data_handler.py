import random
from torch.utils.data import DataLoader
from config import opt as cfg
from utils import read_hdf5, read_json, get_labels
import numpy as np


class DataHandler:

    def __init__(self):
        super().__init__()

    def load_train_query_db_data(self):
        """
        Load and split (train, query, db)

        :return: tuples of (images, captions, labels), each element is array
        """
        random.seed(cfg.seed)
        images, captions, labels = load_dataset()

        train, query, db = self.split_data(images, captions, labels)

        return train, query, db

    @staticmethod
    def split_data(images, captions, labels):
        """
        Split dataset to get training, query and db subsets

        :param: images: image embeddings array
        :param: captions: caption embeddings array
        :param: labels: labels array

        :return: tuples of (images, captions, labels), each element is array
        """
        idx_tr, idx_q, idx_db = get_split_idxs(len(images))
        idx_tr_cap, idx_q_cap, idx_db_cap = get_caption_idxs(idx_tr, idx_q, idx_db)

        train = images[idx_tr], captions[idx_tr_cap], labels[idx_tr], (idx_tr, idx_tr_cap)
        query = images[idx_q], captions[idx_q_cap], labels[idx_q], (idx_q, idx_q_cap)
        db = images[idx_db], captions[idx_db_cap], labels[idx_db], idx_db, (idx_db, idx_db_cap)

        return train, query, db


class DataHandlerAugmentedTxt:

    def __init__(self):
        super().__init__()

    def load_train_query_db_data(self):
        """
        Load and split (train, query, db)

        :return: tuples of (images, captions, labels), each element is array
        """
        random.seed(cfg.seed)
        images, captions, labels, captions_aug = load_dataset(txt_aug=True)

        train, query, db = self.split_data(images, captions, labels, captions_aug)

        return train, query, db

    @staticmethod
    def split_data(images, captions, labels, captions_aug):
        """
        Split dataset to get training, query and db subsets

        :param: images: image embeddings array
        :param: captions: caption embeddings array
        :param: labels: labels array
        :param: captions_aug: augmented caption embeddings

        :return: tuples of (images, captions, labels), each element is array
        """
        idx_tr, idx_q, idx_db = get_split_idxs(len(images))
        idx_tr_cap, idx_q_cap, idx_db_cap = get_caption_idxs(idx_tr, idx_q, idx_db)

        train = images[idx_tr], captions[idx_tr_cap], labels[idx_tr], (idx_tr, idx_tr_cap), captions_aug[idx_tr_cap]
        query = images[idx_q], captions[idx_q_cap], labels[idx_q], (idx_q, idx_q_cap), captions_aug[idx_q_cap]
        db = images[idx_db], captions[idx_db_cap], labels[idx_db], (idx_db, idx_db_cap), captions_aug[idx_db_cap]

        return train, query, db


class DataHandlerAugmentedTxtImg:

    def __init__(self):
        super().__init__()

    def load_train_query_db_data(self):
        """
        Load and split (train, query, db)

        :return: tuples of (images, captions, labels), each element is array
        """
        random.seed(cfg.seed)
        images, captions, labels, captions_aug, images_aug = load_dataset(img_aug=True, txt_aug=True)

        train, query, db = self.split_data(images, captions, labels, captions_aug, images_aug)

        return train, query, db

    @staticmethod
    def split_data(images, captions, labels, captions_aug, images_aug):
        """
        Split dataset to get training, query and db subsets

        :param: images: image embeddings array
        :param: captions: caption embeddings array
        :param: labels: labels array
        :param: captions_aug: augmented caption embeddings
        :param: images_aug: augmented image embeddings

        :return: tuples of (images, captions, labels), each element is array
        """
        idx_tr, idx_q, idx_db = get_split_idxs(len(images))
        idx_tr_cap, idx_q_cap, idx_db_cap = get_caption_idxs(idx_tr, idx_q, idx_db)

        train = images[idx_tr], captions[idx_tr_cap], labels[idx_tr], (idx_tr, idx_tr_cap), captions_aug[idx_tr_cap], \
                images_aug[idx_tr]
        query = images[idx_q], captions[idx_q_cap], labels[idx_q], (idx_q, idx_q_cap), captions_aug[idx_q_cap], \
                images_aug[idx_q]
        db = images[idx_db], captions[idx_db_cap], labels[idx_db], (idx_db, idx_db_cap), captions_aug[idx_db_cap], \
             images_aug[idx_db]

        return train, query, db


def load_dataset(img_aug=False, txt_aug=False):
    """
    Load dataset

    :return: images and captions embeddings, labels
    """
    images = read_hdf5(cfg.image_emb_for_model, 'image_emb', normalize=True)
    captions = read_hdf5(cfg.caption_emb_for_model, 'caption_emb', normalize=True)
    labels = np.array(get_labels(read_json(cfg.dataset_json_for_model), suppress_console_info=True))

    if img_aug and txt_aug:
        captions_aug = read_hdf5(cfg.caption_emb_aug_for_model, 'caption_emb', normalize=True)
        images_aug = read_hdf5(cfg.image_emb_aug_for_model, 'image_emb', normalize=True)
        return images, captions, labels, captions_aug, images_aug
    elif img_aug:
        images_aug = read_hdf5(cfg.image_emb_aug_for_model, 'image_emb', normalize=True)
        return images, captions, labels, images_aug
    elif txt_aug:
        captions_aug = read_hdf5(cfg.caption_emb_aug_for_model, 'caption_emb', normalize=True)
        return images, captions, labels, captions_aug
    else:
        return images, captions, labels


def get_split_idxs(arr_len):
    """
    Get indexes for training, query and db subsets

    :param: arr_len: array length

    :return: indexes for training, query and db subsets
    """
    idx_all = list(range(arr_len))
    idx_train, idx_eval = split_indexes(idx_all, cfg.dataset_train_split)
    idx_query, idx_db = split_indexes(idx_eval, cfg.dataset_query_split)

    return idx_train, idx_query, idx_db


def split_indexes(idx_all, split):
    """
    Splits list in two parts.

    :param idx_all: array to split
    :param split: portion to split
    :return: splitted lists
    """
    idx_length = len(idx_all)
    selection_length = int(idx_length * split)

    idx_selection = sorted(random.sample(idx_all, selection_length))

    idx_rest = sorted(list(set(idx_all).difference(set(idx_selection))))

    return idx_selection, idx_rest


def get_caption_idxs(idx_train, idx_query, idx_db):
    """
    Get caption indexes.

    :param: idx_train: train image (and label) indexes
    :param: idx_query: query image (and label) indexes
    :param: idx_db: db image (and label) indexes

    :return: caption indexes for corresponding index sets
    """
    idx_train_cap = get_caption_idxs_from_img_idxs(idx_train)
    idx_query_cap = get_caption_idxs_from_img_idxs(idx_query)
    idx_db_cap = get_caption_idxs_from_img_idxs(idx_db)
    return idx_train_cap, idx_query_cap, idx_db_cap


def get_caption_idxs_from_img_idxs(img_idxs):
    """
    Get caption indexes. There are 5 captions for each image (and label).
    Say, img indexes - [0, 10, 100]
    Then, caption indexes - [0, 1, 2, 3, 4, 50, 51, 52, 53, 54, 100, 501, 502, 503, 504]

    :param: img_idxs: image (and label) indexes

    :return: caption indexes
    """
    caption_idxs = []
    for idx in img_idxs:
        for i in range(5):  # each image has 5 captions
            caption_idxs.append(idx * 5 + i)
    return caption_idxs


def get_dataloaders(data_handler, ds_train, ds_query, ds_db):
    """
    Initializes dataloaders

    :param data_handler: data handler instance
    :param ds_train: class of train dataset
    :param ds_query: class of query dataset
    :param ds_db: class of database dataset

    :return: dataloaders
    """
    data_handler = data_handler()

    # data tuples: (images, captions, labels, (idxs, idxs_cap))
    # or (images, captions, labels, (idxs, idxs_cap), augmented_captions)
    # or (images, captions, labels, (idxs, idxs_cap), augmented_captions, augmented_images)
    train_tuple, query_tuple, db_tuple = data_handler.load_train_query_db_data()

    # train dataloader
    dataset_triplets = ds_train(*train_tuple, seed=cfg.seed)
    dataloader_train = DataLoader(dataset_triplets, batch_size=cfg.batch_size, shuffle=True)

    # query dataloader
    dataset_q = ds_query(*query_tuple, seed=cfg.seed)
    dataloader_q = DataLoader(dataset_q, batch_size=cfg.batch_size)

    # database dataloader
    dataset_db = ds_db(*db_tuple, seed=cfg.seed)
    dataloader_db = DataLoader(dataset_db, batch_size=cfg.batch_size)

    return dataloader_train, dataloader_q, dataloader_db
