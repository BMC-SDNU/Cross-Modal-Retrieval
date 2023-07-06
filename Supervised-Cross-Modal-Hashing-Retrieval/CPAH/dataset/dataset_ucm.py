import itertools
import torch
import numpy as np

from utils import select_idxs


class AbstractDataset(torch.utils.data.Dataset):

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        self.seed = seed
        self.image_replication_factor = 1  # default value, how many times we need to replicate image

        self.images = images
        self.captions = captions
        self.labels = labels

        self.captions_aug = captions_aug
        self.images_aug = images_aug

        self.idxs = np.array(idxs[0])
        self.idxs_cap = np.array(idxs[1])

    def __getitem__(self, index):
        return

    def __len__(self):
        return


class DatasetDuplet5(AbstractDataset):
    """
    Class for dataset representation.

    Each image has 5 corresponding captions

    Duplet dataset sample - img-txt (image and corresponding caption)
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization.

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        self.image_replication_factor = 5  # how many times we need to replicate image

    def __getitem__(self, index):
        """
        Returns a tuple (img, txt, label) - image and corresponding caption

        :param index: index of sample
        :return: tuple (img, txt, label)
        """
        img_idx, txt_idx = self.get_idx_combination_duplet(index)
        return (
            index,
            (self.idxs[img_idx], self.idxs_cap[txt_idx]),
            torch.from_numpy(self.images[img_idx].astype('float32')),
            torch.from_numpy(self.captions[txt_idx].astype('float32')),
            self.labels[img_idx]
        )

    def __len__(self):
        return len(self.captions)  # len(self.images) * self.image_replication

    def get_idx_combination_duplet(self, index):
        """
        Returns combination of indexes for each item of dataset.

        Each image has 5 corresponding captions.

        Thus, dataset ((img, txt) tuples) is 5 times larger than number of unique images.

        :param index:
        :return:
        """
        return index // self.image_replication_factor, index


class DatasetTriplet5(AbstractDataset):
    """
    Class for dataset representation.

    Each image has 5 corresponding captions

    Triplet dataset sample - img-txt-txt (image and 2 corresponding captions)
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        self.image_replication_factor = 10  # how many times we need to replicate image

    def __getitem__(self, index):
        """
        Returns a tuple (img, txt1, txt2, label) - image and 2 corresponding captions

        :param index: index of sample
        :return: tuple (img, txt1, txt2, label)
        """
        img_idx, txt1_idx, txt2_idx = self.get_idx_combination_tripet(index)
        return (
            index,
            (self.idxs[img_idx], self.idxs_cap[txt1_idx], self.idxs_cap[txt2_idx]),
            torch.from_numpy(self.images[img_idx].astype('float32')),
            torch.from_numpy(self.captions[txt1_idx].astype('float32')),
            torch.from_numpy(self.captions[txt2_idx].astype('float32')),
            self.labels[img_idx]
        )

    def __len__(self):
        return len(self.images) * self.image_replication_factor

    def get_idx_combination_tripet(self, index):
        """
        Returns combination of indexes for each item of dataset.

        Each image has 5 corresponding captions. We need to select 2 captions per image.
        For each unique image we have C(5, 2) = 10 combinations of captions.

        Thus, dataset ((img, txt, txt) tuples) is 10 times larger than number of unique images.

        :param index:
        :return:
        """
        # {0: (0, 1), 1: (0, 2), 2: (0, 3), 3: (0, 4), 4: (1, 2), 5: (1, 3), 6: (1, 4), 7: (2, 3), 8: (2, 4), 9: (3, 4)}
        combinations = {k: v for k, v in enumerate(itertools.combinations(range(5), 2))}  # dict with C(5, 2)

        img_idx = index // self.image_replication_factor
        caption_base_idx = img_idx * 5
        combination = combinations[index % self.image_replication_factor]

        txt1_idx = caption_base_idx + combination[0]
        txt2_idx = caption_base_idx + combination[1]
        return img_idx, txt1_idx, txt2_idx


class DatasetQuadrupletAugmentedTxtImg(AbstractDataset):
    """
    Class for dataset representation.

    Quadruplet dataset sample - img-img-txt-txt
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        caption_idxs = select_idxs(len(self.captions), 1, 5, seed=self.seed)[0]
        self.captions = self.captions[caption_idxs]
        self.captions_aug = self.captions_aug[caption_idxs]
        self.idxs_cap = self.idxs_cap[caption_idxs]

    def __getitem__(self, index):
        """
        Returns a tuple (img1, img2, txt1, txt2, label)

        :param index: index of sample
        :return: tuple (img1, img2, txt1, txt2, label)
        """
        return (
            index,
            (self.idxs[index], self.idxs[index], self.idxs_cap[index], self.idxs_cap[index]),
            torch.from_numpy(self.images[index].astype('float32')),
            torch.from_numpy(self.images_aug[index].astype('float32')),
            torch.from_numpy(self.captions[index].astype('float32')),
            torch.from_numpy(self.captions_aug[index].astype('float32')),
            self.labels[index]
        )

    def __len__(self):
        return len(self.images)


class DatasetQuadrupletAugmentedTxtImgDouble(AbstractDataset):
    """
    Class for dataset representation.

    Quadruplet dataset sample - img-img-txt-txt
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        caption_idxs = select_idxs(len(self.captions), 1, 5, seed=self.seed)[0]
        self.captions = self.captions[caption_idxs]
        self.captions_aug = self.captions_aug[caption_idxs]
        self.idxs_cap = self.idxs_cap[caption_idxs]

        self.images = np.vstack((self.images, self.images_aug))
        self.labels = np.hstack((self.labels, self.labels))
        self.idxs = np.hstack((np.array(self.idxs), np.array(self.idxs)))
        self.captions = np.vstack((self.captions, self.captions_aug))
        self.idxs_cap = np.hstack((np.array(self.idxs_cap), np.array(self.idxs_cap)))
        print()

    def __getitem__(self, index):
        """
        Returns a tuple (img1, img2, txt1, txt2, label)

        :param index: index of sample
        :return: tuple (img1, img2, txt1, txt2, label)
        """
        return (
            index,
            (self.idxs[index], self.idxs[index], self.idxs_cap[index], self.idxs_cap[index]),
            torch.from_numpy(self.images[index].astype('float32')),
            0,
            torch.from_numpy(self.captions[index].astype('float32')),
            0,
            self.labels[index]
        )

    def __len__(self):
        return len(self.images)


class DatasetQuadrupletAugmentedImg(AbstractDataset):
    """
    Class for dataset representation.

    Quadruplet dataset sample - img-img-txt-txt
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        idxs1, idxs2 = select_idxs(len(self.captions), 2, 5, seed=self.seed)
        self.captions1 = self.captions[idxs1]
        self.captions2 = self.captions_aug[idxs2]
        self.idxs_cap1 = self.idxs_cap[idxs1]
        self.idxs_cap2 = self.idxs_cap[idxs2]

    def __getitem__(self, index):
        """
        Returns a tuple (img1, img2, txt1, txt2, label)

        :param index: index of sample
        :return: tuple (img, txt1, txt2, label)
        """
        return (
            index,
            (self.idxs[index], self.idxs[index], self.idxs_cap1[index], self.idxs_cap2[index]),
            torch.from_numpy(self.images[index].astype('float32')),
            torch.from_numpy(self.images_aug[index].astype('float32')),
            torch.from_numpy(self.captions1[index].astype('float32')),
            torch.from_numpy(self.captions2[index].astype('float32')),
            self.labels[index]
        )

    def __len__(self):
        return len(self.images)


class DatasetDuplet2(AbstractDataset):
    """
    Class for dataset representation.

    Each image has 5 corresponding captions, select only 2

    Duplet dataset sample - img-txt (image and corresponding caption)
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization.

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        self.image_replication_factor = 2  # how many times we need to replicate image

    def __getitem__(self, index):
        """
        Returns a tuple (img, txt, label) - image and corresponding caption

        :param index: index of sample
        :return: tuple (img, txt, label)
        """
        img_idx, txt_idx = self.get_idx_combination_duplet(index)
        return (
            index,
            (self.idxs[img_idx], self.idxs_cap[txt_idx]),
            torch.from_numpy(self.images[img_idx].astype('float32')),
            torch.from_numpy(self.captions[txt_idx].astype('float32')),
            self.labels[img_idx]
        )

    def __len__(self):
        return len(self.captions)  # len(self.images) * self.image_replication_factor

    def get_idx_combination_duplet(self, index):
        """
        Returns combination of indexes for each item of dataset.

        Each image has 5 corresponding captions, we select only 2.

        Thus, dataset ((img, txt) tuples) is 2 times larger than number of unique images.

        :param index:
        :return:
        """

        return index // self.image_replication_factor, index


class DatasetTriplet2(AbstractDataset):
    """
    Class for dataset representation.

    Each image has 5 corresponding captions, select only 2 of them for training

    Triplet dataset sample - img-txt-txt (image and 2 corresponding captions)
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        caption_idxs = select_idxs(len(self.captions), 2, 5, seed=self.seed)
        caption_idxs = sorted(caption_idxs[0].extend(caption_idxs[1]))
        self.captions = self.captions[caption_idxs]
        self.idxs_cap = self.idxs_cap[caption_idxs]

    def __getitem__(self, index):
        """
        Returns a tuple (img, txt1, txt2, label) - image and 2 corresponding captions

        :param index: index of sample
        :return: tuple (img, txt1, txt2, label)
        """
        img_idx, txt1_idx, txt2_idx = self.get_idx_combination_tripet(index)
        return (
            index,
            (self.idxs[img_idx], self.idxs_cap[txt1_idx], self.idxs_cap[txt2_idx]),
            torch.from_numpy(self.images[img_idx].astype('float32')),
            torch.from_numpy(self.captions[txt1_idx].astype('float32')),
            torch.from_numpy(self.captions[txt2_idx].astype('float32')),
            self.labels[img_idx]
        )

    def __len__(self):
        return len(self.images)

    @staticmethod
    def get_idx_combination_tripet(index):
        """
        Returns combination of indexes for each item of dataset.

        :param index:
        :return:
        """
        return index, 2 * index, 2 * index + 1


class DatasetTriplet2AugmentedTxt(AbstractDataset):
    """
    Class for dataset representation.

    Each image has 5 original and 5 augmented captions, select only 1 original and 1 augmented

    Triplet dataset sample - img-txt-txt (image, caption and augmented caption)
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        :param captions_aug: augmented caption embeddings
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        caption_idxs = select_idxs(len(self.captions), 1, 5, seed=self.seed)[0]
        self.captions = self.captions[caption_idxs]
        self.captions_aug = self.captions_aug[caption_idxs]
        self.idxs_cap = self.idxs_cap[caption_idxs]

    def __getitem__(self, index):
        """
        Returns a tuple (img, txt1, txt2, label) - image and 2 corresponding captions

        :param index: index of sample
        :return: tuple (img, txt1, txt2, label)
        """
        return (
            index,
            (self.idxs[index], self.idxs_cap[index], self.idxs_cap[index]),
            torch.from_numpy(self.images[index].astype('float32')),
            torch.from_numpy(self.captions[index].astype('float32')),
            torch.from_numpy(self.captions_aug[index].astype('float32')),
            self.labels[index]
        )

    def __len__(self):
        return len(self.images)


class DatasetDuplet1(AbstractDataset):
    """
    Class for dataset representation.

    Each image has 5 corresponding captions

    Duplet dataset sample - img-txt (image and corresponding caption)
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization.

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        caption_idxs = select_idxs(len(self.captions), 1, 5, seed=self.seed)[0]
        self.captions = self.captions[caption_idxs]

    def __getitem__(self, index):
        """
        Returns a tuple (img, txt, label) - image and corresponding caption

        :param index: index of sample
        :return: tuple (img, txt, label)
        """
        return (
            index,
            (self.idxs[index], self.idxs_cap[index]),
            torch.from_numpy(self.images[index].astype('float32')),
            torch.from_numpy(self.captions[index].astype('float32')),
            self.labels[index]
        )

    def __len__(self):
        return len(self.images)
