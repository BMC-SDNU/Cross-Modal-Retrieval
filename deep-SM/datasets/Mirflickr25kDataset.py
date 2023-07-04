import numpy as np
import torch

from torch.utils.data import Dataset

from utils import Timer


class Mirflickr25kDataset(Dataset):
    def __init__(self, data_root='data/mirflickr25k', transform=None):
        with Timer('Loading npy into Mirflickr25kDataset'):
            self.texts = np.load(f'{data_root}/texts.npy')
            self.labels = np.load(f'{data_root}/labels.npy')
            self.images = np.load(f'{data_root}/images.npy', allow_pickle=True)

        self.transform = transform

        # print(self.texts.shape)
        # print(self.labels.shape)
        # print(self.images.shape)
        # print(self.images[0], self.images[0].shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = self.images[item]

        if self.transform:
            image = self.transform(image)

        return image, self.labels[item], self.texts[item]


def split_dataset(dataset, ratio=0.9, shuffle=True):
    torch.random.manual_seed(0)

    total_size = len(dataset)
    train_size = int(total_size * ratio)

    return torch.utils.data.random_split(dataset, (train_size, total_size - train_size))

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#
#     datasets = Mirflickr25kDataset()
#
#     # id2XXX
#     _, label_map = get_label_map()
#     _, tag_map = get_tag_map()
#
#     for image, label, text in datasets:
#         plt.imshow(image.astype('int'))
#         print(label_map[np.argmax(label)])
#         for index in np.where(text == 1)[0]:
#             print(tag_map[index], end=', ')
#         print()
#
#         plt.show()
#         input('test')
