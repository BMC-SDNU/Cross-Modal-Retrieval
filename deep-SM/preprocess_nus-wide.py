import os
import shutil
import imageio

import numpy as np

from tqdm import tqdm
from pprint import pprint

nus_wide_data_dir = './data/NUS-WIDE'


def get_nus_tag_map():
    with open(nus_wide_data_dir + '/NUS_WID_Tags/Final_Tag_List.txt') as f:
        tag_list = f.readlines()[:1000]

    tag_list = list(map(lambda x: x.strip(), tag_list))

    tag2id = {tag: n for n, tag in enumerate(tag_list)}
    id2tag = {n: tag for n, tag in enumerate(tag_list)}

    return tag2id, id2tag


def get_nus_concept_map():
    with open(nus_wide_data_dir + '/Concepts81.txt') as f:
        concept_list = f.readlines()

    concept_list = list(map(lambda x: x.strip(), concept_list))

    concept2id = {tag: n for n, tag in enumerate(concept_list)}
    id2concept = {n: tag for n, tag in enumerate(concept_list)}

    return concept2id, id2concept


def process_tags():
    tags_dir = nus_wide_data_dir + '/NUS_WID_Tags'

    train_tags = np.loadtxt(tags_dir + '/Train_Tags1k.dat', dtype=int)
    test_tags = np.loadtxt(tags_dir + '/Test_Tags1k.dat', dtype=int)

    print(f'Saving tags with shape: train {train_tags.shape}, test {test_tags.shape}')
    np.save(nus_wide_data_dir + '/train_tags.npy', train_tags)
    np.save(nus_wide_data_dir + '/test_tags.npy', test_tags)


def process_concepts():
    tags_dir = nus_wide_data_dir + '/NUS_WID_Tags'

    train_concepts = np.loadtxt(tags_dir + '/Train_Tags81.txt', dtype=int)
    test_concepts = np.loadtxt(tags_dir + '/Test_Tags81.txt', dtype=int)

    print(f'Saving concepts(labels) with shape: train {train_concepts.shape}, test {test_concepts.shape}')
    np.save(nus_wide_data_dir + '/train_concepts.npy', train_concepts)
    np.save(nus_wide_data_dir + '/test_concepts.npy', test_concepts)


def process_images():
    images_dir = nus_wide_data_dir + '/Flickr'

    with open(nus_wide_data_dir + '/ImageList/TrainImagelist.txt') as f:
        train_list = f.readlines()
        train_list = list(map(lambda x: x.strip().replace('\\', '/'), train_list))
        train_list = list(filter(lambda x: x.endswith('.jpg'), train_list))

    with open(nus_wide_data_dir + '/ImageList/TestImagelist.txt') as f:
        test_list = f.readlines()
        test_list = list(map(lambda x: x.strip().replace('\\', '/'), test_list))
        test_list = list(filter(lambda x: x.endswith('.jpg'), test_list))

    train_images = []
    for filename in tqdm(train_list, desc='Train Image loading'):
        image = imageio.imread(os.path.join(images_dir, filename))
        train_images.append(np.asarray(image))

    test_images = []
    for filename in tqdm(test_list, desc='Test Image loading'):
        image = imageio.imread(os.path.join(images_dir, filename))
        test_images.append(np.asarray(image))

    train_images = np.array(train_images, dtype=object)
    test_images = np.array(train_images, dtype=object)

    print(f'Saving Train images with shape: {train_images.shape}')
    np.save(nus_wide_data_dir + '/train_images.npy', train_images, allow_pickle=True)

    print(f'Saving Test images with shape: {test_images.shape}')
    np.save(nus_wide_data_dir + '/test_images.npy', test_images, allow_pickle=True)


def main():
    # process_tags()
    # process_concepts()
    process_images()
    # clean()


if __name__ == '__main__':
    main()
