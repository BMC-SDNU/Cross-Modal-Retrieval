"""
****************** COPYRIGHT AND CONFIDENTIALITY INFORMATION ******************
Copyright (c) 2018 [Thomson Licensing]
All Rights Reserved
This program contains proprietary information which is a trade secret/business \
secret of [Thomson Licensing] and is protected, even if unpublished, under \
applicable Copyright laws (including French droit d'auteur) and/or may be \
subject to one or more patent(s).
Recipient is to retain this program in confidence and is not permitted to use \
or make copies thereof other than as permitted in a written agreement with \
[Thomson Licensing] unless otherwise expressly allowed by applicable laws or \
by [Thomson Licensing] under express agreement.
Thomson Licensing is a company of the group TECHNICOLOR
*******************************************************************************
This scripts permits one to reproduce training and experiments of:
    Engilberge, M., Chevallier, L., Pérez, P., & Cord, M. (2018, April).
    Finding beans in burgers: Deep semantic-visual embedding with localization.
    In Proceedings of CVPR (pp. 3984-3993)

Author: Martin Engilberge
"""

import json
import os
import re

import numpy as np
import torch
import torch.utils.data as data

from misc.config import path
from misc.utils import encode_sentence, _load_dictionary
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from visual_genome import local as vg


class CocoCaptionsRV(data.Dataset):

    def __init__(self, root=path["COCO_ROOT"], coco_json_file_path=path["COCO_RESTVAL_SPLIT"], word_dict_path=path["WORD_DICT"], sset="train", transform=None):
        self.root = os.path.join(root, "images/")
        self.transform = transform

        # dataset.json come from Karpathy neural talk repository and contain the restval split of coco
        with open(coco_json_file_path, 'r') as f:
            datas = json.load(f)

        if sset == "train":
            self.content = [x for x in datas["images"] if x["split"] == "train"]
        elif sset == "trainrv":
            self.content = [x for x in datas["images"] if x["split"] == "train" or x["split"] == "restval"]
        elif sset == "val":
            self.content = [x for x in datas["images"] if x["split"] == "val"]
        else:
            self.content = [x for x in datas["images"] if x["split"] == "test"]

        self.content = [(os.path.join(y["filepath"], y["filename"]), [x["raw"] for x in y["sentences"]]) for y in self.content]

        path_params = os.path.join(word_dict_path, 'utable.npy')
        self.params = np.load(path_params, encoding='latin1')
        self.dico = _load_dictionary(word_dict_path)

    def __getitem__(self, index, raw=False):
        idx = index / 5

        idx_cap = index % 5

        path = self.content[int(idx)][0]
        target = self.content[int(idx)][1][idx_cap]

        if raw:
            return path, target

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        target = encode_sentence(target, self.params, self.dico)

        return img, target

    def __len__(self):
        return len(self.content) * 5


class VgCaptions(data.Dataset):

    def __init__(self, coco_root=path["COCO_ROOT"], vg_path_ann=path["VG_ANN"], path_vg_img=path["VG_IMAGE"], coco_json_file_path=path["COCO_RESTVAL_SPLIT"], word_dict_path=path["WORD_DICT"], image=True, transform=None):
        self.transform = transform
        self.image = image

        path_params = os.path.join(word_dict_path, 'utable.npy')
        self.params = np.load(path_params, encoding='latin1')
        self.dico = _load_dictionary(word_dict_path)

        self.path_vg_img = path_vg_img

        ids = vg.get_all_image_data(vg_path_ann)
        regions = vg.get_all_region_descriptions(vg_path_ann)

        annFile = os.path.join(coco_root, "annotations/captions_val2014.json")
        coco = COCO(annFile)
        ids_val_coco = list(coco.imgs.keys())

        # Uncomment following bloc to evaluate only on validation set from Rest/Val split
        # with open(coco_json_file_path, 'r') as f: # coco_json_file_path = "/home/wp01/users/engilbergem/dev/trunk/CPLApplications/deep/PytorchApplications/coco/dataset.json"
        #     datas = json.load(f)
        # ids_val_coco = [x['cocoid'] for x in datas["images"] if x["split"] == "val"]  # list(coco.imgs.keys())

        self.data = [x for x in zip(ids, regions) if x[0].coco_id in ids_val_coco]
        self.imgs_paths = [x[0].id for x in self.data]
        self.nb_regions = [len([x.phrase for x in y[1]])
                           for y in self.data]
        self.captions = [x.phrase for y in self.data for x in y[1]]

    def __getitem__(self, index, raw=False):

        if self.image:

            id_vg = self.data[index][0].id
            img = Image.open(os.path.join(self.path_vg_img,
                                          str(id_vg) + ".jpg")).convert('RGB')

            if raw:
                return img

            if self.transform is not None:
                img = self.transform(img)

            return img
        else:
            target = self.captions[index]

            #  If the caption is incomplete we set it to zero
            if len(target) < 3:
                target = torch.FloatTensor(1, 620)
            else:
                target = encode_sentence(target, self.params, self.dico)

            return target

    def __len__(self):
        if self.image:
            return len(self.data)
        else:
            return len(self.captions)


class CocoSemantic(data.Dataset):

    def __init__(self, coco_root=path["COCO_ROOT"], word_dict_path=path["WORD_DICT"], transform=None):
        self.coco_root = coco_root

        annFile = os.path.join(coco_root, "annotations/instances_val2014.json")
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

        path_params = os.path.join(word_dict_path, 'utable.npy')
        params = np.load(path_params, encoding='latin1')
        dico = _load_dictionary(word_dict_path)

        self.categories = self.coco.loadCats(self.coco.getCatIds())
        # repeats category with plural version
        categories_sent = [cat['name'] + " " + cat['name'] + "s" for cat in self.categories]
        self.categories_w2v = [encode_sentence(cat, params, dico, tokenize=True) for cat in categories_sent]

    def __getitem__(self, index, raw=False):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        target = dict()

        path = self.coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.coco_root, "images/val2014/", path)).convert('RGB')
        img_size = img.size

        for ann in anns:
            key = [cat['name'] for cat in self.categories if cat['id'] == ann["category_id"]][0]

            if key not in target:
                target[key] = list()

            if type(ann['segmentation']) != list:
                if type(ann['segmentation']['counts']) == list:
                    rle = maskUtils.frPyObjects(
                        [ann['segmentation']], img_size[0], img_size[1])
                else:
                    rle = [ann['segmentation']]

                target[key] += [("rle", rle)]
            else:
                target[key] += ann["segmentation"]

        if raw:
            return path, target

        if self.transform is not None:
            img = self.transform(img)

        return img, img_size, target

    def __len__(self):
        return len(self.ids)


class FileDataset(data.Dataset):

    def __init__(self, img_dir_paths, imgs=None, transform=None):
        self.transform = transform
        self.root = img_dir_paths
        self.imgs = imgs or [os.path.join(img_dir_paths, f) for f in os.listdir(img_dir_paths) if re.match(r'.*\.jpg', f)]

    def __getitem__(self, index):

        img = Image.open(self.imgs[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def get_image_list(self):
        return self.imgs

    def __len__(self):
        return len(self.imgs)


class TextDataset(data.Dataset):

    def __init__(self, text_path, word_dict_path=path["WORD_DICT"]):

        with open(text_path) as f:
            lines = f.readlines()

        self.sent_list = [line.rstrip('\n') for line in lines]

        path_params = os.path.join(word_dict_path, 'utable.npy')
        self.params = np.load(path_params, encoding='latin1')
        self.dico = _load_dictionary(word_dict_path)

    def __getitem__(self, index):

        caption = self.sent_list[index]

        caption = encode_sentence(caption, self.params, self.dico)

        return caption

    def __len__(self):
        return len(self.sent_list)
