from __future__ import print_function
import torch.utils.data as data
import os
import pickle
import numpy as np
import lmdb
import torch
import torchvision.transforms as transforms
from PIL import Image, ExifTags
from torch.utils.data.dataloader import default_collate
from utils import get_list_of_files


def default_loader(path):
    im = Image.open(path).convert('RGB')
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(im._getexif().items())

        if exif[orientation] == 3:
            im = im.rotate(180, expand=True)
        elif exif[orientation] == 6:
            im = im.rotate(270, expand=True)
        elif exif[orientation] == 8:
            im = im.rotate(90, expand=True)
    except:
        pass
    return im


def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)


def error_catching_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(im._getexif().items())

            if exif[orientation] == 3:
                im = im.rotate(180, expand=True)
            elif exif[orientation] == 6:
                im = im.rotate(270, expand=True)
            elif exif[orientation] == 8:
                im = im.rotate(90, expand=True)
        except:
            pass

        return im
    except:
        # print('bad image: '+path, end =" ")#print(file=sys.stderr)
        return Image.new('RGB', (224, 224), 'white')


class foodSpaceLoader(data.Dataset):
    def __init__(self, transform=None, loader=default_loader, partition=None, opts=None, aug='', language=0):
        
        self.env = lmdb.open(os.path.join(opts.data_path, partition + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with open(os.path.join(opts.data_path, partition + '_keys.pkl'), 'rb') as f:
            self.ids = pickle.load(f)

        # Dictionaries to map between index in vocab and word (word2vec table with four additional keywords at the begining)
        try:
            with open(os.path.join(opts.data_path, 'ingr_vocab.pkl'), 'rb') as f:
                self.ingr_vocab = pickle.load(f)
            with open(os.path.join(opts.data_path, 'vocab_ingr.pkl'), 'rb') as f:
                self.vocab_ingr = pickle.load(f)
        except:
            pass


        self.transform = transform
        self.loader = loader
        self.opts = opts
        self.aug = aug
        self.language = language
        self.partition = partition

    def __getitem__(self, index):
        # read lmdb
        with self.env.begin(write=False) as txn:
            serialized_sample = txn.get(self.ids[index].encode())
        sample = pickle.loads(serialized_sample, encoding='latin1')

        # image
        img_path = self.opts.img_path

        imgs = sample['imgs']
        if self.partition == 'train':
            imgIdx = np.random.choice(range(min(self.opts.maxImgs, len(imgs))))
        else:
            imgIdx = 0
        loader_path = [imgs[imgIdx]['id'][i] for i in range(4)]
        loader_path = os.path.join(*loader_path)
        path = os.path.join(img_path, self.partition, loader_path, imgs[imgIdx]['id'])
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)


        if len(self.aug)>0:
            aug = 0
            if 'english' in self.aug: # random choice between original text ("0"), en-de-en back translation ("1") or en-ru-en back translation ("2")
                tmp = np.random.choice([1,2])
                aug = np.random.choice([aug,tmp]) # 50% chance using original, 25% en-de-en and 25% en-ru-en
            lang = []
            if 'de' in self.aug: lang.append(3)
            if 'ru' in self.aug: lang.append(4)
            if 'fr' in self.aug: lang.append(5)
            if len(lang)>0:
                tmp = np.random.choice(lang)
                aug = np.random.choice([aug,tmp]) # 50% chance using english (orig or back translation) and 50% chance of using different laguage (ko, de, ru, fr)
            # ingrs = torch.tensor(sample['recipe'][aug])
            tmp = torch.tensor(sample['recipe'][aug])
            part_of_recipe = np.array(sample['part_of_recipe'])
            ingrs = torch.tensor([tmp[0]],dtype=int)
            if 'textinput'in self.aug:
                textinputs = np.random.choice(['title','ingr','inst','title,ingr','title,inst','ingr,inst','title,ingr,inst'])
            else:
                textinputs = self.opts.textinputs
            if 'title' in textinputs:
                ingrs = torch.cat((ingrs,tmp[part_of_recipe[aug]==1]))
            if 'ingr' in textinputs:
                ingrs = torch.cat((ingrs,tmp[part_of_recipe[aug]==2]))
            if 'inst' in textinputs:
                ingrs = torch.cat((ingrs,tmp[part_of_recipe[aug]==3]))
            tmp = torch.zeros_like(tmp)
            tmp[:len(ingrs)] = ingrs
            ingrs = tmp
            if 'mask' in self.aug:
                # select randomly up to 25% of words to mask ( [MASK] = 103 )
                inds = np.random.choice((ingrs>0).sum().item(), np.random.choice(int((ingrs>0).sum().item()*0.25)))
                ingrs[inds] = 103
        else:
            tmp = torch.tensor(sample['recipe'][self.language])
            part_of_recipe = np.array(sample['part_of_recipe'])
            ingrs = torch.tensor([tmp[0]],dtype=int)
            if 'title' in self.opts.textinputs:
                ingrs = torch.cat((ingrs,tmp[part_of_recipe[int(self.language)]==1]))
            if 'ingr' in self.opts.textinputs:
                ingrs = torch.cat((ingrs,tmp[part_of_recipe[int(self.language)]==2]))
            if 'inst' in self.opts.textinputs:
                ingrs = torch.cat((ingrs,tmp[part_of_recipe[int(self.language)]==3]))
            if len(ingrs)==512:
                ingrs[-1] = 102
            else:
                ingrs = torch.cat((ingrs,torch.tensor([102])))
            tmp = torch.zeros_like(tmp)
            tmp[:len(ingrs)] = ingrs
            ingrs = tmp

        # recipe id
        rec_id = self.ids[index]

        return [img, ingrs], rec_id


    def __len__(self):
        return len(self.ids)


#--------------------------------------------------------#


def default_image_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        return Image.new('RGB', (224, 224), 'white')

def default_image_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        return Image.new('RGB', (224, 224), 'white')


class ImagerLoader(data.Dataset):
    def __init__(self, img_path, data_path=None, partition=None, text_embedding_path=None, image_loader=default_image_loader, image_transform=None):

        if data_path is None or img_path is None or text_embedding_path is None:
            raise Exception('No data path specified.')

        if partition is None:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition = partition
        # open recipe data to load image & class
        self.env = lmdb.open(os.path.join(data_path, partition + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        # load recipe IDs
        with open(os.path.join(data_path, partition + '_keys.pkl'), 'rb') as f:
            self.ids = pickle.load(f)
        # load recipe embeddings
        self.__load_text_embedding(text_embedding_path)

        self.imgPath = img_path
        self.image_transform = image_transform
        self.image_loader = image_loader


    def __load_text_embedding(self, text_embedding_path):
        emb = np.load(text_embedding_path)
        text_emb = emb['rec_embeds']
        text_id = emb['rec_ids']
        img_emb = emb['img_embeds']
        self.text_embs = {}
        self.img_embs = {}
        for i in range(len(text_id)):
            self.text_embs[text_id[i]] = np.array(text_emb[i,:])
            self.img_embs[text_id[i]] = np.array(img_emb[i,:])


    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            serialized_sample = txn.get(self.ids[index].encode())
        sample = pickle.loads(serialized_sample, encoding='latin1')
        imgs = sample['imgs']
        recipe_id = self.ids[index]

        # image
        if self.partition == 'train':
            imgIdx = np.random.choice(range(min(5, len(imgs))))
        else:
            imgIdx = 0
        loader_path = [imgs[imgIdx]['id'][i] for i in range(4)]
        loader_path = os.path.join(*loader_path)
        path = os.path.join(self.imgPath, self.partition, loader_path, imgs[imgIdx]['id'])

        img = self.image_loader(path)
        if self.image_transform is not None:
            img = self.image_transform(img)
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        re_img = transforms.Resize(128)(img)
        img = normalize(img)
        re_img = normalize(re_img)

        # recipe class label
        class_label = sample['classes'] - 1

        # text embedding
        recipe_emb = torch.FloatTensor(self.text_embs[recipe_id]) # 1x1024
        img_emb = torch.FloatTensor(self.img_embs[recipe_id]) # 1x1024

        return img, class_label, re_img, recipe_emb, img_emb, recipe_id


    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    img, class_label, re_img, recipe_emb, img_emb, food_id = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(img, 0)
    class_label = torch.LongTensor(list(class_label))
    re_imgs = torch.stack(re_img, 0)
    recipe_embs = torch.stack(recipe_emb, 0)
    img_embs = torch.stack(img_emb, 0)
    food_ids = list(food_id)

    return [images, class_label, re_imgs, recipe_embs, img_embs, food_ids]


# def get_loader(img_path, transform, vocab, data_path, partition, batch_size, shuffle, num_workers, pin_memory):
def get_loader(img_path, data_path, partition, text_embedding_path, transform, batch_size, shuffle, num_workers, pin_memory, drop_last=True):
    data_loader = torch.utils.data.DataLoader(ImagerLoader(img_path, data_path, partition, text_embedding_path, image_transform=transform),
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory,
                                              drop_last=drop_last,
                                              collate_fn=collate_fn)
    return data_loader


#--------------------------------------------------------------------------------------------
class ImagerLoaderFromDir(data.Dataset):
    def __init__(self, img_path, image_loader=default_image_loader, image_transform=None):

        self.imgPath = img_path
        self.image_transform = image_transform
        self.image_loader = image_loader
        self.__get_files_and_IDs()

    def __get_files_and_IDs(self):
        self.image_filenames = get_list_of_files(self.imgPath)
        self.IDs = [filename[:10] for filename in self.image_filenames]

    def __getitem__(self, index):
        recipe_id = self.IDs[index]
        image_filename = self.image_filenames[index]
        path = os.path.join(self.imgPath, image_filename)

        img = self.image_loader(path)
        img = self.image_transform(img)

        return img, recipe_id

    def __len__(self):
        return len(self.IDs)


def image_collate_fn(data):
    img, id = zip(*data)
    images = torch.stack(img, 0)
    id = list(id)
    return images, id


def get_image_loader_from_dir(img_path, image_transforms, batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False):
    data_loader = torch.utils.data.DataLoader(
        ImagerLoaderFromDir(img_path, image_transform=image_transforms),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=image_collate_fn)
    return data_loader


#---------------------------------------------------------
class EmbeddingDataset(data.Dataset):
    def __init__(self, rec_ids, rec_embs):
        super(EmbeddingDataset, self).__init__()
        self.rec_ids = rec_ids
        self.rec_embs = rec_embs

    def __len__(self):
        return len(self.rec_ids)

    def __getitem__(self, index):
        rec_id = self.rec_ids[index]
        emb = torch.FloatTensor(self.rec_embs[index])

        return emb, rec_id


def embed_collate_fn(data):
    emb, id = zip(*data)
    emb = torch.stack(emb, 0)
    id = list(id)
    return emb, id


def get_embedding_data_loader(rec_ids, rec_embs, batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False):
    data_loader = data.DataLoader(EmbeddingDataset(rec_ids, rec_embs),
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=drop_last,
                                  collate_fn=embed_collate_fn)
    return data_loader