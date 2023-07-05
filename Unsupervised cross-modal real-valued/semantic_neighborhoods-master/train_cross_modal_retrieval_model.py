from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, strip_non_alphanum
import nltk
import gensim
import numpy as np
import pickle
import re, math
import os
import sys
import json
import random
import glob
import gzip
import time
import traceback
import torch
import json
from torch.autograd import Variable
from torchvision import transforms, models
from enum import Enum
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sklearn.utils
from collections import Counter
import csv, itertools
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from multiprocessing import Pool

csv.field_size_limit(sys.maxsize)
d2v = Doc2Vec.load('./doc2vec_model.gensim')
word_to_vector = d2v.wv
word_to_index = {word : idx for idx, word in enumerate(word_to_vector.index2entity)}
del d2v
word_to_index['<!START!>'] = len(word_to_index)
word_to_index['<!END!>'] = len(word_to_index)

# Load KNN - THIS IS PUT INTO SHARED MEMORY FOR MULTIPROCESSING
neighbors = pickle.load(open('./document_feats_knn.pickle', 'rb'))
_neighbors_pths2idx = {v : k for k,v in enumerate(neighbors['paths'])} # SHARED!
_neighbors_idxs, _neighbors_dists = zip(*neighbors['neighbors'])
_neighbors_idxs = np.stack([np.pad(a.ravel(), (0, 200 - a.size), 'constant', constant_values=0) for a in _neighbors_idxs], axis=0) # SHARED
_neighbors_dists = np.stack([np.pad(a.ravel(), (0, 200 - a.size), 'constant', constant_values=0) for a in _neighbors_dists], axis=0)  # SHARED
_neighbors_pths = neighbors['paths'] # SHARED
del neighbors  # Free python object - use only Numpy variants

def get_db():
    db = pickle.load(open('complete_db.pickle', 'rb'))
    orig_paths_and_text = []
    for politics, issues in db.items():
        for issue, items in issues.items():
            for item in items:
                if len(item['content_text']) > 500:
                    orig_paths_and_text.append(
                        (item['local_path'], item['content_text']))
    random.shuffle(orig_paths_and_text)
    return orig_paths_and_text
def get_doc2vec_neighbors(pth, transform, orig_img):
    assert(pth in _neighbors_pths2idx)    
    root_idx = _neighbors_pths2idx[pth]
    root_neighbor_idxs = _neighbors_idxs[root_idx, :]
    root_neighbor_dists = _neighbors_dists[root_idx, :]
    non_dupe_idxs = np.nonzero(np.abs(root_neighbor_dists) > 0)[0].tolist()
    neighbor_imgs = []
    neighbor_pths = []
    non_dupe_idxs = [ndi for ndi in non_dupe_idxs if _neighbors_pths[root_neighbor_idxs[ndi]] in MyDataset.train_set and _neighbors_pths[root_neighbor_idxs[ndi]] in MyDataset.result_db]
    # Choose one randomly from first N (here we use N=10 most similar neighbors - worked slightly in later tests) but you can also adjust to all 200
    non_dupe_idxs = non_dupe_idxs[:10]
    random.shuffle(non_dupe_idxs)
    # IF YOU WANT - YOU CAN USE MORE THAN ONE NEIGHBOR
    for ndi in non_dupe_idxs:
        neighbor_idx = root_neighbor_idxs[ndi]
        neighbor_pth = _neighbors_pths[neighbor_idx]
        if neighbor_pth not in MyDataset.train_set or root_idx == neighbor_idx or neighbor_pth not in MyDataset.result_db:
            continue
        try:
            img = transform(Image.open(neighbor_pth).convert('RGB'))
        except:
            continue
        neighbor_imgs.append(img)
        neighbor_pths.append(neighbor_pth)
    # USING 1 NEIGHBOR
    return (neighbor_imgs[0], neighbor_pths[0])
def convert_text(tup, sentence_limit=2):
    def normalize_text(text):
        text = text.encode('ascii', errors='replace').decode('ascii')
        return text
    pth, text = tup
    text = normalize_text(text)
    # Sentence limit
    sentences = []
    for sentence in nltk.tokenize.sent_tokenize(text)[:sentence_limit]:
        sentences.append(sentence)
    text = ' '.join(sentences)
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short]
    tokenized_text = preprocess_string(text, CUSTOM_FILTERS)
    tokenized_text = list(filter(lambda x: x in word_to_index, tokenized_text))
    return (pth, tokenized_text)
class MyDataset(torch.utils.data.Dataset):
    result_db = {}
    def __init__(self, mode):
        super().__init__()
        if not MyDataset.result_db:
            train_test_dict = pickle.load(open('train_test_pths.pickle', 'rb'))
            train_set = train_test_dict['train']
            val_set = set(random.sample(train_set, len(train_test_dict['test'])))
            train_set = [pth for pth in train_set if pth not in val_set]
            MyDataset.train_set = list(train_set)
            MyDataset.test_set = list(val_set)
            complete_db = [(pth, txt[:10000]) for pth, txt in get_db() if pth in _neighbors_pths2idx and (pth in MyDataset.train_set or pth in MyDataset.test_set)]
            pool = Pool(processes=48)
            documents_to_words = dict(tqdm(pool.imap_unordered(convert_text, complete_db, chunksize=1), total=len(complete_db), leave=False, desc='Convert to Fixed Dict'))
            pool.close()
            # Convert words to GT word vector
            for pth, sentence in tqdm(documents_to_words.items(), leave=False, desc='Words to Vals'):
                if not sentence:
                    continue
                sentence_to_idx = np.asarray([word_to_index['<!START!>']]+[word_to_index[word] for word in sentence]+[word_to_index['<!END!>']], dtype=np.int32)
                MyDataset.result_db[pth] = sentence_to_idx
            ##########################################################
        if mode == 'train':
            self.dataset = [img for img in MyDataset.train_set if img in MyDataset.result_db]
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        elif mode == 'val':
            self.dataset = [img for img in MyDataset.test_set if img in MyDataset.result_db]
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        while True:
            try:
                img = Image.open(self.dataset[index]).convert('RGB')
                neighbor_data = get_doc2vec_neighbors(self.dataset[index], self.transform, img)
                break
            except:
                index = random.randrange(len(self.dataset))
                continue        
        # img, path, neighbor_img, neighbor_pth        
        return self.transform(img), self.dataset[index], neighbor_data[0], neighbor_data[1]
def weight_init(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.1)
    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        m.weight.data.uniform_()
        m.bias.data.zero_()
class ImageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
        # Projection from Image Features into join space
        self.projector = torch.nn.Linear(2048, 256, bias=True)
        self.projector.apply(weight_init)
    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.shape[0], -1) # flatten, preserving batch dim
        x = self.projector(x)
        # critical to normalize projections
        x = F.normalize(x, dim=1)
        return x        
class RecurrentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=len(word_to_index), embedding_dim=200)
        # INITIALIZE THE EMBEDDING MODEL
        self.embedding.weight.data[:-2,:].copy_(torch.from_numpy(word_to_vector.vectors))
        self.GRU = torch.nn.GRU(input_size=200, hidden_size=512, batch_first=True)
        # Projection 
        self.projector = torch.nn.Linear(512, 256, bias=True)
        self.projector.apply(weight_init)
    def forward(self, input, lengths):        
        embedding = self.embedding(input) # embed the padded sequence
        # Use pack_padded_sequence to make sure the LSTM won’t see the padded items
        embedding = torch.nn.utils.rnn.pack_padded_sequence(input=embedding, lengths=lengths, batch_first=True, enforce_sorted=False)
        # run through recurrent model
        _, h_n = self.GRU(embedding)
        # flatten for linear
        h_n = h_n.contiguous()[0]
        projection = self.projector(h_n)
        # critical to normalize projections
        projection = F.normalize(projection, dim=1)
        return projection
def angular_loss(anchors, positives, negatives, angle_bound=1.):
    """
    Calculates angular loss
    :param anchors: A torch.Tensor, (n, embedding_size)
    :param positives: A torch.Tensor, (n, embedding_size)
    :param negatives: A torch.Tensor, (n, embedding_size)
    :param angle_bound: tan^2 angle
    :return: A scalar
    """
    anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
    positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)
    negatives = torch.unsqueeze(negatives, dim=0).expand(len(positives), -1, -1)

    x = 4. * angle_bound * torch.matmul((anchors + positives), negatives.transpose(1, 2)) \
        - 2. * (1. + angle_bound) * torch.matmul(anchors, positives.transpose(1, 2))  # (n, 1, n-1)

    # Preventing overflow
    with torch.no_grad():
        t = torch.max(x, dim=2)[0]

    x = torch.exp(x - t.unsqueeze(dim=1))
    x = torch.log(torch.exp(-t) + torch.sum(x, 2))
    loss = torch.mean(t + x)

    return loss
def main():
    train_dataloader = torch.utils.data.DataLoader(dataset=MyDataset('train'), batch_size=32, shuffle=True, num_workers=40)
    test_dataloader = torch.utils.data.DataLoader(dataset=MyDataset('val'), batch_size=32, shuffle=False, num_workers=40)
    writer = SummaryWriter(f'models/l_i2t_l_sym_{sys.argv[1].replace(".","_")}_l_img_{sys.argv[2].replace(".","_")}_l_text_{sys.argv[3].replace(".","_")}/')
    img_model = torch.nn.DataParallel(ImageModel()).cuda()
    rnn_model = torch.nn.DataParallel(RecurrentModel()).cuda()
    optimizer = torch.optim.Adam(params=itertools.chain(img_model.parameters(), rnn_model.parameters()), lr=0.0001, weight_decay=1e-5)
    ### LR SCHEDULER
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, patience=5)
    ###
    itr = 0    
    best_loss = sys.maxsize
    for e in tqdm(range(1, 1000), ascii=True, desc='Epoch'):
        img_model.train()
        rnn_model.train()
        random.seed()
        with tqdm(total=len(train_dataloader), ascii=True, leave=False, desc='iter') as pbar:
            for i, (images, paths, neighbor_imgs, neighbor_pths) in enumerate(train_dataloader):
                itr += 1
                optimizer.zero_grad()
                images = images.float().cuda()
                image_projections = img_model(images) # Batch size x 256
                neighbor_imgs = neighbor_imgs.float().cuda()
                neighbor_imgs_projections = img_model(neighbor_imgs)

                neighbor_sentences = [torch.from_numpy(MyDataset.result_db[path]).long() for path in neighbor_pths]
                neighbor_lengths = torch.LongTensor([torch.numel(item) for item in neighbor_sentences])
                neighbor_sentences = torch.nn.utils.rnn.pad_sequence(sequences=neighbor_sentences, batch_first=True, padding_value=word_to_index['<!END!>']).cuda()

                positive_sentences = [torch.from_numpy(MyDataset.result_db[path]).long() for path in paths]
                pos_lengths = torch.LongTensor([torch.numel(item) for item in positive_sentences])
                positive_sentences = torch.nn.utils.rnn.pad_sequence(sequences=positive_sentences, batch_first=True, padding_value=word_to_index['<!END!>']).cuda()
                
                negative_sentences = [torch.from_numpy(MyDataset.result_db[path]).long() for path in random.sample(train_dataloader.dataset.dataset, len(positive_sentences))]
                neg_lengths = torch.LongTensor([torch.numel(item) for item in negative_sentences])
                negative_sentences = torch.nn.utils.rnn.pad_sequence(sequences=negative_sentences, batch_first=True, padding_value=word_to_index['<!END!>']).cuda()

                neighbor_projections = rnn_model(neighbor_sentences, neighbor_lengths)
                positive_projections = rnn_model(positive_sentences, pos_lengths)
                negative_projections = rnn_model(negative_sentences, neg_lengths)

                # Baseline loss
                # Compute n-pairs angular loss
                image_projections_np = torch.repeat_interleave(image_projections, len(negative_projections), dim=0)
                positive_projections_np = torch.repeat_interleave(positive_projections, len(negative_projections), dim=0)
                negative_projections_np = negative_projections.repeat(len(negative_projections), 1)
                l_i2t = angular_loss(anchors=image_projections_np, positives=positive_projections_np, negatives=negative_projections_np)

                # L_img - Image anchor, Neighbor img anchor, negative image neighbors. Angular Npairs
                image_projections_np = torch.repeat_interleave(image_projections, len(negative_projections), dim=0)
                neighbor_imgs_projections_np = torch.repeat_interleave(neighbor_imgs_projections, len(negative_projections), dim=0)
                permute_idxs = torch.from_numpy(np.asarray([j if i != j else (j+1) % len(image_projections) for i in range(len(image_projections)) for j in range(len(image_projections))]))
                image_projections_np2 = image_projections[permute_idxs,...]
                l_img = angular_loss(anchors=image_projections_np, positives=neighbor_imgs_projections_np, negatives=image_projections_np2)

                # L_text - Angular npairs
                positive_projections_np = torch.repeat_interleave(positive_projections, len(negative_projections), dim=0)
                neighbor_projections_np = torch.repeat_interleave(neighbor_projections, len(negative_projections), dim=0)
                negative_projections_np = negative_projections.repeat(len(negative_projections), 1)
                l_text = angular_loss(anchors=positive_projections_np, positives=neighbor_projections_np, negatives=negative_projections_np)

                # Symmetric angular loss npairs (text to image)
                positive_projections_np = torch.repeat_interleave(positive_projections, len(image_projections), dim=0)
                image_projections_np = torch.repeat_interleave(image_projections, len(image_projections), dim=0)                
                permute_idxs = torch.from_numpy(np.asarray([j if i != j else (j+1) % len(image_projections) for i in range(len(image_projections)) for j in range(len(image_projections))]))
                image_projections_np2 = image_projections[permute_idxs,...]
                l_sym = angular_loss(anchors=positive_projections_np, positives=image_projections_np, negatives=image_projections_np2)

                loss = l_i2t + float(sys.argv[1])*l_sym + float(sys.argv[2])*l_img + float(sys.argv[3])*l_text

                loss.backward()
                optimizer.step()
                writer.add_scalar('data/train_loss', loss.item(), itr)
                writer.add_scalar('data/l_i2t', l_i2t.item(), itr)
                writer.add_scalar('data/l_sym', l_sym.item(), itr)
                writer.add_scalar('data/l_img', l_img.item(), itr)
                writer.add_scalar('data/l_text', l_text.item(), itr)

                pbar.update()
        img_model.eval()
        rnn_model.eval()
        losses = []
        random.seed(9485629)
        with tqdm(total=len(test_dataloader), ascii=True, leave=False, desc='eval') as pbar, torch.no_grad():
            for i, (images, paths, _, _) in enumerate(test_dataloader):
                optimizer.zero_grad()
                
                images = images.float().cuda()
                image_projections = img_model(images) # Batch size x 256

                positive_sentences = [torch.from_numpy(MyDataset.result_db[path]).long() for path in paths]
                pos_lengths = torch.LongTensor([torch.numel(item) for item in positive_sentences])
                positive_sentences = torch.nn.utils.rnn.pad_sequence(sequences=positive_sentences, batch_first=True, padding_value=word_to_index['<!END!>']).cuda()

                negative_sentences = [torch.from_numpy(MyDataset.result_db[path]).long() for path in random.sample(test_dataloader.dataset.dataset, len(positive_sentences))]
                neg_lengths = torch.LongTensor([torch.numel(item) for item in negative_sentences])
                negative_sentences = torch.nn.utils.rnn.pad_sequence(sequences=negative_sentences, batch_first=True, padding_value=word_to_index['<!END!>']).cuda()

                positive_projections = rnn_model(positive_sentences, pos_lengths)
                negative_projections = rnn_model(negative_sentences, neg_lengths)

                loss = angular_loss(anchors=image_projections, positives=positive_projections, negatives=negative_projections)
                
                losses.append(loss.item())

                pbar.update()
        curr_loss = np.mean(losses)
        writer.add_scalar('data/val_loss', curr_loss, e)
        scheduler.step(curr_loss)
        # save only the best model
        if curr_loss < best_loss:
            best_loss = curr_loss
            # delete prior models
            prior_models = glob.glob(f'models/l_i2t_l_sym_{sys.argv[1].replace(".","_")}_l_img_{sys.argv[2].replace(".","_")}_l_text_{sys.argv[3].replace(".","_")}/*.pth')
            for pm in prior_models:
                os.remove(pm)
            try:
                torch.save(rnn_model.state_dict(), f'models/l_i2t_l_sym_{sys.argv[1].replace(".","_")}_l_img_{sys.argv[2].replace(".","_")}_l_text_{sys.argv[3].replace(".","_")}/rnn_model_{e}.pth')
                torch.save(img_model.state_dict(), f'models/l_i2t_l_sym_{sys.argv[1].replace(".","_")}_l_img_{sys.argv[2].replace(".","_")}_l_text_{sys.argv[3].replace(".","_")}/img_model_{e}.pth')
            except:
                print('Failed saving')
                continue
if __name__ == '__main__':
    main()