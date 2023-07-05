# -----------------------------------------------------------
# Dual Semantic Relations Attention Network (DSRAN) implementation based on
# "VSE++: Improving Visual-Semantic Embeddings with Hard Negatives"
# "Learning Dual Semantic Relations with Graph Attention for Image-Text Matching"
# Keyu Wen, Xiaodong Gu, and Qingrong Cheng
# IEEE Transactions on Circuits and Systems for Video Technology, 2020
# Writen by Keyu Wen, 2020
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
import time
from GAT import GATLayer
import copy
from resnet import resnet152
import torchtext
import pickle
import os


def l2norm(X, dim=-1, eps=1e-12):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class GATopt(object):
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = 8
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.2


class GAT(nn.Module):
    def __init__(self, config_gat):
        super(GAT, self).__init__()
        layer = GATLayer(config_gat)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config_gat.num_layers)])

    def forward(self, input_graph):
        hidden_states = input_graph
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states)
        return hidden_states  # B, seq_len, D


class RcnnEncoder(nn.Module):
    def __init__(self, opt):
        super(RcnnEncoder, self).__init__()
        self.embed_size = opt.embed_size
        self.fc_image = nn.Linear(opt.img_dim, self.embed_size)
        self.init_weights()
        
    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc_image.in_features +
                                  self.fc_image.out_features)
        self.fc_image.weight.data.uniform_(-r, r)
        self.fc_image.bias.data.fill_(0)

    def forward(self, images, img_pos):  # (b, 100, 2048) (b,100,1601+6)
        img_f = self.fc_image(images)
        return img_f  # (b,100,768)


# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):

    def __init__(self, opt):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = opt.embed_size

        self.cnn = resnet152(pretrained=True)
        # self.fc = nn.Sequential(nn.Linear(2048, self.embed_size), nn.ReLU(), nn.Dropout(0.1))
        self.fc = nn.Linear(opt.img_dim, self.embed_size)
        if not opt.finetune:
            print('image-encoder-resnet no grad!')
            for param in self.cnn.parameters():
                param.requires_grad = False
        else:
            print('image-encoder-resnet fine-tuning !')

        self.init_weights()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        features_orig = self.cnn(images)
        features_top = features_orig[-1]
        features = features_top.view(features_top.size(0), features_top.size(1), -1).transpose(2, 1) # b, 49, 2048
        features = self.fc(features)

        return features


# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.embed_size = opt.embed_size
        # word embedding
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        # caption embedding
        self.rnn = nn.GRU(opt.word_dim, opt.embed_size, opt.num_layers, batch_first=True)
        vocab = pickle.load(open('vocab/'+opt.data_name+'_vocab.pkl', 'rb'))
        word2idx = vocab.word2idx
        # self.init_weights()
        self.init_weights('glove', word2idx, opt.word_dim)
        self.dropout = nn.Dropout(0.1)

    def init_weights(self, wemb_type, word2idx, word_dim):
        if wemb_type.lower() == 'random_init':
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe()
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        # return out
        x = self.embed(x)
        x = self.dropout(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        cap_emb = l2norm(cap_emb, dim=-1)
        cap_emb_mean = torch.mean(cap_emb, 1)
        cap_emb_mean = l2norm(cap_emb_mean)

        return cap_emb, cap_emb_mean


class Fusion(nn.Module):
    def __init__(self, opt):
        super(Fusion, self).__init__()
        self.f_size = opt.embed_size
        self.gate0 = nn.Linear(self.f_size, self.f_size)
        self.gate1 = nn.Linear(self.f_size, self.f_size)

        self.fusion0 = nn.Linear(self.f_size, self.f_size)
        self.fusion1 = nn.Linear(self.f_size, self.f_size)

    def forward(self, vec1, vec2):
        features_1 = self.gate0(vec1)
        features_2 = self.gate1(vec2)
        t = torch.sigmoid(self.fusion0(features_1) + self.fusion1(features_2))
        f = t * features_1 + (1 - t) * features_2
        return f


class DSRAN(nn.Module):

    def __init__(self, opt):
        super(DSRAN, self).__init__()
        self.K = opt.K
        self.img_enc = EncoderImageFull(opt)
        self.rcnn_enc = RcnnEncoder(opt)
        self.txt_enc = EncoderText(opt)
        config_rcnn = GATopt(opt.embed_size, 1)
        config_img= GATopt(opt.embed_size, 1)
        config_cap= GATopt(opt.embed_size, 1)
        config_joint= GATopt(opt.embed_size, 1)
        # SSR
        self.gat_1 = GAT(config_rcnn)
        self.gat_2 = GAT(config_img)
        self.gat_cap = GAT(config_cap)
        # JSR
        self.gat_cat_1 = GAT(config_joint)
        if self.K == 2:
            self.gat_cat_2 = GAT(config_joint)
            self.fusion = Fusion(opt)
        elif self.K == 4:
            self.gat_cat_2 = GAT(config_joint)
            self.gat_cat_3 = GAT(config_joint)
            self.gat_cat_4 = GAT(config_joint)
            self.fusion = Fusion(opt)
            self.fusion2 = Fusion(opt)
            self.fusion3 = Fusion(opt)
        
    def forward(self, images, img_rcnn, img_pos, captions, lengths):
        img_emb_orig = self.gat_2(self.img_enc(images))
        rcnn_emb = self.rcnn_enc(img_rcnn, img_pos)
        rcnn_emb = self.gat_1(rcnn_emb)
        img_cat = torch.cat((img_emb_orig, rcnn_emb), 1)
        img_cat_1 = self.gat_cat_1(img_cat)
        img_cat_1 = torch.mean(img_cat_1, dim=1)
        if self.K == 1:
            img_cat = img_cat_1
        elif self.K == 2:
            img_cat_2 = self.gat_cat_2(img_cat)
            img_cat_2 = torch.mean(img_cat_2, dim=1)
            img_cat = self.fusion(img_cat_1, img_cat_2)
        elif self.K == 4:
            img_cat_2 = self.gat_cat_2(img_cat)
            img_cat_2 = torch.mean(img_cat_2, dim=1)
            img_cat_3 = self.gat_cat_3(img_cat)
            img_cat_3 = torch.mean(img_cat_3, dim=1)
            img_cat_4 = self.gat_cat_4(img_cat)
            img_cat_4 = torch.mean(img_cat_4, dim=1)
            img_cat_1_1 = self.fusion(img_cat_1, img_cat_2)
            img_cat_1_2 = self.fusion2(img_cat_3, img_cat_4)
            img_cat = self.fusion3(img_cat_1_1, img_cat_1_2)
        img_emb = l2norm(img_cat)
        cap_emb, cap_emb_mean = self.txt_enc(captions, lengths)
        cap_gat = self.gat_cap(cap_emb)
        cap_embs = l2norm(torch.mean(cap_gat, dim=1))

        return img_emb, cap_embs


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        im_sn = scores - d1
        c_sn = scores - d2
        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query

        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class VSE(object):
    """
    rkiros/uvs model
    """
    def __init__(self, opt):
        # tutorials/09 - Image Captioning
        # Build Models
        self.grad_clip = opt.grad_clip

        self.DSRAN = DSRAN(opt)
        if torch.cuda.is_available():
            self.DSRAN.cuda()
            cudnn.benchmark = True
        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin)
        params = list(self.DSRAN.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.DSRAN.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.DSRAN.load_state_dict(state_dict[0])

    def train_start(self):
        """switch to train mode
        """
        self.DSRAN.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.DSRAN.eval()

    def forward_emb(self, images, captions, img_rcnn, img_pos, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            img_rcnn = img_rcnn.cuda()
            img_pos = img_pos.cuda()

        img_emb, cap_emb = self.DSRAN(images, img_rcnn, img_pos, captions, lengths)
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.data, img_emb.size(0))
        return loss

    def train_emb(self, images, captions, img_rcnn, img_pos, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, img_rcnn, img_pos, lengths)
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

