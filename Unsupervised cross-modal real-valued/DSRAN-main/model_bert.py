# -----------------------------------------------------------
# Dual Semantic Relations Attention Network (DSRAN) implementation based on
# "VSE++: Improving Visual-Semantic Embeddings with Hard Negatives"
# "Learning Dual Semantic Relations with Graph Attention for Image-Text Matching"
# Keyu Wen, Xiaodong Gu, and Qingrong Cheng
# IEEE Transactions on Circuits and Systems for Video Technology, 2020
# Writen by Keyu Wen & Linyang Li, 2020
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import copy
from resnet import resnet152
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
import time
from GAT import GATLayer


def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class RcnnEncoder(nn.Module):
    def __init__(self, opt):
        super(RcnnEncoder, self).__init__()
        self.embed_size = opt.embed_size
        self.fc_image = nn.Sequential(nn.Linear(opt.img_dim, opt.img_dim),
                                      nn.ReLU(),
                                      nn.Linear(opt.img_dim, self.embed_size),
                                      nn.ReLU(),
                                      nn.Dropout(0.1))
        self.fc_pos = nn.Sequential(nn.Linear(6 + 1601, self.embed_size),
                                    nn.ReLU(),
                                    nn.Dropout(0.1))
        self.fc = nn.Linear(self.embed_size * 2, self.embed_size)

    def forward(self, images, img_pos):  # (b, 100, 2048) (b,100,1601+6)
        img_f = self.fc_image(images)
        img_pe = self.fc_pos(img_pos)
        img_embs = img_f + img_pe
        return img_embs  # (b,100,768)


class ImageEncoder(nn.Module):

    def __init__(self, opt):
        super(ImageEncoder, self).__init__()
        self.embed_size = opt.embed_size
        self.cnn = resnet152(pretrained=True)
        self.fc = nn.Sequential(nn.Linear(opt.img_dim, opt.embed_size), nn.ReLU(), nn.Dropout(0.1))
        if not opt.ft_res:
            print('image-encoder-resnet no grad!')
            for param in self.cnn.parameters():
                param.requires_grad = False
        else:
            print('image-encoder-resnet fine-tuning !')

    # def load_state_dict(self, state_dict):
    #     if 'cnn.classifier.1.weight' in state_dict:
    #         state_dict['cnn.classifier.0.weight'] = state_dict[
    #             'cnn.classifier.1.weight']
    #         del state_dict['cnn.classifier.1.weight']
    #         state_dict['cnn.classifier.0.bias'] = state_dict[
    #             'cnn.classifier.1.bias']
    #         del state_dict['cnn.classifier.1.bias']
    #         state_dict['cnn.classifier.3.weight'] = state_dict[
    #             'cnn.classifier.4.weight']
    #         del state_dict['cnn.classifier.4.weight']
    #         state_dict['cnn.classifier.3.bias'] = state_dict[
    #             'cnn.classifier.4.bias']
    #         del state_dict['cnn.classifier.4.bias']

    #     super(ImageEncoder, self).load_state_dict(state_dict)

    def forward(self, images):
        features_orig = self.cnn(images)
        features_top = features_orig[-1]
        features = features_top.view(features_top.size(0), features_top.size(1), -1).transpose(2, 1) # b, 49, 2048
        features = self.fc(features)
        return features


class TextEncoder(nn.Module):
    def __init__(self, opt):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(opt.bert_path)
        if not opt.ft_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print('text-encoder-bert no grad')
        else:
            print('text-encoder-bert fine-tuning !')
        self.embed_size = opt.embed_size
        self.fc = nn.Sequential(nn.Linear(opt.bert_size, opt.embed_size), nn.ReLU(), nn.Dropout(0.1))

    def forward(self, captions):
        all_encoders, pooled = self.bert(captions)
        out = all_encoders[-1]
        out = self.fc(out)
        return out


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


def cosine_sim(im, s):
    return im.mm(s.t())


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

    def forward(self, im, s):
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        im_sn = scores - d1
        c_sn = scores - d2
        cost_s = (self.margin + scores - d1).clamp(min=0)

        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


def get_optimizer(params, opt, t_total=-1):
    bertadam = BertAdam(params, lr=opt.learning_rate, warmup=opt.warmup, t_total=t_total)
    return bertadam


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
        self.img_enc = ImageEncoder(opt)
        self.txt_enc = TextEncoder(opt)
        self.rcnn_enc = RcnnEncoder(opt)

        config_img = GATopt(opt.embed_size, 1)
        config_cap = GATopt(opt.embed_size, 1)
        config_rcnn = GATopt(opt.embed_size, 1)
        config_joint = GATopt(opt.embed_size, 1)

        self.K = opt.K
        # SSR
        self.gat_1 = GAT(config_img)
        self.gat_2 = GAT(config_rcnn)
        self.gat_cap = GAT(config_cap)
        # JSR
        self.gat_cat = GAT(config_joint)
        if self.K == 2:
            self.gat_cat_1 = GAT(config_joint)
            self.fusion = Fusion(opt)
        elif self.K == 4:
            self.gat_cat_1 = GAT(config_joint)
            self.gat_cat_2 = GAT(config_joint)
            self.gat_cat_3 = GAT(config_joint)
            
            self.fusion = Fusion(opt)
            self.fusion_1 = Fusion(opt)
            self.fusion_2 = Fusion(opt)
        
    def forward(self, images_orig, rcnn_fe, img_pos, captions):

        img_emb_orig = self.gat_1(self.img_enc(images_orig))
        rcnn_emb = self.rcnn_enc(rcnn_fe, img_pos)
        rcnn_emb = self.gat_2(rcnn_emb)
        img_cat = torch.cat((img_emb_orig, rcnn_emb), 1)
        img_cat_1 = self.gat_cat(img_cat)
        img_cat_1 = torch.mean(img_cat_1, dim=1)
        if self.K == 1:
            img_cat = img_cat_1
        elif self.K == 2:
            img_cat_2 = self.gat_cat_1(img_cat)
            img_cat_2 = torch.mean(img_cat_2, dim=1)
            img_cat = self.fusion(img_cat_1, img_cat_2)
        elif self.K == 4:
            img_cat_2 = self.gat_cat_1(img_cat)
            img_cat_2 = torch.mean(img_cat_2, dim=1)
            img_cat_3 = self.gat_cat_2(img_cat)
            img_cat_3 = torch.mean(img_cat_3, dim=1)
            img_cat_4 = self.gat_cat_3(img_cat)
            img_cat_4 = torch.mean(img_cat_4, dim=1)
            img_cat_1_1 = self.fusion_1(img_cat_1, img_cat_2)
            img_cat_1_2 = self.fusion_2(img_cat_3, img_cat_4)
            img_cat = self.fusion(img_cat_1_1, img_cat_1_2)
        img_emb = l2norm(img_cat)
        cap_emb = self.txt_enc(captions)
        cap_gat = self.gat_cap(cap_emb)
        cap_embs = l2norm(torch.mean(cap_gat, dim=1))

        return img_emb, cap_embs


class VSE(object):

    def __init__(self, opt):
        self.DSRAN = DSRAN(opt)
        self.DSRAN = nn.DataParallel(self.DSRAN)
        if torch.cuda.is_available():
            self.DSRAN.cuda()
            cudnn.benchmark = True
        self.criterion = ContrastiveLoss(margin=opt.margin)
        params = list(self.DSRAN.named_parameters())
        param_optimizer = params
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = opt.l_train * opt.num_epochs
        if opt.warmup == -1:
            t_total = -1
        self.optimizer = get_optimizer(params=optimizer_grouped_parameters, opt=opt, t_total=t_total)
        self.Eiters = 0

    def state_dict(self):
        state_dict = self.DSRAN.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.DSRAN.load_state_dict(state_dict)

    def train_start(self):
        self.DSRAN.train()

    def val_start(self):
        self.DSRAN.eval()

    def forward_emb(self, images_orig, rcnn_fe, img_pos, captions):
        if torch.cuda.is_available():
            images_orig = images_orig.cuda()
            rcnn_fe = rcnn_fe.cuda()
            img_pos = img_pos.cuda()
            captions = captions.cuda()

        img_emb, cap_emb = self.DSRAN(images_orig, rcnn_fe, img_pos, captions)

        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.data, img_emb.size(0))
        return loss

    def train_emb(self, images, images_orig, img_pos, captions, ids=None, *args):
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        img_emb, cap_emb = self.forward_emb(images_orig, images, img_pos, captions)

        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        loss.backward()
        self.optimizer.step()
