# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, opt, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False, pos_emb=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.pos_embed_len = 0
        assert pos_emb == False
        if pos_emb:
            self.pos_embed_len = 25
            print("using pos embedding with length: ", self.pos_embed_len)
            self.pos_embed = nn.Embedding(5, self.pos_embed_len)
        else:
            self.pos_embed = None

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim + self.pos_embed_len, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        if self.pos_embed is not None:
            self.pos_embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, x_pos, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)
        if self.pos_embed is not None:
            x_pos_emb = self.pos_embed(x_pos)
            x_emb = torch.cat([x_emb, x_pos_emb], 2)
        packed = pack_padded_sequence(x_emb, lengths.data.tolist(), batch_first=True)

        # Forward propagate RNN
        out, ht = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:,:,:int(cap_emb.size(2)/2)] + cap_emb[:,:,int(cap_emb.size(2)/2):])/2
            ht = (ht[0] + ht[1]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
            ht = l2norm(ht, dim=-1)

        # For multi-GPUs
        if cap_emb.size(1) < x_emb.size(1):
            pad_size = x_emb.size(1) - cap_emb.size(1)
            pad_emb = torch.Tensor(cap_emb.size(0), pad_size, cap_emb.size(2))
            if torch.cuda.is_available():
                pad_emb = pad_emb.cuda()
            cap_emb = torch.cat([cap_emb, pad_emb], 1)

        return ht, cap_emb, cap_len


def func_attention(query, context, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = F.softmax(attn, dim=1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = F.softmax(attn*smooth, dim=1)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):

        diagonal = scores.diag().view(-1, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

class SCAN(nn.Module):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        super(SCAN, self).__init__()
        # Build Models
        self.opt = opt
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt, opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers, 
                                   use_bi_gru=opt.bi_gru,  
                                   no_txtnorm=opt.no_txtnorm,
                                   pos_emb=opt.pos_emb)

    def forward_emb(self, images, captions, captions_pos, lengths, masks):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            #captions_pos = captions_pos.cuda()
            lengths = lengths.cuda()

        # Forward
        img_emb = self.img_enc(images)

        ht, cap_emb, cap_lens = self.txt_enc(captions, captions_pos, lengths)
        return img_emb, ht, cap_emb, lengths

    def forward_score(self, img_emb, ht, cap_emb, cap_len, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        # compute image-sentence score matrix
        if "vsesc" not in self.opt.model_mode:
            if self.opt.cross_attn == 't2i':
                scores = self.xattn_score_t2i(img_emb, ht, cap_emb, cap_len, self.opt)
            elif self.opt.cross_attn == 'i2t':
                scores = self.xattn_score_i2t(img_emb, ht, cap_emb, cap_len, self.opt)
            else:
                raise ValueError("unknown cross_attn", self.opt.cross_attn)
            return scores, None
        else: 
            scores_0 = self.xattn_score_t2i(img_emb, ht, cap_emb, cap_len, self.opt)
            scores_1 = self.xattn_score_i2t(img_emb, ht, cap_emb, cap_len, self.opt)
            return scores_0, scores_1

    def forward(self, images, captions, captions_pos, lengths, masks, ids=None, *args):
        """One training step given images and captions.
        """
        # compute the embeddings
        img_emb, ht, cap_emb, cap_lens = self.forward_emb(images, captions, captions_pos, lengths, masks)
        scores_0, scores_1 = self.forward_score(img_emb, ht, cap_emb, cap_lens)
        return scores_0, scores_1

    def xattn_score_t2i(self, images, caption_ht, captions_all, cap_lens, opt):
        """
        Images: (n_image, n_regions, d) matrix of images
        captions_all: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        similarities = []
        n_image = images.size(0)
        n_caption = captions_all.size(0)
        images = images.float()
        captions_all = captions_all.float()
        caption_ht = caption_ht.float()
        images_fc = images.mean(1, keepdim=True)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            """
                word(query): (n_image, n_word, d)
                image(context): (n_image, n_regions, d)
                weiContext: (n_image, n_word, d)
                attn: (n_image, n_region, n_word)
            """
            weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
            cap_i_expand = cap_i_expand.contiguous()
            weiContext = weiContext.contiguous()

            row_sim = cosine_similarity(cap_i_expand.double(), weiContext.double(), dim=2)

            images_fc_expand = images_fc.expand_as(cap_i_expand)
            if opt.agg_func == 'LogSumExp':
                row_sim.mul_(opt.lambda_lse).exp_()
                row_sim = row_sim.sum(dim=1, keepdim=True)
                row_sim = torch.log(row_sim)/opt.lambda_lse
            elif opt.agg_func == 'Max':
                row_sim = row_sim.max(dim=1, keepdim=True)[0]
            elif opt.agg_func == 'Sum':
                row_sim = row_sim.sum(dim=1, keepdim=True)
            elif opt.agg_func == 'Mean':
                row_sim = row_sim.mean(dim=1, keepdim=True)
            else:
                raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1).double()
        if self.training:
            similarities = similarities.transpose(0,1)
        
        return similarities

    def xattn_score_i2t(self, images, caption_ht, captions_all, cap_lens, opt):
        """
        Images: (batch_size, n_regions, d) matrix of images
        captions_all: (batch_size, max_n_words, d) matrix of captions
        CapLens: (batch_size) array of caption lengths
        """
        #assert 1 == 0
        similarities = []
        n_image = images.size(0)
        n_caption = captions_all.size(0)
        n_region = images.size(1)
        images = images.float()
        captions_all = captions_all.float()
        caption_ht = caption_ht.float()
        images_fc = images.mean(1, keepdim=True)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            cap_h_i = caption_ht[i].unsqueeze(0).unsqueeze(0).contiguous()
            cap_h_i_expand = cap_h_i.expand_as(images)
            """
                word(query): (n_image, n_word, d)
                image(context): (n_image, n_region, d)
                weiContext: (n_image, n_region, d)
                attn: (n_image, n_word, n_region)
            """
            weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
            # (n_image, n_region)
            row_sim = cosine_similarity(images, weiContext, dim=2)
            if opt.agg_func == 'LogSumExp':
                row_sim.mul_(opt.lambda_lse).exp_()
                row_sim = row_sim.sum(dim=1, keepdim=True)
                row_sim = torch.log(row_sim)/opt.lambda_lse
            elif opt.agg_func == 'Max':
                row_sim = row_sim.max(dim=1, keepdim=True)[0]
            elif opt.agg_func == 'Sum':
                row_sim = row_sim.sum(dim=1, keepdim=True)
            elif opt.agg_func == 'Mean':
                row_sim = row_sim.mean(dim=1, keepdim=True)
            else:
                raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1).double()
        if self.training:
            similarities = similarities.transpose(0,1)
        return similarities
