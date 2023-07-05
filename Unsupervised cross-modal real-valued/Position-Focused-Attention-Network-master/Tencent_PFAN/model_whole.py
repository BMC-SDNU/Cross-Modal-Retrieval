# -----------------------------------------------------------
# Position Focused Attention Network (PFAN) implementation based on 
# another network Stacked Cross Attention Network (https://arxiv.org/abs/1803.08024)
# the code of SCAN: https://github.com/kuanghuei/SCAN
# ---------------------------------------------------------------
"""PFAN model"""

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict


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


def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False, need_box_whole=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm, need_box_whole)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False, need_box_whole=False, split_size = 16, position_embed_size = 200):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.split_size = split_size
        self.need_box_whole = need_box_whole
        self.position_size = split_size * split_size
        self.position_embed_size = position_embed_size
        self.no_imgnorm = no_imgnorm
        ###################################
        self.fc = nn.Linear(img_dim + self.position_embed_size, embed_size) # add position_embedding into image feature
        self.fc_whole = nn.Linear(img_dim, embed_size)
        #self.fc = nn.Linear(img_dim, embed_size)
        #print("FC", self.fc)
        self.position_embedding = nn.Embedding(self.position_size + 1, self.position_embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

        r2 = np.sqrt(6.) / np.sqrt(self.fc_whole.in_features + self.fc_whole.out_features)
        self.fc_whole.weight.data.uniform_(-r2, r2)
        self.fc_whole.bias.data.fill_(0)

    def compute_area(self, box, i_x, i_y):
        #compute the area between box and [(i_x,i_y),(i_x+1,i_y+1)]
        p1_x, p1_y, p2_x, p2_y = box
        one_wh = 1.0/self.split_size
        p3_x, p3_y, p4_x, p4_y = i_x * one_wh, i_y * one_wh, (i_x+1) * one_wh, (i_y+1) * one_wh
        if p1_x > p4_x or p2_x < p3_x or p1_y > p4_y or p2_y < p3_y:
            return 0.0
        len = min(p2_x,p4_x) - max(p1_x,p3_x)
        wid = min(p2_y, p4_y) - max(p1_y, p3_y)
        if len < 0 or wid < 0:
            return 0.0
        return len * wid

    def extract_one_box(self, box):
        one_wh = 1.0/self.split_size
        x,y,x2,y2 = box
        x = x.cpu()
        #print("XY", x,y,x2,y2)
        res = torch.zeros(1,self.position_embed_size)
        area_wei = 0.0
        for i_x in range(int(x/one_wh),int(x2/one_wh) + 1):
            for i_y in range(int(y/one_wh), int(y2/one_wh) + 1):
                if i_x >= self.split_size or i_y >= self.split_size:
                    continue
                index = i_x * self.split_size + i_y
                area = self.compute_area(box, i_x, i_y)
                area_wei += area
                res += area * self.position_embedding(Variable(torch.LongTensor([index])))
        return res / area_wei

    def extract_box_feature(self, boxes):
        box_feas = torch.zeros(boxes.size(0), boxes.size(1), self.position_embed_size)
        for i in range(boxes.size(0)): # batch_size
            for j in range(boxes.size(1)): # 36 boxes
                box_feas[i][j] = self.extract_one_box(boxes[i][j])
        return box_feas

    def forward(self, images, whole_images, boxes):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        #print "Images shape in forward", images
        #box_features = self.extract_box_feature(boxes) # not use
        #print("Boxes shape", boxes)

        #print("Whole images", whole_images)

        new_boxes = boxes.view(boxes.size(0)*boxes.size(1),boxes.size(2))
        #print("New boxes shape", new_boxes)
        index_size = new_boxes.size(1)/2
        new_boxes_index = new_boxes[:,:index_size].type(torch.LongTensor)
        new_boxes_weight = new_boxes[:,index_size:2*index_size]
        if torch.cuda.is_available():
            new_boxes_index = new_boxes_index.cuda()
            new_boxes_weight = new_boxes_weight.cuda()
        # new_boxes_index shape: batch_size*36, 15
        # new_boxes_weight shape: batch_size*36, 15
        box_features = self.position_embedding(new_boxes_index)
        # box_features shape: batch_size*36, 15, 200
        # => batch_size*36, 200, 15
        box_features = torch.transpose(box_features, 1, 2)
        # => batch_size*36, 200, 1
        box_features = torch.bmm(box_features, new_boxes_weight.unsqueeze(2))
        # => batch_size, 36, 200
        box_features = box_features.view(boxes.size(0), boxes.size(1),-1)
        #print("Box feature shape", box_features)
        #features = self.fc(images)
        #print("Need box whole", self.need_box_whole)
        if self.need_box_whole:
            whole_position = torch.mean(self.position_embedding.weight, dim=0)
            whole_position = whole_position.repeat(box_features.size(0), 1).unsqueeze(1)
            if box_features.size(1) == images.size(1):
                box_features[:, box_features.size(1)-1, :] = whole_position
            else:
                box_features = torch.cat((box_features, whole_position), dim=1)
            #print("New box feature shape", box_features)
        image_position = torch.cat((images, box_features),dim=2)

        #print("Embedding size", self.position_embedding.weight)
        #print("Image postion shape", image_position.size())
        features = self.fc(image_position) # ##########
        whole_features = self.fc_whole(whole_images)
        #features = self.fc(images)
        #print("Final feature shape", features)
        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)
        if not self.no_imgnorm:
            whole_features = l2norm(whole_features, dim=-1)

        return features, whole_features

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

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        # Hidden_out shape, 2, batch_size, 1024
        out, hidden_out = self.rnn(packed)
        #print("Text Out", out)
        #print("Text Out (2)", hidden_out)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:,:,:cap_emb.size(2)/2] + cap_emb[:,:,cap_emb.size(2)/2:])/2
            hidden_out = (hidden_out[:hidden_out.size(0)/2,:,:] + hidden_out[hidden_out.size(0)/2:,:,:])/2
        hidden_out = hidden_out.squeeze()
        #print("Text cap_emb", cap_emb)
        #print("Hidden out", hidden_out)
        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
            hidden_out = l2norm(hidden_out, dim=-1)
        return cap_emb, cap_len, hidden_out


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
        attn = nn.Softmax()(attn)
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
    attn = nn.Softmax()(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    # attnT is the sim between boxes and words
    attnT = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, sourceL)
    max_attnT = torch.max(attnT,dim=1)[0]
    #print("max AttnT", max_attnT)
    #print("AttnT", attnT)
    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT, max_attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    try:
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
    except:
        print("x1", x1)
        print("x2", x2)
        print("dim", dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, whole_images, captions, cap_lens, final_captions, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Whole images: (n_image, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    Final captions: (n_caption, d) matrix of captions
    n_image == n_caption == batch_size, when training

    Note:
    if opt.need_box_whole is True, the images shape will be to (n_image, n_regions+1, d) matrix of images, the last box (shape: n_image, 1, d) is the 'whole image'
    """
    #print("Box whole", opt.need_box_whole)
    # box_whole_images shape: n_image, d
    #print "OOOOO", opt.need_box_whole
    #print "lambda_whole", opt.lambda_whole
    if opt.need_box_whole:
        box_whole_images = images[:, -1, :]
        images = images[:, :images.size(1)-1, :]
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        #print("CAP",captions[i])
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        final_cap_i = final_captions[i, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        final_cap_i_expand = final_cap_i.repeat(n_image, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
            max_attn: (n_image, n_word)
        """
        weiContext, attn, max_attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        """
            diff_attn: (n_image, n_region, n_word)
        """
        #max_attn2 = max_attn.unsqueeze(0)
        #max_attn2 = torch.transpose(max_attn2, 0, 1)
        #max_attn2 = max_attn2.expand(attn.size()[0],attn.size()[1],attn.size()[2])
        #diff_attn = (max_attn2 - attn) # * max_attn2
        """
            diff_score : (n_image, 1)
        """
        #diff_score = diff_attn.mean(dim=1,keepdim=True).mean(dim=2,keepdim=False)
        cap_i_expand = cap_i_expand.contiguous()
        final_cap_i_expand = final_cap_i_expand.contiguous()
        #print("final_cap_i_expand", final_cap_i_expand)
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        whole_sim = cosine_similarity(whole_images, final_cap_i_expand, dim=1).view(whole_images.size(0), 1)
        if opt.need_box_whole:
            box_whole_sim = cosine_similarity(box_whole_images, final_cap_i_expand, dim=1).view(box_whole_images.size(0), 1)
        #print("Row sim", row_sim)
        #print("Whole sim", whole_sim)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
            max_attn.mul_(opt.lambda_lse).exp_()
            max_attn = max_attn.sum(dim=1, keepdim=True)
            max_attn = torch.log(max_attn)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
            max_attn = max_attn.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
            max_attn = max_attn.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
            max_attn = max_attn.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        #print("Row sim(2)", row_sim+0.5*max_attn+0.5*diff_score)
        #print("Row sim", row_sim)
        #print("Whole sim", whole_sim)
        final_sim = row_sim + 0.5 * max_attn + opt.lambda_whole * whole_sim
        if opt.need_box_whole:
            final_sim += 0.5 * box_whole_sim
        similarities.append(final_sim) # +0.5*diff_score

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


def xattn_score_i2t(images, whole_images, captions, cap_lens, final_captions, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Whole images: (batch_size, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    Final captions: (batch_size, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        final_cap_i = final_captions[i, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        final_cap_i_expand = final_cap_i.repeat(n_image, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn, max_attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        cap_i_expand = cap_i_expand.contiguous()
        final_cap_i_expand = final_cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        row_sim = cosine_similarity(images, weiContext, dim=2)
        whole_sim = cosine_similarity(whole_images, final_cap_i_expand, dim=1).view(whole_images.size(0), 1)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
            max_attn.mul_(opt.lambda_lse).exp_()
            max_attn = max_attn.sum(dim=1, keepdim=True)
            max_attn = torch.log(max_attn)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
            max_attn = max_attn.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
            max_attn = max_attn.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
            max_attn = max_attn.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        final_sim = row_sim + 0.5 * max_attn + opt.lambda_whole * whole_sim
        similarities.append(final_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, w_im, s, s_l, f_s):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores = xattn_score_t2i(im, w_im, s, s_l, f_s, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores = xattn_score_i2t(im, w_im, s, s_l, f_s, self.opt)
        else:
            raise ValueError("unknown first norm type:", opt.raw_feature_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

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
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm, need_box_whole = opt.need_box_whole)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        params += list(self.img_enc.position_embedding.parameters())
        params += list(self.img_enc.fc_whole.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, whole_images, boxes, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        whole_images = Variable(whole_images, volatile=volatile)
        boxes = Variable(boxes, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            whole_images = whole_images.cuda()
            boxes = boxes.cuda()
            captions = captions.cuda()

        # Forward
        img_emb, whole_img_emb = self.img_enc(images, whole_images, boxes)
        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens, final_cap_emb = self.txt_enc(captions, lengths)
        return img_emb, whole_img_emb, cap_emb, cap_lens, final_cap_emb

    def forward_loss(self, img_emb, whole_img_emb, cap_emb, cap_len, final_cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, whole_img_emb, cap_emb, cap_len, final_cap_emb)
        try:
            self.logger.update('Le', loss.data[0], img_emb.size(0))
        except:
            self.logger.update('Le', loss.data, img_emb.size(0))
        return loss

    def train_emb(self, images, whole_images, boxes, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        #print("Image shape", images.shape)
        #print("Boxes shape", boxes.shape)
        # compute the embeddings
        img_emb, whole_img_emb, cap_emb, cap_lens, final_cap_emb = self.forward_emb(images, whole_images, boxes, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, whole_img_emb, cap_emb, cap_lens, final_cap_emb)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
