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

# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

        for param in self.cnn.parameters():
            param.requires_grad = finetune

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model, device_ids=[0,2]).cuda()

        return model

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features,1)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features,1)

        return features


def EncoderImage(data_name, img_dim, embed_size, finetune=False, precomp_enc_type='basic', cnn_type='vgg19',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    oimg_enc = EncoderImageFull(embed_size, finetune, cnn_type, no_imgnorm)

    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc, oimg_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.wl = nn.Linear(img_dim, embed_size)

        self.dim_k = embed_size/8
        #realation
        self.WK = nn.Linear(embed_size, self.dim_k)
        self.WQ = nn.Linear(embed_size, self.dim_k)
        self.WV = nn.Linear(embed_size, embed_size)

        self.birnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.wl.in_features +
                                  self.wl.out_features)
        self.wl.weight.data.uniform_(-r, r)
        self.wl.bias.data.fill_(0)

        r = np.sqrt(6.) / np.sqrt(self.WK.in_features +
                                  self.WK.out_features)
        self.WK.weight.data.uniform_(-r, r)
        self.WK.bias.data.fill_(0)

        self.WQ.weight.data.uniform_(-r, r)
        self.WQ.bias.data.fill_(0)

        r = np.sqrt(6.) / np.sqrt(self.WV.in_features +
                                  self.WV.out_features)
        self.WV.weight.data.uniform_(-r, r)
        self.WV.bias.data.fill_(0)

    def forward(self, images, oimages):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        featuresp = self.wl(images)
        B = featuresp.size(0)
        N = featuresp.size(1)

        if not self.no_imgnorm:
            featuresp = l2norm(featuresp, dim=-1)

        featuresp = featuresp + torch.unsqueeze(oimages,1).repeat(1,N,1)
        #f1 = featuresp
        #version1
        w_k = self.WK(featuresp)
        w_k = w_k.view(B,N,1,self.dim_k)

        w_q = self.WQ(featuresp)
        w_q = w_q.view(B,1,N,self.dim_k)

        scaled_dot = torch.sum((w_k*w_q), -1)
        scaled_dot = scaled_dot / np.sqrt(self.dim_k)

        w_a = scaled_dot.view(B,N,N)

        w_mn = torch.nn.Softmax(dim=2)(w_a)

        w_v = self.WV(featuresp)

        w_mn = w_mn.view(B,N,N,1)
        w_v = w_v.view(B,N,1,-1)

        output = w_mn*w_v

        featuresp = torch.sum(output,-2) + featuresp
        #print 'ffff'
        #version2
        '''
        for i in range(B):
            f_a = f1[i,...]
            print f_a.size()
            if i == 0:
                w_k = self.WK(f_a)
                w_k = w_k.view(N,1,self.dim_k)
                #print 'ss'
                w_q = self.WQ(f_a)
                w_q = w_q.view(1,N,self.dim_k)

                scaled_dot = torch.sum((w_k*w_q),-1 )
                scaled_dot = scaled_dot / np.sqrt(self.dim_k)

                w_a = scaled_dot.view(N,N)
                #print 'ss'
                w_mn = w_a
                w_mn = torch.nn.Softmax(dim=1)(w_mn)

                w_v = self.WV(f_a)

                w_mn = w_mn.view(N,N,1)
                w_v = w_v.view(N,1,-1)
                #print 'ss'
                output = w_mn*w_v

                output = torch.unsqueeze(torch.sum(output,-2),0)
            else:
                w_k = self.WK(f_a)
                w_k = w_k.view(N,1,self.dim_k)

                w_q = self.WQ(f_a)
                w_q = w_q.view(1,N,self.dim_k)

                scaled_dot = torch.sum((w_k*w_q),-1 )
                scaled_dot = scaled_dot / np.sqrt(self.dim_k)
                #print 'dd'
        
                w_a = scaled_dot.view(N,N)

                w_mn = w_a
                w_mn = torch.nn.Softmax(dim=1)(w_mn)

                w_v = self.WV(f_a)

                w_mn = w_mn.view(N,N,1)
                w_v = w_v.view(N,1,-1)
                #print 'dd'
                s = w_mn*w_v

                s = torch.sum(s,-2)
                #print s.size()
                #print output.size()
                output = torch.cat((output,torch.unsqueeze(s,0)),0)
                #print 'hh'
        '''
        #print output.size()

        # normalize in the joint embedding space
        out, _ = self.birnn(featuresp)
        features = out[:,N-1,:]
        
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        # word embedding     
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

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
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            out = l2norm(out, 1)

        return out, x


class DecoderWithoutAttention(nn.Module):
    """
    Decoder.
    """
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=1024, dropout=0.5):
        """
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithoutAttention, self).__init__()
        self.encoder = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, encoder_dim)
        :return: hidden state, cell state
        """
        h = self.init_h(encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(encoder_out)
        return h, c

    def forward(self, embeddings, encoder_out, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, encoder_dim)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths
        """

        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        #A = np.zeros((batch_size,1)).tolist()
        decode_lengths = (np.array(caption_lengths) - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = Variable(torch.zeros(batch_size, max(decode_lengths), vocab_size))
        if torch.cuda.is_available():
            predictions = predictions.cuda()
        # At each time-step, decode by
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h, c = self.decode_step(
                embeddings[:batch_size_t, t, :],
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
        #predictions = predictions.contiguous()
        #print predictions
        return predictions, decode_lengths


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
       
        self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
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
        self.img_enc, self.oimg_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size, finetune=opt.finetune,
                                    precomp_enc_type=opt.precomp_enc_type, cnn_type=opt.cnn_type,
                                    no_imgnorm=opt.no_imgnorm)

        self.img_enc = nn.DataParallel(self.img_enc, device_ids=[0,2]).cuda()

        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,  
                                   no_txtnorm=opt.no_txtnorm)

        self.decoder = DecoderWithoutAttention(embed_dim=opt.word_dim, decoder_dim=opt.decoder_dim, vocab_size=opt.vocab_size)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.oimg_enc.cuda()
            self.decoder.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         max_violation=opt.max_violation)

        self.decode_criterion = nn.CrossEntropyLoss()
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.decoder.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.oimg_enc.state_dict(), self.txt_enc.state_dict(), self.img_enc.state_dict(), self.decoder.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        #self.oimg_enc.load_state_dict(state_dict[0])
        model_dict = self.oimg_enc.state_dict()
        pretrained_dict = {k: v for k, v in state_dict[0].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.oimg_enc.load_state_dict(model_dict)
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.oimg_enc.eval()
        self.decoder.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.oimg_enc.eval()
        self.decoder.eval()

    def forward_emb(self, oimages, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        oimages = Variable(oimages, volatile=volatile)
        images = Variable(images, volatile=volatile)
        captions_no_cuda = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            oimages = oimages.cuda()
            images = images.cuda()
            captions = captions_no_cuda.cuda()

        # Forward
        oimg_emb = self.oimg_enc(oimages)
        img_emb = self.img_enc(images, oimg_emb)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, x = self.txt_enc(captions, lengths)

        if volatile:
            return img_emb, cap_emb, oimg_emb

        scores, decode_lengths = self.decoder(x, img_emb, lengths)
        
        return img_emb, cap_emb, oimg_emb, scores, decode_lengths, captions

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.data[0], img_emb.size(0))
        return loss

    def forward_decode_loss(self, scores, decode_lengths, captions):
        targets = captions[:,1:]
        #print 'starr'
        #print scores
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        #print scores
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        #print scores.size()
        '''
        scores = Variable(scores)
        targets = Variable(scores)
        if torch.cuda.is_available():
            scores = scores.cuda()
            targets = targets.cuda()
        '''
        loss_de = self.decode_criterion(scores, targets)
        self.logger.update('Lcp', loss_de.data[0], captions.size(0))
        return loss_de

    def train_emb(self, oimages, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, oimg_emb, scores, decode_lengths, captions = self.forward_emb(oimages, images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss_vse = self.forward_loss(img_emb, cap_emb)
        #print loss_vse
        loss_de = self.forward_decode_loss(scores, decode_lengths, captions)
        #print loss_de

        loss = loss_vse + loss_de
        self.logger.update('La', loss.data[0], captions.size(0))

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
