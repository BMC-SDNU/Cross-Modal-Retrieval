import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from torch.nn import functional as F
from layers import l2norm
import model

# tutorials/08 - Language Model
# RNN Based Language Model
class GRUTextEncoder(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False, gru_units=1024):
        super(GRUTextEncoder, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.gru_units = gru_units

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, gru_units, num_layers, batch_first=True)
        
        self.fc = None
        if embed_size != gru_units:
            self.fc = nn.Linear(gru_units, embed_size)

        self.init_weights()

    def init_weights(self):
        
        self.embed.weight.data.uniform_(-0.1, 0.1)

        if self.fc:
            r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)

            self.fc.weight.data.uniform_(-r, r)
            self.fc.bias.data.fill_(0)

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
        I = Variable(I.expand(x.size(0), 1, self.gru_units)-1)
        if torch.cuda.is_available():
            I = I.cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        if self.fc:
            out = self.fc(out)

        self.outputs = {'output': out}
        # normalization in the joint embedding space
        outnormed = l2norm(out)
        self.outputs['outnormed'] = outnormed
        # take absolute value, used by order embeddings
        if self.use_abs:
            outnormed = torch.abs(outnormed)

        return outnormed


class EmbeddingLayer(nn.Module):

    def __init__(self, 
                 vocab_size, 
                 embed_dim, 
                 one_hot_input, 
                 trainable=True):

        super(EmbeddingLayer, self).__init__()

        self.one_hot_input = one_hot_input 
        self.vocab_size = vocab_size 
        self.embed_dim = embed_dim
        self.trainable = trainable

        if one_hot_input:
            assert vocab_size == embed_dim
            assert not trainable

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.init_weights()
        
    def init_weights(self,):

        if self.one_hot_input:
            matrix = torch.eye(len(self.embedding.weight.data))
            self.embedding.weight.data = matrix
            self.embedding.weight.data[0,...] = 0. # Padding token             
        else:
            self.embedding.weight.data.uniform_(-0.1, 0.1)

        if not self.trainable:
            self.embedding.weight.requires_grad = False

    def forward(self, x, lens=None):
        return self.embedding(x)


class InceptionEncoder(nn.Module):

    def __init__(self, 
                    num_embeddings, 
                    embedding_size,                     
                    char_level=True, 
                    one_hot_input=True,
                    latent_size=1024,
                    nb_filters_init=192,
                    trainable_emb=False,
                    nb_towers=5,
                    nb_filters=256,
                    pool='max', 
                    use_abs=False):
        
        super(InceptionEncoder, self).__init__()        

        self.use_abs = use_abs
        self.pool = pool
        self.num_embeddings = num_embeddings
        self.char_level = char_level
        self.latent_size = latent_size

        final_features_dim = nb_towers * nb_filters
        
        if char_level:
            embedding_size = num_embeddings

        self.embed = EmbeddingLayer(vocab_size=num_embeddings, 
                                    embed_dim=embedding_size, 
                                    one_hot_input=one_hot_input, 
                                    trainable=trainable_emb)

        init_convs = self.__add_initial_convs__(embedding_size, nb_filters_init)
        
        towers = []
        for f in range(1, nb_towers+1):
            tower = self.__add_conv_tower__(in_channels=nb_filters_init, 
                                    nb_filters=nb_filters, 
                                    nb_layers=f)
            towers.append(tower)

        self.init_convs = nn.ModuleList(init_convs)
        self.towers = nn.ModuleList(towers)

        self.fc = nn.Linear(final_features_dim, self.latent_size)

        global_initializer(self)

    def __add_initial_convs__(self, embedding_size, nb_filters):
        convs = []
        filter_sizes = [3, 5, 7]
        paddings = [1, 2, 3]
        n = len(filter_sizes)
        for k, p in zip(filter_sizes, paddings):
            conv = ConvBlock(in_channels=embedding_size,  
                            out_channels=nb_filters/n, 
                            kernel_size=k, 
                            padding=p, 
                            maxout=True, 
                            activation=None)
            convs.append(conv)

        return convs

    def __add_conv_tower__(self, in_channels, nb_filters, nb_layers):

        layers = []

        for i in range(nb_layers):
            conv = ConvBlock(in_channels=in_channels,  
                            out_channels=nb_filters, 
                            kernel_size=3, 
                            padding=1, 
                            maxout=True, 
                            activation=None)
            in_channels = nb_filters
            layers.append(conv)

        return nn.Sequential(*layers)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)

        x = x.permute(0, 2, 1)

        init_conv = []
        for net_module in self.init_convs:
            init_conv.append(net_module(x))

        x = torch.cat(init_conv, 1)

        multi_aspect_feats = []
        for tower in self.towers:
            _x = tower(x)
            _x = F.max_pool1d(_x, _x.size()[-1])
            _x = _x.view(_x.size(0), _x.size(1))
            multi_aspect_feats.append(_x)

        out = torch.cat(multi_aspect_feats, -1)

        latent = self.fc(out)
        # normalization in the joint embedding space
        out = l2norm(latent)
        
        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        self.outputs = {}
        self.outputs['output'] = out

        return out


class BasicConv(nn.Module):

    def __init__(self, activation=None, batchnorm=False, **kwargs):
        super(BasicConv, self).__init__()

        layers = []

        conv = nn.Conv1d(**kwargs)
        layers += [conv]
        
        if activation is not None:
            layers += [activation()]
        
        if batchnorm:
            layers += [nn.BatchNorm1d(kwargs['out_channels'])]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        
        return self.conv(x)


class ConvBlock(nn.Module):

    def __init__(self, maxout=True, 
                       activation=None,
                       batchnorm=False, 
                       **kwargs):

        super(ConvBlock, self).__init__()
        self.batchnorm = batchnorm
        self.activation = activation
        self.maxout = maxout 

        self.conv1 = BasicConv(activation=activation, **kwargs)
        if maxout:
            self.conv2 = BasicConv(activation=activation, **kwargs)

    def forward(self, x):

        a = self.conv1(x)
        if self.maxout:
            b = self.conv2(x)
            a = torch.max(a, b)

        return a


def global_initializer(net):

    for m in net.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform(m.weight.data)
            # m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


text_encoders_alias = {
    'gru': {'method': GRUTextEncoder, 'args': {}},
    
    'chain-v1': {'method': InceptionEncoder, 'args': {
                    'nb_filters_init': 192,
                    'trainable_emb': False,
                    'nb_towers': 4,
                    'nb_filters': 256,
                    'pool': 'max', 
    }},
}


def get_text_encoder(encoder, opt):

    encoder = encoder.lower()
    vocab_size = opt.vocab_size
    word_dim = opt.word_dim
    num_layers = opt.num_layers
    embed_size = opt.embed_size
    use_abs = opt.use_abs
    model_kwargs = opt.kwargs

    model_args = {}
    model_kwargs = [] if model_kwargs is None else model_kwargs
    for _x in model_kwargs:
        k, _type, v = _x.split(':')
        model_args[k] = eval(_type)(v)

    try:
        gru_units = opt.gru_units
        norm_words = opt.norm_words
    except AttributeError:
        gru_units = embed_size
        norm_words = None


    # Character
    if opt.vocab_path.lower() == 'char':
        params = {
            'num_embeddings': vocab_size,
            'embedding_size': word_dim,
            'char_level': True,
            'latent_size': opt.embed_size,
            'use_abs': opt.use_abs,
            # 'use_dense': False
        }    
    # Word embedding
    else:
        params = {
            'vocab_size': vocab_size, 
            'word_dim': word_dim,
            'gru_units': gru_units,             
            'embed_size': embed_size, 
            'num_layers': num_layers,
            'use_abs': opt.use_abs,            
        }

        if encoder.startswith('attentive'):
            params['att_units'] = opt.att_units 
            params['hops'] = opt.att_hops
            params['norm_words'] = norm_words,

    params.update(text_encoders_alias[encoder]['args'])
    params.update(model_args)
    txt_enc = text_encoders_alias[encoder]['method'](**params)

    model.print_summary(txt_enc)
    return txt_enc

