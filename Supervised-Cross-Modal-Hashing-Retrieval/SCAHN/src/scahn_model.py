"""SCAHN model, implemented in mindspore."""

import numpy as np
import pickle as pk

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.initializer import initializer
from src.base_models.gcn import GCN


class CrossModalJointEncoder(nn.Cell):
    """
    
    """
    def __init__(self,
                 config,
                 image_encoder,
                 text_encoder,
                 post_mean=True,
                 initialization='he_uniform'):
        super(CrossModalJointEncoder, self).__init__()
        
        # initialise configs
        self.mean = post_mean
        self.bs = config.bs
        self.seq_len = config.seq_len
        self.img_seq = config.img_seq

        # for image modality
        self.image_encoder = image_encoder

        # for text modality
        #   backbone
        self.text_encoder = text_encoder

        #   posttransformers
        self.img_post_trans = nn.transformer.TransformerEncoder(config.bs, config.num_layers, config.emb_dim, 2048, config.img_seq, 4, hidden_act='relu')
        self.txt_post_trans = nn.transformer.TransformerEncoder(config.bs, config.num_layers, config.emb_dim, 2048, config.seq_len, 4, hidden_act='relu')
        self.attn_mask = nn.transformer.AttentionMask(self.seq_len)

        #   postprocess
        #     from 768 dims to 1024 dims
        self.aggr = self.aggr_mean if self.mean else self.aggr_first

        self.img_map = nn.Dense(config.img_emb_dim, config.emb_dim)
        self.img_map.weight.set_data(initializer(initialization, [config.emb_dim, config.img_emb_dim]))

        self.txt_map = nn.Dense(config.txt_emb_dim, config.emb_dim)
        self.txt_map.weight.set_data(initializer(initialization, [config.emb_dim, config.txt_emb_dim]))

        zeros = ops.Zeros()
        self.token_type_ids = zeros((config.bs, config.seq_len), mstype.int32)

    def construct(self, img, img_box, txt, txt_mask):
        ones = ops.Ones()
        cast = ops.Cast()
        l2norm = ops.L2Normalize()

        # image modality
        _, image_feat = self.image_encoder(img, img_box)
        image_feat = self.img_map(image_feat)
        img_att_mask = ones((self.bs, self.img_seq, self.img_seq), mstype.float32)
        image_feat = self.img_post_trans(image_feat, img_att_mask)[0]
        image_feat_aggr = self.aggr(image_feat)
        image_feat_aggr = l2norm(image_feat_aggr)

        # text modality
        text_feat, _, _ = self.text_encoder(txt, self.token_type_ids, txt_mask)
        txt_att_mask = self.attn_mask(cast(txt_mask, mstype.float32))
        text_feat = self.txt_map(text_feat)
        text_feat = self.txt_post_trans(text_feat, txt_att_mask)[0]
        text_feat_aggr = self.aggr(text_feat)
        text_feat_aggr = l2norm(text_feat_aggr)
        
        return image_feat_aggr, text_feat_aggr

    def aggr_mean(self, feat):
        return feat.mean(axis=1)

    def aggr_first(self, feat):
        return feat[:, 0, :]


class HashNet(nn.Cell):
    def __init__(self,
                 config,
                 initialization='he_uniform'):
        super(HashNet, self).__init__()
        self.fc = nn.Dense(config.emb_dim, config.hash_bit)
        self.fc.weight.set_data(initializer(initialization, [config.hash_bit, config.emb_dim]))
        self.tanh = nn.Tanh()

    def construct(self, x):
        x = self.fc(x)
        x = self.tanh(x)
        return x


class SCAHNGenerator(nn.Cell):
    """
    
    """
    def __init__(self,
                 config,
                 image_encoder,
                 text_encoder,
                 initialization='he_uniform',
                 ):
        super(SCAHNGenerator, self).__init__()

        # utils
        cast = ops.Cast()
        normalize = ops.L2Normalize()
        randn = np.random.randn
        randint = np.random.randint

        # base encoder
        self.encoder = CrossModalJointEncoder(config, image_encoder, text_encoder)

        # gcn basics
        _adj = self.get_A(config.adj_path, 0.4, config.num_label)
        self.A = Parameter(cast(Tensor.from_numpy(_adj), mstype.float32), "A", False, False, False)
        self.gcn = GCN(config.gcn_input_dim, config.emb_dim, config.gcn_hidden, config.gcn_dropout)
        self.inp = np.load(config.inp_path, allow_pickle=True, encoding="latin1")
        self.inp = Tensor(self.inp, mstype.float32)

        # hash module
        self.img_hash_net = HashNet(config)

        self.txt_hash_net = HashNet(config)

        # contrastive learning module
        self.K = config.K
        self.paco_linear = nn.Dense(config.emb_dim, config.num_label)
        self.paco_linear.weight.set_data(initializer(initialization, [config.num_label, config.emb_dim]))

        queue_i_btw = Parameter(normalize(Tensor(randn(self.K, config.hash_bit), mstype.float32)), 'queue_i_btw', requires_grad=False)
        queue_l_i_btw = Parameter(Tensor(randint(0, 2, (self.K, config.num_label)), mstype.float32), 'queue_l_i_btw', requires_grad=False)
        queue_ptr_i_btw = Parameter(Tensor(0, mstype.int32), 'queue_ptr_i_btw', requires_grad=False)

        queue_t_btw = Parameter(normalize(Tensor(randn(self.K, config.hash_bit), mstype.float32)), 'queue_t_btw', requires_grad=False)
        queue_l_t_btw = Parameter(Tensor(randint(0, 2, (self.K, config.num_label)), mstype.float32), 'queue_l_t_btw', requires_grad=False)
        queue_ptr_t_btw = Parameter(Tensor(0, mstype.int32), 'queue_ptr_t_btw', requires_grad=False)

        queue_i_in = Parameter(normalize(Tensor(randn(self.K, config.hash_bit), mstype.float32)), 'queue_i_in', requires_grad=False)
        queue_l_i_in = Parameter(Tensor(randint(0, 2, (self.K, config.num_label)), mstype.float32), 'queue_l_i_in', requires_grad=False)
        queue_ptr_i_in = Parameter(Tensor(0, mstype.int32), 'queue_ptr_i_in', requires_grad=False)

        queue_t_in = Parameter(normalize(Tensor(randn(self.K, config.hash_bit), mstype.float32)), 'queue_t_in', requires_grad=False)
        queue_l_t_in = Parameter(Tensor(randint(0, 2, (self.K, config.num_label)), mstype.float32), 'queue_l_t_in', requires_grad=False)
        queue_ptr_t_in = Parameter(Tensor(0, mstype.int32), 'queue_ptr_t_in', requires_grad=False)

        self.queue = ParameterTuple((queue_i_btw, queue_t_btw, queue_i_in, queue_t_in))
        self.queue_l = ParameterTuple((queue_l_i_btw, queue_l_t_btw, queue_l_i_in, queue_l_t_in))
        self.queue_ptr = ParameterTuple((queue_ptr_i_btw, queue_ptr_t_btw, queue_ptr_i_in, queue_ptr_t_in))
        self.slide = Tensor(np.arange(0, self.K, 1), mstype.int32)

    def get_queue(self, q, k, labels, f_q, flag):
        cat = ops.Concat(0)
        scatter_update = ops.ScatterUpdate()
        nonzero = ops.nonzero
        logical_and = ops.logical_and
        assign = ops.Assign()
        cast = ops.Cast()

        queue, queue_l, queue_ptr = self.queue[flag], self.queue_l[flag], self.queue_ptr[flag]

        features = cat((q, k, ops.stop_gradient(queue.copy())))
        target = cat((labels, labels, ops.stop_gradient(queue_l.copy())))
        logits_q = self.paco_linear(f_q)

        keys = ops.stop_gradient(k)
        labels = ops.stop_gradient(labels)
        batch_size = keys.shape[0]

        assert self.K % batch_size == 0
        slide = self.slide
        mask = logical_and((slide >= queue_ptr), (slide < (queue_ptr + batch_size)))
        slide = cast(nonzero(mask * (slide + 1)).squeeze(), mstype.int32)
        scatter_update(queue, slide, keys)
        scatter_update(queue_l, slide, labels)
        assign(queue_ptr, (queue_ptr + batch_size) % self.K)
        
        return features, target, logits_q

    def get_A(self, adj_path, t, num_label):
        result = pk.load(open(adj_path, 'rb'))
        _adj = result['adj']
        _nums = result['nums']
        _nums = _nums[:, np.newaxis]
        _adj = _adj / _nums
        _adj[_adj < t] = 0
        _adj[_adj >= t] = 1
        _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
        _adj = _adj + np.identity(num_label, np.int)
        return _adj

    def get_adj(self, A):
        pow = ops.Pow()
        cast = ops.Cast()
        diag = ops.diag
        mm = ops.matmul

        D = diag(pow(cast(A.sum(1), mstype.float32), -0.5))
        adj = mm(mm(A, D).T, D)
        return adj

    def predict_class(self, feat):
        mm = ops.matmul

        adj = ops.stop_gradient(self.get_adj(self.A))
        x = self.gcn(adj, self.inp)
        x = mm(feat, x.T)
        return x

    def construct(self, img, img_box, txt, txt_mask):
        image_feat_aggr, text_feat_aggr = self.encoder(img, img_box, txt, txt_mask)

        img_cls = self.predict_class(image_feat_aggr)
        txt_cls = self.predict_class(text_feat_aggr)

        sigmoid = nn.Sigmoid()
        img_cls = sigmoid(img_cls)
        txt_cls = sigmoid(txt_cls)

        img_hash = self.img_hash_net(image_feat_aggr)
        txt_hash = self.txt_hash_net(text_feat_aggr)

        return img_cls, txt_cls, img_hash, txt_hash, image_feat_aggr, text_feat_aggr


class CrossDiscriminator(nn.Cell):
    def __init__(self,
                 config,
                 ):
        super(CrossDiscriminator, self).__init__()
        self.img_discriminator = Discriminator(config)
        self.txt_discriminator = Discriminator(config)

    def construct(self, img_hash, txt_hash):
        return self.img_discriminator(txt_hash), self.txt_discriminator(img_hash)


class Discriminator(nn.Cell):
    """
    
    """
    def __init__(self,
                 config,
                 initialization='he_uniform',
                 ):
        super(Discriminator, self).__init__()
        self.layer0 = nn.Dense(config.hash_bit, config.dis_hidden_dim)
        self.layer0.weight.set_data(initializer(initialization, [config.dis_hidden_dim, config.hash_bit]))
        self.layer1 = nn.Dense(config.dis_hidden_dim, 1)
        self.layer1.weight.set_data(initializer(initialization, [1, config.dis_hidden_dim]))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        x = self.layer0(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.sigmoid(x)
        return x
