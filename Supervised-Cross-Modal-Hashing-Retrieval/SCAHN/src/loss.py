import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor


class WithLossCellG(nn.Cell):
    def __init__(self, net, dis_net, config):
        super(WithLossCellG, self).__init__(auto_prefix=False)
        self.net = net
        self.dis_net = dis_net
        self.loss_type = 1 if config.loss_type == 'triplet' else 0
        self.paco_loss = PaCoLoss(config.bs, config.paco_alpha, config.paco_beta, config.paco_gamma, 
                                 config.paco_sup_t, config.paco_t, config.paco_base_t, 
                                 config.K, config.num_label)
        self.triplet_loss = TripletLoss(config, 'sum')

    def construct(self, img_txt, txt_mask, label):
        log = ops.log
        mll = MultiLabelLoss()
        paco_loss = self.paco_loss
        triplet_loss = self.triplet_loss
        img, img_box, txt = img_txt[0], img_txt[1], img_txt[2]
        img_cls, txt_cls, img_hash, txt_hash, image_feat_aggr, text_feat_aggr = self.net(img, img_box, txt, txt_mask)
        
        img_hash_g_stoped = self.net.img_hash_net(ops.stop_gradient(image_feat_aggr))
        txt_hash_g_stoped = self.net.txt_hash_net(ops.stop_gradient(text_feat_aggr))
        img_pred, txt_pred = self.dis_net(img_hash_g_stoped, txt_hash_g_stoped)
        
        lose_i = -log(img_pred).mean()
        lose_t = -log(txt_pred).mean()
        loss_a = lose_i + lose_t

        loss_g_i = mll(label, img_cls)
        loss_g_t = mll(label, txt_cls)
        loss_g = loss_g_i + loss_g_t

        if self.loss_type:
            loss_c = triplet_loss(img_hash, label, txt_hash, label) + triplet_loss(txt_hash, label, img_hash, label)
        
        else:
            feature_x_btw, target_x_btw, logits_x_btw = self.net.get_queue(img_hash, ops.stop_gradient(txt_hash), label, image_feat_aggr, 0)
            feature_y_btw, target_y_btw, logits_y_btw = self.net.get_queue(txt_hash, ops.stop_gradient(img_hash), label, text_feat_aggr, 1)
            paco_i2t_btw = paco_loss(feature_x_btw, target_x_btw, logits_x_btw)
            paco_t2i_btw = paco_loss(feature_y_btw, target_y_btw, logits_y_btw)
            paco_btw = paco_i2t_btw + paco_t2i_btw

            feature_x_in, target_x_in, logits_x_in = self.net.get_queue(img_hash, ops.stop_gradient(img_hash), label, image_feat_aggr, 2)
            feature_y_in, target_y_in, logits_y_in = self.net.get_queue(txt_hash, ops.stop_gradient(txt_hash), label, text_feat_aggr, 3)
            paco_i2i_in = paco_loss(feature_x_in, target_x_in, logits_x_in)
            paco_t2t_in = paco_loss(feature_y_in, target_y_in, logits_y_in)
            paco_in = paco_i2i_in + paco_t2t_in

            loss_c = paco_btw + paco_in

        loss = loss_a + loss_g + loss_c

        return loss, loss, img_hash, txt_hash


class WithLossCellD(nn.Cell):
    def __init__(self, dis_net):
        super(WithLossCellD, self).__init__()
        self.dis_net = dis_net
        self.one = Tensor(1.0, dtype=mstype.float32)

    def construct(self, img_hash, txt_hash):
        log = ops.log

        img_pred_txt, txt_pred_img = self.dis_net(img_hash, txt_hash)
        img_pred_img, txt_pred_txt = self.dis_net(txt_hash, img_hash)

        one = self.one.expand_as(img_pred_img)
        
        lose_i2i = -log(img_pred_img).mean()
        lose_i2t = -log(one - img_pred_txt).mean()
        lose_t2i = -log(one - txt_pred_img).mean()
        lose_t2t = -log(txt_pred_txt).mean()
        loss = lose_i2i + lose_i2t + lose_t2i + lose_t2t
        return loss


class MultiLabelLoss(nn.Cell):
    def __init__(self):
        super(MultiLabelLoss, self).__init__()

    def construct(self, true, pred):
        sigmoid = nn.Sigmoid()

        t_loss = true * ops.log(sigmoid(pred)) + (1 - true) * ops.log(1 - sigmoid(pred))
        loss = t_loss.mean(axis=-1)
        return -loss.mean()


class PaCoLoss(nn.Cell):
    def __init__(self, bs=32, alpha=0.05, beta=1.0, gamma=0.0, supt=1.0, temperature=1.0, base_temperature=None, K=128, num_classes=24):
        super(PaCoLoss, self).__init__()
        self.bs = bs
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes

    def construct(self, features, labels, sup_logits):
        mm = ops.matmul
        cast = ops.Cast()
        div = ops.Div()
        cat = ops.Concat(1)
        ones_like = ops.OnesLike()
        eye = ops.Eye()

        batch_size = self.bs

        mask = cast(mm(labels[:batch_size], labels.T) > 0, mstype.float32)
        
        # compute logits
        anchor_dot_contrast = div(
            mm(features[:batch_size], features.T),
            self.temperature)

        # add supervised logits
        anchor_dot_contrast = cat((sup_logits / self.supt, anchor_dot_contrast))
        

        # for numerical stability
        logits_max = anchor_dot_contrast.max(1, True)
        logits = anchor_dot_contrast - ops.stop_gradient(logits_max)

        # mask-out self-contrast cases
        logits_mask = ops.sub(ones_like(mask), eye(batch_size, self.K + 2 * batch_size, mstype.float32))

        mask = mask * logits_mask

        # add ground truth
        mask = cat((labels[:batch_size] * self.beta, mask * self.alpha))

        # compute log_prob
        logits_mask = cat((ops.ones((batch_size, self.num_classes), mstype.float32), self.gamma * logits_mask))
        exp_logits = ops.exp(logits) * logits_mask
        log_prob = logits - ops.log((exp_logits.sum(1, keepdims=True)) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = - loss.mean()

        return loss


class TripletLoss(nn.Cell):
    def __init__(self, config, reduction='mean'):
        super(TripletLoss, self).__init__()
        self.reduction = reduction
        self.alpha = config.triplet_alpha
        self.flag = (config.triplet_beta - 0.1) * config.triplet_gamma + 0.1
        self.bs = config.bs
        self.margin = config.triplet_margin

    def construct(self, source, s_labels, target=None, t_labels=None):
        gt = ops.gt
        cast = ops.Cast()
        clip = ops.clip_by_value

        margin = self.margin

        if target is None:
            target = source
        if t_labels is None:
            t_labels = s_labels

        pairwise_dist = self.cos_distance(source, target)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = pairwise_dist.expand_dims(2)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = pairwise_dist.expand_dims(1)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask, weight = self.get_triplet_mask(s_labels, t_labels)
        if self.alpha == 10:
            triplet_loss = 10 * weight * mask * triplet_loss
        else:
            triplet_loss = mask * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = clip(triplet_loss, Tensor(0), triplet_loss.max())

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = cast(gt(triplet_loss, 1e-16), mstype.float32)
        num_positive_triplets = valid_triplets.sum()

        if self.reduction == 'mean':
            triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        elif self.reduction == 'sum':
            triplet_loss = triplet_loss.sum()

        return triplet_loss

    
    def cos_distance(self, source, target):
        clip = ops.clip_by_value
        sqrt = ops.Sqrt()
        source = source.expand_dims(1)
        denominator = sqrt(source.pow(2).sum(-1)) * sqrt(target.pow(2).sum(-1))
        cos_sim = (source * target).sum(-1) / clip(denominator, Tensor(1e-8), denominator.max())
        distances = clip(1 - cos_sim, Tensor(0), (1 - cos_sim).max())
        return distances


    def get_triplet_mask(self, s_labels, t_labels):
        reshape = ops.reshape
        sort = ops.Sort(1, descending=True)
        cast = ops.Cast()
        mm = ops.matmul

        flag = self.flag
        sim_origin = mm(s_labels, t_labels.T)
        sim = cast((sim_origin > 0), mstype.float32)
        ideal_list = sort(sim_origin)[0]
        ph = mnp.arange(0., self.bs) + 2
        ph = reshape(mnp.tile(ph, (1, self.bs)), (self.bs, self.bs))
        th = mnp.log2(ph)
        Z = reshape((((2 ** ideal_list - 1) / th).sum(1)), (-1, 1))
        sim_origin = 2 ** sim_origin - 1
        sim_origin = sim_origin / Z

        i_equal_j = sim.expand_dims(2)
        i_equal_k = sim.expand_dims(1)
        sim_pos = sim_origin.expand_dims(2)
        sim_neg = sim_origin.expand_dims(1)
        weight = (sim_pos - sim_neg) * flag
        mask = i_equal_j * (1 - i_equal_k) * flag

        return mask, weight
