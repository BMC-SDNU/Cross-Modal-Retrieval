import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import ParameterTuple


class WithLossCell(nn.Cell):
    """Wrap the network with loss function to return generator loss"""

    def __init__(self, net):
        super().__init__(auto_prefix=False)
        self.net = net

    def construct(self, img_txt, txt_mask, label):
        """Construct forward graph"""
        output = self.net(img_txt, txt_mask, label)
        return output[0]


class TrainOneStepCellG(nn.Cell):
    def __init__(self, net, optimizer, config):
        super(TrainOneStepCellG, self).__init__(auto_prefix=False)
        self.grad_clip = config.grad_clip

        self.optimizer = optimizer
        self.net = net
        self.net.net.set_grad()
        self.net.net.set_train()
        self.net.dis_net.set_train(False)
        self.grad = ops.grad
        all = net.net.encoder.trainable_params()
        tails = net.net.encoder.img_post_trans.trainable_params() + \
                    net.net.encoder.txt_post_trans.trainable_params() + \
                    net.net.encoder.img_map.trainable_params() + \
                    net.net.encoder.txt_map.trainable_params()
        encoder_params = all if config.train_all else tails
        if not config.use_raw_img:
            encoder_params += net.net.encoder.image_encoder.trainable_params()
        params = encoder_params + net.net.img_hash_net.trainable_params() + \
                 net.net.txt_hash_net.trainable_params() + net.net.gcn.trainable_params()
        if config.loss_type == 'paco':
            params += net.net.paco_linear.trainable_params()
        self.weights = ParameterTuple(params)
        
    def construct(self, img, img_box, txt, txt_mask, label):
        clip = ops.clip_by_global_norm
        weights = self.weights
        img_txt = (img, img_box, txt)
        grads_g, aux = self.grad(self.net, 0, weights, True)(img_txt, txt_mask, label)
        grads_g = grads_g[1]
        if self.grad_clip != 0:
            grads_g = clip(grads_g, self.grad_clip)
        opt_res = self.optimizer(grads_g)
        return ops.depend(aux[0], opt_res), ops.depend(aux[1], opt_res), ops.depend(aux[2], opt_res)


class TrainOneStepCellD(nn.Cell):
    def __init__(self, net, optimizer):
        super(TrainOneStepCellD, self).__init__()
        self.optimizer = optimizer
        self.net = net
        self.net.set_grad()
        self.net.set_train()
        self.grad = ops.GradOperation(get_all=True, get_by_list=True)
        self.weights = ParameterTuple(net.trainable_params())
        
    def construct(self, img_hash, txt_hash):
        weights = self.weights
        loss = self.net(img_hash, txt_hash)
        grads_d = self.grad(self.net, weights)(img_hash, txt_hash)[1]
        opt_res = self.optimizer(grads_d)
        return ops.depend(loss, opt_res)