from src.scahn_model import SCAHNGenerator, CrossDiscriminator
from src.loss import WithLossCellG, WithLossCellD
from src.cell import TrainOneStepCellG, TrainOneStepCellD
from src.config import get_config
from src.dataset import get_dataset
from src.lr import SelfMadeDynamicLR
from src.base_models.vit import get_network as get_img_network, VitArgs
from src.base_models.bert_model import BertModel, BertConfig
from src.base_models.buf_transformer import BUFPostProcessing

import time

import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as np
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import context
from tqdm import tqdm


def run_train(config):
    context.set_context(mode=ms.GRAPH_MODE if config.mindspore_mode == 'GRAPH' else ms.PYNATIVE_MODE,
                         device_target=config.device_target, device_id=config.device_id)

    with open(config.log_file, 'w') as f:
        f.write('')
    
    print("-----------initializing dataset-----------")
    train_dataset = get_dataset(config, 'train')
    num_epochs = config.num_epochs
    train_dataloader = train_dataset.create_dict_iterator(num_epochs=num_epochs)

    query_dataset = get_dataset(config, 'query')
    query_dataloader = query_dataset.create_dict_iterator(num_epochs=-1)
    
    db_dataset = get_dataset(config, 'db')
    db_dataloader = db_dataset.create_dict_iterator(num_epochs=-1)


    print("-----------initializing base models-----------")
    args = VitArgs(config.img_size, config.emb_dim)
    if config.use_raw_img:
        image_encoder = get_img_network(config.vit_type, args)
        if config.vit_ckpt:
            ms.load_checkpoint(config.vit_ckpt, image_encoder)
    else:
        image_encoder = BUFPostProcessing(config)

    bert_config = BertConfig(seq_length=config.seq_len, hidden_size=config.txt_emb_dim)
    text_encoder = BertModel(bert_config, is_training=True, use_one_hot_embeddings=False)
    if config.bert_ckpt:
        ms.load_checkpoint(config.bert_ckpt, text_encoder)

    scahn_base = SCAHNGenerator(config, image_encoder, text_encoder)
    
    cross_discriminator = CrossDiscriminator(config)


    print("-----------initializing withloss models-----------")
    with_loss_G = WithLossCellG(scahn_base, cross_discriminator, config)
    with_loss_D = WithLossCellD(cross_discriminator)


    print("-----------initializing dynamic LRs-----------")
    dynamic_lr = SelfMadeDynamicLR(config.lr, config.decay_rate, config.decay_steps, config.warmup_steps)
    finetun_dynamic_lr = SelfMadeDynamicLR(config.finetun_lr, config.decay_rate, config.decay_steps, config.warmup_steps)
    gcn_dynamic_lr = SelfMadeDynamicLR(config.gcn_lr, config.decay_rate, config.decay_steps, config.warmup_steps)
    dis_dynamic_lr = SelfMadeDynamicLR(config.dis_lr, config.decay_rate, config.decay_steps, config.warmup_steps)

   
    print("-----------initializing optimizers-----------")
    all = with_loss_G.net.encoder.trainable_params()
    tails = with_loss_G.net.encoder.img_post_trans.trainable_params() + \
                with_loss_G.net.encoder.txt_post_trans.trainable_params() + \
                with_loss_G.net.encoder.img_map.trainable_params() + \
                with_loss_G.net.encoder.txt_map.trainable_params()
    encoder_params = all if config.train_all else tails
    if not config.use_raw_img:
        encoder_params += with_loss_G.net.encoder.image_encoder.trainable_params()
    params = [{'params': encoder_params, 'lr': finetun_dynamic_lr},
                {'params': with_loss_G.net.img_hash_net.trainable_params(), 'lr': dynamic_lr},
                {'params': with_loss_G.net.txt_hash_net.trainable_params(), 'lr': dynamic_lr},
                {'params': with_loss_G.net.gcn.trainable_params(), 'lr': gcn_dynamic_lr}
                ]
    if config.loss_type == 'paco':
        params.append({'params': with_loss_G.net.paco_linear.trainable_params(), 'lr': dynamic_lr})
    optimizer_G = nn.Adam(
        params=params,
        learning_rate=config.finetun_lr
    )
    optimizer_D = nn.Adam(
        params=with_loss_D.trainable_params(),
        learning_rate=dis_dynamic_lr,
        beta1=0.5,
        beta2=0.9,
        weight_decay=0.0001
    )


    print("-----------initializing TrainOneStepCell models-----------")
    base_net = TrainOneStepCellG(with_loss_G, optimizer_G, config)
    dis_net = TrainOneStepCellD(with_loss_D, optimizer_D)

    base_net.set_train()
    dis_net.set_train()


    print("-----------start training-----------")
    mapi2tlist, mapt2ilist = [], []
    for i in range(1, num_epochs + 1):
        loss = []
        start_time = time.time()
        for data in tqdm(train_dataloader):
            if config.use_raw_img:
                img, img_box = data['img'], None
            else:
                img, img_box = data['img_feat'], data['img_box']
            base_loss, img_hash, txt_hash = base_net(img, img_box, data['txt'], data['txt_mask'], data['label'])
            dis_loss = dis_net(img_hash, txt_hash)
            loss.append(base_loss)
        cost_time = time.time() - start_time

        print("num_epoch: ", i, "loss: ", sum(loss)/len(loss), "epoch time: ", cost_time)

        with open(config.log_file, 'a+') as f:
            f.write('epoch:')
            f.write(str(i))
            f.write(' loss:')
            f.write(str(sum(loss)/len(loss)))
            f.write(' epoch time:')
            f.write('%.4f' % cost_time)
            f.write('\n')

        if i % config.valid_freq == 0:
            scahn_base.set_train(False)
            start_time = time.time()
            mapi2t, mapt2i = valid(scahn_base, query_dataloader, db_dataloader, config)
            cost_time = time.time() - start_time
            scahn_base.set_train()

            print("num_epoch: ", i, "mapi2t: ", mapi2t, "mapt2i: ", mapt2i, "valid time: ", cost_time)
            mapi2tlist.append(mapi2t)
            mapt2ilist.append(mapt2i)

            with open(config.log_file, 'a+') as f:
                f.write('max MAP: MAP(i->t):')
                f.write('%3.4f' % mapi2t)
                f.write(' MAP(t->i):')
                f.write('%3.4f' % mapt2i)
                f.write(' valid time:')
                f.write('%.4f' % cost_time)
                f.write('\n')

    with open(config.log_file, 'a+') as f:
        f.write('max MAP: MAP(i->t):')
        f.write('%3.4f' % max(mapi2tlist))
        f.write(' MAP(t->i):')
        f.write('%3.4f' % max(mapt2ilist))


def generate_code(net, dataloader, num, bit, batch_size, num_label, use_raw_img):
    zeros = ops.Zeros()
    sign = ops.Sign()
    X = zeros((num, bit), mstype.float32)
    Y = zeros((num, bit), mstype.float32)
    labels = zeros((num, num_label), mstype.float32)
    i = 0
    for data in tqdm(dataloader):
        if use_raw_img:
            img, img_box = data['img'], None
        else:
            img, img_box = data['img_feat'], data['img_box']
        _, _, img_hash, txt_hash, _, _ = net(img, img_box, data['txt'], data['txt_mask'])
        label = data['label']
        idx_end = min(num, (i + 1) * batch_size)
        X[i * batch_size: idx_end, :] = img_hash.copy()
        Y[i * batch_size: idx_end, :] = txt_hash.copy()
        labels[i * batch_size: idx_end, :] = label.copy()
        i += 1

    X = sign(X)
    Y = sign(Y)
    return X, Y, labels


def calc_hamming_dist(B1, B2):
    mm = ops.matmul
    expand_dims = ops.ExpandDims()

    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = expand_dims(B1, 0)
    distH = 0.5 * (q - mm(B1, B2.T))
    return distH
    

def calc_map_k(qB, rB, query_label, retrieval_label):
    mm = ops.matmul
    sort = ops.Sort()
    expand_dims = ops.ExpandDims()
    squeeze = ops.squeeze
    cast = ops.Cast()
    nonzero = ops.nonzero
    
    num_query = query_label.shape[0]
    map = 0.
    k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = squeeze(mm(expand_dims(query_label[i], 0), retrieval_label.T) > 0)
        tsum = gnd.sum()
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = sort(hamm)
        ind = squeeze(ind)
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = np.arange(1, total + 1, dtype=mstype.float32)
        tindex = squeeze(cast(nonzero(gnd)[:total], mstype.float32)) + 1.0    
        map += (count / tindex).mean()
    map = map / num_query
    return map


def valid(net, query_dataloader, db_dataloader, config):
    qX, qY, qlabels = generate_code(net, query_dataloader, config.query_num, config.hash_bit, config.bs, config.num_label, config.use_raw_img)
    dX, dY, dlabels = generate_code(net, db_dataloader, config.db_num, config.hash_bit, config.bs, config.num_label, config.use_raw_img)

    mapi2t = calc_map_k(qX, dY, qlabels, dlabels)
    mapt2i = calc_map_k(qY, dX, qlabels, dlabels)

    return mapi2t, mapt2i


if __name__ == '__main__':
    config = get_config()
    config.db_num = config.db_num // config.bs * config.bs
    config.query_num = config.query_num // config.bs * config.bs
    config.log_file += time.strftime('%Y%m%d_%H-%M-%S', time.localtime(time.time())) + '.log'
    run_train(config)