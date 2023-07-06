import os
import torch
from torch import autograd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset import Dataset
from config import opt
from models.CPAH import CPAH
from torch.optim import Adam
from utils import calc_map_k, pr_curve, p_top_k, Visualizer, write_pickle, pr_curve3, p_top_k2, calc_map_k2, \
    calc_map_rad2
from datasets.data_handler import load_data
import time

"""
Xie, De, et al. "Multi-Task Consistency-Preserving Adversarial Hashing for Cross-Modal Retrieval."
IEEE Transactions on Image Processing 29 (2020): 3626-3637.
DOI: 10.1109/TMM.2020.2969792
"""

from dataset.data_handler import get_dataloaders, DataHandlerAugmentedTxtImg
from dataset.dataset_ucm import *
from torch.nn.functional import one_hot


def logger():
    """
    Instantiate logger

    :return:
    """
    import logging

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    log_name = 'log.txt'
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    txt_log = logging.FileHandler(os.path.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    return logger


log = logger()


def stack_idxs(idxs, idxs_batch):
    if len(idxs) == 0:
        return [ib for ib in idxs_batch]
    else:
        return [torch.hstack(i).detach() for i in zip(idxs, idxs_batch)]


def generate_codes_from_dataloader(model, dataloader):
    """
    Generate binary codes from duplet dataloader

    :param: dataloader: duplet dataloader

    :returns: hash codes for given duplet dataloader, image replication factor of dataset
    """
    num = len(dataloader.dataset)

    irf = dataloader.dataset.image_replication_factor

    Bi = torch.zeros(num, opt.bit).to(opt.device)
    Bt = torch.zeros(num, opt.bit).to(opt.device)
    L = torch.zeros(num, opt.num_label).to(opt.device)

    dataloader_idxs = []

    # for i, input_data in tqdm(enumerate(test_dataloader)):
    for i, (idx, sample_idxs, img, txt, label) in enumerate(dataloader):
        dataloader_idxs = stack_idxs(dataloader_idxs, sample_idxs)
        img = img.to(opt.device)
        txt = txt.to(opt.device)
        if len(label.shape) == 1:
            label = one_hot(label, num_classes=opt.num_label).to(opt.device)
        else:
            label.to(opt.device)
        bi = model.generate_img_code(img)
        bt = model.generate_txt_code(txt)
        idx_end = min(num, (i + 1) * opt.batch_size)
        Bi[i * opt.batch_size: idx_end, :] = bi.data
        Bt[i * opt.batch_size: idx_end, :] = bt.data
        L[i * opt.batch_size: idx_end, :] = label.data

    Bi = torch.sign(Bi)
    Bt = torch.sign(Bt)
    return Bi, Bt, L, irf, dataloader_idxs


def get_each_nth_element(arr, n):
    """
    intentionally ugly solution, needed to avoid query replications during test/validation

    :return: array
    """
    return arr[::n]


def get_codes_labels_indexes(model, dataloader_q, dataloader_db, remove_replications=True):
    """
    Generate binary codes from duplet dataloaders for query and response

    :param: remove_replications: remove replications from dataset

    :returns: hash codes and labels for query and response, sample indexes
    """
    # hash X, hash Y, labels X/Y, image replication factor, indexes X, indexes Y
    qBX, qBY, qLXY, irf_q, (qIX, qIY) = generate_codes_from_dataloader(model, dataloader_q)
    # hash X, hash Y, labels X/Y, image replication factor, indexes X, indexes Y
    rBX, rBY, rLXY, irf_db, (rIX, rIY) = generate_codes_from_dataloader(model, dataloader_db)

    # get Y Labels
    qLY = qLXY
    rLY = rLXY

    # X modality sometimes contains replicated samples (see datasets), remove them by selecting each nth element
    # remove replications for hash codes
    qBX = get_each_nth_element(qBX, irf_q)
    rBX = get_each_nth_element(rBX, irf_db)
    # remove replications for labels
    qLX = get_each_nth_element(qLXY, irf_q)
    rLX = get_each_nth_element(rLXY, irf_db)
    # remove replications for indexes
    qIX = get_each_nth_element(qIX, irf_q)
    rIX = get_each_nth_element(rIX, irf_db)

    return qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, (qIX, qIY, rIX, rIY)


def calc_maps_k_glob(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, k):
    mapi2t = calc_map_k2(qBX, rBY, qLX, rLY, k)
    mapt2i = calc_map_k2(qBY, rBX, qLY, rLX, k)
    mapi2i = calc_map_k2(qBX, rBX, qLX, rLX, k)
    mapt2t = calc_map_k2(qBY, rBY, qLY, rLY, k)

    avg = (mapi2t.item() + mapt2i.item() + mapi2i.item() + mapt2t.item()) * 0.25

    mapi2t, mapt2i, mapi2i, mapt2t, mapavg = mapi2t.item(), mapt2i.item(), mapi2i.item(), mapt2t.item(), avg

    s = 'Valid: mAP@{:2d}, avg: {:3.3f}, i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
    log.info(s.format(k, mapavg, mapi2t, mapt2i, mapi2i, mapt2t))

    return mapi2t, mapt2i, mapi2i, mapt2t, mapavg


def calc_maps_rad_glob(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, rs):
    mapsi2t = calc_map_rad2(qBX, rBY, qLX, rLY)
    mapst2i = calc_map_rad2(qBY, rBX, qLY, rLX)
    mapsi2i = calc_map_rad2(qBX, rBX, qLX, rLX)
    mapst2t = calc_map_rad2(qBY, rBY, qLY, rLY)

    mapsi2t, mapst2i, mapsi2i, mapst2t = mapsi2t.numpy(), mapst2i.numpy(), mapsi2i.numpy(), mapst2t.numpy()

    s = 'Valid: mAP HR{}, i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
    for r in rs:
        log.info(s.format(r, mapsi2t[r], mapst2i[r], mapsi2i[r], mapst2t[r]))

    return mapsi2t, mapst2i, mapsi2i, mapst2t


def calc_p_top_k_glob(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY):
    k = [1, 5, 10, 20, 50] + list(range(100, 1001, 100))

    pk_i2t = p_top_k2(qBX, rBY, qLX, rLY, k, tqdm_label='I2T')
    pk_t2i = p_top_k2(qBY, rBX, qLY, rLX, k, tqdm_label='T2I')
    pk_i2i = p_top_k2(qBX, rBX, qLX, rLX, k, tqdm_label='I2I')
    pk_t2t = p_top_k2(qBY, rBY, qLY, rLY, k, tqdm_label='T2T')

    pk_dict = {'k': k,
               'pki2t': pk_i2t,
               'pkt2i': pk_t2i,
               'pki2i': pk_i2i,
               'pkt2t': pk_t2t}

    log.info('P@K values: {}'.format(pk_dict))

    return pk_dict


def update_maps_dict(maps, mapi2t, mapt2i, mapi2i, mapt2t, mapavg):
    maps['i2t'].append(mapi2t)
    maps['t2i'].append(mapt2i)
    maps['i2i'].append(mapi2i)
    maps['t2t'].append(mapt2t)
    maps['avg'].append(mapavg)


def update_max_maps_dict(maps_max, mapi2t, mapt2i, mapi2i, mapt2t, mapavg):
    """
    Update max MAPs dictionary (replace values)

    :param: mapi2t: I-to-T MAP
    :param: mapt2i: T-to-I MAP
    :param: mapi2i: I-to-I MAP
    :param: mapt2t: T-to-T MAP
    :param: mapavg: average MAP
    """
    maps_max['i2t'] = mapi2t
    maps_max['t2i'] = mapt2i
    maps_max['i2i'] = mapi2i
    maps_max['t2t'] = mapt2t
    maps_max['avg'] = mapavg


def train2(**kwargs):
    since = time.time()
    opt.parse(kwargs)

    s = 'Init ({}): {}, {} bits, proc: {}, {}'
    log.info(s.format('CPAH', opt.flag.upper(), opt.bit, opt.proc, 'TRAIN'))

    if (opt.device is None) or (opt.device == 'cpu'):
        opt.device = torch.device('cpu')
    else:
        opt.device = torch.device(opt.device)

    if opt.use_aug_data:
        train_class = DatasetQuadrupletAugmentedTxtImgDouble
    else:
        train_class = DatasetQuadrupletAugmentedTxtImg

    dl_train, dl_q, dl_db = get_dataloaders(DataHandlerAugmentedTxtImg, train_class, DatasetDuplet1, DatasetDuplet1)

    model = CPAH(opt.image_dim, opt.text_dim, opt.hidden_dim, opt.bit, opt.num_label).to(opt.device)

    optimizer_gen = Adam([
        {'params': model.image_module.parameters()},
        {'params': model.text_module.parameters()},
        {'params': model.hash_module.parameters()},
        {'params': model.mask_module.parameters()},
        {'params': model.consistency_dis.parameters()},
        {'params': model.classifier.parameters()},
    ], lr=opt.lr, weight_decay=0.0005)

    optimizer_dis = Adam(model.feature_dis.parameters(), lr=opt.lr, betas=(0.5, 0.9), weight_decay=0.0001)

    # tri_loss = TripletLoss(opt, reduction='sum')
    loss_bce = torch.nn.BCELoss(reduction='sum')
    loss_ce = torch.nn.CrossEntropyLoss(reduction='sum')

    loss = []
    losses = []

    dataset_size = len(dl_train.dataset)

    B = torch.randn(dataset_size, opt.bit).sign().to(opt.device)

    H_i = torch.zeros(dataset_size, opt.bit).to(opt.device)
    H_t = torch.zeros(dataset_size, opt.bit).to(opt.device)

    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit) + str(opt.proc)
    maps_max = {'i2t': 0., 't2i': 0., 'i2i': 0., 't2t': 0., 'avg': 0.}
    maps = {'i2t': [], 't2i': [], 'i2i': [], 't2t': [], 'avg': []}

    #################################################################

    for epoch in range(opt.max_epoch):
        t1 = time.time()
        e_loss = 0
        e_losses = {'adv': 0, 'class': 0, 'quant': 0, 'pairwise': 0}
        # for i, (ind, img, txt, label) in tqdm(enumerate(train_dataloader)):
        for i, (ind, sample_idx, img, img_aug, txt, txt_aug, label) in enumerate(dl_train):
            # print(i)
            imgs = img.to(opt.device)
            txt = txt.to(opt.device)
            labels = one_hot(torch.tensor(label), num_classes=opt.num_label)
            labels = labels.to(opt.device).float()
            # labels = label.to(opt.device)

            batch_size = len(ind)

            h_img, h_txt, f_rc_img, f_rc_txt, f_rp_img, f_rp_txt = model(imgs, txt)

            H_i[ind, :] = h_img
            H_t[ind, :] = h_txt

            ###################################
            # train discriminator. CPAH paper: (5)
            ###################################
            # IMG - real, TXT - fake
            # train with real (IMG)
            optimizer_dis.zero_grad()

            d_real = model.dis_D(f_rc_img.detach())
            d_real = -torch.log(torch.sigmoid(d_real)).mean()
            d_real.backward()

            # train with fake (TXT)
            d_fake = model.dis_D(f_rc_txt.detach())
            d_fake = -torch.log(torch.ones(batch_size).to(opt.device) - torch.sigmoid(d_fake)).mean()
            d_fake.backward()

            # train with gradient penalty (GP)
            # interpolate real and fake data
            alpha = torch.rand(batch_size, opt.hidden_dim).to(opt.device)
            interpolates = alpha * f_rc_img.detach() + (1 - alpha) * f_rc_txt.detach()
            interpolates.requires_grad_()
            disc_interpolates = model.dis_D(interpolates)
            # get gradients with respect to inputs
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).to(opt.device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            # calculate penalty
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  # 10 is GP hyperparameter
            gradient_penalty.backward()

            optimizer_dis.step()

            ###################################
            # train generator
            ###################################

            # adversarial loss, CPAH paper: (6)
            loss_adver = -torch.log(torch.sigmoid(model.dis_D(f_rc_txt))).mean()  # don't detach from graph

            # consistency classification loss, CPAH paper: (7)
            f_r = torch.vstack([f_rc_img, f_rc_txt, f_rp_img, f_rp_txt])
            l_r = [1] * len(ind) * 2 + [0] * len(ind) + [2] * len(ind)  # labels
            l_r = torch.tensor(l_r).to(opt.device)
            loss_consistency_class = loss_ce(f_r, l_r)

            # classification loss, CPAH paper: (8)
            l_f_rc_img = model.dis_classify(f_rc_img, 'img')
            l_f_rc_txt = model.dis_classify(f_rc_txt, 'txt')
            loss_class = loss_bce(l_f_rc_img, labels) + loss_bce(l_f_rc_txt, labels)
            # loss_class = torch.tensor(0).to(opt.device)

            # pairwise loss, CPAH paper: (10)
            S = (labels.mm(labels.T) > 0).float()
            # theta = 0.5 * ((h_img.mm(h_txt.T) + h_txt.mm(h_img.T)) / 2)  # not completely sure
            theta = 0.5 * h_img.mm(h_txt.T)
            # theta.retain_grad()
            # theta.register_hook(lambda x: print("theta  :", torch.max(x), torch.min(x), torch.mean(x)))
            e_theta = torch.exp(theta)
            # e_theta.retain_grad()
            # e_theta.register_hook(lambda x: print("theta  :", torch.max(x), torch.min(x), torch.mean(x)))
            loss_pairwise = -torch.sum(S * theta - torch.log(1 + e_theta))

            # quantization loss, CPAH paper: (11)
            loss_quant = torch.sum(torch.pow(B[ind, :] - h_img, 2)) + torch.sum(torch.pow(B[ind, :] - h_txt, 2))
            # loss_quant = torch.tensor(0).to(opt.device)

            err = 100 * loss_adver + opt.alpha * (
                    loss_consistency_class + loss_class) + loss_pairwise + opt.beta * loss_quant

            e_losses['adv'] += 100 * loss_adver.detach().cpu().numpy()
            e_losses['class'] += (opt.alpha * (loss_consistency_class + loss_class)).detach().cpu().numpy()
            e_losses['pairwise'] += loss_pairwise.detach().cpu().numpy()
            e_losses['quant'] += loss_quant.detach().cpu().numpy()

            optimizer_gen.zero_grad()
            err.backward()
            optimizer_gen.step()

            e_loss = err + e_loss

        loss.append(e_loss.item())
        e_losses['sum'] = sum(e_losses.values())
        losses.append(e_losses)

        B = (0.5 * (H_i.detach() + H_t.detach())).sign()

        delta_t = time.time() - t1
        print('Epoch: {:4d}/{:4d}, time, {:3.3f}s, loss: {:15.3f},'.format(epoch + 1, opt.max_epoch, delta_t,
                                                                           loss[-1]) + 5 * ' ' + 'losses:', e_losses)

        # validate

        if opt.valid and (epoch + 1) % opt.valid_freq == 0:

            model.eval()

            qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes = get_codes_labels_indexes(model, dl_q, dl_db)

            mapi2t, mapt2i, mapi2i, mapt2t, mapavg = calc_maps_k_glob(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 5)

            map_k_5 = (mapi2t, mapt2i, mapi2i, mapt2t, mapavg)
            map_k_10 = calc_maps_k_glob(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 10)
            map_k_20 = calc_maps_k_glob(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 20)
            map_r = calc_maps_rad_glob(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, [0, 1, 2, 3, 4, 5])
            p_at_k = calc_p_top_k_glob(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
            maps_eval = (map_k_5, map_k_10, map_k_20, map_r, p_at_k)

            # visualize_retrieval(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes, 'UNHD')

            update_maps_dict(maps, mapi2t, mapt2i, mapi2i, mapt2t, mapavg)

            if mapavg > maps_max['avg']:
                update_max_maps_dict(maps_max, mapi2t, mapt2i, mapi2i, mapt2t, mapavg)

            save_model(model)
            write_pickle(os.path.join(path, 'maps_eval.pkl'), maps_eval)

            with torch.cuda.device(opt.device):
                torch.save([H_i, H_t], os.path.join(path, 'hash_maps_i_t.pth'))
            with torch.cuda.device(opt.device):
                torch.save(B, os.path.join(path, 'code_map.pth'))

            model.train()

        # decrease the lr to its one fifth every 30 epochs
        if epoch % 30 == 0:
            for params in optimizer_gen.param_groups:
                params['lr'] = max(params['lr'] * 0.2, 1e-6)

    if not opt.valid:
        save_model(model)

    time_elapsed = time.time() - since
    log.info('\n   Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    s = 'Max Avg MAP: {avg:3.3f}, Max MAPs: i->t: {i2t:3.3f}, t->i: {t2i:3.3f}, i->i: {i2i:3.3f}, t->t: {t2t:3.3f}\n\n\n\n\n'
    log.info(s.format(**maps_max))

    # write_pickle(os.path.join(path, 'res_dict.pkl'), res_dict)


def train(**kwargs):
    since = time.time()
    opt.parse(kwargs)

    if (opt.device is None) or (opt.device == 'cpu'):
        opt.device = torch.device('cpu')
    else:
        opt.device = torch.device(opt.device)

    images, tags, labels = load_data(opt.data_path, type=opt.dataset)
    train_data = Dataset(opt, images, tags, labels)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    L = train_data.get_labels()
    L = L.to(opt.device)
    # test
    i_query_data = Dataset(opt, images, tags, labels, test='image.query')
    i_db_data = Dataset(opt, images, tags, labels, test='image.db')
    t_query_data = Dataset(opt, images, tags, labels, test='text.query')
    t_db_data = Dataset(opt, images, tags, labels, test='text.db')

    i_query_dataloader = DataLoader(i_query_data, opt.batch_size, shuffle=False)
    i_db_dataloader = DataLoader(i_db_data, opt.batch_size, shuffle=False)
    t_query_dataloader = DataLoader(t_query_data, opt.batch_size, shuffle=False)
    t_db_dataloader = DataLoader(t_db_data, opt.batch_size, shuffle=False)

    query_labels, db_labels = i_query_data.get_labels()
    query_labels = query_labels.to(opt.device)
    db_labels = db_labels.to(opt.device)

    model = CPAH(opt.image_dim, opt.text_dim, opt.hidden_dim, opt.bit, opt.num_label).to(opt.device)

    # discriminator = DisModel(opt.hidden_dim, opt.num_label).to(opt.device)

    optimizer_gen = Adam([
        {'params': model.image_module.parameters()},
        {'params': model.text_module.parameters()},
        {'params': model.hash_module.parameters()},
        {'params': model.mask_module.parameters()},
        {'params': model.consistency_dis.parameters()},
        {'params': model.classifier.parameters()},
    ], lr=opt.lr, weight_decay=0.0005)

    optimizer_dis = Adam(model.feature_dis.parameters(), lr=opt.lr, betas=(0.5, 0.9), weight_decay=0.0001)

    # tri_loss = TripletLoss(opt, reduction='sum')
    loss_bce = torch.nn.BCELoss(reduction='sum')
    loss_ce = torch.nn.CrossEntropyLoss(reduction='sum')

    loss = []
    losses = []

    max_mapi2t = 0.
    max_mapt2i = 0.
    max_mapi2i = 0.
    max_mapt2t = 0.
    max_average = 0.

    mapt2i_list = []
    mapi2t_list = []
    mapi2i_list = []
    mapt2t_list = []
    train_times = []

    B = torch.randn(opt.training_size, opt.bit).sign().to(opt.device)

    H_i = torch.zeros(opt.training_size, opt.bit).to(opt.device)
    H_t = torch.zeros(opt.training_size, opt.bit).to(opt.device)

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(opt.max_epoch):
        t1 = time.time()
        e_loss = 0
        e_losses = {'adv': 0, 'class': 0, 'quant': 0, 'pairwise': 0}
        # for i, (ind, img, txt, label) in tqdm(enumerate(train_dataloader)):
        for i, (ind, img, txt, label) in enumerate(train_dataloader):
            # print(i)
            imgs = img.to(opt.device)
            txt = txt.to(opt.device)
            labels = label.to(opt.device)

            batch_size = len(ind)

            h_img, h_txt, f_rc_img, f_rc_txt, f_rp_img, f_rp_txt = model(imgs, txt)

            H_i[ind, :] = h_img
            H_t[ind, :] = h_txt

            ###################################
            # train discriminator. CPAH paper: (5)
            ###################################
            # IMG - real, TXT - fake
            # train with real (IMG)
            optimizer_dis.zero_grad()

            d_real = model.dis_D(f_rc_img.detach())
            d_real = -torch.log(torch.sigmoid(d_real)).mean()
            d_real.backward()

            # train with fake (TXT)
            d_fake = model.dis_D(f_rc_txt.detach())
            d_fake = -torch.log(torch.ones(batch_size).to(opt.device) - torch.sigmoid(d_fake)).mean()
            d_fake.backward()

            # train with gradient penalty (GP)
            # interpolate real and fake data
            alpha = torch.rand(batch_size, opt.hidden_dim).to(opt.device)
            interpolates = alpha * f_rc_img.detach() + (1 - alpha) * f_rc_txt.detach()
            interpolates.requires_grad_()
            disc_interpolates = model.dis_D(interpolates)
            # get gradients with respect to inputs
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).to(opt.device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            # calculate penalty
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  # 10 is GP hyperparameter
            gradient_penalty.backward()

            optimizer_dis.step()

            ###################################
            # train generator
            ###################################

            # adversarial loss, CPAH paper: (6)
            loss_adver = -torch.log(torch.sigmoid(model.dis_D(f_rc_txt))).mean()  # don't detach from graph

            # consistency classification loss, CPAH paper: (7)
            f_r = torch.vstack([f_rc_img, f_rc_txt, f_rp_img, f_rp_txt])
            l_r = [1] * len(ind) * 2 + [0] * len(ind) + [2] * len(ind)  # labels
            l_r = torch.tensor(l_r).to(opt.device)
            loss_consistency_class = loss_ce(f_r, l_r)

            # classification loss, CPAH paper: (8)
            l_f_rc_img = model.dis_classify(f_rc_img, 'img')
            l_f_rc_txt = model.dis_classify(f_rc_txt, 'txt')
            loss_class = loss_bce(l_f_rc_img, labels) + loss_bce(l_f_rc_txt, labels)
            # loss_class = torch.tensor(0).to(opt.device)

            # pairwise loss, CPAH paper: (10)
            S = (labels.mm(labels.T) > 0).float()
            # theta = 0.5 * ((h_img.mm(h_txt.T) + h_txt.mm(h_img.T)) / 2)  # not completely sure
            theta = 0.5 * h_img.mm(h_txt.T)
            # theta.retain_grad()
            # theta.register_hook(lambda x: print("theta  :", torch.max(x), torch.min(x), torch.mean(x)))
            e_theta = torch.exp(theta)
            # e_theta.retain_grad()
            # e_theta.register_hook(lambda x: print("theta  :", torch.max(x), torch.min(x), torch.mean(x)))
            loss_pairwise = -torch.sum(S * theta - torch.log(1 + e_theta))

            # quantization loss, CPAH paper: (11)
            loss_quant = torch.sum(torch.pow(B[ind, :] - h_img, 2)) + torch.sum(torch.pow(B[ind, :] - h_txt, 2))
            # loss_quant = torch.tensor(0).to(opt.device)

            err = 100 * loss_adver + opt.alpha * (
                        loss_consistency_class + loss_class) + loss_pairwise + opt.beta * loss_quant

            e_losses['adv'] += 100 * loss_adver.detach().cpu().numpy()
            e_losses['class'] += (opt.alpha * (loss_consistency_class + loss_class)).detach().cpu().numpy()
            e_losses['pairwise'] += loss_pairwise.detach().cpu().numpy()
            e_losses['quant'] += loss_quant.detach().cpu().numpy()

            optimizer_gen.zero_grad()
            err.backward()
            optimizer_gen.step()

            e_loss = err + e_loss

        loss.append(e_loss.item())
        e_losses['sum'] = sum(e_losses.values())
        losses.append(e_losses)

        B = (0.5 * (H_i.detach() + H_t.detach())).sign()

        delta_t = time.time() - t1
        print('Epoch: {:4d}/{:4d}, time, {:3.3f}s, loss: {:15.3f},'.format(epoch + 1, opt.max_epoch, delta_t,
                                                                           loss[-1]) + 5 * ' ' + 'losses:', e_losses)
        # validate
        if opt.valid and (epoch + 1) % opt.valid_freq == 0:
            mapi2t, mapt2i, mapi2i, mapt2t = valid(model, i_query_dataloader, i_db_dataloader, t_query_dataloader,
                                                   t_db_dataloader, query_labels, db_labels)
            print(
                'Epoch: {:4d}/{:4d}, validation MAP: MAP(i->t) = {:3.4f}, MAP(t->i) = {:3.4f}, MAP(i->i) = {:3.4f}, MAP(t->t) = {:3.4f}'.format(
                    epoch + 1, opt.max_epoch, mapi2t, mapt2i, mapi2i, mapt2t))

            mapi2t_list.append(mapi2t)
            mapt2i_list.append(mapt2i)
            mapi2i_list.append(mapi2i)
            mapt2t_list.append(mapt2t)
            train_times.append(delta_t)

            if 0.5 * (mapi2t + mapt2i) > max_average:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                max_mapi2i = mapi2i
                max_mapt2t = mapt2t
                max_average = 0.5 * (mapi2t + mapt2i)
                save_model(model)
                path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit) + str(opt.proc)
                with torch.cuda.device(opt.device):
                    torch.save([H_i, H_t], os.path.join(path, 'hash_maps_i_t.pth'))
                with torch.cuda.device(opt.device):
                    torch.save(B, os.path.join(path, 'code_map.pth'))

        # decrease the lr to its one fifth every 30 epochs
        if epoch % 30 == 0:
            for params in optimizer_gen.param_groups:
                params['lr'] = max(params['lr'] * 0.2, 1e-6)

        if epoch % 100 == 0:
            pass

    if not opt.valid:
        save_model(model)

    time_elapsed = time.time() - since
    print('\n   Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if opt.valid:
        print('   Max MAP: MAP(i->t) = {:3.4f}, MAP(t->i) = {:3.4f}, MAP(i->i) = {:3.4f}, MAP(t->t) = {:3.4f}'.format(
            max_mapi2t, max_mapt2i, max_mapi2i, max_mapt2t))
    else:
        mapi2t, mapt2i, mapi2i, mapt2t = valid(model, i_query_dataloader, i_db_dataloader, t_query_dataloader,
                                               t_db_dataloader, query_labels, db_labels)
        print('   Max MAP: MAP(i->t) = {:3.4f}, MAP(t->i) = {:3.4f}, MAP(i->i) = {:3.4f}, MAP(t->t) = {:3.4f}'.format(
            mapi2t, mapt2i, mapi2i, mapt2t))

    res_dict = {'mapi2t': mapi2t_list,
                'mapt2i': mapt2i_list,
                'mapi2i': mapi2i_list,
                'mapt2t': mapt2t_list,
                'epoch_times': train_times,
                'losses': losses}

    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit) + str(opt.proc)
    write_pickle(os.path.join(path, 'res_dict.pkl'), res_dict)


def test(**kwargs):
    opt.parse(kwargs)

    if opt.device is not None:
        opt.device = torch.device(opt.device)
    elif opt.gpus:
        opt.device = torch.device(0)
    else:
        opt.device = torch.device('cpu')

    with torch.no_grad():
        model = CPAH(opt.image_dim, opt.text_dim, opt.hidden_dim, opt.bit, opt.num_label).to(opt.device)

        path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit) + str(opt.proc)
        load_model(model, path)

        model.eval()

        images, tags, labels = load_data(opt.data_path, opt.dataset)

        i_query_data = Dataset(opt, images, tags, labels, test='image.query')
        i_db_data = Dataset(opt, images, tags, labels, test='image.db')
        t_query_data = Dataset(opt, images, tags, labels, test='text.query')
        t_db_data = Dataset(opt, images, tags, labels, test='text.db')

        i_query_dataloader = DataLoader(i_query_data, opt.batch_size, shuffle=False)
        i_db_dataloader = DataLoader(i_db_data, opt.batch_size, shuffle=False)
        t_query_dataloader = DataLoader(t_query_data, opt.batch_size, shuffle=False)
        t_db_dataloader = DataLoader(t_db_data, opt.batch_size, shuffle=False)

        qBX = generate_img_code(model, i_query_dataloader, opt.query_size)
        qBY = generate_txt_code(model, t_query_dataloader, opt.query_size)
        rBX = generate_img_code(model, i_db_dataloader, opt.db_size)
        rBY = generate_txt_code(model, t_db_dataloader, opt.db_size)

        query_labels, db_labels = i_query_data.get_labels()
        query_labels = query_labels.to(opt.device)
        db_labels = db_labels.to(opt.device)

        # K = [1, 10, 100, 1000]
        # p_top_k(qBX, rBY, query_labels, db_labels, K, tqdm_label='I2T')
        # pr_curve2(qBY, rBX, query_labels, db_labels)

        p_i2t, r_i2t = pr_curve(qBX, rBY, query_labels, db_labels, tqdm_label='I2T')
        p_t2i, r_t2i = pr_curve(qBY, rBX, query_labels, db_labels, tqdm_label='T2I')
        p_i2i, r_i2i = pr_curve(qBX, rBX, query_labels, db_labels, tqdm_label='I2I')
        p_t2t, r_t2t = pr_curve(qBY, rBY, query_labels, db_labels, tqdm_label='T2T')

        K = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
             10000]
        pk_i2t = p_top_k(qBX, rBY, query_labels, db_labels, K, tqdm_label='I2T')
        pk_t2i = p_top_k(qBY, rBX, query_labels, db_labels, K, tqdm_label='T2I')
        pk_i2i = p_top_k(qBX, rBX, query_labels, db_labels, K, tqdm_label='I2I')
        pk_t2t = p_top_k(qBY, rBY, query_labels, db_labels, K, tqdm_label='T2T')

        mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels)
        mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels)
        mapi2i = calc_map_k(qBX, rBX, query_labels, db_labels)
        mapt2t = calc_map_k(qBY, rBY, query_labels, db_labels)

        pr_dict = {'pi2t': p_i2t.cpu().numpy(), 'ri2t': r_i2t.cpu().numpy(),
                   'pt2i': p_t2i.cpu().numpy(), 'rt2i': r_t2i.cpu().numpy(),
                   'pi2i': p_i2i.cpu().numpy(), 'ri2i': r_i2i.cpu().numpy(),
                   'pt2t': p_t2t.cpu().numpy(), 'rt2t': r_t2t.cpu().numpy()}

        pk_dict = {'k': K,
                   'pki2t': pk_i2t.cpu().numpy(),
                   'pkt2i': pk_t2i.cpu().numpy(),
                   'pki2i': pk_i2i.cpu().numpy(),
                   'pkt2t': pk_t2t.cpu().numpy()}

        map_dict = {'mapi2t': float(mapi2t.cpu().numpy()),
                    'mapt2i': float(mapt2i.cpu().numpy()),
                    'mapi2i': float(mapi2i.cpu().numpy()),
                    'mapt2t': float(mapt2t.cpu().numpy())}

        print('   Test MAP: MAP(i->t) = {:3.4f}, MAP(t->i) = {:3.4f}, MAP(i->i) = {:3.4f}, MAP(t->t) = {:3.4f}'.format(
            mapi2t, mapt2i, mapi2i, mapt2t))

        path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit) + str(opt.proc)
        write_pickle(os.path.join(path, 'pr_dict.pkl'), pr_dict)
        write_pickle(os.path.join(path, 'pk_dict.pkl'), pk_dict)
        write_pickle(os.path.join(path, 'map_dict.pkl'), map_dict)


def generate_img_code(model, test_dataloader, num):
    B = torch.zeros(num, opt.bit).to(opt.device)

    # for i, input_data in tqdm(enumerate(test_dataloader)):
    for i, input_data in enumerate(test_dataloader):
        input_data = input_data.to(opt.device)
        b = model.generate_img_code(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def generate_txt_code(model, test_dataloader, num):
    B = torch.zeros(num, opt.bit).to(opt.device)

    # for i, input_data in tqdm(enumerate(test_dataloader)):
    for i, input_data in enumerate(test_dataloader):
        input_data = input_data.to(opt.device)
        b = model.generate_txt_code(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def valid(model, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader, query_labels, db_labels):
    model.eval()

    qBX = generate_img_code(model, x_query_dataloader, opt.query_size)
    qBY = generate_txt_code(model, y_query_dataloader, opt.query_size)
    rBX = generate_img_code(model, x_db_dataloader, opt.db_size)
    rBY = generate_txt_code(model, y_db_dataloader, opt.db_size)

    mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels)
    mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels)

    mapi2i = calc_map_k(qBX, rBX, query_labels, db_labels)
    mapt2t = calc_map_k(qBY, rBY, query_labels, db_labels)

    model.train()
    return mapi2t.item(), mapt2i.item(), mapi2i.item(), mapt2t.item()


def load_model(model, path):
    if path is not None:
        model.load(os.path.join(path, model.module_name + '.pth'))


def save_model(model):
    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit) + str(opt.proc)
    model.save(model.module_name + '.pth', path, cuda_device=opt.device)


def test2(**kwargs):
    opt.parse(kwargs)

    s = 'Init ({}): {}, {} bits, proc: {}, {}'
    log.info(s.format('CPAH', opt.flag.upper(), opt.bit, opt.proc, 'TEST'))

    if opt.device is not None:
        opt.device = torch.device(opt.device)
    elif opt.gpus:
        opt.device = torch.device(0)
    else:
        opt.device = torch.device('cpu')

    with torch.no_grad():
        # pretrain_model = load_pretrain_model(opt.pretrain_model_path)

        # generator = GEN(opt.image_dim, opt.text_dim, opt.hidden_dim, opt.bit, opt.num_label).to(opt.device)
        model = CPAH(opt.image_dim, opt.text_dim, opt.hidden_dim, opt.bit, opt.num_label).to(opt.device)

        path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit) + str(opt.proc)
        load_model(model, path)

        model.eval()

        dl_train, dl_q, dl_db = get_dataloaders(DataHandlerAugmentedTxtImg, DatasetQuadrupletAugmentedTxtImg,
                                                DatasetDuplet1, DatasetDuplet1)
        qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes = get_codes_labels_indexes(model, dl_q, dl_db)

        mapi2t, mapt2i, mapi2i, mapt2t, mapavg = calc_maps_k_glob(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 5)

        map_k_5 = (mapi2t, mapt2i, mapi2i, mapt2t, mapavg)
        map_k_10 = calc_maps_k_glob(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 10)
        map_k_20 = calc_maps_k_glob(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 20)
        map_r = calc_maps_rad_glob(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, [0, 1, 2, 3, 4, 5])
        p_at_k = calc_p_top_k_glob(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
        maps_eval = (map_k_5, map_k_10, map_k_20, map_r, p_at_k)

        model.train()



if __name__ == '__main__':
    import fire

    fire.Fire()
