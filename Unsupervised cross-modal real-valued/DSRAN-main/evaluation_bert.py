# coding:utf-8
# -----------------------------------------------------------
# Dual Semantic Relations Attention Network (DSRAN) implementation based on
# "VSE++: Improving Visual-Semantic Embeddings with Hard Negatives"
# "Learning Dual Semantic Relations with Graph Attention for Image-Text Matching"
# Keyu Wen, Xiaodong Gu, and Qingrong Cheng
# IEEE Transactions on Circuits and Systems for Video Technology, 2020
# Writen by Keyu Wen, 2020
# ------------------------------------------------------------

from __future__ import print_function
import numpy
from data_bert import get_test_loader
import time
import numpy as np
import torch
import argparse
from model_bert import VSE
from collections import OrderedDict


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        if self.count == 0:
            return str(self.val)
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    def __init__(self):
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    batch_time = AverageMeter()
    val_logger = LogCollector()
    model.val_start()

    end = time.time()

    img_embs = None
    cap_embs = None
    time_encode_start = time.time()
    # device = torch.device("cuda:0")
    with torch.no_grad():
        for i, (images, images_orig, img_pos, captions, ids) in enumerate(data_loader):
            model.logger = val_logger

            img_emb, cap_emb = model.forward_emb(images_orig, images, img_pos, captions)

            if img_embs is None:
                img_embs = torch.zeros(len(data_loader.dataset), img_emb.size(1)).cuda()
                cap_embs = torch.zeros(len(data_loader.dataset), cap_emb.size(1)).cuda()

            img_embs[ids] = img_emb
            cap_embs[ids] = cap_emb

            model.forward_loss(img_emb, cap_emb)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_step == 0:
                logging('Test: [{0}/{1}]\t'
                        '{e_log}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                    i, len(data_loader), batch_time=batch_time,
                    e_log=str(model.logger)))
            del images, captions
    time_encode_end = time.time()
    print('encode_time:%f' % (time_encode_end - time_encode_start))
    img_emb_new = img_embs[0:img_embs.size(0):5]
    sims = torch.mm(img_emb_new, cap_embs.t())
    sims = sims.cpu().numpy()
        
    return img_embs, cap_embs, sims


def evalrank(model_path, data_path=None, split='dev', fold5=False, region_bbox_file=None, feature_path=None):
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    
    if data_path is not None:
        opt.data_path = data_path
    if data_path is not None:
        opt.region_bbox_file = region_bbox_file
    if data_path is not None:
        opt.feature_path = feature_path

    print(opt)
    model = VSE(opt)

    model.load_state_dict(checkpoint['model'])  #

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs, sims = encode_data(model, data_loader)

    time_sim_start = time.time()

    if not fold5:
        img_emb_new = img_embs[0:img_embs.size(0):5]
        print(img_emb_new.size())
        sims = torch.mm(img_emb_new, cap_embs.t())
        sims_T = torch.mm(cap_embs, cap_embs.t())
        sims_T = sims_T.cpu().numpy()

        sims = sims.cpu().numpy()
        np.save('sims_f.npy',sims)
        np.save('sims_f_T.npy',sims_T)

        print('Images: %d, Captions: %d' %
              (img_embs.shape[0] / 5, cap_embs.shape[0]))

        r = simrank(sims)

        time_sim_end = time.time()
        print('sims_time:%f' % (time_sim_end - time_sim_start))
        del sims
    else: # fold5-especially for coco
        print('5k---------------')
        img_emb_new = img_embs[0:img_embs.size(0):5]
        print(img_emb_new.size())

        sims = torch.mm(img_emb_new, cap_embs.t())
        sims_T = torch.mm(cap_embs, cap_embs.t())

        sims = sims.cpu().numpy()
        sims_T = sims_T.cpu().numpy()

        np.save('sims_full_5k.npy',sims)
        np.save('sims_full_T_5k.npy',sims_T)
        print('Images: %d, Captions: %d' %
              (img_embs.shape[0] / 5, cap_embs.shape[0]))

        r = simrank(sims)

        time_sim_end = time.time()
        print('sims_time:%f' % (time_sim_end - time_sim_start))
        del sims, sims_T
        print('1k---------------')
        r_ = [0, 0, 0, 0, 0, 0, 0]
        for i in range(5):
            print(i)
            img_emb_new = img_embs[i * 5000 : int(i * 5000 + img_embs.size(0)/5):5]
            cap_emb_new = cap_embs[i * 5000 : int(i * 5000 + cap_embs.size(0)/5)]

            sims = torch.mm(img_emb_new, cap_emb_new.t())
            sims_T = torch.mm(cap_emb_new, cap_emb_new.t())
            sims_T = sims_T.cpu().numpy()
            sims = sims.cpu().numpy()
            np.save('sims_full_%d.npy'%i,sims)
            np.save('sims_full_T_%d'%i,sims_T)

            print('Images: %d, Captions: %d' %
                  (img_emb_new.size(0), cap_emb_new.size(0)))

            r = simrank(sims)
            r_ = np.array(r_) + np.array(r)

            del sims
            print('--------------------')
        r_ = tuple(r_/5)
        print('I2T:%.1f %.1f %.1f' % r_[0:3])
        print('T2I:%.1f %.1f %.1f' % r_[3:6])
        print('Rsum:%.1f' % r_[-1])


def i2t(images, captions, npts=None, return_ranks=False):
    if npts is None:
        npts = int(images.shape[0] / 5)
        print(npts)
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)

    for index in range(npts):

        im = images[5 * index].reshape(1, images.shape[1])

        d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, return_ranks=False):
    if npts is None:
        npts = int(images.shape[0] / 5)
        print(npts)
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)

    for index in range(npts):
        queries = captions[5 * index:5 * index + 5]
        print('3')

        d = np.dot(queries, ims.T)

        inds = numpy.zeros(d.shape)
        print('5')
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def simrank(similarity):
    sims = similarity  # similarity matrix 1k*5k
    # print(sims)
    img_size, cap_size = sims.shape
    print("imgs: %d, caps: %d" % (img_size, cap_size))
    # time.sleep(10)
    # i2t
    index_list = []
    ranks = numpy.zeros(img_size)
    top1 = numpy.zeros(img_size)
    for index in range(img_size):
        d = sims[index]
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    print('i2t:r1: %.1f, r5: %.1f, r10: %.1f' % (r1, r5, r10))  # , medr, meanr)
    rs = r1 + r5 + r10
    # t2i
    sims_t2i = sims.T
    ranks = numpy.zeros(cap_size)
    top1 = numpy.zeros(cap_size)
    for index in range(img_size):

        d = sims_t2i[5 * index:5 * index + 5]  # 5*1000
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    r1_ = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5_ = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10_ = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr_ = numpy.floor(numpy.median(ranks)) + 1
    meanr_ = ranks.mean() + 1
    rs_ = r1_ + r5_ + r10_
    print('t2i:r1: %.1f, r5: %.1f, r10: %.1f' % (r1_, r5_, r10_))
    rsum = rs + rs_
    print('rsum=%.1f' % rsum)
    return [r1, r5, r10, r1_, r5_, r10_, rsum]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='single_model', help='model name')
    parser.add_argument('--fold', action='store_true', help='fold5')
    parser.add_argument('--name', default='model_best', help='checkpoint name')
    parser.add_argument('--data_path', default='data', help='data path')
    parser.add_argument('--region_bbox_file', default='data/joint-pretrain/flickr30k/region_feat_gvd_wo_bgd/flickr30k_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5', type=str, metavar='PATH',
                        help='path to region features bbox file')
    parser.add_argument('--feature_path', default='data/joint-pretrain/flickr30k/region_feat_gvd_wo_bgd/trainval/', type=str, metavar='PATH',
                        help='path to region features')
    opt = parser.parse_args()

    evalrank('runs/' + opt.model + '/' + opt.name + ".pth.tar", data_path = opt.data_path, split="test", fold5=opt.fold, region_bbox_file=opt.region_bbox_file, feature_path=opt.feature_path)

if __name__ == '__main__':
    main()
