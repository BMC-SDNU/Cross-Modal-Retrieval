"""Evaluation"""

from __future__ import print_function
import os
import sys
import torch
import numpy as np

from model.DECL import DECL
from data import get_test_loader
from vocab import deserialize_vocab
from collections import OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""

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
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print, sub=0):
    """Encode all images and captions loadable by `data_loader`
    """
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    max_n_word = 0
    for i, (images, captions, lengths, ids, _) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))
    ids_ = []
    for i, (images, captions, lengths, ids, _) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        ids_ += ids
        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids, :, :] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        del images, captions
        if sub > 0:
            print(f"===>batch {i}")
        if sub > 0 and i > sub:
            break
    if sub > 0:
        return np.array(img_embs)[ids_].tolist(), np.array(cap_embs)[ids_].tolist(), np.array(cap_lens)[
            ids_].tolist(), ids_
    else:
        return img_embs, cap_embs, cap_lens


def shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100, mode="sim"):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                if mode == "sim":
                    sim = model.forward_sim(im, ca, l, mode)
                else:
                    _, sim, _ = model.forward_sim(im, ca, l, mode)  # Calculate evidence for retrieval

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return sims


def t2i(npts, sims, per_captions=1, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(per_captions * npts)
    top1 = np.zeros(per_captions * npts)
    top5 = np.zeros((per_captions * npts, 5), dtype=int)

    # --> (per_captions * N(caption), N(image))
    sims = sims.T
    retreivaled_index = []
    for index in range(npts):
        for i in range(per_captions):
            inds = np.argsort(sims[per_captions * index + i])[::-1]
            retreivaled_index.append(inds)
            ranks[per_captions * index + i] = np.where(inds == index)[0][0]
            top1[per_captions * index + i] = inds[0]
            top5[per_captions * index + i] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


def i2t(npts, sims, per_captions=1, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    top5 = np.zeros((npts, 5), dtype=int)
    retreivaled_index = []
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        retreivaled_index.append(inds)
        # Score
        rank = 1e20
        for i in range(per_captions * index, per_captions * index + per_captions, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        top5[index] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


def validation(opt, val_loader, model, fold=False):
    # compute the encoding for all the validation images and captions
    if opt.data_name == 'cc152k_precomp':
        per_captions = 1
    elif opt.data_name in ['coco_precomp', 'f30k_precomp']:
        per_captions = 5
    else:
        print(f"No dataset")
        return 0

    model.val_start()
    print('Encoding with model')
    img_embs, cap_embs, cap_lens = encode_data(model.similarity_model, val_loader)
    # clear duplicate 5*images and keep 1*images FIXME
    if not fold:
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])
        # record computation time of validation
        print('Computing similarity from model')
        sims_mean = shard_attn_scores(model.similarity_model, img_embs, cap_embs, cap_lens, opt, shard_size=1000,
                                      mode="not sim")
        print("Calculate similarity time with model")
        (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions, return_ranks=False)
        print("Average i2t Recall: %.2f" % ((r1 + r5 + r10) / 3))
        print("Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1, r5, r10, medr, meanr))
        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions, return_ranks=False)
        print("Average t2i Recall: %.2f" % ((r1i + r5i + r10i) / 3))
        print("Text to image: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1i, r5i, r10i, medri, meanr))
        r_sum = r1 + r5 + r10 + r1i + r5i + r10i
        print("Sum of Recall: %.2f" % (r_sum))
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
            sims = shard_attn_scores(model.similarity_model, img_embs_shard, cap_embs_shard, cap_lens_shard, opt,
                                     shard_size=1000,
                                     mode="not sim")

            print('Computing similarity from model')
            r, rt = i2t(img_embs_shard.shape[0], sims, per_captions, return_ranks=True)
            ri, rti = t2i(img_embs_shard.shape[0], sims, per_captions, return_ranks=True)

            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]
        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        a = np.array(mean_metrics)
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])
        print("rsum: %.1f" % (a[0:3].sum() + a[5:8].sum()))


def validation_dul(opt, val_loader, models, fold=False):
    # compute the encoding for all the validation images and captions
    if opt.data_name == 'cc152k_precomp':
        per_captions = 1
    elif opt.data_name in ['coco_precomp', 'f30k_precomp']:
        per_captions = 5
    else:
        print(f"No dataset")
        return 0

    models[0].val_start()
    models[1].val_start()
    print('Encoding with model')
    img_embs, cap_embs, cap_lens = encode_data(models[0].similarity_model, val_loader, opt.log_step)
    img_embs1, cap_embs1, cap_lens1 = encode_data(models[1].similarity_model, val_loader, opt.log_step)
    if not fold:
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])
        img_embs1 = np.array([img_embs1[i] for i in range(0, len(img_embs1), per_captions)])
        # record computation time of validation
        print('Computing similarity from model')
        sims_mean = shard_attn_scores(models[0].similarity_model, img_embs, cap_embs, cap_lens, opt, shard_size=1000,
                                      mode="not sim")
        sims_mean += shard_attn_scores(models[1].similarity_model, img_embs1, cap_embs1, cap_lens1, opt,
                                       shard_size=1000, mode="not sim")
        sims_mean /= 2
        print("Calculate similarity time with model")
        # caption retrieval
        (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions, return_ranks=False)
        print("Average i2t Recall: %.2f" % ((r1 + r5 + r10) / 3))
        print("Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1, r5, r10, medr, meanr))
        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions, return_ranks=False)
        print("Average t2i Recall: %.2f" % ((r1i + r5i + r10i) / 3))
        print("Text to image: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1i, r5i, r10i, medri, meanr))
        r_sum = r1 + r5 + r10 + r1i + r5i + r10i
        print("Sum of Recall: %.2f" % (r_sum))
        return r_sum
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]

            img_embs_shard1 = img_embs1[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard1 = cap_embs1[i * 5000:(i + 1) * 5000]
            cap_lens_shard1 = cap_lens1[i * 5000:(i + 1) * 5000]
            sims = shard_attn_scores(models[0].similarity_model, img_embs_shard, cap_embs_shard, cap_lens_shard, opt,
                                     shard_size=1000,
                                     mode="not sim")
            sims += shard_attn_scores(models[1].similarity_model, img_embs_shard1, cap_embs_shard1, cap_lens_shard1,
                                      opt,
                                      shard_size=1000,
                                      mode="not sim")
            sims /= 2

            print('Computing similarity from model')
            r, rt0 = i2t(img_embs_shard.shape[0], sims, per_captions, return_ranks=True)
            ri, rti0 = t2i(img_embs_shard.shape[0], sims, per_captions, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        a = np.array(mean_metrics)

        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])
        print("rsum: %.1f" % (a[0:3].sum() + a[5:8].sum()))


def eval_DECL(checkpoint_paths, avg_SGRAF=True, data_path=None, vocab_path=None):
    if avg_SGRAF is False:
        print(f"Load checkpoint from '{checkpoint_paths[0]}'")
        checkpoint = torch.load(checkpoint_paths[0])
        opt = checkpoint['opt']
        print(
            f"Noise ratio is {opt.noise_ratio}, module is {opt.module_name}, best validation epoch is {checkpoint['epoch']} ({checkpoint['best_rsum']})")
        if vocab_path != None:
            opt.vocab_path = vocab_path
        if data_path != None:
            opt.data_path = data_path
        vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
        model = DECL(opt)
        model.load_state_dict(checkpoint['model'])
        if 'coco' in opt.data_name:
            test_loader = get_test_loader('testall', opt.data_name, vocab, 100, 0, opt)
            validation(opt, test_loader, model=model, fold=True)
            validation(opt, test_loader, model=model, fold=False)
        else:
            test_loader = get_test_loader('test', opt.data_name, vocab, 100, 0, opt)
            validation(opt, test_loader, model=model, fold=False)
    else:
        assert len(checkpoint_paths) == 2
        print(f"Load checkpoint from '{checkpoint_paths}'")
        checkpoint0 = torch.load(checkpoint_paths[0])
        checkpoint1 = torch.load(checkpoint_paths[1])
        opt0 = checkpoint0['opt']
        opt1 = checkpoint1['opt']
        print(
            f"Noise ratios are {opt0.noise_ratio} and {opt1.noise_ratio}, "
            f"modules are {opt0.module_name} and {opt1.module_name}, best validation epochs are {checkpoint0['epoch']}"
            f" ({checkpoint0['best_rsum']}) and {checkpoint1['epoch']} ({checkpoint1['best_rsum']})")
        vocab = deserialize_vocab(os.path.join(opt0.vocab_path, '%s_vocab.json' % opt0.data_name))
        model0 = DECL(opt0)
        model1 = DECL(opt1)

        model0.load_state_dict(checkpoint0['model'])
        model1.load_state_dict(checkpoint1['model'])
        if 'coco' in opt0.data_name:
            test_loader = get_test_loader('testall', opt0.data_name, vocab, 100, 0, opt0)
            print(f'=====>model {opt0.module_name} fold:True')
            validation(opt0, test_loader, model0, fold=True)
            print(f'=====>model {opt1.module_name} fold:True')
            validation(opt0, test_loader, model1, fold=True)
            print(f'=====>model SGRAF fold:True')
            validation_dul(opt0, test_loader, models=[model0, model1], fold=True)

            print(f'=====>model {opt0.module_name} fold:False')
            validation(opt0, test_loader, model0, fold=False)
            print(f'=====>model {opt1.module_name} fold:False')
            validation(opt0, test_loader, model1, fold=False)
            print('=====>model SGRAF fold:False')
            validation_dul(opt0, test_loader, models=[model0, model1], fold=False)

        else:
            test_loader = get_test_loader('test', opt0.data_name, vocab, 100, 0, opt0)
            print(f'=====>model {opt0.module_name} fold:False')
            validation(opt0, test_loader, model0, fold=False)
            print(f'=====>model {opt1.module_name} fold:False')
            validation(opt0, test_loader, model1, fold=False)
            print('=====>model SGRAF fold:False')
            validation_dul(opt0, test_loader, models=[model0, model1], fold=False)
