# -----------------------------------------------------------
# Dual Semantic Relations Attention Network (DSRAN) implementation based on
# "VSE++: Improving Visual-Semantic Embeddings with Hard Negatives"
# "Learning Dual Semantic Relations with Graph Attention for Image-Text Matching"
# Keyu Wen, Xiaodong Gu, and Qingrong Cheng
# IEEE Transactions on Circuits and Systems for Video Technology, 2020
# Writen by Keyu Wen, 2020
# ------------------------------------------------------------

import pickle
import os
import time
import shutil
import torch
import data_bert as data
from model_bert import VSE
from evaluation_bert import i2t, t2i, AverageMeter, LogCollector, encode_data, simrank
import numpy as np
import logging
import tensorboard_logger as tb_logger
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data',
                        help='path to datasets')
    parser.add_argument('--data_name', default='coco',
                        help='{coco,f30k}')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=12, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--learning_rate', default=2e-5, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=6, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--logger_name', default='runs/grg',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--ft_res', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--bert_path', default='uncased_L-12_H-768_A-12/',
                        help='path of pre-trained BERT.')
    parser.add_argument('--ft_bert', action='store_true',
                        help='Fine-tune the text encoder.')
    parser.add_argument('--bert_size', default=768, type=int,
                        help='Dimensionality of the text embedding')
    parser.add_argument('--warmup', default=-1, type=float)
    parser.add_argument('--K', default=2, type=int,help='num of JSR.')
    parser.add_argument('--feature_path', default='data/joint-pretrain/flickr30k/region_feat_gvd_wo_bgd/trainval/',
                        type=str, help='path to the pre-computed image features')
    parser.add_argument('--region_bbox_file',
                        default='data/joint-pretrain/flickr30k/region_feat_gvd_wo_bgd/flickr30k_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5',
                        type=str, help='path to the region_bbox_file(.h5)')
    opt = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    train_loader, val_loader = data.get_loaders(opt.data_name, opt.batch_size, opt.workers, opt)
    opt.l_train = len(train_loader)
    print(opt)
    model = VSE(opt)
    best_rsum = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)[-1]
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    for epoch in range(opt.num_epochs):

        adjust_learning_rate(opt, model.optimizer, epoch)

        train(opt, train_loader, model, epoch, val_loader)

        rsum = validate(opt, val_loader, model)[-1]

        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, epoch, prefix=opt.logger_name + '/')


def train(opt, train_loader, model, epoch, val_loader):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    model.train_start()

    end = time.time()
    for i, train_data in enumerate(train_loader):

        data_time.update(time.time() - end)
        model.logger = train_logger
        model.train_emb(*train_data)
        batch_time.update(time.time() - end)
        end = time.time()

        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def validate(opt, val_loader, model):
    _, _, sims = encode_data(
        model, val_loader, opt.log_step, logging.info)
    rs = simrank(sims)
    del sims
    return rs


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
