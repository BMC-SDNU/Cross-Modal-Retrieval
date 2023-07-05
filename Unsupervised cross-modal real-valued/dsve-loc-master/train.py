"""
****************** COPYRIGHT AND CONFIDENTIALITY INFORMATION ******************
Copyright (c) 2018 [Thomson Licensing]
All Rights Reserved
This program contains proprietary information which is a trade secret/business \
secret of [Thomson Licensing] and is protected, even if unpublished, under \
applicable Copyright laws (including French droit d'auteur) and/or may be \
subject to one or more patent(s).
Recipient is to retain this program in confidence and is not permitted to use \
or make copies thereof other than as permitted in a written agreement with \
[Thomson Licensing] unless otherwise expressly allowed by applicable laws or \
by [Thomson Licensing] under express agreement.
Thomson Licensing is a company of the group TECHNICOLOR
*******************************************************************************
This scripts permits one to reproduce training and experiments of:
    Engilberge, M., Chevallier, L., Pérez, P., & Cord, M. (2018, April).
    Finding beans in burgers: Deep semantic-visual embedding with localization.
    In Proceedings of CVPR (pp. 3984-3993)

Author: Martin Engilberge
"""

import argparse
import os
import time

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from misc.dataset import CocoCaptionsRV
from misc.evaluation import eval_recall
from misc.loss import HardNegativeContrastiveLoss
from misc.model import joint_embedding
from misc.utils import AverageMeter, save_checkpoint, collate_fn_padded, log_epoch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR


device = torch.device("cuda")
# device = torch.device("cpu") # uncomment to run with cpu


def train(train_loader, model, criterion, optimizer, epoch, print_freq=1000):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (imgs, caps, lengths) in enumerate(train_loader):

        input_imgs, input_caps = imgs.to(device, non_blocking=True), caps.to(device, non_blocking=True)

        data_time.update(time.time() - end)

        optimizer.zero_grad()
        output_imgs, output_caps = model(input_imgs, input_caps, lengths)
        loss = criterion(output_imgs, output_caps)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), imgs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or i == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

    return losses.avg, batch_time.avg, data_time.avg


def validate(val_loader, model, criterion, print_freq=1000):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    imgs_enc = list()
    caps_enc = list()
    end = time.time()
    for i, (imgs, caps, lengths) in enumerate(val_loader):

        input_imgs, input_caps = imgs.to(device, non_blocking=True), caps.to(device, non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        with torch.no_grad():
            output_imgs, output_caps = model(input_imgs, input_caps, lengths)
            loss = criterion(output_imgs, output_caps)

        imgs_enc.append(output_imgs.cpu().data.numpy())
        caps_enc.append(output_caps.cpu().data.numpy())
        losses.update(loss.item(), imgs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or i == (len(val_loader) - 1):
            print('Data: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      i, len(val_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

    recall = eval_recall(imgs_enc, caps_enc)
    print(recall)
    return losses.avg, batch_time.avg, data_time.avg, recall


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("-n", '--name', default="model", help='Name of the model')
    parser.add_argument("-pf", dest="print_frequency", help="Number of element processed between print", type=int, default=1000)
    parser.add_argument("-bs", "--batch_size", help="The size of the batches", type=int, default=160)
    parser.add_argument("-lr", "--learning_rate", dest="lr", help="Initialization of the learning rate", type=float, default=0.001)
    parser.add_argument("-lrd", "--learning_rate_decrease", dest="lrd",
                        help="List of epoch where the learning rate is decreased (multiplied by first arg of lrd)", nargs='+', type=float, default=[0.5, 2, 3, 4, 5, 6])
    parser.add_argument("-fepoch", dest="fepoch", help="Epoch start finetuning resnet", type=int, default=8)
    parser.add_argument("-mepoch", dest="max_epoch", help="Max epoch", type=int, default=60)
    parser.add_argument('-sru', dest="sru", type=int, default=4)
    parser.add_argument("-de", dest="dimemb", help="Dimension of the joint embedding", type=int, default=2400)

    args = parser.parse_args()

    logger = SummaryWriter(os.path.join("./logs/", args.name))

    end = time.time()
    print("Initializing embedding ...", end=" ")
    join_emb = joint_embedding(args)

    # Text pipeline frozen at the begining
    for param in join_emb.cap_emb.parameters():
        param.requires_grad = False

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    prepro = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    prepro_val = transforms.Compose([
        transforms.Resize((350, 350)),
        transforms.ToTensor(),
        normalize,
    ])

    print("Done in: " + str(time.time() - end) + "s")

    end = time.time()
    print("Loading Data ...", end=" ")

    coco_data_train = CocoCaptionsRV(sset="trainrv", transform=prepro)
    coco_data_val = CocoCaptionsRV(sset="val", transform=prepro_val)

    train_loader = DataLoader(coco_data_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn_padded, pin_memory=True)
    val_loader = DataLoader(coco_data_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn_padded, pin_memory=True)
    print("Done in: " + str(time.time() - end) + "s")

    criterion = HardNegativeContrastiveLoss()

    join_emb.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, join_emb.parameters()), lr=args.lr)
    lr_scheduler = MultiStepLR(optimizer, args.lrd[1:], gamma=args.lrd[0])

    best_rec = 0
    for epoch in range(0, args.max_epoch):
        is_best = False

        train_loss, batch_train, data_train = train(train_loader, join_emb, criterion, optimizer, epoch, print_freq=args.print_frequency)
        val_loss, batch_val, data_val, recall = validate(val_loader, join_emb, criterion, print_freq=args.print_frequency)

        if(sum(recall[0]) + sum(recall[1]) > best_rec):
            best_rec = sum(recall[0]) + sum(recall[1])
            is_best = True

        state = {
            'epoch': epoch,
            'state_dict': join_emb.state_dict(),
            'best_rec': best_rec,
            'args_dict': args,
            'optimizer': optimizer.state_dict(),
        }

        log_epoch(logger, epoch, train_loss, val_loss, optimizer.param_groups[0]
                  ['lr'], batch_train, batch_val, data_train, data_val, recall)
        save_checkpoint(state, is_best, args.name, epoch)

        # Optimizing the text pipeline after one epoch
        if epoch == 1:
            for param in join_emb.cap_emb.parameters():
                param.requires_grad = True
            optimizer.add_param_group({'params': join_emb.cap_emb.parameters(), 'lr': optimizer.param_groups[0]
                                       ['lr'], 'initial_lr': args.lr})
            lr_scheduler = MultiStepLR(optimizer, args.lrd[1:], gamma=args.lrd[0])

        # Starting the finetuning of the whole model
        if epoch == args.fepoch:
            print("Sarting finetuning")
            finetune = True
            for param in join_emb.parameters():
                param.requires_grad = True

            # Keep the first layer of resnet frozen
            for i in range(0, 6):
                for param in join_emb.img_emb.module.base_layer[0][i].parameters():
                    param.requires_grad = False

            optimizer.add_param_group({'params': filter(lambda p: p.requires_grad, join_emb.img_emb.module.base_layer.parameters()), 'lr': optimizer.param_groups[0]
                                       ['lr'], 'initial_lr': args.lr})
            lr_scheduler = MultiStepLR(optimizer, args.lrd[1:], gamma=args.lrd[0])

        lr_scheduler.step(epoch)

    print('Finished Training')
