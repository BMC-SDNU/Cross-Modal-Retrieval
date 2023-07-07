import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CubDataset, CubTextDataset
from model import resnet50
from centerloss import CenterLoss
from train import *
from validate import *

def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch HSE Deployment')
    parser.add_argument('--gpu', default=4, type=int, required=True, help='GPU nums to use')
    parser.add_argument('--workers', default=4, type=int, required=True, metavar='N',help='number of data loading workers')
    parser.add_argument('--epochs', default=50, type=int, required=True, metavar='N',help='number of total epochs to run')
    parser.add_argument('--snapshot', default='./pretrained/', type=str, required=True, metavar='PATH',help='path to latest checkpoint')
    parser.add_argument('--batch_size', default=4, type=int,metavar='N', required=True, help='mini-batch size')
    parser.add_argument('--data_path', default='./dataset/', type=str, required=True, help='path to dataset')
    parser.add_argument('--model_path', default='./model/', type=str, required=True, help='path to model')
    parser.add_argument('--crop_size', default=448, type=int, help='crop size')
    parser.add_argument('--scale_size', default=512, type=int, help='the size of the rescale image')
    parser.add_argument('--print_freq', default=1000, type=int, metavar='N', help='print frequency')
    parser.add_argument('--eval_epoch', default=1, type=int, help='every eval_epoch we will evaluate')
    parser.add_argument('--eval_epoch_thershold', default=2, type=int, help='eval_epoch_thershold')
    parser.add_argument('--loss_choose', default='c', type=str, required=True, help='choose loss(c:centerloss, r:rankingloss)')
    
    args = parser.parse_args()
    return args

def print_args(args):
    print ("==========================================")
    print ("==========       CONFIG      =============")
    print ("==========================================")
    for arg,content in args.__dict__.items():
        print("{}:{}".format(arg,content))
    print ("\n")

def main():
    args = arg_parse()
    print_args(args)

    print("==> Creating dataloader...")
    
    data_dir = args.data_path
    train_list = './list/image/train.txt'
    train_loader = get_train_set(data_dir, train_list, args)
    train_list1 = './list/video/train.txt'
    train_loader1 = get_train_set(data_dir, train_list1, args)
    train_list = './list/audio/train.txt'
    train_loader2 = get_train_set(data_dir, train_list, args)
    train_list3 = './list/text/train.txt'
    train_loader3 = get_text_set(data_dir, train_list3, args, 'train')

    test_list = './list/image/test.txt'
    test_loader = get_test_set(data_dir, test_list, args)
    test_list1 = './list/video/test.txt'
    test_loader1 = get_test_set(data_dir, test_list1, args)
    test_list = './list/audio/test.txt'
    test_loader2 = get_test_set(data_dir, test_list, args)
    test_list3 = './list/text/test.txt'
    test_loader3 = get_text_set(data_dir, test_list3, args, 'test')

    print("==> Loading the network ...")
    model = resnet50(num_classes=200)

    if args.gpu is not None:
        model = nn.DataParallel(model, device_ids=range(args.gpu))
        model = model.cuda()
        cudnn.benchmark = True

    if os.path.isfile(args.snapshot):
        print("==> loading checkpoint '{}'".format(args.snapshot))
        checkpoint = torch.load(args.snapshot)
        model_dict = model.state_dict()
        restore_param = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(restore_param)
        model.load_state_dict(model_dict)
        print("==> loaded checkpoint '{}'".format(args.snapshot))
    else:
        print("==> no checkpoint found at '{}'".format(args.snapshot))
        exit()

    criterion = nn.CrossEntropyLoss()
    center_loss = CenterLoss(200, 200, True)

    params = list(model.parameters()) + list(center_loss.parameters())
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    savepath = args.model_path
    if not os.path.exists(savepath):
       os.makedirs(savepath)

    for epoch in range(args.epochs):
        scheduler.step()
        
        train(train_loader, train_loader1, train_loader2, train_loader3, args, model, criterion, center_loss, optimizer, epoch, args.epochs)
        
        print('-' * 20)
        print("Image Acc:")
        image_acc = validate(test_loader, model, args, False)
        print("Text Acc:")
        text_acc = validate(test_loader3, model, args, True)
    
        save_model_path = savepath + 'epoch_' + str(epoch) + '_' + str(image_acc) +'.pkl'
        torch.save(model.state_dict(), save_model_path)

def get_train_set(data_dir, train_list, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_size = args.crop_size
    scale_size = args.scale_size
    train_data_transform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    train_set = CubDataset(data_dir, train_list, train_data_transform)
    train_loader = DataLoader(dataset=train_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    return train_loader

def get_test_set(data_dir, test_list, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_size = args.crop_size
    scale_size = args.scale_size
    test_data_transform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    test_set = CubDataset(data_dir, test_list, test_data_transform)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    return test_loader

def get_text_set(data_dir, test_list, args, split):
    data_set = CubTextDataset(data_dir, test_list, split)
    data_loader = DataLoader(dataset=data_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    return data_loader

if __name__ == "__main__":
    main()