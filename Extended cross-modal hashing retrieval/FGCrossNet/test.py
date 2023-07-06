import argparse
import os,sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CubDataset, CubTextDataset
from model import resnet50
from retrieval import *

def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch HSE Deployment')
    parser.add_argument('--gpu', default=2, type=int, help='GPU nums to use')
    parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--batch_size', default=25, type=int,metavar='N', help='mini-batch size')
    parser.add_argument('--data_path', default='./dataset/', type=str, required=True, help='path to dataset')
    parser.add_argument('--snapshot', default='./model/rankingloss/model.pkl', type=str, required=True, help='path to latest checkpoint')
    parser.add_argument('--feature', default='./feature', type=str, required=True, help='path to feature')
    parser.add_argument('--crop_size', default=448, type=int, help='crop size')
    parser.add_argument('--scale_size', default=512, type=int, help='the size of the rescale image')

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
    test_list1 = './list/image/test.txt'
    test_loader1 = get_test_set(data_dir, test_list1, args)
    test_list2 = './list/video/test.txt'
    test_loader2 = get_test_set(data_dir, test_list2, args)
    test_list3 = './list/audio/test.txt'
    test_loader3 = get_test_set(data_dir, test_list3, args)
    test_list4 = './list/text/test.txt'
    test_loader4 = get_text_set(data_dir, test_list4, args, 'test')
    
    out_feature_dir1 = os.path.join(args.feature, 'image')
    out_feature_dir2 = os.path.join(args.feature, 'video')
    out_feature_dir3 = os.path.join(args.feature, 'audio')
    out_feature_dir4 = os.path.join(args.feature, 'text')
    
    mkdir(out_feature_dir1)
    mkdir(out_feature_dir2)
    mkdir(out_feature_dir3)
    mkdir(out_feature_dir4)

    print("==> Loading the modelwork ...")
    model = resnet50(num_classes=200)
    model = model.cuda()

    '''
    if args.gpu is not None:
        model = nn.DataParallel(model, device_ids=range(args.gpu))
        model = model.cuda()
        cudnn.benchmark = True
    '''
    
    if args.snapshot:
        if os.path.isfile(args.snapshot):
            print("==> loading checkpoint '{}'".format(args.snapshot))
            checkpoint = torch.load(args.snapshot)
            model.load_state_dict(checkpoint)
            print("==> loaded checkpoint '{}'".format(args.snapshot))
        else:
            print("==> no checkpoint found at '{}'".format(args.snapshot))
            exit()

    model.eval()
    #model = model.module

    print("Image Features ...")
    img = extra(model, test_loader1, out_feature_dir1, args, flag='i')
    print("Video Features ...")
    vid = extra(model, test_loader2, out_feature_dir2, args, flag='v')
    print("Audio Features ...")
    aud = extra(model, test_loader3, out_feature_dir3, args, flag='a')
    print("Text Features ...")
    txt = extra(model, test_loader4, out_feature_dir4, args, flag='t')
    
    compute_mAP(img, vid, aud, txt)

def mkdir(out_feature_dir):
    if not os.path.exists(out_feature_dir):
       os.makedirs(out_feature_dir)

def extra(model, test_loader, out_feature_dir, args, flag):
    size = args.batch_size
    num = 0
    if(flag == 'v'):
        size = 1
        f = np.zeros((len(test_loader),200))
    else:
        f = np.zeros((len(test_loader)*size,200))
    for i, (input, target) in enumerate(test_loader):
        target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input).cuda()
        if(flag == 't'):
            output = model.forward_txt(input_var)
        else:
            output = model.forward_share(input_var)
        if(flag == 'v'):
            output = torch.mean(output,0).reshape(1,200)#video frame average
        output = F.softmax(output, dim=1).detach().cpu().numpy()
        num += output.shape[0]
        if(i == len(test_loader)-1):
            f[i*size:num,:] = output
        else:
            f[i*size:(i+1)*size,:] = output

    np.savetxt(out_feature_dir + '/features_te.txt', f[:num,:])
    return out_feature_dir + '/features_te.txt'

def get_test_set(data_dir,test_list,args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    crop_size = args.crop_size
    scale_size = args.scale_size
    test_data_transform = transforms.Compose([
          transforms.Resize((scale_size,scale_size)),
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

if __name__=="__main__":
    main()
