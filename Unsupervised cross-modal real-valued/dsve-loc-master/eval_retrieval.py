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
import time

import torch
import torchvision.transforms as transforms

from misc.dataset import CocoCaptionsRV
from misc.evaluation import eval_recall
from misc.model import joint_embedding
from misc.utils import collate_fn_padded
from torch.utils.data import DataLoader


device = torch.device("cuda")
# device = torch.device("cpu") # uncomment to run with cpu

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate the model on cross modal retrieval task')
    parser.add_argument("-p", '--path', dest="model_path", help='Path to the weights of the model to evaluate')
    parser.add_argument("-bs", "--batch_size", help="The size of the batches", type=int, default=64)
    parser.add_argument('-tr', "--train", dest="dset", action='store_const', const="train", help="Using training dataset instead of validation", default="val")
    parser.add_argument('-te', "--test", dest="dset", action='store_const', const="test", help="Using test dataset instead of validation", default="val")

    args = parser.parse_args()

    print("Loading model from:", args.model_path)
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)

    join_emb = joint_embedding(checkpoint['args_dict'])
    join_emb.load_state_dict(checkpoint["state_dict"])

    for param in join_emb.parameters():
        param.requires_grad = False

    join_emb.to(device)
    join_emb.eval()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    prepro_val = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = CocoCaptionsRV(sset=args.dset, transform=prepro_val)

    print("Dataset size: ", len(dataset))

    dataset_loader = DataLoader(dataset, batch_size=args.batch_size,
                                num_workers=6, collate_fn=collate_fn_padded, pin_memory=True)

    imgs_enc = list()
    caps_enc = list()

    print("### Beginning of evaluation ###")
    end = time.time()
    for i, (imgs, caps, lengths) in enumerate(dataset_loader, 0):
        input_imgs, input_caps = imgs.to(device), caps.to(device)

        with torch.no_grad():
            output_imgs, output_caps = join_emb(input_imgs, input_caps, lengths)

        imgs_enc.append(output_imgs.cpu().data.numpy())
        caps_enc.append(output_caps.cpu().data.numpy())

        if i % 100 == 99:
            print(str((i + 1) * args.batch_size) + "/" + str(len(dataset)) + " pairs encoded - Time per batch: " + str((time.time() - end)) + "s")

        end = time.time()

    print(args.model_path, args.dset, eval_recall(imgs_enc, caps_enc))
