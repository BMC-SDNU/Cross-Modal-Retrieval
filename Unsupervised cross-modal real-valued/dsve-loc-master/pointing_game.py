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

import numpy as np
import torch
import torchvision.transforms as transforms

from misc.dataset import VgCaptions
from misc.localization import compute_pointing_game_acc
from misc.model import joint_embedding
from misc.utils import collate_fn_img_padded, collate_fn_cap_padded
from torch.utils.data import DataLoader


device = torch.device("cuda")
# device = torch.device("cpu") # uncomment to run with cpu

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate the model on the pointing game task')
    parser.add_argument("-p", '--path', dest="model_path", help='Path to the weight of the model to evaluate')
    parser.add_argument("-bs", "--batch_size", help="The size of the batches", type=int, default=64)
    parser.add_argument('-rv', dest="restval", help="use the restval dataset", action='store_true', default=False)

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

    in_dim = (416.0, 416.0)
    prepro_val = transforms.Compose([
        transforms.Resize((int(in_dim[0]), int(in_dim[1]))),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = VgCaptions(transform=prepro_val)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn_img_padded, pin_memory=True)

    imgs_enc = list()

    print("### Starting image embedding ###")
    end = time.time()
    for i, imgs in enumerate(loader, 0):

        input_imgs = imgs.to(device)
        _, output_imgs = join_emb.img_emb.module.get_activation_map(input_imgs)
        imgs_enc.append(output_imgs.cpu().data.numpy())

        if i % 100 == 99:
            print(str((i + 1) * args.batch_size) + "/" + str(len(dataset)) + " images encoded - Time per batch: " + str((time.time() - end)) + "s")

        end = time.time()

    dataset.image = False
    loader = DataLoader(dataset, batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=8, collate_fn=collate_fn_cap_padded, pin_memory=True)

    caps_enc = list()
    # process captions
    print("### Starting caption embedding ###")
    end = time.time()
    try:
        for i, (caps, lengths) in enumerate(loader, 0):

            input_caps = caps.to(device)
            with torch.no_grad():
                _, output_caps = join_emb(None, input_caps, lengths)
                caps_enc.append(output_caps.cpu().data.numpy())

            if i % 1000 == 999:
                print(str((i + 1) * args.batch_size) + "/" + str(len(dataset)) + " captions encoded - Time per batch: " + str((time.time() - end)) + "s")

            end = time.time()
    except IndexError:
        print(i)

    imgs_stack = np.vstack(imgs_enc)
    caps_stack = np.vstack(caps_enc)

    print("Dimension of images maps:", imgs_stack.shape)
    print("Dimension of captions embeddings:", caps_stack.shape)

    fc_w = join_emb.fc.module.weight.cpu().data.numpy()

    print("Pointing game score", compute_pointing_game_acc(imgs_stack, caps_stack, dataset.nb_regions, dataset.data, fc_w, in_dim))
