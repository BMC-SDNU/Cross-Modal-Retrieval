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

from misc.dataset import TextDataset
from misc.model import joint_embedding
from misc.utils import save_obj, collate_fn_cap_padded
from torch.utils.data import DataLoader


device = torch.device("cuda")
# device = torch.device("cpu") # uncomment to run with cpu

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract embedding representation for images')
    parser.add_argument("-p", '--path', dest="model_path", help='Path to the weights of the model to evaluate')
    parser.add_argument("-d", '--data', dest="data_path", help='path to the file containing the sentence to embed')
    parser.add_argument("-o", '--output', dest="output_path", help='path of the output file', default="./text_embedding")
    parser.add_argument("-bs", "--batch_size", help="The size of the batches", type=int, default=64)

    args = parser.parse_args()

    print("Loading model from:", args.model_path)
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)

    join_emb = joint_embedding(checkpoint['args_dict'])
    join_emb.load_state_dict(checkpoint["state_dict"])

    for param in join_emb.parameters():
        param.requires_grad = False

    join_emb.to(device)
    join_emb.eval()

    dataset = TextDataset(args.data_path)
    print("Dataset size: ", len(dataset))

    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=3, pin_memory=True, collate_fn=collate_fn_cap_padded)

    caps_enc = list()

    print("### Starting sentence embedding ###")
    end = time.time()
    for i, (caps, length) in enumerate(dataset_loader, 0):

        input_caps = caps.to(device)

        with torch.no_grad():
            _, output_emb = join_emb(None, input_caps, length)

        caps_enc.append(output_emb.cpu().data.numpy())

        if i % 100 == 99:
            print(str((i + 1) * args.batch_size) + "/" + str(len(dataset)) + " captions encoded - Time per batch: " + str((time.time() - end)) + "s")

        end = time.time()

    print("Processing done -> saving")
    caps_stack = np.vstack(caps_enc)

    save_obj(caps_stack, args.output_path)
    print("The data has been save to ", args.output_path)
