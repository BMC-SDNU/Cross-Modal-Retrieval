# -*- coding: utf-8 -*-

import os
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--critic-itrs', type=int, default=2, help='critic iteration number')
parser.add_argument('--batch-size', type=int, default=30, help='training batch size')
parser.add_argument('--max-steps', type=int, default=50000, help='max number of training iteration')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
parser.add_argument('--h-dim', type=int, default=64, help='dimension of hidden space')
parser.add_argument('--txt-dim', type=int, default=64, help='dimension of txtual feature')
parser.add_argument('--img-dim', type=int, default=64, help='dimension of visual feature')
parser.add_argument('--label-num', type=int, default=80, help='number of labels')
parser.add_argument('--beta', type=float, default=5, help='beta')
parser.add_argument('--gamma', type=float, default=0.0001, help='gamma')
parser.add_argument('--dir', type=str, default='txt2img', help='img2txt or txt2img')
parser.add_argument('--save-interval', type=int, default='10000', help='save interval')
parser.add_argument('--log-interval', type=int, default='500', help='log interval')
parser.add_argument('--save-dir', type=str, default='checkpoint', help='checkpoint save root')
parser.add_argument('--log-dir', type=str, default='log', help='log save root')
opt = parser.parse_args()

