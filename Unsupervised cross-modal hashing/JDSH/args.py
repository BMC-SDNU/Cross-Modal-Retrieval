import argparse
from easydict import EasyDict as edict
import json

parser = argparse.ArgumentParser(description='JDSH')
parser.add_argument('--Train', default=True, help='train or test', type=bool)
parser.add_argument('--Config', default='./config/JDSH_MIRFlickr.json', help='Configure path', type=str)
parser.add_argument('--Dataset', default='MIRFlickr', help='MIRFlickr or NUSWIDE', type=str)
parser.add_argument('--Checkpoint', default='MIRFlickr_BIT_128.pth', help='checkpoint name', type=str)
parser.add_argument('--Bit', default=16, help='hash bit', type=int)

args = parser.parse_args()

# load basic settings
with open(args.Config, 'r') as f:
    config = edict(json.load(f))

# update settings
config.TRAIN = args.Train
config.DATASET = args.Dataset
config.CHECKPOINT = args.Checkpoint
config.HASH_BIT = args.Bit
