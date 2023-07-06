#!/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: load_data.py
# @Author: Qing-Yuan Jiang
# @Mail: qyjiang24 AT gmail.com
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import os

import numpy as np
import scipy.io as sio

from utils.args import args


def load_data():
    if args.dataname == 'flickr25k':
        datapath = os.path.join(args.datapath, 'flickr-25k.mat')
    elif args.dataname == 'iaprtc12':
        datapath = os.path.join(args.datapath, 'iapr-tc12.mat')
    elif args.dataname == "nuswidetc10":
        datapath = os.path.join(args.datapath, 'nus-widetc10.mat')
    else:
        raise NameError('unsupported data set: {}'.format(args.dataname))

    data = sio.loadmat(datapath)
    test_text = data['YTest'].astype(np.float)
    database_text = data['YDatabase'].astype(np.float)
    test_label = data['testL'].astype(np.float)
    database_label = data['databaseL'].astype(np.float)
    if args.no_deep_feature:
        test_image = data['XTest']
        database_image = data['XDatabase']
    else:
        test_image = data['VTest']
        database_image = data['VDatabase']
    return test_text, test_image, test_label, database_text, database_image, database_label


