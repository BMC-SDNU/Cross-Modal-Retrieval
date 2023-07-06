#!/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: hash_func.py
# @Author: Qing-Yuan Jiang
# @Mail: qyjiang24 AT gmail.com
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np

from utils.args import args


def linear_hash(train_data, binary_codes):
    dim = train_data.shape[-1]
    wx =  np.linalg.inv(train_data.T.dot(train_data) + args.gamma * np.eye(dim)).dot(train_data.T).dot(binary_codes)
    return wx


