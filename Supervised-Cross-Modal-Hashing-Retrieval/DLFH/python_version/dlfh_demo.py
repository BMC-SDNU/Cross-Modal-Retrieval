#!/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: DLFH_demo.py
# @Author: Qing-Yuan Jiang
# @Mail: qyjiang24 AT gmail.com
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import time
import sys

import numpy as np

from data.load_data import load_data
from utils.args import args
from utils.logger import logger
from utils.calc_hamming_ranking import calc_hamming_ranking
from dlfh_algo import dlfh_algo
from hash_func import linear_hash


if __name__ == "__main__":
    np.random.seed(args.seed)

    '''load dataset'''
    logger.info('load dataset: {}'.format(args.dataname))
    test_text, test_image, test_label, database_text, database_image, database_label = load_data()

    '''learning procedure'''
    logger.info('start training procedure')
    start_t = time.time()
    db_image_codes, db_text_codes = dlfh_algo(train_labels=database_label)

    '''out-of-sample extension'''
    wx = linear_hash(database_image, db_image_codes)
    wy = linear_hash(database_text, db_text_codes)

    end_t = time.time() - start_t

    '''start encoding'''
    test_image_codes = np.sign(test_image.dot(wx))
    test_text_codes = np.sign(test_text.dot(wy))

    param = {'topk': [100]}

    metrici2t = calc_hamming_ranking(test_image_codes, db_text_codes, test_label, database_label, param)
    metrict2i = calc_hamming_ranking(test_text_codes, db_image_codes, test_label, database_label, param)

    mapi2t = metrici2t['map']
    mapt2i = metrict2i['map']
    topprei2t = float(metrici2t['topkpre'][0])
    toppret2i = float(metrict2i['topkpre'][0])
    logger.info('map@i2t: {:>.4f}, map@t2i: {:>.4f}.'.format(mapi2t, mapt2i))
    logger.info('precision@i2t: {:>.4f}, precision@t2i: {:>.4f}.'.format(topprei2t, toppret2i))



