#!/usr/bin/env python
# coding=utf-8

#########################################################################
#
# Copyright (c) 2017 Tencent Inc. All Rights Reserved
#
#########################################################################


"""
File: predict.py
Author: bingxinqu@tencent.com
Date: 2018/09/06 19:09:24
Brief:
"""
import jieba
import numpy as np
import torch
import sys, os
from vocab import Vocabulary, deserialize_vocab
from data_whole import get_test_loader
from model_whole import SCAN
from evaluation_whole import shard_xattn_t2i, shard_xattn_i2t
import StringIO
import pycurl
import json
import base64
from flask import Flask, request
app = Flask(__name__)

checkpoint = None

def get_data(model, data_loader, log_step=10):
    img_embs = None
    whole_img_embs = None
    cap_embs = None
    cap_lens = None
    final_cap_embs = None

    max_n_word = 0
    for i, (images, whole_images, boxes, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))
        #print("For", images.shape[0], captions.shape[0], len(lengths))
        #print("CAP", captions[0], captions[1])
    #print("max_n_word", max_n_word)

    for i, (images, whole_images, boxes, captions, lengths, ids) in enumerate(data_loader):
        if i == 0:
            print("IDS", ids)
        img_emb, whole_img_emb, cap_emb, cap_len, final_cap_emb = model.forward_emb(images, whole_images, boxes, captions, lengths, volatile=True)
        #print("Emb shape", img_emb, cap_emb, len(cap_len))

        if img_embs is None:
            if img_emb.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            whole_img_embs = np.zeros((len(data_loader.dataset), whole_img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
            final_cap_embs = np.zeros((len(data_loader.dataset), final_cap_emb.size(1)))
        img_embs[ids,:,:] = img_emb.data.cpu().numpy().copy()
        whole_img_embs[ids,:] = whole_img_emb.data.cpu().numpy().copy()
        cap_embs[ids,:max(lengths),:] = cap_emb.data.cpu().numpy().copy()
        final_cap_embs[ids,:] = final_cap_emb.data.cpu().numpy().copy()
        for j, nid in enumerate(ids):
            cap_lens[nid] = lengths[j]
    return img_embs, whole_img_embs, cap_embs, cap_lens, final_cap_embs

def compute_area(box, i_x, i_y, split_size):
    p1_x, p1_y, p2_x, p2_y = box
    one_wh = 1.0/split_size
    p3_x, p3_y, p4_x, p4_y = i_x * one_wh, i_y * one_wh, (i_x+1) * one_wh, (i_y+1) * one_wh
    if p1_x > p4_x or p2_x < p3_x or p1_y > p4_y or p2_y < p3_y:
        return 0.0
    len = min(p2_x,p4_x) - max(p1_x,p3_x)
    wid = min(p2_y,p4_y) - max(p1_y,p3_y)
    if len < 0 or wid < 0:
        return 0.0
    return len * wid

def retain_top_n_area(indexes, weights, topn=15, split_size=16):
    sorted_index = np.argsort(weights)[::-1]
    res_indexes, res_weights = [], []
    for i in range(min(topn,len(weights))):
        idx = sorted_index[i]
        res_indexes.append(indexes[idx])
        res_weights.append(weights[idx])
    if len(res_indexes) < topn:
        for i in range(topn-len(res_indexes)):
            res_indexes.append(split_size*split_size)
            res_weights.append(0.0)
    return res_indexes, res_weights

def extract_one_box(box, split_size, box_area):
    one_wh = 1.0/split_size
    x,y,x2,y2 = box
    indexes, weights = [], []
    #print "XY", x,y,x2,y2
    for i_x in range(int(x/one_wh),int(x2/one_wh) + 1):
        for i_y in range(int(y/one_wh), int(y2/one_wh) + 1):
    #for i_x in range(0, split_size):
    #    for i_y in range(0, split_size):
            if i_x >= split_size or i_y >= split_size:
                continue
            index = i_x * split_size + i_y
            area = compute_area(box, i_x, i_y, split_size)
            #print "Box", i_x, i_y, area
            if area > 0:
                indexes.append(index)
                weights.append(area)
    indexes, weights = retain_top_n_area(indexes, weights, 15, split_size)
    wei_all = 0.0
    for w in weights:
        wei_all += w

    for i in range(len(weights)):
        weights[i] /= wei_all

    return indexes + weights + [box_area]

def translate_box_to_index(norm_box_data, norm_box_area, split_size=16):
    # norm_box_data shape: image_num, box_num, 4
    # norm_box_area shape: image_num, box_num
    res = []
    for i in range(norm_box_data.shape[0]):
        tmp_res = []
        for j in range(norm_box_data.shape[1]):
            tmp_res.append(extract_one_box(norm_box_data[i][j], split_size, norm_box_area[i][j]))
        res.append(tmp_res)
    res = np.array(res)
    print "Final res shape", res.shape
    return res

def get_box_numpy(boxes, boxes_shape):
    # boxes_numpy shape: image_num, box_num, 4
    # shape_numpy shape: image_num, 2 (height, width)
    print "Box shape", boxes.shape
    print "Box_shape shape", boxes_shape.shape
    assert(boxes.shape[0] == boxes_shape.shape[0])
    norm_box_data = []
    norm_box_area = []
    for i in range(boxes.shape[0]):
        img_h, img_w = boxes_shape[i]
        img_area = img_h * img_w
        tmp_box_data = []
        tmp_box_area = []
        for j in range(boxes.shape[1]):
            x1, y1, x2, y2 = boxes[i,j]
            box_area = (x2-x1)*(y2-y1)
            x1 = x1 / img_w
            y1 = y1 / img_h
            x2 = x2 / img_w
            y2 = y2 / img_h
            if x1 > 1.0:
                print "Error-x1", i, j, x1
                x1 = 1.0
            if y1 > 1.0:
                print "Error-y1", i, j, y1
                y1 = 1.0
            if x2 > 1.0:
                print "Error-x2", i, j, x2
                x2 = 1.0
            if y2 > 1.0:
                print "Error-y2", i, j, y2
                y2 = 1.0
            tmp_box_data.append((x1,y1,x2,y2))
            tmp_box_area.append(box_area/img_area)
        norm_box_data.append(tmp_box_data)
        norm_box_area.append(tmp_box_area)
    norm_box_data = np.array(norm_box_data)
    norm_box_area = np.array(norm_box_area)
    print "Shape norm box data", norm_box_data.shape
    print "Shape norm box area", norm_box_area.shape
    return translate_box_to_index(norm_box_data, norm_box_area)

def reverse_tsv_to_numpy(boxes_tsv):
    # boxes_tsv是一个数组，每个元素是一个dict
    # 包含image_id, image_h, image_w, num_boxes, boxes 和 features
    # 只需要features及boxes
    res, res_box, box_shape = [], [], []
    for box in boxes_tsv:
        feature = box['features']
        buf = base64.decodestring(feature)
        temp = np.frombuffer(buf, dtype=np.float32)
        temp = temp.reshape((-1,2048))
        res.append(temp)

        boxes = box['boxes']
        buf = base64.decodestring(boxes)
        temp = np.frombuffer(buf, dtype=np.float32)
        temp = temp.reshape((-1,4))
        res_box.append(temp)

        height, width = box['image_h'], box['image_w']
        box_shape.append((height, width))
    return np.array(res), get_box_numpy(np.array(res_box), np.array(box_shape))

def write_cap_image(feature_numpy, boxes_numpy, whole_numpy, sentence):
    opt = checkpoint['opt']
    cur_dir = os.getcwd()
    data_dir = os.path.join(cur_dir, 'data/' + opt.data_name)
    cap_path = os.path.join(data_dir, g_split_name + "_sent_caps.txt")
    np_path = os.path.join(data_dir, g_split_name + "_ims.npy")
    np_box_path = os.path.join(data_dir, g_split_name + "_boxes.npy")
    np_whole_path = os.path.join(data_dir, g_split_name + "_who_ims.npy")
    #print "Cap path", cap_path
    #print "Numpy path", np_path
    fw = open(cap_path, 'w')
    fw.write(" ".join(sentence).encode('utf8') + '\n')
    fw.close()
    np.save(np_path, feature_numpy)
    np.save(np_box_path, boxes_numpy)
    np.save(np_whole_path, whole_numpy)

def reverse_sim(sims):
    res = []
    n = sims.shape[0]
    for i in range(n):
        res.append(sims[i][0])
    return res

@app.route('/scan_sentence_image_score', methods=['POST'])
def predict():
    if request.content_type != 'application/json':
        print "application error"
        return '{"error":"application error"}'

    body = request.get_json()
    boxes_tsv = body['boxes']
    sentence = body['sentence']
    whole_items = body['whole_items']

    print "Len in boxes", len(boxes_tsv)
    print "Len in whole_items", len(whole_items)
    print "Sentence", " ".join(sentence).encode('utf8')

    feature_numpy, boxes_numpy = reverse_tsv_to_numpy(boxes_tsv)
    whole_numpy = np.array(whole_items)
    write_cap_image(feature_numpy, boxes_numpy, whole_numpy, sentence)

    print "Boxes numpy", feature_numpy.shape, boxes_numpy.shape
    print "Whole numpy", whole_numpy.shape

    split = g_split_name
    opt = checkpoint['opt']
    #print(opt)
    opt.data_path = 'data'

    model = g_model

    data_loader = get_test_loader(split, opt.data_name, g_vocab, opt.batch_size, opt.workers, opt)

    img_embs, whole_img_embs, cap_embs, cap_lens, final_cap_embs = get_data(model, data_loader) #encode_data(model, data_loader)

    cap_embs = np.array([cap_embs[0]]) # one text to multiply images, so cap_embs has the same

    sims = shard_xattn_t2i(img_embs, whole_img_embs, cap_embs, cap_lens, final_cap_embs, opt, shard_size=128)
    print "Sims (t2i)", sims
    sims2 = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
    print "Sims (i2t)", sims2
    result = {'sim_t2i':reverse_sim(sims),'sim_i2t':reverse_sim(sims2)}
    return json.dumps(result)

g_model = None
g_split_name = 'test_server'
g_vocab = None

if __name__ == "__main__":
    model_path = sys.argv[1]
    port = 5091

    if len(sys.argv) > 2:
        port = int(sys.argv[2])

    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print "Option", opt
    g_model = SCAN(opt)
    g_model.load_state_dict(checkpoint['model'])
    g_vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_sent_vocab.json' % opt.data_name))
    opt.vocab_size = len(g_vocab)
    print "Vocab size", opt.vocab_size
    app.run(host='0.0.0.0', port=port, debug=False)
