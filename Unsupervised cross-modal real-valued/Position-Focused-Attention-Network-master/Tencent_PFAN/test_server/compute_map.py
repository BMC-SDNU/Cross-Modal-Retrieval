#!/usr/bin/env python
# coding=utf-8

#########################################################################
#
# Copyright (c) 2017 Tencent Inc. All Rights Reserved
#
#########################################################################


"""
File: compute_map.py
Author: applehyang@tencent.com
Date: 2018/11/07 11:11:08
Brief:
"""
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import numpy as np

def load_data(dist_path, flag_path, ori_flag = False):
    dist_list = []
    with open(dist_path, 'r') as f:
        for line in f:
            line = line.strip()
            d = json.loads(line.strip(), encoding='utf8')
            dist = d['Dist']
            dist_d = []
            for d in dist:
                dist_d.append(float(d))
            dist_list.append(dist_d)
    flag_list, url_list = [], []
    with open(flag_path, 'r') as f:
        for line in f:
            line = line.strip()
            d = json.loads(line.strip(), encoding='utf8')
            flags = d['Img_flags']
            urls = d['Img_urls']
            flag_list.append(flags)
            url_list.append(urls)

    max_dist = 100000
    for i in range(len(dist_list)):
        for j in range(len(dist_list[i])):
            url = url_list[i][j]
            if url.find('inews.gtimg.com') >= 0:
                dist_list[i][j] = max_dist
    if ori_flag:
        for i in range(len(dist_list)):
            for j in range(len(dist_list[i])):
                dist_list[i][j] = len(dist_list[i]) - j

    return dist_list, flag_list

def load_dist_from_file(dist_file):
    url_dist_dict = {}
    t2i_ratio = 1
    with open(dist_file, 'r') as f:
        line_id = 1
        for line in f:
            d = json.loads(line.strip(), encoding='utf8')
            dist1, dist2 = d['dist_t2i'], d['dist_i2t']
            dist = (np.array(dist1)*t2i_ratio + np.array(dist2)) / (t2i_ratio+1)
            dist = dist.tolist()
            if dist_file.endswith("_t2i.json"):
                dist = dist1
            elif dist_file.endswith("_i2t.json"):
                dist = dist2
            urls = d['urls']
            assert(len(dist)) == len(urls)
            for i, url in enumerate(urls):
                url_dist_dict[url+'@'+str(line_id)] = dist[i]
            line_id += 1
    return url_dist_dict

def load_data_from_PFAN(dist_paths, weights, flag_path):
    assert(len(dist_paths) == len(weights))
    url_list = []
    dist_list = []
    flag_list = []
    url_dist_dict, url_count = {}, {}
    for i, dist_path in enumerate(dist_paths):
        wei = weights[i]
        tmp_dict = load_dist_from_file(dist_path)
        for key in tmp_dict:
            if key not in url_dist_dict:
                url_dist_dict[key] = wei*tmp_dict[key]
                url_count[key] = wei
            else:
                url_dist_dict[key] += wei*tmp_dict[key]
                url_count[key] += wei
    for key in url_dist_dict:
        url_dist_dict[key] /= url_count[key]
    with open(flag_path, 'r') as f:
        line_id = 1
        for line in f:
            line = line.strip()
            d = json.loads(line.strip(), encoding='utf8')
            flags = d['Img_flags']
            img_urls = d['Img_urls']
            url_d = []
            dist_d = []
            flag_d = []
            for i, url in enumerate(img_urls):
                key = url + '@' + str(line_id)
                if key in url_dist_dict:
                    url_d.append(url)
                    dist_d.append(url_dist_dict[key])
                    flag_d.append(flags[i])
            line_id += 1
            assert(len(dist_d) == len(flag_d) and len(url_d) == len(dist_d))
            url_list.append(url_d)
            dist_list.append(dist_d)
            flag_list.append(flag_d)
    assert(len(dist_list) == len(flag_list) and len(url_list) == len(dist_list))
    return url_list, dist_list, flag_list

def compute_map(dist_list, flag_list, line_num, Top=3):
    #dist_list = np.random.rand(len(dist_list))
    sorted_index = np.argsort(dist_list)[::-1]
    temp = 0
    ap = 0
    acc_top1 = -1
    for i in range(min(Top,len(sorted_index))):
        idx = sorted_index[i]
        label = flag_list[idx]
        temp += label
        ap += temp * 1.0 / (i+1)
        if i == 0 and label == 1:
            acc_top1 = line_num
    return ap / min(Top,len(sorted_index)), acc_top1

def compute_correct_num(dist_list, flag_list, Top=3):
    #dist_list = np.random.rand(len(dist_list))
    sorted_index = np.argsort(dist_list)[::-1]
    correct = 0
    for i in range(min(Top, len(sorted_index))):
        idx = sorted_index[i]
        correct += flag_list[idx]
    return correct, min(Top, len(sorted_index))

def compute_correct_score(dist_list, flag_list, score=0.65):
    correct, all = 0, 0
    for i in range(len(dist_list)):
        if dist_list[i] >= score:
            all += 1
            if flag_list[i] == 1:
                correct += 1
    return correct, all

def compute_incorrect_score(dist_list, flag_list, score=0.65):
    incorrect, all = 0, 0
    for i in range(len(dist_list)):
        if dist_list[i] < score:
            all += 1
            if flag_list[i] == 0:
                incorrect += 1
    return incorrect, all

def get_right_count(flag_list):
    res = 0
    for flag in flag_list:
        res += flag
    return res

if __name__ == "__main__":
    dist_files = ["./dist_tag_t2i.json", "./dist_sentence_t2i.json", "./dist_tag_new_t2i.json"]
    weights = [1.0, 1.0, 1.5]
    url_list, dist_list, flag_list = load_data_from_PFAN(dist_files, weights, "../data/tencent_data_meta/test_list.json")
    print "Len", len(dist_list), len(flag_list)
    ANS = 0.0000001
    line_nums_top1 = {}
    for top in range(1,4):
        MAP = 0.0
        count = 0
        correct_count, all_count = 0,0
        for i in range(len(dist_list)):
            if len(dist_list[i]) == 0:
                continue
            if get_right_count(flag_list[i]) >= top:
                map, acc_top1 = compute_map(dist_list[i], flag_list[i], i+1, top)
                if acc_top1 != -1:
                    line_nums_top1[acc_top1] = 1
                MAP += map
                count += 1
                c1, c2 = compute_correct_num(dist_list[i], flag_list[i], top)
                correct_count += c1
                all_count += c2
        print "MAP@"+str(top)+":", MAP/(count+ANS), count
        print "Acc@"+str(top)+":", correct_count*1.0/(all_count+ANS), all_count

    score = 1.0
    while score >= 0.0:
        correct_count, all_count = 0, 0
        for i in range(len(dist_list)):
            c1, c2 = compute_correct_score(dist_list[i], flag_list[i], score)
            correct_count += c1
            all_count += c2
        print "Acc@"+str(score)+":", correct_count*1.0/(all_count+ANS), all_count
        score -= 0.05
    score = 1.0
    while score >= 0.0:
        incorrect_count, all_count = 0, 0
        for i in range(len(dist_list)):
            c1, c2 = compute_incorrect_score(dist_list[i], flag_list[i], score)
            incorrect_count += c1
            all_count += c2
        print "Error rate@"+str(score)+":", incorrect_count*1.0/(all_count+ANS), all_count
        score -= 0.05
