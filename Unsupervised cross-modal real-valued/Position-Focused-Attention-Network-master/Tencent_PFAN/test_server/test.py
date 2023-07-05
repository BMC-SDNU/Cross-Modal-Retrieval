#!/usr/bin/env python
# coding=utf-8

#########################################################################
#
# Copyright (c) 2017 Tencent Inc. All Rights Reserved
#
#########################################################################


"""
File: test.py
Author: applehyang@tencent.com
Date: 2018/11/27 14:11:05
Brief:
"""

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import time
import requests
import StringIO
import pycurl
import csv
import numpy as np

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


def get_sim(data):
    ip = '0.0.0.0'
    port = str(g_port)
    r = requests.post("http://"+str(ip)+":"+str(port)+"/scan_sentence_image_score", data=json.dumps(data, ensure_ascii=False), headers = {'content-type': 'application/json'})
    return r.text

def process_tsv_file(tsv_file, topn=1000000):
    all_items = []
    with open(tsv_file, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            try:
                item['image_id'] = int(item['image_id'])
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])
                item['num_boxes'] = int(item['num_boxes'])
                all_items.append(item)
                if len(all_items) >= topn:
                    break
            except:
                continue
    return all_items

def load_img_url_to_id(img_id_file):
    img_url_to_id = {}
    with open(img_id_file, 'r') as f:
        for line in f:
            line = line.strip()
            url, id = line.split("\t")
            img_url_to_id[url] = int(id)
    return img_url_to_id

def process_json_file(json_file):
    with open(json_file, 'r') as f:
        lines = f.readlines()

    img_id_to_whole_items = {}
    for line in lines:
        d = json.loads(line.strip(), encoding='utf8')
        url = d[0]
        img_feature = d[1]
        url = url[url.rfind('/') + 1:]
        img_id = url[:url.find('.')]
        img_id_to_whole_items[int(img_id)] = img_feature
    return img_id_to_whole_items

def process(text_input_flag = 'tag'):
    output_path = g_output_path
    all_items = process_tsv_file("../data/tencent_data_meta/test_own.tsv.0")
    img_id_to_whole_items = process_json_file("../data/tencent_data_meta/test_features.json")
    img_id_file = "../data/tencent_data_meta/img_id.txt"
    img_id_to_items = {}
    img_url_to_id = load_img_url_to_id(img_id_file)
    for item in all_items:
        img_id_to_items[int(item['image_id'])] = item
    print "Len in all_items", len(img_id_to_items), len(all_items)

    # load input json File
    with open("../data/tencent_data_meta/test_list.json", 'r') as f:
        lines = f.readlines()

    fw = open(output_path, 'w')

    line_id = 0
    line_dict = {}
    line_dict[1] = 1

    for line in lines:
        line_id += 1
        d = json.loads(line.strip(), encoding='utf8')
        if text_input_flag == 'tag':
            tag_ids = d['Tag']
        else:
            tag_ids = d['Content']
        tags = tag_ids
        for i in range(len(tags)):
            tags[i] = tags[i].encode('utf8')
        #tags = tags[:1]
        urls, input_urls = d['Img_urls'], []
        img_ids, items, whole_items = [], [], []
        for url in urls:
            img_id = img_url_to_id[url]
            img_ids.append(img_id)
            if img_id in img_id_to_items and img_id in img_id_to_whole_items:
                items.append(img_id_to_items[img_id])
                input_urls.append(url)
                whole_items.append(img_id_to_whole_items[img_id])
        data = {}
        data['sentence'] = tags
        data['boxes'] = items
        data['whole_items'] = whole_items
        print "Input data", data['sentence'], len(data['boxes']), len(data['whole_items'])
        res = get_sim(data)
        res = json.loads(res, encoding='utf8')
        print "Res",res

        sim_t2i = res['sim_t2i']
        sim_i2t = res['sim_i2t']
        assert(len(sim_t2i) == len(data['boxes']) and len(sim_i2t) == len(data['boxes']) and len(input_urls) == len(data['boxes']))

        write_data = {}
        write_data["urls"] = input_urls
        write_data["dist_t2i"] = sim_t2i
        write_data["dist_i2t"] = sim_i2t
        fw.write(json.dumps(write_data, ensure_ascii=False)+'\n')

    fw.close()

g_output_path = "./dist.json"
g_port = 5091
if __name__ == "__main__":
    text_input_flag = 'tag'
    if len(sys.argv) > 1:
        g_output_path = sys.argv[1]
    if len(sys.argv) > 2:
        text_input_flag = sys.argv[2]
    if len(sys.argv) > 3:
        g_port = int(sys.argv[3])
    process(text_input_flag)
