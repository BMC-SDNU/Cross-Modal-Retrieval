#!/bin/bash
#
# Copyright (c) 2017 Tencent Inc. All Rights Reserved
#
# File: run_server.sh
# Author: bingxinqu@tencent.com
# Date: 2018/10/24 15:10:33
# Brief: 

#For GPU
#CUDA_VISIBLE_DEVICES=0 python scan_server_whole.py ./runs/tencent_data/model_best.pth.tar

#For CPU only (sentence model)
nohup python scan_server_whole.py ./runs/sentence_t2i.pth.tar 5091 &

#For CPU only (tag model)
nohup python scan_server_whole.py ./runs/tag_t2i.pth.tar 5092 &
