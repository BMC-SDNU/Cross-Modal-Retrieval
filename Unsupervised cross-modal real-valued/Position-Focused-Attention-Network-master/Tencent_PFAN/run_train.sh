#!/bin/bash
#
# Copyright (c) 2017 Tencent Inc. All Rights Reserved
#
# File: run_train.sh
# Author: applehyang@tencent.com
# Date: 2018/09/06 11:09:05
# Brief: 

#For GPU
#CUDA_VISIBLE_DEVICES=0 python ./train_whole.py --data_path ./data/ --data_name tencent_data --vocab_path ./vocab/ --logger_name ./runs/tencent_whole_all_w2/ --model_name ./runs/tencent_whole_all_w2/ --max_violation --bi_gru --agg_func=Mean --cross_attn=t2i --lambda_softmax=9 --num_epochs=30 --lr_update=15 --learning_rate=.0002 --embed_size 512 --val_step=2000000 --batch_size=128 --lambda_whole 2

#For CPU only
python ./train_whole.py --data_path ./data/ --data_name tencent_data --vocab_path ./vocab/ --logger_name ./runs/tencent_data/ --model_name ./runs/tencent_data/ --max_violation --bi_gru --agg_func=Mean --cross_attn=t2i --lambda_softmax=9 --num_epochs=30 --lr_update=15 --learning_rate=.0002 --embed_size 512 --val_step=2000000 --batch_size=128 --lambda_whole 2
