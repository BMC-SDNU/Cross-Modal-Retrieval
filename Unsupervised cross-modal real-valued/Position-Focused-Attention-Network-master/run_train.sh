#!/bin/bash
#
# Copyright (c) 2017 Tencent Inc. All Rights Reserved
#
# File: run_train.sh
# Author: applehyang@tencent.com
# Date: 2018/09/06 11:09:05
# Brief: 

#Use GPU:
#CUDA_VISIBLE_DEVICES=0 python train_attention.py --data_path ./data/ --data_name f30k_precomp --vocab_path ./vocab/ --logger_name ./runs/f30k_precomp/ --model_name ./runs/f30k_precomp/ --max_violation --bi_gru --agg_func=Mean --cross_attn=t2i --lambda_softmax=9 --num_epochs=30 --lr_update=15 --learning_rate=.0002 --embed_size=1024 --val_step=2000000 --batch_size=128

#Use CPU only:
python train_attention.py --data_path ./data/ --data_name f30k_precomp --vocab_path ./vocab/ --logger_name ./runs/f30k_precomp/ --model_name ./runs/f30k_precomp/ --max_violation --bi_gru --agg_func=Mean --cross_attn=t2i --lambda_softmax=9 --num_epochs=30 --lr_update=15 --learning_rate=.0002 --embed_size=1024 --val_step=2000000 --batch_size=128
