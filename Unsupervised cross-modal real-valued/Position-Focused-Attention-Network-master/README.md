## Position-Focused-Attention-Network
Position Focused Attention Network for Image-Text Matching, which is published in ijcai-2019. The paper can be downloaded from [arXiv](https://arxiv.org/abs/1907.09748).

## Introduction

This is the source code of Position Focused Attention Network, an approch for Image-Text Matching based on position attention from [Tencent](https://github.com/Tencent). It is built on top of the SCAN (by [Kuang-Huei Lee](https://github.com/kuanghuei/SCAN)) in PyTorch.

## Requirements and Installation
We recommended the following dependencies.

* Python 2.7
* [PyTorch](http://pytorch.org/) 0.3
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

The workflow of PFAN

<img src="https://github.com/HaoYang0123/Position-Focused-Attention-Network/blob/master/workflow_figures/workflow1.jpg" width="745" alt="workflow" /> 
Position attention network in PFAN

<img src="https://github.com/HaoYang0123/Position-Focused-Attention-Network/blob/master/workflow_figures/workflow2.jpg" width="515" alt="position attention" />

## Download data
Download the dataset files. We use the dataset files created by SCAN [Kuang-Huei Lee](https://github.com/kuanghuei/SCAN). The position information of images can be downloaded from [here](https://drive.google.com/open?id=1ZiF1IoeExPcn9V9L78X6jEYuMxR96OLO) (for Flickr30K) and [here](https://drive.google.com/open?id=1DaCZxeXOCm05u-Gf-_MG_zSNKO1UxBat) (for MS-COCO). Noting that we only upload the position information and caption in MS-COCO dataset, while the image feature is not uploaded because of its huge storage. The original image feature can be downloaded from [SCAN](https://github.com/kuanghuei/SCAN). When using the original image features, we should reorder these samples from the sample ids or sample captions.
The Tencent-News dataset files can be downloaded from [here](https://drive.google.com/open?id=1WKq05mhSMc2u0SLtCWkUzgmqTLx95kXR) and [here](https://drive.google.com/open?id=1dPyo2EBHQoHkqx-Dl4R7ISb8t-rVG_KK).

```bash
#For Flickr30K dataset
wget https://drive.google.com/open?id=1ZiF1IoeExPcn9V9L78X6jEYuMxR96OLO
#For MS-COCO dataset
wget https://drive.google.com/open?id=1DaCZxeXOCm05u-Gf-_MG_zSNKO1UxBat
#For Tencent-News training dataset
wget https://drive.google.com/open?id=1WKq05mhSMc2u0SLtCWkUzgmqTLx95kXR
#For Tencent-News testing dataset
wget https://drive.google.com/open?id=1dPyo2EBHQoHkqx-Dl4R7ISb8t-rVG_KK
```

## Training new models

To train Flickr30K and MS-COCO models:
```bash
sh run_train.sh
```
In order to further improve the performance of PFAN on Tencent-News dataset, the whole image feautre is also considered. The details are shown in Tencent_PFAN code:
```bash
sh Tencent_PFAN/run_train.sh
```

Arguments used to train Flickr30K models and MS-COCO models are as same as those of SCAN:

For Flickr30K:

| Method      | Arguments |
| :---------: | :-------: |
|  t-i     | `--max_violation --bi_gru --agg_func=Mean --cross_attn=t2i --lambda_softmax=9 --num_epoches=30 --lr_update=15 --learning_rate=.0002 --embed_size=1024 --batch_size=128 `|
|  i-t     | `--max_violation --bi_gru --agg_func=Mean --cross_attn=i2t --lambda_softmax=4 --num_epoches=30 --lr_update=15 --learning_rate=.0002 --embed_size=1024 --batch_size=128 `|

For MS-COCO:

| Method      | Arguments |
| :---------: | :-------: |
|  t-i    | `--max_violation --bi_gru --agg_func=Mean --cross_attn=t2i --lambda_softmax=9 --num_epoches=30 --lr_update=15 --learning_rate=.0005 --embed_size=1024 --batch_size=128 `|
|  i-t    | `--max_violation --bi_gru --agg_func=Mean --cross_attn=i2t --lambda_softmax=4 --num_epoches=30 --lr_update=15 --learning_rate=.0005 --embed_size=1024 --batch_size=128 `|

For Tencent-News:

| Method      | Arguments |
| :---------: | :-------: |
|  t-i    | `--max_violation --bi_gru --agg_func=Mean --cross_attn=t2i --lambda_softmax=9 --num_epoches=30 --lr_update=15 --learning_rate=.0002 --embed_size=512 --batch_size=128 --lambda_whole=2 `|
|  i-t    | `--max_violation --bi_gru --agg_func=Mean --cross_attn=i2t --lambda_softmax=4 --num_epoches=30 --lr_update=15 --learning_rate=.0002 --embed_size=512 --batch_size=128 --lambda_whole=2 `|

The models on Tencent-News can be downloaded from [here](https://drive.google.com/open?id=1SA2J8-m6w6HvbXyDkOtjLcD7Tw4A9hPn).

## Evaluate trained models on Flickr30K and MS-COCO

```python
from vocab import Vocabulary
import evaluation
evaluation.evalrank("$RUN_PATH/f30k_precomp/model_best.pth.tar", data_path="$DATA_PATH", split="test")
```

## Evaluate position-attention (PFAN-A) and position-only (PFAN-P) models
|            | i2t-1    |i2t-5    |i2t-10    |t2i-1    |t2i-5    |t2i-10    |
| :---------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| PFAN-A   | 70.0  | 91.8  | 95.0  | 50.4  | 78.7  | 86.1  |
| PFAN-P   | 66.0  | 89.4  | 94.1  | 48.6  | 76.9  | 85.1  |

## Evaluate trained models on Tencent-News

First, start the server to process requests
```bash
sh run_server.sh # port 5091 is sentence model and port 5092 is tag model
```
Then, send requests to get results from the server
```bash
cd test_server
python test.py dist_sentence_t2i.json sentence 5091 # to get the results using sentence model and sentence data
python test.py dist_tag_t2i.json tag 5091 # to get the results using sentence model and tag data
python test.py dist_tag_new_t2i.json tag 5092 # to get the results using tag model and tag data
```
Finally, get the MAP@1-3 and A@1-3
```bash
cd test_server
python compute_map.py
```
