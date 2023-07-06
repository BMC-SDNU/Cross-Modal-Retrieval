# DSAH
> Source code of our ICMR 2020 paper "[Deep Semantic-Alignment Hashing for Unsupervised Cross-Modal Retrieval](https://dl.acm.org/doi/abs/10.1145/3372278.3390673)"

## Introduction

Deep hashing methods have achieved tremendous success in cross-modal retrieval, due to its low storage consumption and fast retrieval speed. In real cross-modal retrieval applications, it's hard to obtain label information. Recently, increasing attention has been paid to unsupervised cross-modal hashing. However, existing methods fail to exploit the intrinsic connections between images and their corresponding descriptions or tags (text modality). In this paper, we propose a novel Deep Semantic-Alignment Hashing (DSAH) for unsupervised cross-modal retrieval, which sufficiently utilizes the co-occurred image-text pairs. DSAH explores the similarity information of different modalities and we elaborately design a semantic-alignment loss function, which elegantly aligns the similarities between features with those between hash codes. Moreover, to further bridge the modality gap, we innovatively propose to reconstruct features of one modality with hash codes of the other one. Extensive experiments on three cross-modal retrieval datasets demonstrate that DSAH achieves the state-of-the-art performance.
![Framework](https://github.com/idejie/pics/raw/master/WX20200627-190524.png)
## Requirements
- Python: 3.x
- other dependencies: [env.yaml](https://github.com/idejie/DSAH/blob/master/env.yaml)
## Run
- Update the [setting.py](https://github.com/idejie/DSAH/blob/master/settings.py) with your `data_dir`. And change the value [` EVAL`](https://github.com/idejie/DSAH/blob/be1f3edba30015b164bc41994067a71273cbeb30/settings.py#L6), for **train** setting it with `False`
- run the `train.py`
  ```shell
  python train.py
  ```
 - For test, set the value [` EVAL`](https://github.com/idejie/DSAH/blob/be1f3edba30015b164bc41994067a71273cbeb30/settings.py#L6)  with `True`. And the model will load the `checkpoint/DATASET_CODEBIT_bit_best_epoch.pth`
## Datasets
For datasets, we follow [Deep Cross-Modal Hashing's Github (Jiang, CVPR 2017)](https://github.com/jiangqy/DCMH-CVPR2017/tree/master/DCMH_matlab/DCMH_matlab). You can download these datasets from:

- Wikipedia articles, [[Link](http://www.svcl.ucsd.edu/projects/crossmodal/)]

- MIRFLICKR25K, [[OneDrive]](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EpLD8yNN2lhIpBgQ7Kl8LKABzM68icvJJahchO7pYNPV1g?e=IYoeqn), [[Baidu Pan](https://pan.baidu.com/s/1o5jSliFjAezBavyBOiJxew), password: 8dub]

- NUS-WIDE (top-10 concept), [[OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EoPpgpDlPR1OqK-ywrrYiN0By6fdnBvY4YoyaBV5i5IvFQ?e=kja8Kj)], [[Baidu Pan](https://pan.baidu.com/s/1GFljcAtWDQFDVhgx6Jv_nQ), password: ml4y]
## Citation
If you find this code useful, please cite our paper:
```bibtex
@inproceedings{10.1145/3372278.3390673,
author = {Yang, Dejie and Wu, Dayan and Zhang, Wanqian and Zhang, Haisu and Li, Bo and Wang, Weiping},
title = {Deep Semantic-Alignment Hashing for Unsupervised Cross-Modal Retrieval},
year = {2020},
isbn = {9781450370875},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3372278.3390673},
doi = {10.1145/3372278.3390673},
booktitle = {Proceedings of the 2020 International Conference on Multimedia Retrieval},
pages = {44–52},
numpages = {9},
keywords = {cross-modal hashing, cross-media retrieval, semantic-alignment},
location = {Dublin, Ireland},
series = {ICMR ’20}
}
```
All rights are reserved by the authors.
## References
- [zzs1994/DJSRH](https://github.com/zzs1994/DJSRH)
