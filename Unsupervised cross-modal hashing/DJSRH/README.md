# DJSRH
***********************************************************************************************************

This repository is for ["Deep Joint-Semantics Reconstructing Hashing for Large-Scale Unsupervised Cross-Modal Retrieval"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Su_Deep_Joint-Semantics_Reconstructing_Hashing_for_Large-Scale_Unsupervised_Cross-Modal_Retrieval_ICCV_2019_paper.pdf) 

(to appear in ICCV 2019, Oral)

By Shupeng Su\*, [Zhisheng Zhong](https://zzs1994.github.io)\*, Chao Zhang (\* Authors contributed equally).


***********************************************************************************************************
## Table of contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Ablation studies](#ablation-studies)
- [Comparisons with SOTAs](#comparisons-with-SOTAs)
***********************************************************************************************************

## Introduction

Cross-modal hashing encodes the multimedia data into a common binary hash space in which the correlations among the samples from different modalities can be effectively measured. Deep cross-modal hashing further improves the retrieval performance as the deep neural networks can generate more semantic relevant features and hash codes. In this paper, we study the unsupervised deep cross-modal hash coding and propose Deep JointSemantics Reconstructing Hashing (DJSRH), which has the following two main advantages. First, to learn binary codes that preserve the neighborhood structure of the original data, DJSRH constructs a novel joint-semantics affinity matrix which elaborately integrates the original neighborhood information from different modalities and accordingly is capable to capture the latent intrinsic semantic affinity for the input multi-modal instances. Second, DJSRH later trains the networks to generate binary codes that maximally reconstruct above joint-semantics relations via the proposed reconstructing framework, which is more competent for the batch-wise training as it reconstructs the specific similarity value unlike the common Laplacian constraint merely preserving the similarity order. Extensive experiments demonstrate the significant improvement by DJSRH in various cross-modal retrieval tasks.

<div align=center><img src="https://github.com/zzs1994/DJSRH/blob/master/page_image/DJRSH.png" width="90%" height="90%"></div align=center>

***********************************************************************************************************

## Usage
### Requirements
- python == 2.7.x
- pytorch == 0.3.1
- torchvision
- CV2
- PIL
- h5py

### Datasets
For datasets, we follow [Deep Cross-Modal Hashing's Github (Jiang, CVPR 2017)](https://github.com/jiangqy/DCMH-CVPR2017/tree/master/DCMH_matlab/DCMH_matlab). You can download these datasets from:
- Wikipedia articles, [Link](http://www.svcl.ucsd.edu/projects/crossmodal/)
- MIRFLICKR25K, [[OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EpLD8yNN2lhIpBgQ7Kl8LKABzM68icvJJahchO7pYNPV1g?e=IYoeqn)], [[Baidu Pan](https://pan.baidu.com/s/1o5jSliFjAezBavyBOiJxew), password: 8dub]
- NUS-WIDE (top-10 concept), [[OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EoPpgpDlPR1OqK-ywrrYiN0By6fdnBvY4YoyaBV5i5IvFQ?e=kja8Kj)], [[Baidu Pan](https://pan.baidu.com/s/1GFljcAtWDQFDVhgx6Jv_nQ), password: ml4y]


### Process

__The following experiment results are the average values, if you demand for better results, please run the experiment a few more times (2~5).__

- Clone this repo: `git clone https://github.com/zzs1994/DJSRH.git`.
- Change the 'DATASET_DIR' in `settings.py` to where you place the datasets.
- An example to train a model:
```bash
python train.py
```
- Modify the parameter `EVAL = True` in `settings.py` for validation.
- Ablation studies (__optional__): if you want to evaluate other components of our DJSRH, please refer to our paper and `settings.py`.

***********************************************************************************************************

## Ablation studies
Table 1. The mAP@50 results on NUS-WIDE to evaluate the effectiveness of each component in DJSRH.

<center>

Model|Configuration|64bits (I→T)|64bits (T→I)|128bits (I→T)|128bits (T→I)|
|:---------:|:---:|:-----:|:----:|:----:|:----:|
DJSRH-1|S=S<sub>I</sub>|0.717|0.712|0.741|0.735|
DJSRH-2|S=S<sub>T</sub>|0.702|0.606|0.734|0.581|
DJSRH-3|&beta;S<sub>I</sub>+(1−&beta;)S<sub>T</sub>|0.724|0.720|0.747|0.738|
DJSRH-4|+(&eta;=0.4)|0.790|0.745|0.803|0.757|
DJSRH-5|+(&mu;=1.5)|0.793|0.747|0.812|0.768|
DJSRH|+(&lambda;<sub>1</sub>=&lambda;<sub>2</sub>=0.1)|__0.798__|__0.771__|__0.817__|__0.789__|
DJSRH-6|−(&alpha;=1)|0.786|0.770|0.811|0.782|

</center>

From the table we can observe that each of our proposed components plays a certain role for our final results.

***********************************************************************************************************

## Comparisons with SOTAs
Table 2. The mAP@50 results on image query text (I→T) and text query image (T→I) retrieval tasks at various encoding lengths and
datasets. The best performances are shown as <font color="red">Red</font> while the suboptimal as <font color="blue">Blue</font>.
<div align=center><img src="https://github.com/zzs1994/DJSRH/blob/master/page_image/results.png" width="90%" height="90%"></div align=center>

Figure 1. The precision@top-R curves on different datasets at 128 encoding length.
<div align=center><img src="https://github.com/zzs1994/DJSRH/blob/master/page_image/results_curve.png" width="90%" height="90%"></div align=center>

***********************************************************************************************************
## Citation

If you find this code useful, please cite our paper:

	@inproceedings{su2019deep,
		title={Deep Joint-Semantics Reconstructing Hashing for Large-Scale Unsupervised Cross-Modal Retrieval},
		author={Shupeng Su, Zhisheng Zhong, Chao Zhang},
		booktitle={International Conference on Computer Vision},
		year={2019}
	}

All rights are reserved by the authors.
***********************************************************************************************************