# A New Benchmark and Approach for Fine-grained Cross-media Retrieval

# Introduction
This repository contains the pytorch codes, trained models, and the new benchmarks described in our ACM MM 2019 paper "[A New Benchmark and Approach for Fine-grained Cross-media Retrieval](https://arxiv.org/abs/1907.04476)".

For more details, please visit our [project page](http://59.108.48.34/tiki/FGCrossNet/).

**Results**

- The MAP scores of bi-modality fine-grained cross-media retrieval of our FGCrossNet

|        | I->T | I->A | I->V | T->I | T->A | T->V | A->I | A->T | A->V | V->I | V->T | V->A | Average |
| :----: | :---: | :----: | :---: | :---: | :----: | :---: | :----: | :---: | :---: | :----: | :---: | :----: | :---: |
|FGCrossNet(ours)|  0.210  |  0.526 |  0.606 |  0.255  |  0.181 |  0.208 |  0.553  |  0.159 |  0.443 |  0.629  |  0.195 |  0.437 |  0.366 |



- The MAP scores of multi-modality fine-grained cross-media retrieval of our FGCrossNet

|        | I->All | T->All | V->All | A->All | Average |
| :----: | :----: | :-------: | :---: | :-----: | :-----: |
|FGCrossNet(ours)| 0.549  |   0.196    |  0.416 |  0.485   |  0.412   |


# Installation

**Requirement**

- pytorch, tested on [v1.0]
- CUDA, tested on v9.0
- Language: Python 3.6

## 1. Download dataset

Please visit our [project page](http://59.108.48.34/tiki/FGCrossNet/).

## 2. Download trained model
The trained models of our FGCrossNet framework can be downloaded from [OneDrive](https://1drv.ms/u/s!AvXsEBcM-dJyaVcIN0jU0SJ_YRU?e=a4ZTcZ), [Google Drive](https://drive.google.com/open?id=1Cyfh6073MXm-jOjUWU0HFGI4esEGh3KM) or [Baidu Cloud](https://pan.baidu.com/s/1oFofvfoIvsNwXhb-b8i9Og).


## 3. Prepare audio data
python audio.py


## 4. Training
sh train.sh


## 5. Testing
sh test.sh


# Citing
```
@inproceedings{he2019fine,
    Author = {Xiangteng He, Yuxin Peng, Liu Xie},
    Title = {A New Benchmark and Approach for Fine-grained Cross-media Retrieval},
    Booktitle = {Proc. of ACM International Conference on Multimedia (ACM MM)},
    Year = {2019}
} 
```

# Contributing
For any questions, feel free to open an issue or contact us. ([hexiangteng@pku.edu.cn]())
