# Introduction

This is the source code of our TOMM 2019 paper "CM-GANs: Cross-modal Generative Adversarial Networks for Common Representation Learning", Please cite the following paper if you find our code useful.

Yuxin Peng, Jinwei Qi, "CM-GANs: Cross-modal Generative Adversarial Networks for Common Representation Learning", ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), Vol.15, No.1, pp.22:1-22:24, 2019. [[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=201827)

# Preparation
Our code is based on [torch](http://torch.ch/docs/getting-started.html), and tested on Ubuntu 14.04.5 LTS, Lua 5.1.

# Usage
Data Preparation: we use pascal dataset as example, and the data should be put in `./data/`.
The data files can be download from the [link](http://59.108.48.34/tiki/tiki-download_file.php?fileId=1011) and unzipped to the above path.

run `sh run.sh` to train models, extract features and calculate mAP.

# Our Related Work
If you are interested in cross-media retrieval, you can check our recently published overview paper on IEEE TCSVT:

Yuxin Peng, Xin Huang, and Yunzhen Zhao, "An Overview of Cross-media Retrieval: Concepts, Methodologies, Benchmarks and Challenges", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol.28, No.9, pp.2372-2385, 2018. [[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=201823)

Welcome to our [Benchmark Website](http://www.icst.pku.edu.cn/mipl/xmedia) and [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information about our papers, source codes, and datasets.
