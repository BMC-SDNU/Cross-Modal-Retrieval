# Introduction
This is the source code of our TCYB 2018 paper "SCH-GAN: Semi-supervised Cross-modal Hashing by Generative Adversarial Network", Please cite the following paper if you use our code.

Jian Zhang, Yuxin Peng and Mingkuan Yuan, "SCH-GAN: Semi-supervised Cross-modal Hashing by Generative Adversarial Network", IEEE Transactions on Cybernetics (TCYB), 2018. [[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=201810)


# Usage
For NUSWIDE dataset:

1. Pretrain the model by using the code under pretrain directory (pretrain-nus): python train.py
2. Train the model by using the code under SCHGAN-nus: python train.py
3. Generate hash codes for query and database samples by using the code under SCHGAN-nus: python test.py

Wikipedia and MIRFlickr datasets are similar to the NUSWIDE dataset.

For more information, please refer to our [TCYB paper](http://59.108.48.34/tiki/download_paper.php?fileId=201810).

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information about our papers, source codes, and datasets.
