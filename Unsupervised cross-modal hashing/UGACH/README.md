# Introduction
This is the source code of our AAAI 2018 paper "Unsupervised Generative Adversarial Cross-modal Hashing", Please cite the following paper if you use our code.

Jian Zhang, Yuxin Peng, and Mingkuan Yuan, "Unsupervised Generative Adversarial Cross-modal Hashing", 32th AAAI Conference on Artificial Intelligence (AAAI), New Orleans, Louisiana, USA, Feb. 2–7, 2018. [[PDF]](http://59.108.48.34/mipl/tiki-download_file.php?fileId=461)

# Usage
For NUSWIDE dataset:

1. Generate KNN graph by the codes under KNN directory: python knn_nus_cross5.py
2. Pretrain the model by using the code under pretrain directory: python train_16.py
3. Train the model by using the code under UGACH-nus: python train_16.py
4. Generate hash codes for query and database samples: python test_16.py

MIRFlickr dataset is similar to the NUSWIDE dataset.

The samples of the MIRFlickr input files have been under samples directory. Each line of the files indicates a feature vector of the training data, which have been detailed in the paper. The test and the validation datasets have the same format as the training dataset. 

We use vgg19 model and adopt the output of fc7 layers as the images features, the text feature of NUS datasets come from the official dataset, the text feature of MIRFlickr come from https://github.com/jiangqy/DCMH-CVPR2017

We provide the training features in NUS-WIDE, the feature extreaction tools (from https://github.com/cvjena/cnn-models) and the final model of 16 bit hash in nus-wide (dis_best__nn_16.model). You can download and use the feature extract tools to extract the feature for test set by yourself. We resized the images to 255x225 and crop into 224x224(crop is achieved by Caffe). the query set is randomly selected from the test set.

download:https://pan.baidu.com/s/17vV-LThqKR2cStOHoa26jg

passwd:oiaz

For more information, please refer to our [AAAI paper](http://59.108.48.34/mipl/tiki-download_file.php?fileId=461).

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information about our papers, source codes, and datasets.
