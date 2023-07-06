---
Source code for paper "Discrete Latent Factor Model for Cross-Modal Hashing"
---
## Introduction
### 0. About the paper
This repo is the source code for the paper "Discrete Latent Factor Model for Cross-Modal Hashing" on TIP-2019. The authors are: [Qing-Yuan Jiang](https://jiangqy.github.io/) and [Wu-Jun Li](http://cs.nju.edu.cn/lwj). If you have any questions about the source code, please contact: qyjiang24#gmail.com.

We provide two versions for DLFH, i.e., matlab version and python version. Please see  matlab_version and python_version folder , respectively.
### 1. Running Environment
For matlab version.

```matlab
Matlab 2016
```

For python version.

```python
python 3
```

### 2. Datasets
We use three datasets to perform our experiments, i.e., MIRFLICKR-25K, IAPR-TC12 and NUS-WIDE datasets.

#### 2.1. Dataset download:

You can download all dataset from pan.baidu.com. The links are listed as follows:


- [FLICKR-25K](https://pan.baidu.com/s/14WkNMvfTdobZ_t29RShXpA ) Passwd: pism

- [IAPR-TC12](https://pan.baidu.com/s/1k17NEH-F0NColkBkTRoupA) Passwd: hf8j

- [NUS-WIDE](https://pan.baidu.com/s/1l_m3ktrrCJIEQshA-ezOuw) Password: cmdi


### 3. Run demo

#### 3.1. Matlab version:

Run matlab\_version/DLFH_demo.m.

```matlab
DLFH_demo
```

#### 3.2. Python version:

Run python\_version/dlfh_demo.py

```python
python dlfh_demo.py --bit 8
```

Please note that we only present a simple example for python version. Hence we only implement DLFH for python version. All exmperiments in the paper are based on matlab version.
