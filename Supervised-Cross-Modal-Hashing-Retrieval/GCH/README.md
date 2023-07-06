# GCH
Graph Convolutional Network Hashing for Cross-Modal Retrieval, IJCAI2019


### Introduction
we propose a Graph Convolutional Hashing (GCH) approach, which learns modality-unified binary codes via an affinity graph. For more details, please refer to our
[paper](https://see.xidian.edu.cn/faculty/chdeng/Welcome%20to%20Cheng%20Deng's%20Homepage_files/Papers/Conference/IJCAI2019_Ruiqing.pdf).

<!-- ![alt text](http://cs.rochester.edu/u/zyang39/VG_ICCV19.jpg 
"Framework") -->
<p align="center">
  <img src=fig/framework.png width="75%"/>
</p>

### Citation

    @inproceedings{xu2019graph,
    title={Graph Convolutional Network Hashing for Cross-Modal Retrieval.},
    author={Xu, Ruiqing and Li, Chao and Yan, Junchi and Deng, Cheng and Liu, Xianglong},
    booktitle={Ijcai},
    pages={982--988},
    year={2019}
    }

### Prerequisites

* Python 2.7
* Tensorflow 1.2.0
* Others (numpy, scipy, h5py, etc.)

## Installation

1. Clone the repository

    ```
    git clone https://github.com/DeXie0808/GCH.git
    ```

2. Prepare the dataset and the pretrained model.

* Dataset: Flickr25k dataset
Please download Flickr25k dataset: [FLICKR-25k.mat](https://drive.google.com/file/d/14OqnNZGEzbSt4oc0WvH1qeSxtlhpPsmF/view?usp=sharing) and place the data under ``./data``.

* Pretrained model: vggf
Please download the pretrained vggf model: [imagenet-vgg-f.mat](https://drive.google.com/file/d/12BnVxvuumYfLi9afHmgJGTePrp4Wg_q7/view?usp=sharing) and place the data under ``./data/weight``.

* Mean of ImageNet: mean
Please download the mean of the ImageNet: [Mean.h5](https://drive.google.com/file/d/1tMaQBW_PwJIZPSkeNmnApFiIC7rF7KUO/view?usp=sharing) and place the data under ``./data/weight``.


### Training
3. Train the model, run the code under main folder. 
Change ``setting.py``, use ``phase='train'``

    ```
    python main_itpair.py
    ```

4. Evaluate the model, run the code under main folder. 
Change ``setting.py``, use ``phase='test'``

    ```
    python main_itpair.py
    ```
