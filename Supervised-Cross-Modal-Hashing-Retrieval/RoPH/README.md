# [Cross-modal Hashing via Rank-order Preserving](https://ieeexplore.ieee.org/document/7737053)

This code implements the following paper: 

```
@article{ding2016cross,
  title={Cross-modal hashing via rank-order preserving},
  author={Ding, Kun and Fan, Bin and Huo, Chunlei and Xiang, Shiming and Pan, Chunhong},
  journal={IEEE Transactions on Multimedia},
  volume={19},
  number={3},
  pages={571--585},
  year={2017}
}
```



The code is tested on 64-bit CentOS Linux 7.1.1503 (Core) system with MATLAB 2014b and 64-bit Windows 10 system with MATLAB 2014a. It includes:

1. [demo.m](demo.m): demo code for RoPH.
2. [RoPH_train.m](codes/RoPH/RoPH_train.m): implements the training of RoPH.
3. [RoPH_test.m](code/RoPH/RoPH_test.m): implements the testing of RoPH.
4. [gen_triplets_ml.m](code/RoPH/gen_triplets_ml.m): generates triplet information for multi-label data.

To try the code, please run [demo.m](demo.m) under the main folder. If all goes well, it will print the NDCG of image and text query for RoPH and one state-of-the-art method SePH, CVPR'15. 



If you have any questions, please contact me by Email: kun.ding AT ia.ac.cn.
