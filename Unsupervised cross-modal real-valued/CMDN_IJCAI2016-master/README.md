# Introduction
This is the source code of our IJCAI 2016 paper "Cross-media Shared Representation by Hierarchical Learning with Multiple Deep Networks", Please cite the following paper if you use our code.

Yuxin Peng, Xin Huang, and Jinwei Qi, "Cross-media Shared Representation by Hierarchical Learning with Multiple Deep Networks", 25th International Joint Conference on Artificial Intelligence (IJCAI), pp. 3846-3853 , New York City, New York, USA, Jul. 9-15, 2016. [[PDF]](http://59.108.48.34/mipl/tiki-download_file.php?fileId=314)

# Usage
1.Environment

Set up deepnet as the instruction of deepnet-master/INSTALL.txt.
  
2.Data

cd to the deepnet-master/deepnet/examples/CMDN/feature dir.  
put the data with matlab format in this folder, and run mat2npy.py to convert matlab format to numpy format. Detailed data format please see in mat2npy.py.
  
3.Set

parameter 'size' and 'dimensions' in the following files need to be modified according to the data scale:  
-sae_img/data/wikipedia.pbtxt  
-sae_txt/data/wikipedia.pbtxt  
-multimodal_dbn/data/wikipedia.pbtxt  
where parameter 'size' means the number of data, parameter 'dimensions' means the dimension of data.  
  
4.Run

	$sh runall.sh

For more information, please refer to our [paper](http://59.108.48.34/mipl/tiki-download_file.php?fileId=314)

# Our Related Work
If you are interested in cross-media retrieval, you can check our recently published overview paper on IEEE TCSVT:

Yuxin Peng, Xin Huang, and Yunzhen Zhao, "An Overview of Cross-media Retrieval: Concepts, Methodologies, Benchmarks and Challenges", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2017.[[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=201823)

Welcome to our [Benchmark Website](http://59.108.48.34/mipl/xmedia) and [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information about our papers, source codes, and datasets.
