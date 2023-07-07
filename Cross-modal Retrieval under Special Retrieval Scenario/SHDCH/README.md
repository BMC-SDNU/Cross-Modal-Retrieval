# our method 
Demo on FashionVC dataset. 

## Description 
This is a simplified demo for the paper：Supervised Hierarchical Deep Hashing for Cross-Modal Retrieval, including:  
load_data.py: function to load data.  
img_net.py: net structure for image modality.  
txt_net.py: net structure for text modality.   
demo.py: a demo for our method on FashionVC dataset.  
other: tools to evaluate the preformance of the method.  

## Version 
* Python  3.5 
* Tensorflow  1.13.1 

## How-to run 
1. Download the [FashionVC](https://dl.acm.org/doi/10.1145/3123266.3123314) dataset, and put it into ./DataSet  
    you can download dataset from pan.baidu.com  
    	link: [https://pan.baidu.com/s/1VZwdU8MhWkvVmpMjrFJktw](https://pan.baidu.com/s/1VZwdU8MhWkvVmpMjrFJktw)  
    	password: sreu   
2. Download the [Ssense](https://dl.acm.org/doi/10.1145/3331184.3331229) dataset, and put it into ./DataSet  
    you can download dataset from pan.baidu.com  
    	link: [https://pan.baidu.com/s/1RZsSZY5pY2GSAQEu5ciqAw](https://pan.baidu.com/s/1RZsSZY5pY2GSAQEu5ciqAw)  
    	password: qwmq
3. Download the imagenet-vgg-f.mat, and put it into ./DataSet  
    you can download it from pan.baidu.com  
    	link: [https://pan.baidu.com/s/1jM9ZPXGLIykw4SLzk3_3Wg](https://pan.baidu.com/s/1jM9ZPXGLIykw4SLzk3_3Wg)  
     	password: 9i4a
4. run demo.py
