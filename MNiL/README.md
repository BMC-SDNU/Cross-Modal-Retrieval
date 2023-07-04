# Multi-Networks-Joint-Learning-for-Large-Scale-Cross-Modal-Retrieval

This code is provided for the ACM MM'17 paper "Multi-networks Joint Learning for Large Scale Cross-Model Retrieval".

This code is written in Lua and requires Torch. The preprocssinng code is in Python. If you want process your task, please domnload the Deep Residual Network "resnet-50.t7"

%Train the model
We have everything ready to train the model. Back to the main folder

th coco_train.lua

% Test the model
When you complete the training, you should run the following code:
th coco_test.lua

This code will calculate the Mean Average Precision for your testing set.

Please refer to the following paper:
Liang Zhang, Bingpeng Ma, Guorong Li, Qingming Huang, Qi Tian. 
Multi-networks Joint Learning for Large Scale Cross-Model Retrieval.
In Proc. ACM MM'17

If you have some questions, please contact liang.zhang@vipl.ict.ac.cn
