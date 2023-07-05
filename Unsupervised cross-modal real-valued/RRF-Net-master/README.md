## Code for our ICCV paper 

"Learning a Recurrent Residual Fusion Network for Multimodal Matching".

please cite this paper, if the codes have been used for your research.

## The pipeline of RRF-Net
![architecture](https://github.com/yuLiu24/RRF-Net/blob/master/models/RRF-Net.jpg)

## Data
- Please download the Flickr30K and MS-COCO datasets from their websites.
- The data is stored in LMDB, like "image feature + text feature + image ID".
- Each image has five sentence-level descriptions, and so five descriptions corresponds to the same image ID.
- It is necessary to randomly shuffle the data when converting them into LMDB.

## Train 
- The training prototxt files are in ./models. 
- We do not test the model during the training stage.

## Test
- extract_features.py: extract the L2-norm image and text embedding featutres.
- computePR.m: evaluate the performance of image-to-text and text-to-image retrieval.

## References
"Learning Deep Structure-Preserving Image-Text Embeddings", CVPR 2016.

"Fisher vectors derived from hybrid gaussian-laplacian mixture models for image annotation", CVPR 2015.

## Acknowledgments: 
This code is based on Caffe. Thanks to the contributors of Caffe. 





