This is the implementation of the paper 
"Generalized Semantic Preserving Hashing for N-Label Cross-Modal Retrieval"
in CVPR 2017
The article can be found here :
http://openaccess.thecvf.com/content_cvpr_2017/papers/Mandal_Generalized_Semantic_Preserving_CVPR_2017_paper.pdf

***********************************************************
The implementations provided are for the Wiki and NUS-Wide 
dataset. You can replicate the results of the paper using that.

[1] Please unzip the markSchmidt.zip file to get started.
[2] Put the data in .mat file in the folder datasets.

Kindly look into the following programs to better understand
essence of the algorithm
(1) generate_hash_codes2.m
(2) generate_hash_codes6.m
(3) generate_hash_codes7.m

This implementation uses the post-unification startegy.

Please change the number of iterations (as you seem fit) or run
the algorithm until convergence.
In case you need the data kindly contact me separately.
***********************************************************

***********************************************************
In case you are needed to use this code for unpaired scenarios
you just need to remove the post-unification strategy.
***********************************************************

***********************************************************
I am unable to give the codes for the normal cross-modal operations
due to license issues. Please download the data from the places 
as instructed in the paper. For the CNN features you need to download 
the dataset and extract the features yourself.
I had used matconvnet at http://www.vlfeat.org/matconvnet/ to do that

However to get you started, I have provided the following codes to you
so that the evaluation can be done quickly
(1) retrieval.m - code to compute the PrecisionatK and NDCGatK
***********************************************************
