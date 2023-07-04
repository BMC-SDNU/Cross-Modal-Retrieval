### codes for "Adversarial Cross-modal Retrieval (ACMR)"

## Environment
python2.7(highlight)
tensorflow

## Dataset
Wikipedia and NUS-WIDE-10k datasets are in the "data" folder
Please "unzip" the datasets in "data" folder

To run the demo: 

    # Wikipedia dataset
    python train_adv_crossmodal_simple_wiki.py # with contrastive loss
    python train_adv_crossmodal_triplet_wiki.py # with triplet loss

	# NUS-WIDE-10K
    python train_adv_crossmodal_simple_nuswide.py # with contrastive loss



Note: The codes were modified from the implementation of "Unsupervised Cross-modal Retrieval through Adversarial Learning", written by <a href="https://www.linkedin.com/in/ritsu1228/">Li He</a>. The code is largely based on [this repo](https://github.com/sunpeng981712364/ACMR_demo). If you use the codes, please cite the following two papers: 

[1]  Li He, Xing Xu, Huimin Lu, Yang Yang, Fumin Shen and Heng Tao Shen.  "Unsupervised Cross-modal Retrieval through Adversarial Learning". IEEE International Conference on Multimedia and Expo (ICME), 2017. 

[2]  Bokun Wang, Yang Yang, Xing Xu, Alan Hanjalic, and Heng Tao Shen. "Adversarial Cross-Modal Retrieval". In Proceedings of 25th ACM International Conference on Multimedia (ACM MM), 2017.
