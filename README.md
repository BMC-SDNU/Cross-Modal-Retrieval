# Cross-modal Retrieval

- [Introduction](#introduction)
- [Supported Methods](#supported-methods)
    - [Unsupercised-cross-modal-hashing-retrieval](#Unsupercised-cross-modal-hashing-retrieval)
        - [Matrix Factorization](#Matrix-Factorization)
        - [Graph Theory](#Graph-Theory)
        - [Other Shallow](#Other-Shallow)
        - [Quantization](#Quantization)
        - [Naive Network](#Naive-Network)
        - [GAN](#GAN)
        - [Graph Model](#Graph-Model)
        - [Knowledge Distillation](#Knowledge-Distillation)
    - [supercised-cross-modal-hashing-retrieval](#supercised-cross-modal-hashing-retrieval)

    - [Unsupervised-cross-modal-real-valued](#Unsupervised-cross-modal-real-valued)
        - [CCA](#cca)
        - [Topic Model](#topic-model)
        - [Other Shallow](#other-shallow)
        - [Neural Network](#neural-network)
        - [Naive Network](#native-network)
        - [Dot-product Attention](#dot-product-attention)
        - [Graph Model](#graph-model)
        - [Transformer](#transformer)
        - [Cross-modal Generation](#cross-modal-generation)
    - [Supervised-cross-modal-real-valued](#Supervised-cross-modal-real-valued)
        - [CCA](#CCA)
        - [Dictionary Learning](#Dictionary-Learning)
        - [Feature Mapping](#Feature-Mapping)
        - [Topic Model](#Topic-Model)
        - [Other Shallow](#Other-Shallow)
        - [Naive Network](#Naive-Network)
        - [GAN](#GAN)
        - [Graph Model](#Graph-Model)
        - [Transformer](#Transformer)
- [Usage](#usage)

# Introduction
This library is an open-source repository that contains Unsupervised cross-modal real-valued methods and codes.

# Supported Methods
The currently supported algorithms include:

## Unsupervised-cross-modal-real-valued

### CCA
#### 2017

- **ICCA:Towards Improving Canonical Correlation Analysis for Cross-modal Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/abs/10.1145/3126686.3126726)

#### 2015

- **DCMIT:Deep Correlation for Matching Images and Text(CVPR)** [[PDF]](https://ieeexplore.ieee.org/document/7298966)

- **RCCA:Learning Query and Image Similarities with Ranking Canonical Correlation Analysis(ICCV)** [[PDF]](https://ieeexplore.ieee.org/document/7410369)

#### 2014


- **MCCA:A Multi-View Embedding Space for Modeling Internet Images, Tags, and Their Semantics(IJCV)** [[PDF]](https://arxiv.org/abs/1212.4522)

#### 2013

- **KCCA:Framing Image Description as a Ranking Task Data, Models and Evaluation Metrics(JAIR)** [[PDF]](https://www.ijcai.org/Proceedings/15/Papers/593.pdf)

- **DCCA:Deep Canonical Correlation Analysis(ICML)** [[PDF]](https://proceedings.mlr.press/v28/andrew13.html) [[Code]](https://github.com/Michaelvll/DeepCCA)

#### 2012

- **CR:Continuum Regression for Cross-modal Multimedia Retrieval(ICIP)** [[PDF]](https://ieeexplore.ieee.org/document/6467268)

#### 2010


- **CCA:A New Approach to Cross-Modal Multimedia Retrieval(MM)** [[PDF]](http://www.mit.edu/~rplevy/papers/rasiwasia-etal-2010-acm.pdf)[[Code]](https://github.com/emanuetre/crossmodal)

### Topic Model

#### 2011

- **MDRF:Learning Cross-modality Similarity for Multinomial Data(ICCV)** [[PDF]](https://ieeexplore.ieee.org/document/6126524)

#### 2010

- **tr-mmLDA:Topic Regression Multi-Modal Latent Dirichlet Allocation for Image Annotation(CVPR)** [[PDF]](https://ieeexplore.ieee.org/document/5540000)

#### 2003

- **Corr-LDA:Modeling Annotated Data(SIGIR)** [[PDF]](https://www.cs.columbia.edu/~blei/papers/BleiJordan2003.pdf)

### Other Shallow

#### 2013

- **Bi-CMSRM:Cross-Media Semantic Representation via Bi-directional Learning to Rank(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/2502081.2502097)

- **CTM:Cross-media Topic Mining on Wikipedia(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/2502081.2502180)

#### 2012

- **CoCA:Dimensionality Reduction on Heterogeneous Feature Space(ICDM)** [[PDF]](https://ieeexplore.ieee.org/document/6413864)

#### 2011

- **MCU:Maximum Covariance Unfolding: Manifold Learning for Bimodal Data(NIPS)** [[PDF]](https://proceedings.neurips.cc/paper/2011/file/daca41214b39c5dc66674d09081940f0-Paper.pdf)

#### 2008

- **PAMIR:A Discriminative Kernel-Based Model to Rank Images from Text Queries(TPAMI)** [[PDF]](https://ieeexplore.ieee.org/document/4359384)

#### 2003

- **CFA:Multimedia Content Processing through Cross-Modal Association(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/957013.957143)

### Neural Network

#### 2018

- **CDPAE:Comprehensive Distance-Preserving Autoencoders for Cross-Modal Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3240508.3240607)[[Code]](https://github.com/Atmegal/Comprehensive-Distance-Preserving-Autoencoders-for-Cross-Modal-Retrieval)


#### 2016

- **CMDN:Cross-Media Shared Representation by Hierarchical Learning with Multiple Deep Networks(IJCAI)** [[PDF]](https://www.ijcai.org/Proceedings/16/Papers/541.pdf)[[Code]]()

- **MSAE:Effective deep learning-based multi-modal retrieval(VLDB)** [[PDF]](https://dl.acm.org/doi/10.1007/s00778-015-0391-4)

#### 2014

- **Corr-AE:Cross-modal Retrieval with Correspondence Autoencoder(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/2647868.2654902)

#### 2013

- **RGDBN:Latent Feature Learning in Social Media Network(MM)** [[PDF]](https://dl.acm.org/doi/abs/10.1145/2502081.2502284)

#### 2012

- **MDBM:Multimodal Learning with Deep Boltzmann Machines(NIPS)** [[PDF]](https://jmlr.org/papers/volume15/srivastava14b/srivastava14b.pdf)


### Native Network

#### 2022

- **UWML:Universal Weighting Metric Learning for Cross-Modal Retrieval (TPAMI)** [[PDF]](https://ieeexplore.ieee.org/document/9454290)[[Code]](https://github.com/wayne980/PolyLoss)

- **LESS:Learning to Embed Semantic Similarity for Joint Image-Text Retrieval (TPAMI)**[[PDF]](https://ieeexplore.ieee.org/document/9633145)

- **CMCM:Cross-Modal Coherence for Text-to-Image Retrieval (AAAI)** [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/download/21285/version/19572/21034)

- **P2RM:Point to Rectangle Matching for Image Text Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/abs/10.1145/3503161.3548237)

#### 2020

- **DPCITE:Dual-path Convolutional Image-Text Embeddings with Instance Loss(TOMM)** [[PDF]](https://arxiv.org/abs/1711.05535) [[code]](https://github.com/layumi/Image-Text-Embedding)

- **PSN:Preserving Semantic Neighborhoods for Robust Cross-Modal Retrieval(ECCV)** [[PDF]](https://arxiv.org/abs/2007.08617) [[Code]](https://github.com/CLT29/semantic_neighborhoods)

#### 2019

- **LDR:Learning Disentangled Representation for Cross-Modal Retrieval with Deep Mutual Information Estimation(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3343031.3351053)

#### 2018

- **CHAIN-VSE:Bidirectional Retrieval Made Simple(CVPR)** [[PDF]](https://ieeexplore.ieee.org/document/8578903) [[Code]](https://github.com/jwehrmann/chain-vse)

#### 2017

- **CRC:Cross-media Relevance Computation for Multimedia Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3123266.3123963)

- **VSE++: Improving Visual-Semantic Embeddings with Hard Negatives:(Arxiv)** [[PDF]](https://arxiv.org/abs/1707.05612)[[Code]](https://github.com/fartashf/vsepp)

- **RRF-Net:Learning a Recurrent Residual Fusion Network for Multimodal Matching(ICCV)** [[PDF]](https://ieeexplore.ieee.org/document/8237704)[[Code]](https://github.com/yuLiu24/RRF-Net)

#### 2016

- **DBRLM:Cross-Modal Retrieval via Deep and Bidirectional Representation Learning(TMM)** [[PDF]](https://ieeexplore.ieee.org/abstract/document/7460254)

#### 2015

- **MSDS:Image-Text Cross-Modal Retrieval via Modality-Speciﬁc Feature Learning(ICMR)** [[PDF]](https://dl.acm.org/doi/10.1145/2671188.2749341)

#### 2014

- **DT-RNN:Grounded Compositional Semantics for Finding and Describing Images with Sentences(TACL)** [[PDF]](https://aclanthology.org/Q14-1017.pdf)

### Dot-product Attention

#### 2020

- **SMAN: Stacked Multimodal Attention Network for Cross-Modal Image-Text Retrieval(TC)** [[PDF]](https://ieeexplore.ieee.org/document/9086164)

- **CAAN:Context-Aware Attention Network for Image-Text Retrieval(CVPR)** [[PDF]](https://ieeexplore.ieee.org/document/9157657)

- **IMRAM: Iterative Matching with Recurrent Attention Memory for Cross-Modal Image-Text Retrieval(CVPR)** [[PDF]](https://arxiv.org/abs/2003.03772) [[Code]](https://github.com/HuiChen24/IMRAM)

#### 2019

- **PFAN:Position Focused Attention Network for Image-Text Matching (IJCAI)** [[PDF]](https://arxiv.org/abs/1907.09748)[[Code]](https://github.com/HaoYang0123/Position-Focused-Attention-Network)

- **CAMP: Cross-Modal Adaptive Message Passing for Text-Image Retrieval(ICCV)** [[PDF]](https://arxiv.org/abs/1909.05506) [[Code]](https://github.com/ZihaoWang-CV/CAMP_iccv19)

- **CMRSC:Cross-Modal Image-Text Retrieval with Semantic Consistency(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3343031.3351055) [[Code]](https://github.com/HuiChen24/MM_SemanticConsistency)

#### 2018

- **MCSM:Modality-specific Cross-modal Similarity Measurement with Recurrent Attention Network(TIP)** [[PDF]](https://arxiv.org/abs/1708.04776)[[Code]](https://github.com/PKU-ICST-MIPL/MCSM_TIP2018)

- **DSVEL:Finding beans in burgers: Deep semantic-visual embedding with localization(CVPR)** [[PDF]](https://arxiv.org/abs/1804.01720)[[Code]](https://github.com/technicolor-research/dsve-loc)

- **CRAN:Cross-media Multi-level Alignment with Relation Attention Network(IJCAI)**[[PDF]](https://www.ijcai.org/proceedings/2018/124)

- **SCAN:Stacked Cross Attention for Image-Text Matching(ECCV)** [[PDF]](https://arxiv.org/abs/1803.08024) [[Code]](https://github.com/kuanghuei/SCAN)

#### 2017

- **sm-LSTM:Instance-aware Image and Sentence Matching with Selective Multimodal LSTM(CVPR)** [[PDF]](https://arxiv.org/abs/1611.05588)

### Graph Model

#### 2022

- **LHSC:Learning Hierarchical Semantic Correspondences for Cross-Modal Image-Text Retrieval(ICMR)** [[PDF]](https://dl.acm.org/doi/10.1145/3512527.3531358)

- **IFRFGF:Improving Fusion of Region Features and Grid Features via Two-Step Interaction for Image-Text Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3503161.3548223)

- **CODER:Coupled Diversity-Sensitive Momentum Contrastive Learning for Image-Text Retrieval(ECCV)** [[PDF]](https://dl.acm.org/doi/abs/10.1007/978-3-031-20059-5_40)

#### 2021

- **HSGMP: Heterogeneous Scene Graph Message Passing for Cross-modal Retrieval(ICMR)** [[PDF]](https://dl.acm.org/doi/10.1145/3460426.3463650)

- **WCGL：Wasserstein Coupled Graph Learning for Cross-Modal Retrieval(ICCV)**[[PDF]](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Wasserstein_Coupled_Graph_Learning_for_Cross-Modal_Retrieval_ICCV_2021_paper.html)

#### 2020


- **DSRAN:Learning Dual Semantic Relations with Graph Attention for Image-Text Matching(TCSVT)** [[PDF]](https://arxiv.org/abs/2010.11550) [[code]](https://github.com/kywen1119/DSRAN)

- **VSM:Visual-Semantic Matching by Exploring High-Order Attention and Distraction(CVPR)** [[PDF]](https://ieeexplore.ieee.org/document/9157630)

- **SGM:Cross-modal Scene Graph Matching for Relationship-aware Image-Text Retrieval(WACV)** [[PDF]](https://arxiv.org/abs/1910.05134)

#### 2019

- **KASCE:Knowledge Aware Semantic Concept Expansion for Image-Text Matching(IJCAI)** [[PDF]](https://www.ijcai.org/proceedings/2019/720)

- **VSRN:Visual Semantic Reasoning for Image-Text Matching(ICCV)** [[PDF]](https://arxiv.org/abs/1909.02701) [[Code]](https://github.com/KunpengLi1994/VSRN)

### Transformer

#### 2022

- **DREN:Dual-Level Representation Enhancement on Characteristic and Context for Image-Text Retrieval(TCSVT)** [[PDF]](https://ieeexplore.ieee.org/document/9794669)

- **M2D-BERT:Multi-scale Multi-modal Dictionary BERT For Effective Text-image Retrieval in Multimedia Advertising(CIKM)** [[PDF]](https://dl.acm.org/doi/abs/10.1145/3511808.3557653)

- **ViSTA:ViSTA: Vision and Scene Text Aggregation for Cross-Modal Retrieval(CVPR)** [[PDF]](https://arxiv.org/abs/2203.16778)

- **COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval(CVPR)** [[PDF]](https://arxiv.org/abs/2204.07441)

- **EI-CLIP: Entity-aware Interventional Contrastive Learning for E-commerce Cross-modal Retrieval(CVPR)** [[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_EI-CLIP_Entity-Aware_Interventional_Contrastive_Learning_for_E-Commerce_Cross-Modal_Retrieval_CVPR_2022_paper.pdf)

- **SSAMT:Constructing Phrase-level Semantic Labels to Form Multi-Grained Supervision for Image-Text Retrieval(ICMR)** [[PDF]](https://arxiv.org/abs/2109.05523)

- **TEAM:Token Embeddings Alignment for Cross-Modal Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3503161.3548107)

- **CAliC: Accurate and Efficient Image-Text Retrieval via Contrastive Alignment and Visual Contexts Modeling(MM)** [[PDF]](https://dl.acm.org/doi/abs/10.1145/3503161.3548320)


#### 2021

- **GRAN:Global Relation-Aware Attention Network for Image-Text Retrieval(ICMR)** [[PDF]](https://dl.acm.org/doi/abs/10.1145/3460426.3463615)

- **PCME:Probabilistic Embeddings for Cross-Modal Retrieval(CVPR)** [[PDF]](https://arxiv.org/abs/2101.05068) [[code]](https://github.com/naver-ai/pcme)

#### 2020

- **FashionBERT: Text and Image Matching with Adaptive Loss for Cross-modal Retrieval(SIGIR)** [[PDF]](https://arxiv.org/abs/2005.09801)

#### 2019

- **PVSE:Polysemous Visual-Semantic Embedding for Cross-Modal Retrieval(CVPR)** [[PDF]](https://arxiv.org/abs/1906.04402) [[Code]](https://github.com/yalesong/pvse)

### Cross-modal Generation

#### 2022

- **PCMDA:Paired Cross-Modal Data Augmentation for Fine-Grained Image-to-Text Retrieval(MM)**[[PDF]](https://arxiv.org/abs/2207.14428)

#### 2021

- **CRGN:Deep Relation Embedding for Cross-Modal Retrieval(TIP)** [[PDF]](https://ieeexplore.ieee.org/document/9269483)[[Code]](https://github.com/zyfsa/CRGN)

- **X-MRS:Cross-Modal Retrieval and Synthesis (X-MRS): Closing the Modality Gapin Shared Representation Learning(MM)** [[PDF]](https://arxiv.org/abs/2012.01345)[[Code]](https://github.com/SamsungLabs/X-MRS)

#### 2020

- **AACR:Augmented Adversarial Training for Cross-Modal Retrieval(TMM)** [[PDF]](https://ieeexplore.ieee.org/document/9057710) [[Code]](https://github.com/yiling2018/aacr)

#### 2018

- **LSCO:Learning Semantic Concepts and Order for Image and Sentence Matching(CVPR)** [[PDF]](https://arxiv.org/abs/1712.02036)

- **TCCM:Towards Cycle-Consistent Models for Text and Image Retrieval(CVPR)** [[PDF]](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11132/Cornia_Towards_Cycle-Consistent_Models_for_Text_and_Image_Retrieval_ECCVW_2018_paper.pdf)

- **GXN:Look, Imagine and Match: Improving Textual-Visual Cross-Modal Retrieval with Generative Models(CVPR)** [[PDF]](https://arxiv.org/abs/1711.06420)

#### 2017

- **2WayNet:Linking Image and Text with 2-Way Nets(CVPR)** [[PDF]](https://arxiv.org/abs/1608.07973)

#### 2015

- **DVSA:Deep Visual-Semantic Alignments for Generating Image Descriptions(CVPR)** [[PDF]](https://arxiv.org/abs/1412.2306)

## Supervised-cross-modal-real-valued

### CCA

#### 2022

- **MVMLCCA: Multi-view Multi-label Canonical Correlation Analysis for Cross-modal Matching and Retrieval**  [[PDF]](https://openaccess.thecvf.com/content/CVPR2022W/MULA/html/Sanghavi_Multi-View_Multi-Label_Canonical_Correlation_Analysis_for_Cross-Modal_Matching_and_Retrieval_CVPRW_2022_paper.html) [[Code]](https://github.com/Rushil231100/MVMLCCA)

#### 2015

- **ml-CCA: Multi-Label Cross-modal Retrieval**  [[PDF]](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Ranjan_Multi-Label_Cross-Modal_Retrieval_ICCV_2015_paper.html) [[Code]](https://github.com/Viresh-R/ml-CCA)

#### 2014

- **cluster-CCA: Cluster Canonical Correlation Analysis**  [[PDF]](https://proceedings.mlr.press/v33/rasiwasia14.html)

#### 2012

- **GMA: Generalized Multiview Analysis: A Discriminative Latent Space**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/6247923) [[Code]](https://github.com/huyt16/Twitter100k/tree/master/code/GMA-CVPR2012)

### Dictionary Learning

#### 2018
- **JDSLC: Joint Dictionary Learning and Semantic Constrained Latent Subspace Projection for Cross-Modal Retrieval**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3269206.3269296)

#### 2016
- **DDL: Discriminative Dictionary Learning With Common Label Alignment for Cross-Modal Retrieval**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7353179)

#### 2014

- **CMSDL: Cross-Modality Submodular Dictionary Learning for Information Retrieval**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2661829.2661926)

#### 2013

- **SliM2: Supervised Coupled Dictionary Learning with Group Structures for Multi-Modal Retrieval**  [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/8603)

### Feature Mapping

#### 2017

- **MDSSL: Cross-Modal Retrieval Using Multiordered Discriminative Structured Subspace Learning**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7801820)

- **JLSLR: Joint Latent Subspace Learning and Regression for Cross-Modal Retrieval**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3077136.3080678)

#### 2016

- **JFSSL: Joint Feature Selection and Subspace Learning for Cross-Modal Retrieval**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7346492) [[Code]](https://github.com/2012013382/JFSSL-Cross-Modal-Retrieval)

- **MDCR: Modality-Dependent Cross-Media Retrieval**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2775109)

- **CRLC: Cross-modal Retrieval with Label Completion**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2964284.2967231)

#### 2013

- **JGRHML: Heterogeneous Metric Learning with Joint Graph Regularization for Cross-Media Retrieval**  [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/8464) [[Code]](https://github.com/PKU-ICST-MIPL/JGRHML_AAAI2013)

- **LCFS: Learning Coupled Feature Spaces for Cross-modal Matching**  [[PDF]](https://openaccess.thecvf.com/content_iccv_2013/html/Wang_Learning_Coupled_Feature_2013_ICCV_paper.html)

#### 2011

- **Multi-NPP: Learning Multi-View Neighborhood Preserving Projections**  [[PDF]](https://icml.cc/2011/papers/304_icmlpaper.pdf)

### Topic Model

#### 2014

- **M3R: Multi-modal Mutual Topic Reinforce Modeling for Cross-media Retrieval**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2647868.2654901)

- **NPBUS: Nonparametric Bayesian Upstream Supervised Multi-Modal Topic Models**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2556195.2556238)

### Other Shallow

#### 2019

- **CMOS: Online Asymmetric Metric Learning With Multi-Layer Similarity Aggregation for Cross-Modal Retrieval**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/8680035)

#### 2017

- **CMOS: Online Asymmetric Similarity Learning for Cross-Modal Retrieval**  [[PDF]](https://openaccess.thecvf.com/content_cvpr_2017/html/Wu_Online_Asymmetric_Similarity_CVPR_2017_paper.html)

#### 2016

- **PL-ranking: A Novel Ranking Method for Cross-Modal Retrieval**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2964284.2964336)

- **RL-PLS: Cross-modal Retrieval by Real Label Partial Least Squares**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2964284.2967216)

#### 2013

- **PFAR: Parallel Field Alignment for Cross Media Retrieval**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2502081.2502087)

### Naive Network

#### 2022

- **C3CMR: Cross-Modality Cross-Instance Contrastive Learning for Cross-Media Retrieval**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3503161.3548263)

#### 2020

- **ED-Net: Event-Driven Network for Cross-Modal Retrieval**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3340531.3412081)

#### 2019

- **DSCMR: Deep Supervised Cross-modal Retrieval**  [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhen_Deep_Supervised_Cross-Modal_Retrieval_CVPR_2019_paper.html) [[Code]](https://github.com/penghu-cs/DSCMR)

- **SAM: Cross-Modal Subspace Learning with Scheduled Adaptive Margin Constraints**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3343031.3351030)

#### 2017

- **deep-SM: Cross-Modal Retrieval With CNN Visual Features: A New Baseline**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7428926) [[Code]](https://github.com/zh-plus/CMR-CNN-New-Baseline)

- **CCL: Cross-modal Correlation Learning With Multigrained Fusion by Hierarchical Network**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/8013822)

- **MSFN: Cross-media Retrieval by Learning Rich Semantic Embeddings of Multimedia**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3123266.3123369)

- **MNiL: Multi-Networks Joint Learning for Large-Scale Cross-Modal Retrieval**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3123266.3123317) [[Code]](https://github.com/liangzhang1407/Multi-Networks-Joint-Learning-for-Large-Scale-Cross-Modal-Retrieval)

#### 2016

- **MDNN: Effective deep learning-based multi-modal retrieval**  [[PDF]](https://link.springer.com/article/10.1007/s00778-015-0391-4)

#### 2015

- **RE-DNN: Deep Semantic Mapping for Cross-Modal Retrieval**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7372141)

- **C2MLR: Deep Compositional Cross-modal Learning to Rank via Local-Global Alignment**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2733373.2806240)

### GAN

#### 2022

- **JFSE: Joint Feature Synthesis and Embedding: Adversarial Cross-Modal Retrieval Revisited**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9296975) [[Code]](https://github.com/CFM-MSG/Code_JFSE)

#### 2021

- **AACR: Augmented Adversarial Training for Cross-Modal Retrieval**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9057710) [[Code]](https://github.com/yiling2018/aacr)

#### 2018

- **CM-GANs: Cross-modal Generative Adversarial Networks for Common Representation Learning**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3284750) [[Code]](https://github.com/PKU-ICST-MIPL/CM-GANS_TOMM2019)

#### 2017

- **ACMR: Adversarial Cross-Modal Retrieval**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3123266.3123326) [[Code]](https://github.com/cuishuhao/ACMR)

### Graph Model

#### 2022

- **AGCN: Adversarial Graph Convolutional Network for Cross-Modal Retrieval**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9411880)

- **ALGCN: Adaptive Label-Aware Graph Convolutional Networks for Cross-Modal Retrieval**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9508809)

- **HGE: Cross-Modal Retrieval with Heterogeneous Graph Embedding**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3503161.3548195)

#### 2021

- **GCR: Exploring Graph-Structured Semantics for Cross-Modal Retrieval**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3474085.3475567) [[Code]](https://github.com/neoscheung/GCR)

- **DAGNN: Dual Adversarial Graph Neural Networks for Multi-label Cross-modal Retrieval**  [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/16345)

#### 2018

- **SSPE: Learning Semantic Structure-preserved Embeddings for Cross-modal Retrieval**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3240508.3240521)

### Transformer

#### 2021

- **RLCMR: Rethinking Label-Wise Cross-Modal Retrieval from A Semantic Sharing Perspective**  [[PDF]](https://www.ijcai.org/proceedings/2021/0454.pdf)

# Usage

## Datasets

- **Graph Model--GCR**

Dataset Link:

    Baidu Yun Link: https://pan.baidu.com/s/1YmW8Zz2uK3AgCs6pDEoA8A?pwd=21xh
    Code: 21xh

- **Unsupervised cross-modal real-valued**

Dataset link:

    Baidu Yun Link：https://pan.baidu.com/s/1hBNo8gBSyLbik0ka1POhiQ 
    Code：cc53
