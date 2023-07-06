# Cross-modal Retrieval

- [1. Introduction](#1-introduction)
- [2. Supported Methods](#2-supported-methods)
    - [2.1 Unsupercised-cross-modal-hashing-retrieval](#21-unsupervised-cross-modal-hashing-retrieval)
        - [2.1.1 Matrix Factorization](#211-matrix-factorization)
        - [2.1.2 Graph Theory](#212-graph-theory)
        - [2.1.3 Other Shallow](#213-other-shallow)
        - [2.1.4 Quantization](#214-quantization)
        - [2.1.5 Naive Network](#215-naive-network)
        - [2.1.6 GAN](#216-gan)
        - [2.1.7 Graph Model](#217-graph-model)
        - [2.1.8 Knowledge Distillation](#218-knowledge-distillation)
    - [2.2 Supercised-cross-modal-hashing-retrieval](#22-supercised-cross-modal-hashing-retrieval)
        - [2.2.1 Matrix Factorization](#221-matrix-factorization)
        - [2.2.2 Dictionary Learning](#222-Dictionary-Learning)
        - [2.2.3 Feature Mapping-Sample-Constraint-Label-Constraint](#223-Feature-Mapping-Sample-Constraint-Label-Constraint)
        - [2.2.4 Feature Mapping-Sample-Constraint-Separate-Hamming](#224-Feature-Mapping-Sample-Constraint-Separate-Hamming)
        - [2.2.5 Feature Mapping-Sample-Constraint-Common-Hamming](#225-Feature-Mapping-Sample-Constraint-Common-Hamming)
        - [2.2.6 Feature Mapping-Relation-Constraint](#226-Feature-Mapping-Relation-Constraint)
        - [2.2.7 Other Shallow](#227-Other-Shallow)
        - [2.2.8 Naive Network-Distance-Constraint](#228-Naive-Network-Distance-Constraint)
        - [2.2.9 Naive Network-Similarity-Constraint](#229-Naive-Network-Similarity-Constraint)
        - [2.2.10 Naive Network-Negative-Log-Likelihood](#2210-Naive-Network-Negative-Log-Likel-ihood)
        - [2.2.11 Naive Network-Triplet-Constraint](#2211-Naive-Network-Triplet-Constraint)
        - [2.2.12 GAN](#2212-gan)
        - [2.2.13 Graph Model](#2213-graph-model)
        - [2.2.14 Transformer](#2214-transformer)
        - [2.2.15 Memory Network](#2215-memory-Network)
        - [2.2.16 Quantization](#2216-quantization)
    - [2.3 Unsupervised-cross-modal-real-valued](#23-unsupervised-cross-modal-real-valued)
        - [2.3.1 CCA](#231-cca)
        - [2.3.2 Topic Model](#232-topic-model)
        - [2.3.3 Other Shallow](#233-other-shallow)
        - [2.3.4 Neural Network](#234-neural-network)
        - [2.3.5 Naive Network](#235-native-network)
        - [2.3.6 Dot-product Attention](#236-dot-product-attention)
        - [2.3.7 Graph Model](#237-graph-model)
        - [2.3.8 Transformer](#238-transformer)
        - [2.3.9 Cross-modal Generation](#239-cross-modal-generation)
    - [2.4 Supervised-cross-modal-real-valued](#24-supervised-cross-modal-real-valued)
        - [2.4.1 CCA](#241-cca)
        - [2.4.2 Dictionary Learning](#242-dictionary-learning)
        - [2.4.3 Feature Mapping](#243-feature-mapping)
        - [2.4.4 Topic Model](#244-topic-model)
        - [2.4.5 Other Shallow](#245-other-shallow)
        - [2.4.6 Naive Network](#246-naive-network)
        - [2.4.7 GAN](#247-gan)
        - [2.4.8 Graph Model](#248-graph-model)
        - [2.4.9 Transformer](#249-transformer)
    - [2.5 Extended cross-modal hashing retrieval](#25-extended-cross-modal-hashing-retrieval)
        - [2.5.1 Semi-Supervised (Real-valued)](#251-semi-supervised-real-valued)
        - [2.5.2Semi-Supervised (Hashing)](#252-semi-supervised-hashing)
        - [2.5.3Imbalance (Real-valued)](#253-imbalance-real-valued)
        - [2.5.4Imbalance (Hashing)](#254-imbalance-hashing)
        - [2.5.5Incremental](#255-incremental)

- [3. Usage](#3-usage)

# 1. Introduction
This library is an open-source repository that contains Unsupervised cross-modal real-valued methods and codes.

# 2. Supported Methods
The currently supported algorithms include:

## 2.1 Unsupervised cross-modal hashing retrieval

### 2.1.1 Matrix Factorization

#### 2017

- **RFDH：Robust and Flexible Discrete Hashing for Cross-Modal Similarity Search(TCSVT)** [[PDF]](https://ieeexplore.ieee.org/abstract/document/7967838) [[Code]](https://github.com/Wangdi-Xidian/RFDH)

#### 2015

- **STMH:Semantic Topic Multimodal Hashing for Cross-Media Retrieval(IJCAI)**[[PDF]](https://www.ijcai.org/Proceedings/15/Papers/546.pdf)

#### 2014

- **LSSH:Latent Semantic Sparse Hashing for Cross-Modal Similarity Search(SIGIR)**[[PDF]](https://dl.acm.org/doi/10.1145/2600428.2609610)

- **CMFH:Collective Matrix Factorization Hashing for Multimodal Data(CVPR)**[[PDF]](https://ieeexplore.ieee.org/document/6909664)

### 2.1.2 Graph Theory

#### 2018

- **HMR:Hetero-Manifold Regularisation for Cross-Modal Hashing(TPAMI)**[[PDF]](https://ieeexplore.ieee.org/abstract/document/7801124)

#### 2017

- **FSH:Cross-Modality Binary Code Learning via Fusion Similarity Hashing(CVPR)**[[PDF]](https://ieeexplore.ieee.org/abstract/document/8100155)[[Code]](https://github.com/LynnHongLiu/FSH)

#### 2014

- **SM2H:Sparse Multi-Modal Hashing(TMM)**[[PDF]](https://ieeexplore.ieee.org/document/6665155)

#### 2013

- **IMH:Inter-Media Hashing for Large-scale Retrieval from Heterogeneous Data Sources(SIGMOD)**[[PDF]](https://dl.acm.org/doi/10.1145/2463676.2465274)

- **LCMH:Linear Cross-Modal Hashing for Efﬁcient Multimedia Search(MM)**[[PDF]](https://dl.acm.org/doi/10.1145/2502081.2502107)

#### 2011

- **CVH:Learning Hash Functions for Cross-View Similarity Search(IJCAI)**[[PDF]](https://dl.acm.org/doi/10.5555/2283516.2283623)

### 2.1.3 Other Shallow

#### 2019

- **CRE:Collective Reconstructive Embeddings for Cross-Modal Hashing(TIP)**[[PDF]](https://ieeexplore.ieee.org/document/8594588)

#### 2018

- **HMR:Hetero-Manifold Regularisation for Cross-Modal Hashing(TPAMI)**[[PDF]](https://ieeexplore.ieee.org/abstract/document/7801124)

#### 2015

- **FS-LTE:Full-Space Local Topology Extraction for Cross-Modal Retrieval(TIP)**[[PDF]](https://ieeexplore.ieee.org/document/7076613)

#### 2014

- **IMVH:Iterative Multi-View Hashing for Cross Media Indexing(MM)**[[PDF]](https://dl.acm.org/doi/10.1145/2647868.2654906)

#### 2013

- **PDH:Predictable Dual-View Hashing(ICML)**[[PDF]](https://dl.acm.org/doi/10.5555/3042817.3043085)

### 2.1.4 Quantization

#### 2016

- **CCQ:Composite Correlation Quantization for Efﬁcient Multimodal Retrieval(SIGIR)**[[PDF]](https://arxiv.org/abs/1504.04818)

- **CMCQ:Collaborative Quantization for Cross-Modal Similarity Search(CVPR)**[[PDF]](https://arxiv.org/abs/1902.00623)

#### 2015

- **ACQ:Alternating Co-Quantization for Cross-modal Hashing(ICCV)**[[PDF]](https://ieeexplore.ieee.org/document/7410576)

### 2.1.5 Naive Network

#### 2019

- **UDFCH:Unsupervised Deep Fusion Cross-modal Hashing(ICMI)**[[PDF]](https://dl.acm.org/doi/fullHtml/10.1145/3340555.3353752)

#### 2018

- **UDCMH:Unsupervised Deep Hashing via Binary Latent Factor Models for Large-scale Cross-modal Retrieval(IJCAI)**[[PDF]](https://www.ijcai.org/proceedings/2018/396)

#### 2017 

- **DBRC:Deep Binary Reconstruction for Cross-modal Hashing(MM)**[[PDF]](https://arxiv.org/abs/1708.05127)

#### 2015

- **DMHOR:Learning Compact Hash Codes for Multimodal Representations Using Orthogonal Deep Structure(TMM)**[[PDF]](https://ieeexplore.ieee.org/document/7154455)

### 2.1.6 GAN 

#### 2020

- **MGAH:Multi-Pathway Generative Adversarial Hashing for Unsupervised Cross-Modal Retrieval(TMM)**[[PDF]](https://ieeexplore.ieee.org/document/8734835)

#### 2019

- **CYC-DGH:Cycle-Consistent Deep Generative Hashing for Cross-Modal Retrieval(TIP)**[[PDF]](https://arxiv.org/abs/1804.11013)

- **UCH:Coupled CycleGAN: Unsupervised Hashing Network for Cross-Modal Retrieval(AAAI)**[[PDF]](https://arxiv.org/abs/1903.02149)

#### 2018

- **UGACH:Unsupervised Generative Adversarial Cross-modal Hashing(AAAI)**[[PDF]](https://arxiv.org/abs/1712.00358)[[Code]](https://github.com/PKU-ICST-MIPL/UGACH_AAAI2018)

### 2.1.7 Graph Model

#### 2022

- **ASSPH:Adaptive Structural Similarity Preserving for Unsupervised Cross Modal Hashing(MM)**[[PDF]](https://dl.acm.org/doi/abs/10.1145/3503161.3548431)

#### 2021

- **AGCH:Aggregation-based Graph Convolutional Hashing for Unsupervised Cross-modal Retrieval(TMM)**[[PDF]](https://ieeexplore.ieee.org/document/9335490)

- **DGCPN:Deep Graph-neighbor Coherence Preserving Network for Unsupervised Cross-modal Hashing(AAAI)**[[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/16592)[[Code]](https://github.com/Atmegal/DGCPN)

#### 2020

- **DCSH:Unsupervised Deep Cross-modality Spectral Hashing(TIP)**[[PDF]](https://arxiv.org/abs/2008.00223)

- **SRCH:Set and Rebase: Determining the Semantic Graph Connectivity for Unsupervised Cross-Modal Hashing(IJCAI)**[[PDF]](https://www.ijcai.org/proceedings/2020/0119.pdf)

- **JDSH:Joint-modal Distribution-based Similarity Hashing for Large-scale Unsupervised Deep Cross-modal Retrieval(SIGIR)**[[PDF]](https://dl.acm.org/doi/10.1145/3397271.3401086)[[Code]](https://github.com/KaiserLew/JDSH)

- **DSAH:Deep Semantic-Alignment Hashing for Unsupervised Cross-Modal Retrieval(ICMR)**[[PDF]](https://dl.acm.org/doi/10.1145/3372278.3390673)[[Code]](https://github.com/idejie/DSAH)

#### 2019

- **DJSRH:Deep Joint-Semantics Reconstructing Hashing for Large-Scale Unsupervised Cross-Modal Retrieval(ICCV)**[[PDF]](https://ieeexplore.ieee.org/document/9009571)[[Code]](https://github.com/zs-zhong/DJSRH)

### 2.1.8 Knowledge Distillation

#### 2022

- **DAEH:Deep Adaptively-Enhanced Hashing With Discriminative Similarity Guidance for Unsupervised Cross-Modal Retrieval(TCSVT)**[[PDF]](https://ieeexplore.ieee.org/document/9768805)

#### 2021

- **KDCMH:Unsupervised Deep Cross-Modal Hashing by Knowledge Distillation for Large-scale Cross-modal Retrieval(ICMR)**[[PDF]](https://dl.acm.org/doi/10.1145/3460426.3463626)

- **JOG:Joint-teaching: Learning to Refine Knowledge for Resource-constrained Unsupervised Cross-modal Retrieval(MM)**[[PDF]](https://dl.acm.org/doi/10.1145/3474085.3475286)

#### 2020

- **UKD:Creating Something from Nothing: Unsupervised Knowledge Distillation for Cross-Modal Hashing(CVPR)**[[PDF]](https://arxiv.org/abs/2004.00280)

## 2.2 Supercised-cross-modal-hashing-retrieval

### 2.2.1 Matrix Factorization

#### 2022

- **SCLCH: Joint Specifics and Consistency Hash Learning for Large-Scale Cross-Modal Retrieval(TIP)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9850431)

#### 2020

- **BATCH: A Scalable Asymmetric Discrete Cross-Modal Hashing(TKDE)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9001235) [[Code]](https://github.com/yxinwang/BATCH-TKDE2020)

#### 2019

- **LCMFH: Label Consistent Matrix Factorization Hashing for Large-Scale Cross-Modal Similarity Search(TPAMI)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/8423193)

- **TECH: A Two-Step Cross-Modal Hashing by Exploiting Label Correlations and Preserving Similarity in Both Steps(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3343031.3350862)

#### 2018

- **SCRATCH: A Scalable Discrete Matrix Factorization Hashing for Cross-Modal Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3240508.3240547)

#### 2017

- **DCH: Learning Discriminative Binary Codes for Large-scale Cross-modal Retrieval(TIP)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7867785)

#### 2016

- **SMFH: Supervised Matrix Factorization for Cross-Modality Hashing(IJCAI)**  [[PDF]](https://arxiv.org/abs/1603.05572)

- **SMFH: Supervised Matrix Factorization Hashing for Cross-Modal Retrieval(TIP)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7466099)

### 2.2.2 Dictionary Learning

#### 2016

- **DCDH: Discriminative Coupled Dictionary Hashing for Fast Cross-Media Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2600428.2609563)

#### 2014

- **DLCMH: Dictionary Learning Based Hashing for Cross-Modal Retrieval(SIGIR)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2964284.2967206)

### 2.2.3 Feature Mapping-Sample-Constraint-Label-Constraint

#### 2022

- **DJSAH: Discrete Joint Semantic Alignment Hashing for Cross-Modal Image-Text Search(TCSVT)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9808188)

#### 2020

- **FUH: Fast Unmediated Hashing for Cross-Modal Retrieval(TCSVT)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9285286)

#### 2016

- **MDBE: Multimodal Discriminative Binary Embedding for Large-Scale Cross-Modal Retrieval(TIP)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/7515190) [[Code]](https://github.com/Wangdi-Xidian/MDBE)

### 2.2.4 Feature Mapping-Sample-Constraint-Separate-Hamming

#### 2017

- **CSDH: Sequential Discrete Hashing for Scalable Cross-Modality Similarity Retrieval(TIP)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/7600368)

#### 2016

- **DASH: Frustratingly Easy Cross-Modal Hashing(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2964284.2967218)

#### 2015

- **QCH: Quantized Correlation Hashing for Fast Cross-Modal Search(IJCAI)** [(PDF)](https://dl.acm.org/doi/abs/10.5555/2832747.2832799)

### 2.2.5 Feature Mapping-Sample-Constraint-Common Hamming

#### 2021

- **ASCSH: Asymmetric Supervised Consistent and Speciﬁc Hashing for Cross-Modal Retrieval(TIP)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9269445) [[Code]](https://github.com/minmengzju/ASCSH)

#### 2019

- **SRDMH: Supervised Robust Discrete Multimodal Hashing for Cross-Media Retrieval(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8695043)

#### 2018

- **FDCH: Fast Discrete Cross-modal Hashing With Regressing From Semantic Labels(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3240508.3240683)

#### 2017

- **SRSH: Semi-Relaxation Supervised Hashing for Cross-Modal Retrieval(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3123266.3123320) [[Code]](https://github.com/sduzpf/SRSH)

- **RoPH: Cross-Modal Hashing via Rank-Order Preserving(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/7737053) [[Code]](https://github.com/kding1225/RoPH)

#### 2016

- **SRDMH: Supervised Robust Discrete Multimodal Hashing for Cross-Media Retrieval(CIKM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2983323.2983743)

### 2.2.6 Feature Mapping-Relation-Constraint

#### 2017

- **LSRH: Linear Subspace Ranking Hashing for Cross-Modal Retrieval(TPAMI)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/7571151)

#### 2014

- **SCM: Large-Scale Supervised Multimodal Hashing with Semantic Correlation Maximization(AAAI)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/8995)

- **HTH: Scalable Heterogeneous Translated Hashing(KDD)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2623330.2623688)

#### 2013

- **PLMH: Parametric Local Multimodal Hashing for Cross-View Similarity Search(IJCAI)** [(PDF)](https://repository.hkust.edu.hk/ir/Record/1783.1-60904)

- **RaHH: Comparing Apples to Oranges: A Scalable Solution with Heterogeneous Hashing(KDD)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2487575.2487668) [[Code]](https://github.com/boliu68/RaHH)

#### 2012

- **CRH: Co-Regularized Hashing for Multimodal Data(CRH)** [(PDF)](https://proceedings.neurips.cc/paper_files/paper/2012/hash/5c04925674920eb58467fb52ce4ef728-Abstract.html)

### 2.2.7 Other Shallow

#### 2019

- **DLFH: Discrete Latent Factor Model for Cross-Modal Hashing(TIP)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8636536) [[Code]](https://github.com/jiangqy/DLFH-TIP2019)

#### 2018

- **SDMCH: Supervised Discrete Manifold-Embedded Cross-Modal Hashing(IJCAI)** [(PDF)](https://dl.acm.org/doi/abs/10.5555/3304889.3305010)

#### 2015

- **SePH: Semantics-Preserving Hashing for Cross-View Retrieval(CVPR)** [(PDF)](https://openaccess.thecvf.com/content_cvpr_2015/html/Lin_Semantics-Preserving_Hashing_for_2015_CVPR_paper.html)

#### 2012

- **MLBE: A Probabilistic Model for Multimodal Hash Function Learning(KDD)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2339530.2339678)

#### 2010

- **CMSSH: Data Fusion through Cross-modality Metric Learning using Similarity-Sensitive Hashing(CVPR)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/5539928)

### 2.2.8 Naive Network-Distance-Constraint

#### 2019

- **MCITR: Cross-modal Image-Text Retrieval with Multitask Learning(CIKM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3357384.3358104)

#### 2016

- **CAH: Correlation Autoencoder Hashing for Supervised Cross-Modal Search(ICMR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2911996.2912000)

#### 2014

- **CMNNH: Cross-Media Hashing with Neural Networks(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2647868.2655059)

- **MMNN: Multimodal Similarity-Preserving Hashing(TPAMI)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/6654144)

### 2.2.9 Naive Network-Similarity-Constraint

#### 2022

- **Bi-CMR: Bidirectional Reinforcement Guided Hashing for Effective Cross-Modal Retrieval(AAAI)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/21268) [[Code]](https://github.com/lty4869/Bi-CMR)

- **Bi-NCMH: Deep Normalized Cross-Modal Hashing with Bi-Direction Relation Reasoning(CVPR)** [(PDF)](https://openaccess.thecvf.com/content/CVPR2022W/ODRUM/html/Sun_Deep_Normalized_Cross-Modal_Hashing_With_Bi-Direction_Relation_Reasoning_CVPRW_2022_paper.html)

#### 2021

- **OTCMR: Bridging Heterogeneity Gap with Optimal Transport for Cross-modal Retrieval(CIKM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3459637.3482158)

- **DUCMH: Deep Uniﬁed Cross-Modality Hashing by Pairwise Data Alignment(IJCAI)** [(PDF)](https://cs.nju.edu.cn/_upload/tpl/01/0c/268/template268/pdf/IJCAI-2021-Wang.pdf)

#### 2020

- **NRDH: Nonlinear Robust Discrete Hashing for Cross-Modal Retrieval(SIGIR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3397271.3401152)

- **DCHUC: Deep Cross-Modal Hashing with Hashing Functions and Uniﬁed Hash Codes Jointly Learning(TKDE)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9069300) [[Code]](https://github.com/rongchengtu1/DCHUC)

#### 2017

- **CHN: Correlation Hashing Network for Efﬁcient Cross-Modal Retrieval(BMVC)** [(PDF)](https://arxiv.org/abs/1602.06697)

#### 2016

- **DVSH: Deep Visual-Semantic Hashing for Cross-Modal Retrieval(KDD)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2939672.2939812)

### 2.2.10 Naive Network-Negative-Log-Likelihood

#### 2022

- **MSSPQ: Multiple Semantic Structure-Preserving Quantization for Cross-Modal Retrieval(ICMR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3512527.3531417)

#### 2021

- **DMFH: Deep Multiscale Fusion Hashing for Cross-Modal Retrieval(TCSVT)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9001018)

- **TEACH: Attention-Aware Deep Cross-Modal Hashing(ICMR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3460426.3463625)

#### 2020

- **MDCH: Mask Cross-modal Hashing Networks(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9055057)

#### 2019

- **EGDH: Equally-Guided Discriminative Hashing for Cross-modal Retrieval(IJCAI)** [(PDF)](https://dl.acm.org/doi/abs/10.5555/3367471.3367706)

#### 2018

- **DDCMH: Dual Deep Neural Networks Cross-Modal Hashing(AAAI)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/11249)

- **CMHH: Cross-Modal Hamming Hashing(ECCV)** [(PDF)](https://openaccess.thecvf.com/content_ECCV_2018/html/Yue_Cao_Cross-Modal_Hamming_Hashing_ECCV_2018_paper.html)

#### 2017

- **PRDH: Pairwise Relationship Guided Deep Hashing for Cross-Modal Retrieval(AAAI)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/10719)

- **DCMH: Deep Cross-Modal Hashing(CVPR)** [(PDF)](https://openaccess.thecvf.com/content_cvpr_2017/html/Jiang_Deep_Cross-Modal_Hashing_CVPR_2017_paper.html) [[Code]](https://github.com/WendellGul/DCMH)

### 2.2.11 Naive Network-Triplet-Constraint

#### 2019

- **RDCMH: Multiple Semantic Structure-Preserving Quantization for Cross-Modal Retrieval(AAAI)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/4351)

#### 2018

- **MCSCH: Multi-Scale Correlation for Sequential Cross-modal Hashing Learning(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3240508.3240560)

- **TDH: Triplet-Based Deep Hashing Network for Cross-Modal Retrieval(TIP)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8331146)

### 2.2.12 GAN

#### 2022

- **SCAHN: Semantic Structure Enhanced Contrastive Adversarial Hash Network for Cross-media Representation Learning(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3503161.3548391) [[Code]](https://github.com/YI1219/SCAHN-MindSpore)

#### 2021

- **TGCR: Multiple Semantic Structure-Preserving Quantization for Cross-Modal Retrieval(TCSVT)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9257382)

#### 2020

- **CPAH: Multi-Task Consistency-Preserving Adversarial Hashing for Cross-Modal Retrieval(TIP)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8954946) [[Code]](https://github.com/comrados/cpah)

- **MLCAH: Multi-Level Correlation Adversarial Hashing for Cross-Modal Retrieval(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8970562)

- **DADH: Deep Adversarial Discrete Hashing for Cross-Modal Retrieval(ICMR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3372278.3390711) [[Code]](https://github.com/Zjut-MultimediaPlus/DADH)

#### 2019

- **AGAH: Adversary Guided Asymmetric Hashing for Cross-Modal Retrieval(ICMR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3323873.3325045) [[Code]](https://github.com/WendellGul/AGAH)

#### 2018

- **SSAH: Self-Supervised Adversarial Hashing Networks for Cross-Modal Retrieval(CVPR)** [(PDF)](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Self-Supervised_Adversarial_Hashing_CVPR_2018_paper.html) [[Code]](https://github.com/lelan-li/SSAH)

### 2.2.13 Graph Model

#### 2022

- **HMAH: Multi-Task Consistency-Preserving Adversarial Hashing for Cross-Modal Retrieval(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9782694)

- **SCAHN: Semantic Structure Enhanced Contrastive Adversarial Hash Network for Cross-media Representation Learning(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3503161.3548391) [[Code]](https://github.com/lelan-li/SSAH)

#### 2021

- **LGCNH: Local Graph Convolutional Networks for Cross-Modal Hashing(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3474085.3475346) [[Code]](https://github.com/chenyd7/LGCNH)

#### 2019

- **GCH: Graph Convolutional Network Hashing for Cross-Modal Retrieval(IJCAI)** [(PDF)](https://dl.acm.org/doi/abs/10.5555/3367032.3367172) [[Code]](https://github.com/DeXie0808/GCH)

### 2.2.14 Transformer

#### 2022

- **DCHMT: Differentiable Cross-modal Hashing via Multimodal Transformers(CIKM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3503161.3548187) [[Code]](https://github.com/kalenforn/DCHMT)

- **UniHash: Contrastive Label Correlation Enhanced Unified Hashing Encoder for Cross-modal Retrieval(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3511808.3557265) [[Code]](https://github.com/idealwhite/UniHash)

### 2.2.15 Memory Network

#### 2021

- **CMPD: Using Cross Memory Network With Pair Discrimination for Image-Text Retrieval(TCSVT)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9169915)

#### 2019

- **CMMN: Deep Memory Network for Cross-Modal Retrieval(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8506385)

### 2.2.16 Quantization

#### 2022

- **ACQH: Asymmetric Correlation Quantization Hashing for Cross-Modal Retrieval(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9517000)

#### 2017

- **CDQ: Collective Deep Quantization for Efﬁcient Cross-Modal Retrieval(AAAI)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/11218) [[Code]](https://github.com/caoyue10/aaai17-cdq)

## 2.3 Unsupervised-cross-modal-real-valued

### 2.3.1 CCA
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

### 2.3.2 Topic Model

#### 2011

- **MDRF:Learning Cross-modality Similarity for Multinomial Data(ICCV)** [[PDF]](https://ieeexplore.ieee.org/document/6126524)

#### 2010

- **tr-mmLDA:Topic Regression Multi-Modal Latent Dirichlet Allocation for Image Annotation(CVPR)** [[PDF]](https://ieeexplore.ieee.org/document/5540000)

#### 2003

- **Corr-LDA:Modeling Annotated Data(SIGIR)** [[PDF]](https://www.cs.columbia.edu/~blei/papers/BleiJordan2003.pdf)

### 2.3.3 Other Shallow

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

### 2.3.4 Neural Network

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


### 2.3.5 Native Network

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

### 2.3.6 Dot-product Attention

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

### 2.3.7 Graph Model

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

### 2.3.8 Transformer

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

### 2.3.9 Cross-modal Generation

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


## 2.4 Supervised-cross-modal-real-valued

### 2.4.1 CCA

#### 2022

- **MVMLCCA: Multi-view Multi-label Canonical Correlation Analysis for Cross-modal Matching and Retrieval(CVPRW)**  [[PDF]](https://openaccess.thecvf.com/content/CVPR2022W/MULA/html/Sanghavi_Multi-View_Multi-Label_Canonical_Correlation_Analysis_for_Cross-Modal_Matching_and_Retrieval_CVPRW_2022_paper.html) [[Code]](https://github.com/Rushil231100/MVMLCCA)

#### 2015

- **ml-CCA: Multi-Label Cross-modal Retrieval(ICCV)**  [[PDF]](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Ranjan_Multi-Label_Cross-Modal_Retrieval_ICCV_2015_paper.html) [[Code]](https://github.com/Viresh-R/ml-CCA)

#### 2014

- **cluster-CCA: Cluster Canonical Correlation Analysis(ICAIS)**  [[PDF]](https://proceedings.mlr.press/v33/rasiwasia14.html)

#### 2012

- **GMA: Generalized Multiview Analysis: A Discriminative Latent Space(CVPR)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/6247923) [[Code]](https://github.com/huyt16/Twitter100k/tree/master/code/GMA-CVPR2012)

### 2.4.2 Dictionary Learning

#### 2018
- **JDSLC: Joint Dictionary Learning and Semantic Constrained Latent Subspace Projection for Cross-Modal Retrieval(CIKM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3269206.3269296)

#### 2016
- **DDL: Discriminative Dictionary Learning With Common Label Alignment for Cross-Modal Retrieval(TMM)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7353179)

#### 2014

- **CMSDL: Cross-Modality Submodular Dictionary Learning for Information Retrieval(CIKM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2661829.2661926)

#### 2013

- **SliM2: Supervised Coupled Dictionary Learning with Group Structures for Multi-Modal Retrieval(AAAI)**  [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/8603)

### 2.4.3 Feature Mapping

#### 2017

- **MDSSL: Cross-Modal Retrieval Using Multiordered Discriminative Structured Subspace Learning(TMM)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7801820)

- **JLSLR: Joint Latent Subspace Learning and Regression for Cross-Modal Retrieval(SIGIR)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3077136.3080678)

#### 2016

- **JFSSL: Joint Feature Selection and Subspace Learning for Cross-Modal Retrieval(TPAIMI)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7346492) [[Code]](https://github.com/2012013382/JFSSL-Cross-Modal-Retrieval)

- **MDCR: Modality-Dependent Cross-Media Retrieval(TIST)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2775109)

- **CRLC: Cross-modal Retrieval with Label Completion(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2964284.2967231)

#### 2013

- **JGRHML: Heterogeneous Metric Learning with Joint Graph Regularization for Cross-Media Retrieval(AAAI)**  [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/8464) [[Code]](https://github.com/PKU-ICST-MIPL/JGRHML_AAAI2013)

- **LCFS: Learning Coupled Feature Spaces for Cross-modal Matching(ICCV)**  [[PDF]](https://openaccess.thecvf.com/content_iccv_2013/html/Wang_Learning_Coupled_Feature_2013_ICCV_paper.html)

#### 2011

- **Multi-NPP: Learning Multi-View Neighborhood Preserving Projections(ICML)**  [[PDF]](https://icml.cc/2011/papers/304_icmlpaper.pdf)

### 2.4.4 Topic Model

#### 2014

- **M3R: Multi-modal Mutual Topic Reinforce Modeling for Cross-media Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2647868.2654901)

- **NPBUS: Nonparametric Bayesian Upstream Supervised Multi-Modal Topic Models(WSDM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2556195.2556238)

### 2.4.5 Other Shallow

#### 2019

- **CMOS: Online Asymmetric Metric Learning With Multi-Layer Similarity Aggregation for Cross-Modal Retrieval(TIP)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/8680035)

#### 2017

- **CMOS: Online Asymmetric Similarity Learning for Cross-Modal Retrieval(CVPR)**  [[PDF]](https://openaccess.thecvf.com/content_cvpr_2017/html/Wu_Online_Asymmetric_Similarity_CVPR_2017_paper.html)

#### 2016

- **PL-ranking: A Novel Ranking Method for Cross-Modal Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2964284.2964336)

- **RL-PLS: Cross-modal Retrieval by Real Label Partial Least Squares(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2964284.2967216)

#### 2013

- **PFAR: Parallel Field Alignment for Cross Media Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2502081.2502087)

### 2.4.6 Naive Network

#### 2022

- **C3CMR: Cross-Modality Cross-Instance Contrastive Learning for Cross-Media Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3503161.3548263)

#### 2020

- **ED-Net: Event-Driven Network for Cross-Modal Retrieval(CIKM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3340531.3412081)

#### 2019

- **DSCMR: Deep Supervised Cross-modal Retrieval(CVPR)**  [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhen_Deep_Supervised_Cross-Modal_Retrieval_CVPR_2019_paper.html) [[Code]](https://github.com/penghu-cs/DSCMR)

- **SAM: Cross-Modal Subspace Learning with Scheduled Adaptive Margin Constraints(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3343031.3351030)

#### 2017

- **deep-SM: Cross-Modal Retrieval With CNN Visual Features: A New Baseline(TCYB)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7428926) [[Code]](https://github.com/zh-plus/CMR-CNN-New-Baseline)

- **CCL: Cross-modal Correlation Learning With Multigrained Fusion by Hierarchical Network(TMM)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/8013822)

- **MSFN: Cross-media Retrieval by Learning Rich Semantic Embeddings of Multimedia(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3123266.3123369)

- **MNiL: Multi-Networks Joint Learning for Large-Scale Cross-Modal Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3123266.3123317) [[Code]](https://github.com/liangzhang1407/Multi-Networks-Joint-Learning-for-Large-Scale-Cross-Modal-Retrieval)

#### 2016

- **MDNN: Effective deep learning-based multi-modal retrieval(VLDB)**  [[PDF]](https://link.springer.com/article/10.1007/s00778-015-0391-4)

#### 2015

- **RE-DNN: Deep Semantic Mapping for Cross-Modal Retrieval(ICTAI)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7372141)

- **C2MLR: Deep Compositional Cross-modal Learning to Rank via Local-Global Alignment(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2733373.2806240)

### 2.4.7 GAN

#### 2022

- **JFSE: Joint Feature Synthesis and Embedding: Adversarial Cross-Modal Retrieval Revisited(TPAMI)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9296975) [[Code]](https://github.com/CFM-MSG/Code_JFSE)

#### 2021

- **AACR: Augmented Adversarial Training for Cross-Modal Retrieval(TMM)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9057710) [[Code]](https://github.com/yiling2018/aacr)

#### 2018

- **CM-GANs: Cross-modal Generative Adversarial Networks for Common Representation Learning(TMM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3284750) [[Code]](https://github.com/PKU-ICST-MIPL/CM-GANS_TOMM2019)

#### 2017

- **ACMR: Adversarial Cross-Modal Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3123266.3123326) [[Code]](https://github.com/cuishuhao/ACMR)

### 2.4.8 Graph Model

#### 2022

- **AGCN: Adversarial Graph Convolutional Network for Cross-Modal Retrieval(TCSVT)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9411880)

- **ALGCN: Adaptive Label-Aware Graph Convolutional Networks for Cross-Modal Retrieval(TMM)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9508809)

- **HGE: Cross-Modal Retrieval with Heterogeneous Graph Embedding(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3503161.3548195)

#### 2021

- **GCR: Exploring Graph-Structured Semantics for Cross-Modal Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3474085.3475567) [[Code]](https://github.com/neoscheung/GCR)

- **DAGNN: Dual Adversarial Graph Neural Networks for Multi-label Cross-modal Retrieval(AAAI)**  [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/16345)

#### 2018

- **SSPE: Learning Semantic Structure-preserved Embeddings for Cross-modal Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3240508.3240521)

### 2.4.9 Transformer

#### 2021

- **RLCMR: Rethinking Label-Wise Cross-Modal Retrieval from A Semantic Sharing Perspective(IJCAI)**  [[PDF]](https://www.ijcai.org/proceedings/2021/0454.pdf)

## 2.5 Extended cross-modal hashing retrieval

### 2.5.1 Semi-Supervised (Real-valued)

#### 2020

- **SSCMR:Semi-Supervised Cross-Modal Retrieval With Label Prediction(TMM)** [[PDF]](https://ieeexplore.ieee.org/document/8907496)

#### 2019

- **A3VSE:Annotation Efficient Cross-Modal Retrieval with Adversarial Attentive Alignment(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3343031.3350894)

- **ASFS:Adaptive Semi-Supervised Feature Selection for Cross-Modal Retrieval(TMM)** [[PDF]](https://ieeexplore.ieee.org/document/8501586)

#### 2018

- **GSS-SL:Generalized Semi-supervised and Structured Subspace Learning for Cross-Modal Retrieval(TMM)** [[PDF]](https://ieeexplore.ieee.org/document/7968473)

#### 2017

- **SSDC:Semi-supervised Distance Consistent Cross-modal Retrieval(VSCC)**[[PDF]](https://dl.acm.org/doi/abs/10.1145/3132734.3132735)

#### 2013

- **JRL:Learning Cross-Media Joint Representation With Sparse and Semisupervised Regularization(TCSVT)** [[PDF]](https://ieeexplore.ieee.org/document/6587747)[[Code]](https://github.com/PKU-ICST-MIPL/JRL_TCSVT2014)

#### 2012

- **MVML-GL:Multiview Metric Learning with Global Consistency and Local Smoothness(TIST)** [[PDF]](https://dl.acm.org/doi/10.1145/2168752.2168767)

### 2.5.2 Semi-Supervised (Hashing)

#### 2020

- **SCH-GAN：Semi-Supervised Cross-Modal Hashing by Generative Adversarial Network(TC)** [[PDF]](https://ieeexplore.ieee.org/document/8472802) [[Code]](https://github.com/PKU-ICST-MIPL/SCHGAN_TCYB2018)

- **SGCH:Semi-supervised graph convolutional hashing network for large-scale cross-modal retrieval(ICIP)** [[PDF]](https://ieeexplore.ieee.org/document/9190641)

#### 2019

- **SSDQ:Semi-supervised Deep Quantization for Cross-modal Search(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3343031.3350934)

- **S3PH:Semi-supervised semantic-preserving hashing for efficient cross-modal retrieval(ICME)** [[PDF]](https://ieeexplore.ieee.org/document/8784930)

#### 2017

- **AUSL:Adaptively Uniﬁed Semi-supervised Learning for Cross-Modal Retrieval(IJCAI)** [[PDF]](https://www.ijcai.org/proceedings/2017/0476.pdf)

#### 2016

- **NPH:Neighborhood-Preserving Hashing for Large-Scale Cross-Modal Search(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/2964284.2967241)

### 2.5.3 Imbalance (Real-valued)

#### 2021

- **PAN: Prototype-based Adaptive Network for Robust Cross-modal Retrieval(SIGIR)** [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3404835.3462867)

- **MCCN: Multimodal Coordinated Clustering Network for Large-Scale Cross-modal Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3474085.3475670)

#### 2020

- **DAVAE:Incomplete Cross-modal Retrieval with Dual-Aligned Variational Autoencoders(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3394171.3413676)

#### 2015

- **SCDL:Semi-supervised Coupled Dictionary Learning for Cross-modal Retrieval in Internet Images and Texts(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/2733373.2806346)

- **LGCFL:Learning Consistent Feature Representation for Cross-Modal Multimedia Retrieval(TMM)** [[PDF]](https://ieeexplore.ieee.org/document/7006724)

### 2.5.4 Imbalance (Hashing)

#### 2020

- **RUCMH:Robust Unsupervised Cross-modal Hashing for Multimedia Retrieval(TOIS)** [[PDF]](https://dl.acm.org/doi/10.1145/3389547)

- **ATFH-N:Adversarial Tri-Fusion Hashing Network for Imbalanced Cross-Modal Retrieval(TETCI)** [[PDF]](https://ieeexplore.ieee.org/document/9139424)

- **FlexCMH:Flexible Cross-Modal Hashing(TNNLS)** [[PDF]](https://ieeexplore.ieee.org/document/9223723)

#### 2019

- **TFNH:Triplet Fusion Network Hashing for Unpaired Cross-Modal Retrieval(ICMR)** [[PDF]](https://www.comp.hkbu.edu.hk/~ymc/papers/conference/icmr19-publication-version.pdf) [[Code]](https://github.com/hutt94/TFNH)

- **CALM:Collective Afﬁnity Learning for Partial Cross-Modal Hashing(TIP)** [[PDF]](https://ieeexplore.ieee.org/document/8846593)

- **MTFH: A Matrix Tri-Factorization Hashing Framework for Efﬁcient Cross-Modal Retrieval:(TIP)** [[PDF]](https://arxiv.org/abs/1805.01963) [[Code]](https://github.com/starxliu/MTFH)

- **GSPH:Generalized Semantic Preserving Hashing for Cross-Modal Retrieval(TIP)** [[PDF]](https://ieeexplore.ieee.org/document/8425016)

#### 2018

- **DAH:Dense Auto-Encoder Hashing for Robust Cross-Modality Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3240508.3240684)

#### 2017

- **GSPH:Generalized Semantic Preserving Hashing for n-Label Cross-Modal Retrieval(CVPR)** [[PDF]](https://ieeexplore.ieee.org/document/8099765) [[Code]](https://github.com/devraj89/Generalized-Semantic-Preserving-Hashing-for-N-Label-Cross-Modal-Retrieval)

### 2.5.5 Incremental

#### 2021

- **MARS: Learning Modality-Agnostic Representation for Scalable Cross-Media Retrieval(TCSVT)** [[PDF]](https://ieeexplore.ieee.org/document/9654230)

- **CCMR:Continual learning in cross-modal retrieval(CVPR)** [[PDF]](https://arxiv.org/abs/2104.06806)

- **SCML:Real-world Cross-modal Retrieval via Sequential Learning(TMM)** [[PDF]](https://ieeexplore.ieee.org/document/9117043)

#### 2020

- **ATTL-CEL:Adaptive Temporal Triplet-loss for Cross-modal Embedding Learning(MM)**[[PDF]](https://dl.acm.org/doi/10.1145/3394171.3413540)

#### 2019

- **SVHNs:Separated Variational Hashing Networks for Cross-Modal Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3343031.3351078)

- **ECMH:Extensible Cross-Modal Hashing(IJCAI)** [[PDF]](https://www.ijcai.org/proceedings/2019/0292.pdf) [[Code]](https://github.com/3andero/Extensible-Cross-Modal-Hashing)

#### 2018

- **TempXNet:Temporal Cross-Media Retrieval with Soft-Smoothing(MM)** [[PDF]](https://arxiv.org/abs/1810.04547)

# 3. Usage

## 3.1 Datasets

- **Graph Model--GCR**

Dataset Link:

    Baidu Yun Link: https://pan.baidu.com/s/1YmW8Zz2uK3AgCs6pDEoA8A?pwd=21xh
    Code: 21xh

- **Unsupervised cross-modal real-valued**

Dataset link:

    Baidu Yun Link：https://pan.baidu.com/s/1hBNo8gBSyLbik0ka1POhiQ 
    Code：cc53

- **Quantization--CDQ**

Dataset Link:

    Baidu Yun Link: https://pan.baidu.com/s/1mO1hdsJR2FN5xEAv2e7eaw?pwd=us9v
    Code: us9v

- **GAN--CPAH**

Dataset Link:

    Baidu Yun Link: https://pan.baidu.com/s/145Zool0FUb3758EeSxtHBw?pwd=mxt7
    Code: mxt7

- **Transformer--DCHMT**

Dataset Link:

    Baidu Yun Link: https://pan.baidu.com/s/1UHr2NVjFkTjLXXQ8Izy5WA?pwd=qfsj
    Code: qfsj

- **Feature Mapping(Sample Constraint)(Label Constraint)--MDBE**

Dataset Link:

    Baidu Yun Link: https://pan.baidu.com/s/15BtQ_Zz7UihZBW6KXTTodA?pwd=ir7g
    Code: ir7g

- **Feature Mapping(Sample Constraint)(Common Hamming)--RoPH**

Dataset Link:

    Baidu Yun Link: https://pan.baidu.com/s/1_uIulkuxcIcubvl5u3zsOA?pwd=46c4
    Code: 46c4
