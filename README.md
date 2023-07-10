# Cross-modal Retrieval


- [1. Introduction](#1-introduction)
- [2. Supported Methods](#2-supported-methods)
    - [2.1 Unsupercised-cross-modal-hashing-retrieval](#21-unsupervised-cross-modal-hashing-retrieval)
        - [2.1.1 Unsupervised shallow cross-modal hashing retrieval](#211-unsupervised-shallow-cross-modal-hashing-retrieval)
            - [2.1.1.1 Matrix Factorization](#2111-matrix-factorization)
            - [2.1.1.2 Graph Theory](#2112-graph-theory)
            - [2.1.1.3 Other Shallow](#2113-other-shallow)
            - [2.1.1.4 Quantization](#2114-quantization)
        - [2.1.2 Supervised shallow cross-modal hashing retrieval](#212-unsupervised-deep-cross-modal-hashing-retrieval)
            - [2.1.2.1 Naive Network](#2121-naive-network)
            - [2.1.2.2 GAN](#2122-gan)
            - [2.1.2.3 Graph Model](#2123-graph-model)
            - [2.1.2.4 Knowledge Distillation](#2124-knowledge-distillation)
    - [2.2 Supervised-cross-modal-hashing-retrieval](#22-supervised-cross-modal-hashing-retrieval)
        - [2.2.1 Supervised shallow cross-modal hashing retrieval](#221-supervised-shallow-cross-modal-hashing-retrieval)
            - [2.2.1.1 Matrix Factorization](#2211-matrix-factorization)
            - [2.2.1.2 Dictionary Learning](#2212-dictionary-learning)
            - [2.2.1.3 Feature Mapping-Sample-Constraint-Label-Constraint](#2213-feature-mapping-sample-constraint-label-constraint)
            - [2.2.1.4 Feature Mapping-Sample-Constraint-Separate-Hamming](#2214-Feature-Mapping-Sample-Constraint-Separate-Hamming)
            - [2.2.1.5 Feature Mapping-Sample-Constraint-Common-Hamming](#2215-Feature-Mapping-Sample-Constraint-Common-Hamming)
            - [2.2.1.6 Feature Mapping-Relation-Constraint](#2216-Feature-Mapping-Relation-Constraint)
            - [2.2.1.7 Other Shallow](#2217-Other-Shallow)
        - [2.2.2 Supervised deep cross-modal hashing retrieval](#222-supervised-deep-cross-modal-hashing-retrieval)
            - [2.2.2.1 Naive Network-Distance-Constraint](#2221-Naive-Network-Distance-Constraint)
            - [2.2.2.2 Naive Network-Similarity-Constraint](#2222-Naive-Network-Similarity-Constraint)
            - [2.2.2.3 Naive Network-Negative-Log-Likelihood](#2223-Naive-Network-Negative-Log-Likel-ihood)
            - [2.2.2.4 Naive Network-Triplet-Constraint](#2224-Naive-Network-Triplet-Constraint)
            - [2.2.2.5 GAN](#2225-gan)
            - [2.2.2.6 Graph Model](#2226-graph-model)
            - [2.2.2.7 Transformer](#2227-transformer)
            - [2.2.2.8 Memory Network](#2228-memory-Network)
            - [2.2.2.9 Quantization](#2229-quantization)
    - [2.3 Unsupervised-cross-modal-real-valued](#23-unsupervised-cross-modal-real-valued)
        - [2.3.1 Early unsupervised cross-modal real-valued retrieval](#231-early-unsupervised-cross-modal-real-valued-retrieval)
            - [2.3.1.1 CCA](#2311-cca)
            - [2.3.1.2 Topic Model](#2312-topic-model)
            - [2.3.1.3 Other Shallow](#2313-other-shallow)
            - [2.3.1.4 Neural Network](#2314-neural-network)
        - [2.3.2 Image-text matching retrieval](#232-image-text-matching-retrieval)
            - [2.3.2.1 Naive Network](#2321-native-network)
            - [2.3.2.2 Dot-product Attention](#2322-dot-product-attention)
            - [2.3.2.3 Graph Model](#2323-graph-model)
            - [2.3.2.4 Transformer](#238-transformer)
            - [2.3.2.5 Cross-modal Generation](#2324-cross-modal-generation)
    - [2.4 Supervised-cross-modal-real-valued](#24-supervised-cross-modal-real-valued)
        - [2.4.1 Supervised shallow cross-modal real-valued retrieval](#241-supervised-shallow-cross-modal-real-valued-retrieval)
            - [2.4.1.1 CCA](#2411-cca)
            - [2.4.1.2 Dictionary Learning](#2412-dictionary-learning)
            - [2.4.1.3 Feature Mapping](#2413-feature-mapping)
            - [2.4.1.4 Topic Model](#2414-topic-model)
            - [2.4.1.5 Other Shallow](#2415-other-shallow)
        - [2.4.2 Supervised deep cross-modal real-valued retrieval](#242-supervised-deep-cross-modal-real-valued-retrieval)
            - [2.4.2.1 Naive Network](#2421-naive-network)
            - [2.4.2.2 GAN](#2422-gan)
            - [2.4.2.3 Graph Model](#2423-graph-model)
            - [2.4.2.4 Transformer](#2424-transformer)
    - [2.5 Cross-modal-Retrieval-under-Special-Retrieval-Scenario](#25-Cross-modal-Retrieval-under-Special-Retrieval-Scenario)
        - [2.5.1 Semi-Supervised (Real-valued)](#251-semi-supervised-real-valued)
        - [2.5.2 Semi-Supervised (Hashing)](#252-semi-supervised-hashing)
        - [2.5.3 Imbalance (Real-valued)](#253-imbalance-real-valued)
        - [2.5.4 Imbalance (Hashing)](#254-imbalance-hashing)
        - [2.5.5 Incremental](#255-incremental)
        - [2.5.6 Noise](#256-Noise)
        - [2.5.7 Cross-Domain](#257-Cross-Domain)
        - [2.5.8 Zero-Shot](#258-Zero-Shot)
        - [2.5.9 Few-Shot](#259-Few-Shot)
        - [2.5.10 Online Learning](#2510-Online-Learning)
        - [2.5.11 Hierarchical](#2511-Hierarchical)
        - [2.5.12 Fine-grained](#2512-Fine-grained)
- [3. Usage](#3-usage)

</details>

# 1. Introduction
This library is an open-source repository that contains cross-modal retrieval methods and codes.

# 2. Supported Methods
The currently supported algorithms include:

<details><summary>+Details</summary>

## 2.1 Unsupervised cross-modal hashing retrieval

<details><summary>++Details</summary>

### 2.1.1 Unsupervised shallow cross-modal hashing retrieval

<details><summary>+++Details</summary>

#### 2.1.1.1 Matrix Factorization

<details><summary>++++Details</summary>

##### 2017

- **RFDH：Robust and Flexible Discrete Hashing for Cross-Modal Similarity Search(TCSVT)** [[PDF]](https://ieeexplore.ieee.org/abstract/document/7967838) [[Code]](https://github.com/Wangdi-Xidian/RFDH)

##### 2015

- **STMH:Semantic Topic Multimodal Hashing for Cross-Media Retrieval(IJCAI)**[[PDF]](https://www.ijcai.org/Proceedings/15/Papers/546.pdf)

##### 2014

- **LSSH:Latent Semantic Sparse Hashing for Cross-Modal Similarity Search(SIGIR)**[[PDF]](https://dl.acm.org/doi/10.1145/2600428.2609610)

- **CMFH:Collective Matrix Factorization Hashing for Multimodal Data(CVPR)**[[PDF]](https://ieeexplore.ieee.org/document/6909664)

</details>

#### 2.1.1.2 Graph Theory

<details>

##### 2018

- **HMR:Hetero-Manifold Regularisation for Cross-Modal Hashing(TPAMI)**[[PDF]](https://ieeexplore.ieee.org/abstract/document/7801124)

##### 2017

- **FSH:Cross-Modality Binary Code Learning via Fusion Similarity Hashing(CVPR)**[[PDF]](https://ieeexplore.ieee.org/abstract/document/8100155)[[Code]](https://github.com/LynnHongLiu/FSH)

##### 2014

- **SM2H:Sparse Multi-Modal Hashing(TMM)**[[PDF]](https://ieeexplore.ieee.org/document/6665155)

##### 2013

- **IMH:Inter-Media Hashing for Large-scale Retrieval from Heterogeneous Data Sources(SIGMOD)**[[PDF]](https://dl.acm.org/doi/10.1145/2463676.2465274)

- **LCMH:Linear Cross-Modal Hashing for Efﬁcient Multimedia Search(MM)**[[PDF]](https://dl.acm.org/doi/10.1145/2502081.2502107)

##### 2011

- **CVH:Learning Hash Functions for Cross-View Similarity Search(IJCAI)**[[PDF]](https://dl.acm.org/doi/10.5555/2283516.2283623)

</details>

#### 2.1.1.3 Other Shallow

<details>

##### 2019

- **CRE:Collective Reconstructive Embeddings for Cross-Modal Hashing(TIP)**[[PDF]](https://ieeexplore.ieee.org/document/8594588)

##### 2018

- **HMR:Hetero-Manifold Regularisation for Cross-Modal Hashing(TPAMI)**[[PDF]](https://ieeexplore.ieee.org/abstract/document/7801124)

##### 2015

- **FS-LTE:Full-Space Local Topology Extraction for Cross-Modal Retrieval(TIP)**[[PDF]](https://ieeexplore.ieee.org/document/7076613)

##### 2014

- **IMVH:Iterative Multi-View Hashing for Cross Media Indexing(MM)**[[PDF]](https://dl.acm.org/doi/10.1145/2647868.2654906)

##### 2013

- **PDH:Predictable Dual-View Hashing(ICML)**[[PDF]](https://dl.acm.org/doi/10.5555/3042817.3043085)

</details>

#### 2.1.1.4 Quantization

<details>

##### 2016

- **CCQ:Composite Correlation Quantization for Efﬁcient Multimodal Retrieval(SIGIR)**[[PDF]](https://arxiv.org/abs/1504.04818)

- **CMCQ:Collaborative Quantization for Cross-Modal Similarity Search(CVPR)**[[PDF]](https://arxiv.org/abs/1902.00623)

##### 2015

- **ACQ:Alternating Co-Quantization for Cross-modal Hashing(ICCV)**[[PDF]](https://ieeexplore.ieee.org/document/7410576)

</details>

</details>

### 2.1.2 Unsupervised deep cross-modal hashing retrieval

<details>

#### 2.1.2.1 Naive Network

<details>

##### 2019

- **UDFCH:Unsupervised Deep Fusion Cross-modal Hashing(ICMI)**[[PDF]](https://dl.acm.org/doi/fullHtml/10.1145/3340555.3353752)

##### 2018

- **UDCMH:Unsupervised Deep Hashing via Binary Latent Factor Models for Large-scale Cross-modal Retrieval(IJCAI)**[[PDF]](https://www.ijcai.org/proceedings/2018/396)

##### 2017 

- **DBRC:Deep Binary Reconstruction for Cross-modal Hashing(MM)**[[PDF]](https://arxiv.org/abs/1708.05127)

##### 2015

- **DMHOR:Learning Compact Hash Codes for Multimodal Representations Using Orthogonal Deep Structure(TMM)**[[PDF]](https://ieeexplore.ieee.org/document/7154455)

</details>

#### 2.1.2.2 GAN 

<details>

##### 2020

- **MGAH:Multi-Pathway Generative Adversarial Hashing for Unsupervised Cross-Modal Retrieval(TMM)**[[PDF]](https://ieeexplore.ieee.org/document/8734835)

##### 2019

- **CYC-DGH:Cycle-Consistent Deep Generative Hashing for Cross-Modal Retrieval(TIP)**[[PDF]](https://arxiv.org/abs/1804.11013)

- **UCH:Coupled CycleGAN: Unsupervised Hashing Network for Cross-Modal Retrieval(AAAI)**[[PDF]](https://arxiv.org/abs/1903.02149)

##### 2018

- **UGACH:Unsupervised Generative Adversarial Cross-modal Hashing(AAAI)**[[PDF]](https://arxiv.org/abs/1712.00358)[[Code]](https://github.com/PKU-ICST-MIPL/UGACH_AAAI2018)

</details>

#### 2.1.2.3 Graph Model

<details>

##### 2022

- **ASSPH:Adaptive Structural Similarity Preserving for Unsupervised Cross Modal Hashing(MM)**[[PDF]](https://dl.acm.org/doi/abs/10.1145/3503161.3548431)

##### 2021

- **AGCH:Aggregation-based Graph Convolutional Hashing for Unsupervised Cross-modal Retrieval(TMM)**[[PDF]](https://ieeexplore.ieee.org/document/9335490)

- **DGCPN:Deep Graph-neighbor Coherence Preserving Network for Unsupervised Cross-modal Hashing(AAAI)**[[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/16592)[[Code]](https://github.com/Atmegal/DGCPN)

##### 2020

- **DCSH:Unsupervised Deep Cross-modality Spectral Hashing(TIP)**[[PDF]](https://arxiv.org/abs/2008.00223)

- **SRCH:Set and Rebase: Determining the Semantic Graph Connectivity for Unsupervised Cross-Modal Hashing(IJCAI)**[[PDF]](https://www.ijcai.org/proceedings/2020/0119.pdf)

- **JDSH:Joint-modal Distribution-based Similarity Hashing for Large-scale Unsupervised Deep Cross-modal Retrieval(SIGIR)**[[PDF]](https://dl.acm.org/doi/10.1145/3397271.3401086)[[Code]](https://github.com/KaiserLew/JDSH)

- **DSAH:Deep Semantic-Alignment Hashing for Unsupervised Cross-Modal Retrieval(ICMR)**[[PDF]](https://dl.acm.org/doi/10.1145/3372278.3390673)[[Code]](https://github.com/idejie/DSAH)

##### 2019

- **DJSRH:Deep Joint-Semantics Reconstructing Hashing for Large-Scale Unsupervised Cross-Modal Retrieval(ICCV)**[[PDF]](https://ieeexplore.ieee.org/document/9009571)[[Code]](https://github.com/zs-zhong/DJSRH)

</details>

#### 2.1.2.4 Knowledge Distillation

<details>

##### 2022

- **DAEH:Deep Adaptively-Enhanced Hashing With Discriminative Similarity Guidance for Unsupervised Cross-Modal Retrieval(TCSVT)**[[PDF]](https://ieeexplore.ieee.org/document/9768805)

##### 2021

- **KDCMH:Unsupervised Deep Cross-Modal Hashing by Knowledge Distillation for Large-scale Cross-modal Retrieval(ICMR)**[[PDF]](https://dl.acm.org/doi/10.1145/3460426.3463626)

- **JOG:Joint-teaching: Learning to Refine Knowledge for Resource-constrained Unsupervised Cross-modal Retrieval(MM)**[[PDF]](https://dl.acm.org/doi/10.1145/3474085.3475286)

##### 2020

- **UKD:Creating Something from Nothing: Unsupervised Knowledge Distillation for Cross-Modal Hashing(CVPR)**[[PDF]](https://arxiv.org/abs/2004.00280)

</details>

</details>

</details>

## 2.2 Supervised-cross-modal-hashing-retrieval

<details>

### 2.2.1 Supervised shallow cross-modal hashing retrieval

<details>

#### 2.2.1.1 Matrix Factorization

<details>

##### 2022

- **SCLCH: Joint Specifics and Consistency Hash Learning for Large-Scale Cross-Modal Retrieval(TIP)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9850431)

##### 2020

- **BATCH: A Scalable Asymmetric Discrete Cross-Modal Hashing(TKDE)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9001235) [[Code]](https://github.com/yxinwang/BATCH-TKDE2020)

##### 2019

- **LCMFH: Label Consistent Matrix Factorization Hashing for Large-Scale Cross-Modal Similarity Search(TPAMI)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/8423193)

- **TECH: A Two-Step Cross-Modal Hashing by Exploiting Label Correlations and Preserving Similarity in Both Steps(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3343031.3350862)

##### 2018

- **SCRATCH: A Scalable Discrete Matrix Factorization Hashing for Cross-Modal Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3240508.3240547)

##### 2017

- **DCH: Learning Discriminative Binary Codes for Large-scale Cross-modal Retrieval(TIP)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7867785)

##### 2016

- **SMFH: Supervised Matrix Factorization for Cross-Modality Hashing(IJCAI)**  [[PDF]](https://arxiv.org/abs/1603.05572)

- **SMFH: Supervised Matrix Factorization Hashing for Cross-Modal Retrieval(TIP)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7466099)

</details>

#### 2.2.1.2 Dictionary Learning

<details>

##### 2016

- **DCDH: Discriminative Coupled Dictionary Hashing for Fast Cross-Media Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2600428.2609563)

##### 2014

- **DLCMH: Dictionary Learning Based Hashing for Cross-Modal Retrieval(SIGIR)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2964284.2967206)

</details>

#### 2.2.1.3 Feature Mapping-Sample-Constraint-Label-Constraint

<details>

##### 2022

- **DJSAH: Discrete Joint Semantic Alignment Hashing for Cross-Modal Image-Text Search(TCSVT)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9808188)

##### 2020

- **FUH: Fast Unmediated Hashing for Cross-Modal Retrieval(TCSVT)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9285286)

##### 2016

- **MDBE: Multimodal Discriminative Binary Embedding for Large-Scale Cross-Modal Retrieval(TIP)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/7515190) [[Code]](https://github.com/Wangdi-Xidian/MDBE)

</details>

#### 2.2.1.4 Feature Mapping-Sample-Constraint-Separate-Hamming

<details>

##### 2017

- **CSDH: Sequential Discrete Hashing for Scalable Cross-Modality Similarity Retrieval(TIP)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/7600368)

##### 2016

- **DASH: Frustratingly Easy Cross-Modal Hashing(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2964284.2967218)

##### 2015

- **QCH: Quantized Correlation Hashing for Fast Cross-Modal Search(IJCAI)** [(PDF)](https://dl.acm.org/doi/abs/10.5555/2832747.2832799)

</details>

#### 2.2.1.5 Feature Mapping-Sample-Constraint-Common Hamming

<details>

##### 2021

- **ASCSH: Asymmetric Supervised Consistent and Speciﬁc Hashing for Cross-Modal Retrieval(TIP)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9269445) [[Code]](https://github.com/minmengzju/ASCSH)

##### 2019

- **SRDMH: Supervised Robust Discrete Multimodal Hashing for Cross-Media Retrieval(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8695043)

##### 2018

- **FDCH: Fast Discrete Cross-modal Hashing With Regressing From Semantic Labels(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3240508.3240683)

##### 2017

- **SRSH: Semi-Relaxation Supervised Hashing for Cross-Modal Retrieval(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3123266.3123320) [[Code]](https://github.com/sduzpf/SRSH)

- **RoPH: Cross-Modal Hashing via Rank-Order Preserving(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/7737053) [[Code]](https://github.com/kding1225/RoPH)

##### 2016

- **SRDMH: Supervised Robust Discrete Multimodal Hashing for Cross-Media Retrieval(CIKM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2983323.2983743)

</details>

#### 2.2.1.6 Feature Mapping-Relation-Constraint

<details>

##### 2017

- **LSRH: Linear Subspace Ranking Hashing for Cross-Modal Retrieval(TPAMI)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/7571151)

##### 2014

- **SCM: Large-Scale Supervised Multimodal Hashing with Semantic Correlation Maximization(AAAI)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/8995)

- **HTH: Scalable Heterogeneous Translated Hashing(KDD)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2623330.2623688)

##### 2013

- **PLMH: Parametric Local Multimodal Hashing for Cross-View Similarity Search(IJCAI)** [(PDF)](https://repository.hkust.edu.hk/ir/Record/1783.1-60904)

- **RaHH: Comparing Apples to Oranges: A Scalable Solution with Heterogeneous Hashing(KDD)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2487575.2487668) [[Code]](https://github.com/boliu68/RaHH)

##### 2012

- **CRH: Co-Regularized Hashing for Multimodal Data(CRH)** [(PDF)](https://proceedings.neurips.cc/paper_files/paper/2012/hash/5c04925674920eb58467fb52ce4ef728-Abstract.html)

</details>

#### 2.2.1.7 Other Shallow

<details>

##### 2019

- **DLFH: Discrete Latent Factor Model for Cross-Modal Hashing(TIP)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8636536) [[Code]](https://github.com/jiangqy/DLFH-TIP2019)

##### 2018

- **SDMCH: Supervised Discrete Manifold-Embedded Cross-Modal Hashing(IJCAI)** [(PDF)](https://dl.acm.org/doi/abs/10.5555/3304889.3305010)

##### 2015

- **SePH: Semantics-Preserving Hashing for Cross-View Retrieval(CVPR)** [(PDF)](https://openaccess.thecvf.com/content_cvpr_2015/html/Lin_Semantics-Preserving_Hashing_for_2015_CVPR_paper.html)

##### 2012

- **MLBE: A Probabilistic Model for Multimodal Hash Function Learning(KDD)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2339530.2339678)

##### 2010

- **CMSSH: Data Fusion through Cross-modality Metric Learning using Similarity-Sensitive Hashing(CVPR)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/5539928)

</details>

</details>

### 2.2.2 Supervised deep cross-modal hashing retrieval

<details>

#### 2.2.2.1 Naive Network-Distance-Constraint

<details>

##### 2019

- **MCITR: Cross-modal Image-Text Retrieval with Multitask Learning(CIKM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3357384.3358104)

##### 2016

- **CAH: Correlation Autoencoder Hashing for Supervised Cross-Modal Search(ICMR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2911996.2912000)

##### 2014

- **CMNNH: Cross-Media Hashing with Neural Networks(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2647868.2655059)

- **MMNN: Multimodal Similarity-Preserving Hashing(TPAMI)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/6654144)

</details>

#### 2.2.2.2 Naive Network-Similarity-Constraint

<details>

##### 2022

- **Bi-CMR: Bidirectional Reinforcement Guided Hashing for Effective Cross-Modal Retrieval(AAAI)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/21268) [[Code]](https://github.com/lty4869/Bi-CMR)

- **Bi-NCMH: Deep Normalized Cross-Modal Hashing with Bi-Direction Relation Reasoning(CVPR)** [(PDF)](https://openaccess.thecvf.com/content/CVPR2022W/ODRUM/html/Sun_Deep_Normalized_Cross-Modal_Hashing_With_Bi-Direction_Relation_Reasoning_CVPRW_2022_paper.html)

##### 2021

- **OTCMR: Bridging Heterogeneity Gap with Optimal Transport for Cross-modal Retrieval(CIKM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3459637.3482158)

- **DUCMH: Deep Uniﬁed Cross-Modality Hashing by Pairwise Data Alignment(IJCAI)** [(PDF)](https://cs.nju.edu.cn/_upload/tpl/01/0c/268/template268/pdf/IJCAI-2021-Wang.pdf)

##### 2020

- **NRDH: Nonlinear Robust Discrete Hashing for Cross-Modal Retrieval(SIGIR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3397271.3401152)

- **DCHUC: Deep Cross-Modal Hashing with Hashing Functions and Uniﬁed Hash Codes Jointly Learning(TKDE)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9069300) [[Code]](https://github.com/rongchengtu1/DCHUC)

##### 2017

- **CHN: Correlation Hashing Network for Efﬁcient Cross-Modal Retrieval(BMVC)** [(PDF)](https://arxiv.org/abs/1602.06697)

##### 2016

- **DVSH: Deep Visual-Semantic Hashing for Cross-Modal Retrieval(KDD)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/2939672.2939812)

</details>

#### 2.2.2.3 Naive Network-Negative-Log-Likelihood

<details>

##### 2022

- **MSSPQ: Multiple Semantic Structure-Preserving Quantization for Cross-Modal Retrieval(ICMR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3512527.3531417)

##### 2021

- **DMFH: Deep Multiscale Fusion Hashing for Cross-Modal Retrieval(TCSVT)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9001018)

- **TEACH: Attention-Aware Deep Cross-Modal Hashing(ICMR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3460426.3463625)

##### 2020

- **MDCH: Mask Cross-modal Hashing Networks(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9055057)

##### 2019

- **EGDH: Equally-Guided Discriminative Hashing for Cross-modal Retrieval(IJCAI)** [(PDF)](https://dl.acm.org/doi/abs/10.5555/3367471.3367706)

##### 2018

- **DDCMH: Dual Deep Neural Networks Cross-Modal Hashing(AAAI)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/11249)

- **CMHH: Cross-Modal Hamming Hashing(ECCV)** [(PDF)](https://openaccess.thecvf.com/content_ECCV_2018/html/Yue_Cao_Cross-Modal_Hamming_Hashing_ECCV_2018_paper.html)

##### 2017

- **PRDH: Pairwise Relationship Guided Deep Hashing for Cross-Modal Retrieval(AAAI)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/10719)

- **DCMH: Deep Cross-Modal Hashing(CVPR)** [(PDF)](https://openaccess.thecvf.com/content_cvpr_2017/html/Jiang_Deep_Cross-Modal_Hashing_CVPR_2017_paper.html) [[Code]](https://github.com/WendellGul/DCMH)

</details>

#### 2.2.2.4 Naive Network-Triplet-Constraint

<details>

##### 2019

- **RDCMH: Multiple Semantic Structure-Preserving Quantization for Cross-Modal Retrieval(AAAI)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/4351)

##### 2018

- **MCSCH: Multi-Scale Correlation for Sequential Cross-modal Hashing Learning(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3240508.3240560)

- **TDH: Triplet-Based Deep Hashing Network for Cross-Modal Retrieval(TIP)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8331146)

</details>

#### 2.2.2.5 GAN

<details>

##### 2022

- **SCAHN: Semantic Structure Enhanced Contrastive Adversarial Hash Network for Cross-media Representation Learning(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3503161.3548391) [[Code]](https://github.com/YI1219/SCAHN-MindSpore)

##### 2021

- **TGCR: Multiple Semantic Structure-Preserving Quantization for Cross-Modal Retrieval(TCSVT)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9257382)

##### 2020

- **CPAH: Multi-Task Consistency-Preserving Adversarial Hashing for Cross-Modal Retrieval(TIP)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8954946) [[Code]](https://github.com/comrados/cpah)

- **MLCAH: Multi-Level Correlation Adversarial Hashing for Cross-Modal Retrieval(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8970562)

- **DADH: Deep Adversarial Discrete Hashing for Cross-Modal Retrieval(ICMR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3372278.3390711) [[Code]](https://github.com/Zjut-MultimediaPlus/DADH)

##### 2019

- **AGAH: Adversary Guided Asymmetric Hashing for Cross-Modal Retrieval(ICMR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3323873.3325045) [[Code]](https://github.com/WendellGul/AGAH)

##### 2018

- **SSAH: Self-Supervised Adversarial Hashing Networks for Cross-Modal Retrieval(CVPR)** [(PDF)](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Self-Supervised_Adversarial_Hashing_CVPR_2018_paper.html) [[Code]](https://github.com/lelan-li/SSAH)

</details>

#### 2.2.2.6 Graph Model

<details>

##### 2022

- **HMAH: Multi-Task Consistency-Preserving Adversarial Hashing for Cross-Modal Retrieval(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9782694)

- **SCAHN: Semantic Structure Enhanced Contrastive Adversarial Hash Network for Cross-media Representation Learning(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3503161.3548391) [[Code]](https://github.com/lelan-li/SSAH)

##### 2021

- **LGCNH: Local Graph Convolutional Networks for Cross-Modal Hashing(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3474085.3475346) [[Code]](https://github.com/chenyd7/LGCNH)

##### 2019

- **GCH: Graph Convolutional Network Hashing for Cross-Modal Retrieval(IJCAI)** [(PDF)](https://dl.acm.org/doi/abs/10.5555/3367032.3367172) [[Code]](https://github.com/DeXie0808/GCH)

</details>

#### 2.2.2.7 Transformer

<details>

##### 2022

- **DCHMT: Differentiable Cross-modal Hashing via Multimodal Transformers(CIKM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3503161.3548187) [[Code]](https://github.com/kalenforn/DCHMT)

- **UniHash: Contrastive Label Correlation Enhanced Unified Hashing Encoder for Cross-modal Retrieval(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3511808.3557265) [[Code]](https://github.com/idealwhite/UniHash)

</details>

#### 2.2.2.8 Memory Network

<details>

##### 2021

- **CMPD: Using Cross Memory Network With Pair Discrimination for Image-Text Retrieval(TCSVT)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9169915)

##### 2019

- **CMMN: Deep Memory Network for Cross-Modal Retrieval(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8506385)

</details>

#### 2.2.2.9 Quantization

<details>

##### 2022

- **ACQH: Asymmetric Correlation Quantization Hashing for Cross-Modal Retrieval(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/9517000)

##### 2017

- **CDQ: Collective Deep Quantization for Efﬁcient Cross-Modal Retrieval(AAAI)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/11218) [[Code]](https://github.com/caoyue10/aaai17-cdq)

</details>

</details>

</details>

## 2.3 Unsupervised-cross-modal-real-valued

<details>

### 2.3.1 Early unsupervised cross-modal real-valued retrieval

<details>

#### 2.3.1.1 CCA

<details>

##### 2017

- **ICCA:Towards Improving Canonical Correlation Analysis for Cross-modal Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/abs/10.1145/3126686.3126726)

##### 2015

- **DCMIT:Deep Correlation for Matching Images and Text(CVPR)** [[PDF]](https://ieeexplore.ieee.org/document/7298966)

- **RCCA:Learning Query and Image Similarities with Ranking Canonical Correlation Analysis(ICCV)** [[PDF]](https://ieeexplore.ieee.org/document/7410369)

##### 2014


- **MCCA:A Multi-View Embedding Space for Modeling Internet Images, Tags, and Their Semantics(IJCV)** [[PDF]](https://arxiv.org/abs/1212.4522)

##### 2013

- **KCCA:Framing Image Description as a Ranking Task Data, Models and Evaluation Metrics(JAIR)** [[PDF]](https://www.ijcai.org/Proceedings/15/Papers/593.pdf)

- **DCCA:Deep Canonical Correlation Analysis(ICML)** [[PDF]](https://proceedings.mlr.press/v28/andrew13.html) [[Code]](https://github.com/Michaelvll/DeepCCA)

##### 2012

- **CR:Continuum Regression for Cross-modal Multimedia Retrieval(ICIP)** [[PDF]](https://ieeexplore.ieee.org/document/6467268)

##### 2010


- **CCA:A New Approach to Cross-Modal Multimedia Retrieval(MM)** [[PDF]](http://www.mit.edu/~rplevy/papers/rasiwasia-etal-2010-acm.pdf)[[Code]](https://github.com/emanuetre/crossmodal)

</details>

#### 2.3.1.2 Topic Model

<details>

##### 2011

- **MDRF:Learning Cross-modality Similarity for Multinomial Data(ICCV)** [[PDF]](https://ieeexplore.ieee.org/document/6126524)

##### 2010

- **tr-mmLDA:Topic Regression Multi-Modal Latent Dirichlet Allocation for Image Annotation(CVPR)** [[PDF]](https://ieeexplore.ieee.org/document/5540000)

##### 2003

- **Corr-LDA:Modeling Annotated Data(SIGIR)** [[PDF]](https://www.cs.columbia.edu/~blei/papers/BleiJordan2003.pdf)

</details>

#### 2.3.1.3 Other Shallow

<details>

##### 2013

- **Bi-CMSRM:Cross-Media Semantic Representation via Bi-directional Learning to Rank(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/2502081.2502097)

- **CTM:Cross-media Topic Mining on Wikipedia(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/2502081.2502180)

##### 2012

- **CoCA:Dimensionality Reduction on Heterogeneous Feature Space(ICDM)** [[PDF]](https://ieeexplore.ieee.org/document/6413864)

##### 2011

- **MCU:Maximum Covariance Unfolding: Manifold Learning for Bimodal Data(NIPS)** [[PDF]](https://proceedings.neurips.cc/paper/2011/file/daca41214b39c5dc66674d09081940f0-Paper.pdf)

##### 2008

- **PAMIR:A Discriminative Kernel-Based Model to Rank Images from Text Queries(TPAMI)** [[PDF]](https://ieeexplore.ieee.org/document/4359384)

##### 2003

- **CFA:Multimedia Content Processing through Cross-Modal Association(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/957013.957143)

</details>

#### 2.3.1.4 Neural Network

<details>

##### 2018

- **CDPAE:Comprehensive Distance-Preserving Autoencoders for Cross-Modal Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3240508.3240607)[[Code]](https://github.com/Atmegal/Comprehensive-Distance-Preserving-Autoencoders-for-Cross-Modal-Retrieval)


##### 2016

- **CMDN:Cross-Media Shared Representation by Hierarchical Learning with Multiple Deep Networks(IJCAI)** [[PDF]](https://www.ijcai.org/Proceedings/16/Papers/541.pdf)[[Code]]()

- **MSAE:Effective deep learning-based multi-modal retrieval(VLDB)** [[PDF]](https://dl.acm.org/doi/10.1007/s00778-015-0391-4)

##### 2014

- **Corr-AE:Cross-modal Retrieval with Correspondence Autoencoder(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/2647868.2654902)

##### 2013

- **RGDBN:Latent Feature Learning in Social Media Network(MM)** [[PDF]](https://dl.acm.org/doi/abs/10.1145/2502081.2502284)

##### 2012

- **MDBM:Multimodal Learning with Deep Boltzmann Machines(NIPS)** [[PDF]](https://jmlr.org/papers/volume15/srivastava14b/srivastava14b.pdf)

</details>

</details>

### 2.3.2 Image-text matching retrieval

<details>

#### 2.3.2.1 Native Network

<details>

##### 2022

- **UWML:Universal Weighting Metric Learning for Cross-Modal Retrieval (TPAMI)** [[PDF]](https://ieeexplore.ieee.org/document/9454290)[[Code]](https://github.com/wayne980/PolyLoss)

- **LESS:Learning to Embed Semantic Similarity for Joint Image-Text Retrieval (TPAMI)**[[PDF]](https://ieeexplore.ieee.org/document/9633145)

- **CMCM:Cross-Modal Coherence for Text-to-Image Retrieval (AAAI)** [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/download/21285/version/19572/21034)

- **P2RM:Point to Rectangle Matching for Image Text Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/abs/10.1145/3503161.3548237)

##### 2020

- **DPCITE:Dual-path Convolutional Image-Text Embeddings with Instance Loss(TOMM)** [[PDF]](https://arxiv.org/abs/1711.05535) [[code]](https://github.com/layumi/Image-Text-Embedding)

- **PSN:Preserving Semantic Neighborhoods for Robust Cross-Modal Retrieval(ECCV)** [[PDF]](https://arxiv.org/abs/2007.08617) [[Code]](https://github.com/CLT29/semantic_neighborhoods)

##### 2019

- **LDR:Learning Disentangled Representation for Cross-Modal Retrieval with Deep Mutual Information Estimation(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3343031.3351053)

##### 2018

- **CHAIN-VSE:Bidirectional Retrieval Made Simple(CVPR)** [[PDF]](https://ieeexplore.ieee.org/document/8578903) [[Code]](https://github.com/jwehrmann/chain-vse)

##### 2017

- **CRC:Cross-media Relevance Computation for Multimedia Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3123266.3123963)

- **VSE++: Improving Visual-Semantic Embeddings with Hard Negatives:(Arxiv)** [[PDF]](https://arxiv.org/abs/1707.05612)[[Code]](https://github.com/fartashf/vsepp)

- **RRF-Net:Learning a Recurrent Residual Fusion Network for Multimodal Matching(ICCV)** [[PDF]](https://ieeexplore.ieee.org/document/8237704)[[Code]](https://github.com/yuLiu24/RRF-Net)

##### 2016

- **DBRLM:Cross-Modal Retrieval via Deep and Bidirectional Representation Learning(TMM)** [[PDF]](https://ieeexplore.ieee.org/abstract/document/7460254)

##### 2015

- **MSDS:Image-Text Cross-Modal Retrieval via Modality-Speciﬁc Feature Learning(ICMR)** [[PDF]](https://dl.acm.org/doi/10.1145/2671188.2749341)

##### 2014

- **DT-RNN:Grounded Compositional Semantics for Finding and Describing Images with Sentences(TACL)** [[PDF]](https://aclanthology.org/Q14-1017.pdf)

</details>

#### 2.3.2.2 Dot-product Attention

<details>

##### 2020

- **SMAN: Stacked Multimodal Attention Network for Cross-Modal Image-Text Retrieval(TC)** [[PDF]](https://ieeexplore.ieee.org/document/9086164)

- **CAAN:Context-Aware Attention Network for Image-Text Retrieval(CVPR)** [[PDF]](https://ieeexplore.ieee.org/document/9157657)

- **IMRAM: Iterative Matching with Recurrent Attention Memory for Cross-Modal Image-Text Retrieval(CVPR)** [[PDF]](https://arxiv.org/abs/2003.03772) [[Code]](https://github.com/HuiChen24/IMRAM)

##### 2019

- **PFAN:Position Focused Attention Network for Image-Text Matching (IJCAI)** [[PDF]](https://arxiv.org/abs/1907.09748)[[Code]](https://github.com/HaoYang0123/Position-Focused-Attention-Network)

- **CAMP: Cross-Modal Adaptive Message Passing for Text-Image Retrieval(ICCV)** [[PDF]](https://arxiv.org/abs/1909.05506) [[Code]](https://github.com/ZihaoWang-CV/CAMP_iccv19)

- **CMRSC:Cross-Modal Image-Text Retrieval with Semantic Consistency(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3343031.3351055) [[Code]](https://github.com/HuiChen24/MM_SemanticConsistency)

##### 2018

- **MCSM:Modality-specific Cross-modal Similarity Measurement with Recurrent Attention Network(TIP)** [[PDF]](https://arxiv.org/abs/1708.04776)[[Code]](https://github.com/PKU-ICST-MIPL/MCSM_TIP2018)

- **DSVEL:Finding beans in burgers: Deep semantic-visual embedding with localization(CVPR)** [[PDF]](https://arxiv.org/abs/1804.01720)[[Code]](https://github.com/technicolor-research/dsve-loc)

- **CRAN:Cross-media Multi-level Alignment with Relation Attention Network(IJCAI)**[[PDF]](https://www.ijcai.org/proceedings/2018/124)

- **SCAN:Stacked Cross Attention for Image-Text Matching(ECCV)** [[PDF]](https://arxiv.org/abs/1803.08024) [[Code]](https://github.com/kuanghuei/SCAN)

##### 2017

- **sm-LSTM:Instance-aware Image and Sentence Matching with Selective Multimodal LSTM(CVPR)** [[PDF]](https://arxiv.org/abs/1611.05588)

</details>

#### 2.3.2.3 Graph Model

<details>

##### 2022

- **LHSC:Learning Hierarchical Semantic Correspondences for Cross-Modal Image-Text Retrieval(ICMR)** [[PDF]](https://dl.acm.org/doi/10.1145/3512527.3531358)

- **IFRFGF:Improving Fusion of Region Features and Grid Features via Two-Step Interaction for Image-Text Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3503161.3548223)

- **CODER:Coupled Diversity-Sensitive Momentum Contrastive Learning for Image-Text Retrieval(ECCV)** [[PDF]](https://dl.acm.org/doi/abs/10.1007/978-3-031-20059-5_40)

##### 2021

- **HSGMP: Heterogeneous Scene Graph Message Passing for Cross-modal Retrieval(ICMR)** [[PDF]](https://dl.acm.org/doi/10.1145/3460426.3463650)

- **WCGL：Wasserstein Coupled Graph Learning for Cross-Modal Retrieval(ICCV)**[[PDF]](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Wasserstein_Coupled_Graph_Learning_for_Cross-Modal_Retrieval_ICCV_2021_paper.html)

##### 2020

- **DSRAN:Learning Dual Semantic Relations with Graph Attention for Image-Text Matching(TCSVT)** [[PDF]](https://arxiv.org/abs/2010.11550) [[code]](https://github.com/kywen1119/DSRAN)

- **VSM:Visual-Semantic Matching by Exploring High-Order Attention and Distraction(CVPR)** [[PDF]](https://ieeexplore.ieee.org/document/9157630)

- **SGM:Cross-modal Scene Graph Matching for Relationship-aware Image-Text Retrieval(WACV)** [[PDF]](https://arxiv.org/abs/1910.05134)

##### 2019

- **KASCE:Knowledge Aware Semantic Concept Expansion for Image-Text Matching(IJCAI)** [[PDF]](https://www.ijcai.org/proceedings/2019/720)

- **VSRN:Visual Semantic Reasoning for Image-Text Matching(ICCV)** [[PDF]](https://arxiv.org/abs/1909.02701) [[Code]](https://github.com/KunpengLi1994/VSRN)

</details>

#### 2.3.2.4 Transformer

<details>

##### 2022

- **DREN:Dual-Level Representation Enhancement on Characteristic and Context for Image-Text Retrieval(TCSVT)** [[PDF]](https://ieeexplore.ieee.org/document/9794669)

- **M2D-BERT:Multi-scale Multi-modal Dictionary BERT For Effective Text-image Retrieval in Multimedia Advertising(CIKM)** [[PDF]](https://dl.acm.org/doi/abs/10.1145/3511808.3557653)

- **ViSTA:ViSTA: Vision and Scene Text Aggregation for Cross-Modal Retrieval(CVPR)** [[PDF]](https://arxiv.org/abs/2203.16778)

- **COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval(CVPR)** [[PDF]](https://arxiv.org/abs/2204.07441)

- **EI-CLIP: Entity-aware Interventional Contrastive Learning for E-commerce Cross-modal Retrieval(CVPR)** [[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_EI-CLIP_Entity-Aware_Interventional_Contrastive_Learning_for_E-Commerce_Cross-Modal_Retrieval_CVPR_2022_paper.pdf)

- **SSAMT:Constructing Phrase-level Semantic Labels to Form Multi-Grained Supervision for Image-Text Retrieval(ICMR)** [[PDF]](https://arxiv.org/abs/2109.05523)

- **TEAM:Token Embeddings Alignment for Cross-Modal Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3503161.3548107)

- **CAliC: Accurate and Efficient Image-Text Retrieval via Contrastive Alignment and Visual Contexts Modeling(MM)** [[PDF]](https://dl.acm.org/doi/abs/10.1145/3503161.3548320)


##### 2021

- **GRAN:Global Relation-Aware Attention Network for Image-Text Retrieval(ICMR)** [[PDF]](https://dl.acm.org/doi/abs/10.1145/3460426.3463615)

- **PCME:Probabilistic Embeddings for Cross-Modal Retrieval(CVPR)** [[PDF]](https://arxiv.org/abs/2101.05068) [[code]](https://github.com/naver-ai/pcme)

##### 2020

- **FashionBERT: Text and Image Matching with Adaptive Loss for Cross-modal Retrieval(SIGIR)** [[PDF]](https://arxiv.org/abs/2005.09801)

##### 2019

- **PVSE:Polysemous Visual-Semantic Embedding for Cross-Modal Retrieval(CVPR)** [[PDF]](https://arxiv.org/abs/1906.04402) [[Code]](https://github.com/yalesong/pvse)

</details>

#### 2.3.2.5 Cross-modal Generation

<details>

##### 2022

- **PCMDA:Paired Cross-Modal Data Augmentation for Fine-Grained Image-to-Text Retrieval(MM)**[[PDF]](https://arxiv.org/abs/2207.14428)

##### 2021

- **CRGN:Deep Relation Embedding for Cross-Modal Retrieval(TIP)** [[PDF]](https://ieeexplore.ieee.org/document/9269483)[[Code]](https://github.com/zyfsa/CRGN)

- **X-MRS:Cross-Modal Retrieval and Synthesis (X-MRS): Closing the Modality Gapin Shared Representation Learning(MM)** [[PDF]](https://arxiv.org/abs/2012.01345)[[Code]](https://github.com/SamsungLabs/X-MRS)

##### 2020

- **AACR:Augmented Adversarial Training for Cross-Modal Retrieval(TMM)** [[PDF]](https://ieeexplore.ieee.org/document/9057710) [[Code]](https://github.com/yiling2018/aacr)

##### 2018

- **LSCO:Learning Semantic Concepts and Order for Image and Sentence Matching(CVPR)** [[PDF]](https://arxiv.org/abs/1712.02036)

- **TCCM:Towards Cycle-Consistent Models for Text and Image Retrieval(CVPR)** [[PDF]](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11132/Cornia_Towards_Cycle-Consistent_Models_for_Text_and_Image_Retrieval_ECCVW_2018_paper.pdf)

- **GXN:Look, Imagine and Match: Improving Textual-Visual Cross-Modal Retrieval with Generative Models(CVPR)** [[PDF]](https://arxiv.org/abs/1711.06420)

##### 2017

- **2WayNet:Linking Image and Text with 2-Way Nets(CVPR)** [[PDF]](https://arxiv.org/abs/1608.07973)

##### 2015

- **DVSA:Deep Visual-Semantic Alignments for Generating Image Descriptions(CVPR)** [[PDF]](https://arxiv.org/abs/1412.2306)

</details>


</details>

</details>

## 2.4 Supervised-cross-modal-real-valued

<details>

### 2.4.1 Supervised shallow cross-modal real-valued retrieval

<details>

#### 2.4.1.1 CCA

<details>

##### 2022

- **MVMLCCA: Multi-view Multi-label Canonical Correlation Analysis for Cross-modal Matching and Retrieval(CVPRW)**  [[PDF]](https://openaccess.thecvf.com/content/CVPR2022W/MULA/html/Sanghavi_Multi-View_Multi-Label_Canonical_Correlation_Analysis_for_Cross-Modal_Matching_and_Retrieval_CVPRW_2022_paper.html) [[Code]](https://github.com/Rushil231100/MVMLCCA)

##### 2015

- **ml-CCA: Multi-Label Cross-modal Retrieval(ICCV)**  [[PDF]](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Ranjan_Multi-Label_Cross-Modal_Retrieval_ICCV_2015_paper.html) [[Code]](https://github.com/Viresh-R/ml-CCA)

##### 2014

- **cluster-CCA: Cluster Canonical Correlation Analysis(ICAIS)**  [[PDF]](https://proceedings.mlr.press/v33/rasiwasia14.html)

##### 2012

- **GMA: Generalized Multiview Analysis: A Discriminative Latent Space(CVPR)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/6247923) [[Code]](https://github.com/huyt16/Twitter100k/tree/master/code/GMA-CVPR2012)

</details>

#### 2.4.1.2 Dictionary Learning

<details>

##### 2018
- **JDSLC: Joint Dictionary Learning and Semantic Constrained Latent Subspace Projection for Cross-Modal Retrieval(CIKM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3269206.3269296)

##### 2016
- **DDL: Discriminative Dictionary Learning With Common Label Alignment for Cross-Modal Retrieval(TMM)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7353179)

##### 2014

- **CMSDL: Cross-Modality Submodular Dictionary Learning for Information Retrieval(CIKM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2661829.2661926)

##### 2013

- **SliM2: Supervised Coupled Dictionary Learning with Group Structures for Multi-Modal Retrieval(AAAI)**  [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/8603)

</details>

#### 2.4.1.3 Feature Mapping

<details>

##### 2017

- **MDSSL: Cross-Modal Retrieval Using Multiordered Discriminative Structured Subspace Learning(TMM)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7801820)

- **JLSLR: Joint Latent Subspace Learning and Regression for Cross-Modal Retrieval(SIGIR)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3077136.3080678)

##### 2016

- **JFSSL: Joint Feature Selection and Subspace Learning for Cross-Modal Retrieval(TPAIMI)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7346492) [[Code]](https://github.com/2012013382/JFSSL-Cross-Modal-Retrieval)

- **MDCR: Modality-Dependent Cross-Media Retrieval(TIST)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2775109)

- **CRLC: Cross-modal Retrieval with Label Completion(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2964284.2967231)

##### 2013

- **JGRHML: Heterogeneous Metric Learning with Joint Graph Regularization for Cross-Media Retrieval(AAAI)**  [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/8464) [[Code]](https://github.com/PKU-ICST-MIPL/JGRHML_AAAI2013)

- **LCFS: Learning Coupled Feature Spaces for Cross-modal Matching(ICCV)**  [[PDF]](https://openaccess.thecvf.com/content_iccv_2013/html/Wang_Learning_Coupled_Feature_2013_ICCV_paper.html)

##### 2011

- **Multi-NPP: Learning Multi-View Neighborhood Preserving Projections(ICML)**  [[PDF]](https://icml.cc/2011/papers/304_icmlpaper.pdf)

</details>

#### 2.4.1.4 Topic Model

<details>

##### 2014

- **M3R: Multi-modal Mutual Topic Reinforce Modeling for Cross-media Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2647868.2654901)

- **NPBUS: Nonparametric Bayesian Upstream Supervised Multi-Modal Topic Models(WSDM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2556195.2556238)

</details>

#### 2.4.1.5 Other Shallow

<details>

##### 2019

- **CMOS: Online Asymmetric Metric Learning With Multi-Layer Similarity Aggregation for Cross-Modal Retrieval(TIP)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/8680035)

##### 2017

- **CMOS: Online Asymmetric Similarity Learning for Cross-Modal Retrieval(CVPR)**  [[PDF]](https://openaccess.thecvf.com/content_cvpr_2017/html/Wu_Online_Asymmetric_Similarity_CVPR_2017_paper.html)

##### 2016

- **PL-ranking: A Novel Ranking Method for Cross-Modal Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2964284.2964336)

- **RL-PLS: Cross-modal Retrieval by Real Label Partial Least Squares(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2964284.2967216)

##### 2013

- **PFAR: Parallel Field Alignment for Cross Media Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2502081.2502087)

</details>

</details>

### 2.4.2 Supervised deep cross-modal real-valued retrieval

<details>

#### 2.4.2.1 Naive Network

<details>

##### 2022

- **C3CMR: Cross-Modality Cross-Instance Contrastive Learning for Cross-Media Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3503161.3548263)

##### 2020

- **ED-Net: Event-Driven Network for Cross-Modal Retrieval(CIKM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3340531.3412081)

##### 2019

- **DSCMR: Deep Supervised Cross-modal Retrieval(CVPR)**  [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhen_Deep_Supervised_Cross-Modal_Retrieval_CVPR_2019_paper.html) [[Code]](https://github.com/penghu-cs/DSCMR)

- **SAM: Cross-Modal Subspace Learning with Scheduled Adaptive Margin Constraints(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3343031.3351030)

##### 2017

- **deep-SM: Cross-Modal Retrieval With CNN Visual Features: A New Baseline(TCYB)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7428926) [[Code]](https://github.com/zh-plus/CMR-CNN-New-Baseline)

- **CCL: Cross-modal Correlation Learning With Multigrained Fusion by Hierarchical Network(TMM)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/8013822)

- **MSFN: Cross-media Retrieval by Learning Rich Semantic Embeddings of Multimedia(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3123266.3123369)

- **MNiL: Multi-Networks Joint Learning for Large-Scale Cross-Modal Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3123266.3123317) [[Code]](https://github.com/liangzhang1407/Multi-Networks-Joint-Learning-for-Large-Scale-Cross-Modal-Retrieval)

##### 2016

- **MDNN: Effective deep learning-based multi-modal retrieval(VLDB)**  [[PDF]](https://link.springer.com/article/10.1007/s00778-015-0391-4)

##### 2015

- **RE-DNN: Deep Semantic Mapping for Cross-Modal Retrieval(ICTAI)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/7372141)

- **C2MLR: Deep Compositional Cross-modal Learning to Rank via Local-Global Alignment(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/2733373.2806240)

</details>

#### 2.4.2.2 GAN

<details>

##### 2022

- **JFSE: Joint Feature Synthesis and Embedding: Adversarial Cross-Modal Retrieval Revisited(TPAMI)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9296975) [[Code]](https://github.com/CFM-MSG/Code_JFSE)

##### 2021

- **AACR: Augmented Adversarial Training for Cross-Modal Retrieval(TMM)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9057710) [[Code]](https://github.com/yiling2018/aacr)

##### 2018

- **CM-GANs: Cross-modal Generative Adversarial Networks for Common Representation Learning(TMM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3284750) [[Code]](https://github.com/PKU-ICST-MIPL/CM-GANS_TOMM2019)

##### 2017

- **ACMR: Adversarial Cross-Modal Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3123266.3123326) [[Code]](https://github.com/cuishuhao/ACMR)

</details>

#### 2.4.2.3 Graph Model

<details>

##### 2022

- **AGCN: Adversarial Graph Convolutional Network for Cross-Modal Retrieval(TCSVT)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9411880)

- **ALGCN: Adaptive Label-Aware Graph Convolutional Networks for Cross-Modal Retrieval(TMM)**  [[PDF]](https://ieeexplore.ieee.org/abstract/document/9508809)

- **HGE: Cross-Modal Retrieval with Heterogeneous Graph Embedding(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3503161.3548195)

##### 2021

- **GCR: Exploring Graph-Structured Semantics for Cross-Modal Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3474085.3475567) [[Code]](https://github.com/neoscheung/GCR)

- **DAGNN: Dual Adversarial Graph Neural Networks for Multi-label Cross-modal Retrieval(AAAI)**  [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/16345)

##### 2018

- **SSPE: Learning Semantic Structure-preserved Embeddings for Cross-modal Retrieval(MM)**  [[PDF]](https://dl.acm.org/doi/abs/10.1145/3240508.3240521)

</details>

#### 2.4.2.4 Transformer
<details>

##### 2021

- **RLCMR: Rethinking Label-Wise Cross-Modal Retrieval from A Semantic Sharing Perspective(IJCAI)**  [[PDF]](https://www.ijcai.org/proceedings/2021/0454.pdf)

</details>

</details>

</details>

## 2.5 Cross-modal-Retrieval-under-Special-Retrieval-Scenario

<details>

#### 2.5.1 Semi-Supervised (Real-valued)

<details>

##### 2020

- **SSCMR:Semi-Supervised Cross-Modal Retrieval With Label Prediction(TMM)** [[PDF]](https://ieeexplore.ieee.org/document/8907496)

##### 2019

- **A3VSE:Annotation Efficient Cross-Modal Retrieval with Adversarial Attentive Alignment(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3343031.3350894)

- **ASFS:Adaptive Semi-Supervised Feature Selection for Cross-Modal Retrieval(TMM)** [[PDF]](https://ieeexplore.ieee.org/document/8501586)

##### 2018

- **GSS-SL:Generalized Semi-supervised and Structured Subspace Learning for Cross-Modal Retrieval(TMM)** [[PDF]](https://ieeexplore.ieee.org/document/7968473)

##### 2017

- **SSDC:Semi-supervised Distance Consistent Cross-modal Retrieval(VSCC)**[[PDF]](https://dl.acm.org/doi/abs/10.1145/3132734.3132735)

##### 2013

- **JRL:Learning Cross-Media Joint Representation With Sparse and Semisupervised Regularization(TCSVT)** [[PDF]](https://ieeexplore.ieee.org/document/6587747)[[Code]](https://github.com/PKU-ICST-MIPL/JRL_TCSVT2014)

##### 2012

- **MVML-GL:Multiview Metric Learning with Global Consistency and Local Smoothness(TIST)** [[PDF]](https://dl.acm.org/doi/10.1145/2168752.2168767)

</details>

#### 2.5.2 Semi-Supervised (Hashing)

<details>

##### 2020

- **SCH-GAN：Semi-Supervised Cross-Modal Hashing by Generative Adversarial Network(TC)** [[PDF]](https://ieeexplore.ieee.org/document/8472802) [[Code]](https://github.com/PKU-ICST-MIPL/SCHGAN_TCYB2018)

- **SGCH:Semi-supervised graph convolutional hashing network for large-scale cross-modal retrieval(ICIP)** [[PDF]](https://ieeexplore.ieee.org/document/9190641)

##### 2019

- **SSDQ:Semi-supervised Deep Quantization for Cross-modal Search(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3343031.3350934)

- **S3PH:Semi-supervised semantic-preserving hashing for efficient cross-modal retrieval(ICME)** [[PDF]](https://ieeexplore.ieee.org/document/8784930)

##### 2017

- **AUSL:Adaptively Uniﬁed Semi-supervised Learning for Cross-Modal Retrieval(IJCAI)** [[PDF]](https://www.ijcai.org/proceedings/2017/0476.pdf)

##### 2016

- **NPH:Neighborhood-Preserving Hashing for Large-Scale Cross-Modal Search(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/2964284.2967241)

</details>

#### 2.5.3 Imbalance (Real-valued)

<details>

##### 2021

- **PAN: Prototype-based Adaptive Network for Robust Cross-modal Retrieval(SIGIR)** [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3404835.3462867)

- **MCCN: Multimodal Coordinated Clustering Network for Large-Scale Cross-modal Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3474085.3475670)

##### 2020

- **DAVAE:Incomplete Cross-modal Retrieval with Dual-Aligned Variational Autoencoders(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3394171.3413676)

##### 2015

- **SCDL:Semi-supervised Coupled Dictionary Learning for Cross-modal Retrieval in Internet Images and Texts(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/2733373.2806346)

- **LGCFL:Learning Consistent Feature Representation for Cross-Modal Multimedia Retrieval(TMM)** [[PDF]](https://ieeexplore.ieee.org/document/7006724)

</details>

#### 2.5.4 Imbalance (Hashing)

<details>

##### 2020

- **RUCMH:Robust Unsupervised Cross-modal Hashing for Multimedia Retrieval(TOIS)** [[PDF]](https://dl.acm.org/doi/10.1145/3389547)

- **ATFH-N:Adversarial Tri-Fusion Hashing Network for Imbalanced Cross-Modal Retrieval(TETCI)** [[PDF]](https://ieeexplore.ieee.org/document/9139424)

- **FlexCMH:Flexible Cross-Modal Hashing(TNNLS)** [[PDF]](https://ieeexplore.ieee.org/document/9223723)

##### 2019

- **TFNH:Triplet Fusion Network Hashing for Unpaired Cross-Modal Retrieval(ICMR)** [[PDF]](https://www.comp.hkbu.edu.hk/~ymc/papers/conference/icmr19-publication-version.pdf) [[Code]](https://github.com/hutt94/TFNH)

- **CALM:Collective Afﬁnity Learning for Partial Cross-Modal Hashing(TIP)** [[PDF]](https://ieeexplore.ieee.org/document/8846593)

- **MTFH: A Matrix Tri-Factorization Hashing Framework for Efﬁcient Cross-Modal Retrieval:(TIP)** [[PDF]](https://arxiv.org/abs/1805.01963) [[Code]](https://github.com/starxliu/MTFH)

- **GSPH:Generalized Semantic Preserving Hashing for Cross-Modal Retrieval(TIP)** [[PDF]](https://ieeexplore.ieee.org/document/8425016)

##### 2018

- **DAH:Dense Auto-Encoder Hashing for Robust Cross-Modality Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3240508.3240684)

##### 2017

- **GSPH:Generalized Semantic Preserving Hashing for n-Label Cross-Modal Retrieval(CVPR)** [[PDF]](https://ieeexplore.ieee.org/document/8099765) [[Code]](https://github.com/devraj89/Generalized-Semantic-Preserving-Hashing-for-N-Label-Cross-Modal-Retrieval)

</details>

#### 2.5.5 Incremental

<details>

##### 2021

- **MARS: Learning Modality-Agnostic Representation for Scalable Cross-Media Retrieval(TCSVT)** [[PDF]](https://ieeexplore.ieee.org/document/9654230)

- **CCMR:Continual learning in cross-modal retrieval(CVPR)** [[PDF]](https://arxiv.org/abs/2104.06806)

- **SCML:Real-world Cross-modal Retrieval via Sequential Learning(TMM)** [[PDF]](https://ieeexplore.ieee.org/document/9117043)

##### 2020

- **ATTL-CEL:Adaptive Temporal Triplet-loss for Cross-modal Embedding Learning(MM)**[[PDF]](https://dl.acm.org/doi/10.1145/3394171.3413540)

##### 2019

- **SVHNs:Separated Variational Hashing Networks for Cross-Modal Retrieval(MM)** [[PDF]](https://dl.acm.org/doi/10.1145/3343031.3351078)

- **ECMH:Extensible Cross-Modal Hashing(IJCAI)** [[PDF]](https://www.ijcai.org/proceedings/2019/0292.pdf) [[Code]](https://github.com/3andero/Extensible-Cross-Modal-Hashing)

##### 2018

- **TempXNet:Temporal Cross-Media Retrieval with Soft-Smoothing(MM)** [[PDF]](https://arxiv.org/abs/1810.04547)

</details>

#### 2.5.6 Noise

<details>

##### 2022

- **DECL: Deep Evidential Learning with Noisy Correspondence for Cross-modal Retrieval(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3503161.3547922) [[Code]](https://github.com/QinYang79/DECL)

- **ELRCMR: Early-Learning regularized Contrastive Learning for Cross-Modal Retrieval with Noisy Labels(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3503161.3548066)

- **CMMQ: Mutual Quantization for Cross-Modal Search with Noisy Labels(CVPR)** [(PDF)](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Mutual_Quantization_for_Cross-Modal_Search_With_Noisy_Labels_CVPR_2022_paper.html)

##### 2021

- **MRL: Learning Cross-Modal Retrieval with Noisy Labels(CVPR)** [(PDF)](https://openaccess.thecvf.com/content/CVPR2021/html/Hu_Learning_Cross-Modal_Retrieval_With_Noisy_Labels_CVPR_2021_paper.html) [[Code]](https://github.com/penghu-cs/MRL)

##### 2018

- **WSJE: Webly Supervised Joint Embedding for Cross-Modal Image-Text Retrieval(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3240508.3240712)

</details>

#### 2.5.7 Cross-Domain

<details>

##### 2021

- **M2GUDA: Multi-Metrics Graph-Based Unsupervised Domain Adaptation for Cross-Modal Hashing(ICMR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3460426.3463670)

- **ACP: Adaptive Cross-Modal Prototypes for Cross-Domain Visual-Language Retrieval(CVPR)** [(PDF)](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Adaptive_Cross-Modal_Prototypes_for_Cross-Domain_Visual-Language_Retrieval_CVPR_2021_paper.html)

##### 2020

- **DASG: Unsupervised Cross-Media Retrieval Using Domain Adaptation With Scene Graph(TCSVT)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8902166)

</details>

#### 2.5.8 Zero-Shot

<details>

##### 2020

- **LCALE: Learning Cross-Aligned Latent Embeddings for Zero-Shot Cross-Modal Retrieval(AAAI)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/6817)

- **CFSA: Correlated Features Synthesis and Alignment for Zero-shot Cross-modal Retrieval(SIGIR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3397271.3401149)

##### 2019

- **ZS-CMR: Learning Cross-Aligned Latent Embeddings for Zero-Shot Cross-Modal Retrieval(TIP)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8743557)

</details>

#### 2.5.9 Few-Shot

<details>

##### 2021

- **SOCMH: Know Yourself and Know Others: Efficient Common Representation Learning for Few-shot Cross-modal Retrieval(ICMR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3460426.3463632)

</details>

#### 2.5.10 Online Learning

<details>

##### 2020

- **CMOLRS: Online Fast Adaptive Low-Rank Similarity Learning for Cross-Modal Retrieval(TMM)** [(PDF)](https://ieeexplore.ieee.org/abstract/document/8845601) [[Code]](https://github.com/yiling2018/cmolrs)

- **LEMON: Label Embedding Online Hashing for Cross-Modal Retrieval(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3394171.3413971) [[Code]](https://github.com/yxinwang/LEMON-MM2020)

##### 2019

- **FOMH: Flexible Online Multi-modal Hashing for Large-scale Multimedia Retrieval(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3343031.3350999) [[Code]](https://github.com/lxuu306/FOMH)

##### 2017

- **OCMSR: Online Cross-Modal Scene Retrieval by Binary Representation and Semantic Graph(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3123266.3123311)

##### 2016

- **OCMH: Online cross-modal hashing for web image retrieval(AAAI)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/9982)

</details>

#### 2.5.11 Hierarchical

<details>

##### 2020

- **SHDCH: Supervised Hierarchical Deep Hashing for Cross-Modal Retrieval(MM)** [(PDF)](https://ojs.aaai.org/index.php/AAAI/article/view/9982) [[Code]](https://github.com/SDU-MIMA/SHDCH)

##### 2019

- **HiCHNet: Supervised Hierarchical Cross-Modal Hashing(SIGIR)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3331184.3331229) [[Code]](https://github.com/CCSun-cs/HichNet)

</details>

#### 2.5.12 Fine-grained

<details>

##### 2022

- **PCMDA: Paired Cross-Modal Data Augmentation for Fine-Grained Image-to-Text Retrieval(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3503161.3547809)

##### 2019

- **FGCrossNet: A New Benchmark and Approach for Fine-grained Cross-media Retrieval(MM)** [(PDF)](https://dl.acm.org/doi/abs/10.1145/3343031.3350974) [[Code]](https://github.com/PKU-ICST-MIPL/FGCrossNet_ACMMM2019)

</details>


</details>

</details>

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

- **Online learning--SHDCH**

Dataset Link:

    Baidu Yun Link: https://pan.baidu.com/s/1-CsIJbvz3IFsmDgYk9BwYg?pwd=7hd8
    Code: 7hd8
    
- **Noise--MRL**

Dataset Link:

    Baidu Yun Link: https://pan.baidu.com/s/1FIrB-gXJa9VHKzLRQZf30Q?pwd=g3qt
    Code: g3qt

- **Online learning--LEMON**

Dataset Link:

    Baidu Yun Link: https://pan.baidu.com/s/1s5SnnAXo5wK7cmRs3zNq4w?pwd=jxjo
    Code: jxjo
    
- **Fine-grained--FGCrossNet**

Dataset Link:

    Baidu Yun Link: https://pan.baidu.com/s/1OYxCLmNKvPzwLIs5snTOlA?pwd=r80g
    Code: r80g
        
- **Noise--DECL**

Dataset Link:

    Baidu Yun Link: https://pan.baidu.com/s/1FcxkwOuuiUXnIl1LAatDLA?pwd=nl2z
    Code: nl2z
