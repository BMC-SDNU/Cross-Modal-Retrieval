# Code_JFSE
This is the source code of our paper "Joint Feature Synthesis and Embedding: Adversarial Cross-modal Retrieval Revisited", which is an extension of our previous conference work entitled "Adversarial Cross-modal Retrieval" published in ACM Multimedia Conference (ACM MM) 2017.

## Installation and Requirements
### Installation
We recommended the following dependencies:
- Python 3
- Tensorflow > 1.0
- Numpy
- pickle
  
## Training
### Data Download
Data Preparation: We use [PKU XMediaNet dataset](http://59.108.48.34/tiki/XMediaNet/) as example, and the data should be put in ./data/. The data files can be download from the [link](http://59.108.48.34/mipl/tiki-download_file.php?fileId=1012) and unzipped to the above path.

### Running
Run demo.py to train models and calculate mAP
```
python demo.py
```


### Main Idea

In this paper, we revisit the adversarial learning in existing cross-modal GAN methods and propose **Joint Feature Synthesis and Embedding (JFSE)**, a novel method that jointly performs multimodal feature synthesis and common embedding space learning.

*Unlike most existing cross-modal GAN approaches that focus on the standard retrieval, the proposed JFSE approach can be simultaneously applied to standard retrieval, zero-shot retrieval and generalized zero-shot retrieval tasks.* 
<br>

   ![framework](https://github.com/CFM-MSG/Code_JFSE/blob/master/fig/flowchart.png)

<br>

### What task does our code (method) solve?
The existing cross-modal GAN approaches typically 1) require labeled multimodal data of massive labor cost to establish cross-modal correlation; 2) utilize the vanilla GAN model that results in unstable training procedure and meaningless synthetic features and 3) lack of extensibility for retrieving cross-modal data of new classes.

JFSE proposes an advanced network architecture of coupled conditional WGANs (cWGANs) to synthesize multimodal data in the semantic feature space to complement the true data to overcome the lack of labeled cross-modal data in the available multimodal datasets. Unlike most existing studies only focus on the standard cross-modal retrieval scenario, JFSE further explores the more practical scenarios of zero-shot and generalized zero-shot cross-modal retrieval scenarios. 

### Insight of our JFSE model:
- Effective cross-modal feature synthesis with improved cWGAN structure to produce meaningful synthetic features and to learn more effective common embedding space.
- Advanced common embedding space learning. To support both standard retrieval, zero-shot and generalized Zero-shot retrieval tasks, we develop three advanced distribution alignment schemes to capture cross-modal correlation and enable the knowledge transfer during common embedding space learning. Besides, to enable the knowledge transfer between classes, we introduce an advanced cycle-consistency constraint that preserves the semantic compatibility between original features and the mapped common features of both true and synthetic data.

### Difference with the Other Cross-modal GAN approaches

<img src="https://github.com/CFM-MSG/Code_JFSE/blob/master/fig/comparison_cmgan.png" width="95%" />

- The early work of ACMR [13] takes the feature projection as the “implicit" generator to generate the embedding features, which is not the true meaning of the GAN structure. Instead, it leverages a discriminator to distinguish the source of the projected features from images or text captions, which helps to learn a modality-invariant embedding space.
- The later works of GXN [21], CM-GANS [15], R2GAN [5], DADN [28] and TANSS [22] all have two pairs of generator-discriminator for individual modalities, where a generator is commonly a vanilla GAN model for independent image-image and text-text generation on the feature level or pixel level. Note that these approaches also have a discriminator to discriminate against the modality of an embedding feature.
- Our JFSE approach takes two coupled cWGANs that consider the class embeddings as side information for cross-modal data synthesis on the feature level. Meanwhile, the class embeddings are treated as the common embedding space, which is more effective to correlate the feature synthesis for each modality and encapsulate a rich set of loss functions for effective distribution alignment.
- The proposed three distribution alignment schemes are more general and comprehensive that cover the diverse strategies used in CM-GANS [15], DADN [28] and TANSS [22]. Moreover, our JFSE method enables the knowledge transfer between seen and unseen classes for the practical scenarios of zero-shot and generalized zero-shot retrieval, which have not been investigated in previous cross-modal GAN approaches.

### Three Novel Distribution Alignment Schemes

- **Cross-modal Maximum Mean Discrepancy (CMMD)**. The target of the CMMD scheme is to maximize the mean discrepancy of both the true and synthetic embedding features of pairwise instances of different modalities.
- **Cross-Modal Correlation Alignment (CMCA)**. The CMCA scheme is another scheme the measures the cross-modal correlation by exploring the overall data distribution of all instances in different modalities. Unlike the CMMD scheme that models the overall cross-modal correlation based on pairwise instances, the CMCA scheme treats the embedding features of all the true and synthetic instances as matrix forms, and measure the cross-modal distance with the covariance of a matrix.
- **Cross-Modal Projection Matching (CMPM)**. The CMPM scheme is designed to model the cross-modal correlation by minimizing the Kullback-Leibler (KL) divergence between the normalized matching distributions and projection compatibility distributions of different modalities.

These proposed schemes can align the distributions of both true and synthetic features of two modalities in the common embedding space with transferable knowledge according to their semantics. *They are more advanced than the widely used schemes of correlation loss and triplet ranking loss that consider to learn the cross-modal correlation on the pairwise instance-level.*

### The State-of-the-art Performance 

#### Results on Standard Retrieval

The overall comparison of our JFSE method and the compared methods on the four datasets.

<br>

 <img src="https://github.com/CFM-MSG/Code_JFSE/blob/master/fig/standard.png" width="95%" />

#### Results on Zero-shot Retrieval

The overall comparison of our JFSE method and the compared methods for both seen class retrieval and unseen class retrieval on the four datasets.

<br>

 <img src="https://github.com/CFM-MSG/Code_JFSE/blob/master/fig/zsl.png" width="95%" />

 #### Results on Generalized Zero-Shot Retrieval

The overall comparison results of our JFSE method and the compared methods on Wikipedia and Pascal Sentences datasets.

<br>

 <img src="https://github.com/CFM-MSG/Code_JFSE/blob/master/fig/gzsl.png" width="70%" />

#### Results on Runtime Comparison

The runtime comparison (in seconds) of typical GAN-based and discriminative approaches on Wikipedia and PKU-XMediaNet datasets.

 <img src="https://github.com/CFM-MSG/Code_JFSE/blob/master/fig/runtime.png" width="95%" />