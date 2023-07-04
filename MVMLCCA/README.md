# MVMLCCA
Code for the paper "Multi-view Multi-label Canonical  Correlation Analysis  for Cross-modal Matching and Retrieval", CVPRW 2022

---
Features are uploaded at. : https://drive.google.com/drive/folders/1b4JSQYaiKL6CSQPz-FTWLNluuC8c29Rk?usp=sharing 
*** 

## Training 

1. Arrange the Training Input Features in a Cell Format. {Name it as C_x}
2. Arrange the Corresponding Labels in the similar Cell format {Name it as C_z}
3. Call the UnpairedCCA3 function with the inputs. This will return Wx & D matrix, which are eigenvectors and eigenvalues of the modified covariance matrix.

***
## Testing

Call MyRetrieval3 function, which will return mAP metric for two experiments. In each of the experiment,  One modality is kept as query, and other as target. The inputs are defined as follows.

1. Wx : matrix of eigen vectors , size (sum of features x sum of features)
2. D : diagnol matrix of eigen values , size (sum of features x sum of features)
3. p_each = matrix of size n_modality x 1 , where each row contains number of features in that corresponding modality.
4. <index_1,index_2> = modality index for which the test code is to run (In case of trained on only 2 modalities, and testing on just one modality, the index_1 = index_2 )  
5. X_1_test  : feature matrix for modality 1 for testing
6. X_2_test  : feature matrix for modality 2 for testing
7. Z_1_test  : Label matrix for modality 1 for testing
8. Z_2_test  : Label matrix for modality 2 for testing
9. ld : latent space dimension
10. D_power : integer value as the parameter of the power function in scaling the eigen vector coordinates with coresponding eigenvalues. 

---
Contact : sanghavi.1@iitj.ac.in


