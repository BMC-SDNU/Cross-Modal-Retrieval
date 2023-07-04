function [mAP,mAP21] = MyRetrieval3(Wx,D,p_each,index_1,index_2,X_1_test , X_2_test ,Z_1_test,Z_2_test, ld , D_power)

%Wx : matrix of eigen vectors , size (sum of features x sum of features)
%D : diagnol matrix of eigen values , size (sum of features x sum of features)
%p_each = matrix of size n_modality x 1 , where each row contains number of
%features in that corresponding modality.
%<index_1,index_2> = modality index for which the test code is to run
%X_1_test  : feature matrix for modality 1 for testing
%X_2_test  : feature matrix for modality 2 for testing
%Z_1_test  : Label matrix for modality 1 for testing
%Z_2_test  : Label matrix for modality 2 for testing
%ld : latent space dimension
%D_power : integer value as the parameter of the power function in scaling
%the eigen vector coordinates with coresponding eigenvalues. 
new_Wx = zeros(p_each(index_1,1)+p_each(index_2,1),sum(p_each));
new_Wx(1:p_each(index_1,1),:) = Wx(sum(p_each(1:index_1-1)) + 1 : sum(p_each(1:index_1-1))+p_each(index_1,1) ,:);
new_Wx(p_each(index_1,1)+1 :p_each(index_1,1)+p_each(index_2,1),:) = Wx(sum(p_each(1:index_2-1)) + 1 : sum(p_each(1:index_2-1))+p_each(index_2,1) , :);
% save('new_Wx');
[mAP,mAP21] = MyRetrieval(new_Wx,D,X_1_test , X_2_test ,Z_1_test,Z_2_test, ld , D_power);