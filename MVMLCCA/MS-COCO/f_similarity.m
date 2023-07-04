function [similarity_value] = f_similarity(Z_i_g,Z_j_h,f_type)

%Z_i_g = C x 1 column vector of labels for the gth data point in ith modality (can be made up of 0 1 or some weights w_k)
% Z_j_h = C x 1 column vector of labels for the hth data point in jth modality 
if (f_type == "dot_product")
    similarity_value = dot(Z_i_g,Z_j_h)/(norm(Z_i_g)*norm(Z_j_h));
end
if (f_type == "squared_exponent")
    labelsimilaritysigma = 1;
    similarity_value = exp(-1*(pdist2(Z_i_g,Z_j_h,'euclidean'))/labelsimilaritysigma);%pdist2 is not squared
end





