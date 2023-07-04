function X_out = MyNormalization(X_1)
[n_1,p_1] = size(X_1);

%calculating mean accross the columns, which will give mean of values for a
%particular feature for all the n data samples.
mean_X_1 = mean(X_1); %1*p_1 column vector
%variance across rows
var_X_1 = sqrt(var(X_1));

%making the effective mean 0 and deviation 1. Gaussian distribution
temp_X_1 = (X_1 - repmat(mean_X_1,n_1,1));%./repmat(var_X_1,n_1,1);
%temp_X_1 = X_1;
%L2 normalization for a particular feature generalized for all.
l2_norm_1 = sqrt(sum(temp_X_1.^2));

X_out = temp_X_1;
% if(l2_norm_1 ~= 0)
%     X_out = (temp_X_1)./repmat(l2_norm_1,n_1,1);
% end

