function [B, time] = SePH_test_fuse(X1, X2, model, type)
%encode the training data to binary codes, fuse the results from different
%views
%
%input:
% X1: data from the first view
% X2: data from the second view
% model: trained model
% type: 'binary' or 'uint8'
%
%output:
% B: (compact) binary codes

tic;

assert(strcmp(type,'uint8')||strcmp(type,'binary'));

if model.use_kernel
    D2 = Euclid2(model.A1, X1, 'col', 0);
    X1 = exp(-D2/2/model.squared_sigma1);
    
    D2 = Euclid2(model.A2, X2, 'col', 0);
    X2 = exp(-D2/2/model.squared_sigma2);
end

X1 = bsxfun(@minus, X1, model.mean_vec1);
X2 = bsxfun(@minus, X2, model.mean_vec2);

P1 = sigmoid(model.W1'*X1);
P2 = sigmoid(model.W2'*X2);
B = (P1.*P2-(1-P1).*(1-P2))>0;

if(strcmp(type,'uint8'))
    B = compactbit(B);
end

time = toc;

end