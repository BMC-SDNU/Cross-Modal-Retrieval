function model = SePH_train(X1, X2, Y, opts)
%my implementation of the CVPR'15 paper "Semantics-Preserving Hashing for Cross-View Retrieval"
%
%input:
% X1: data of the first view
% X2: data of the second view
% Y: label matrix
% opts:
%  alpha: regulization parameter
%  K: number of bits
%  use_kernel: if use kernel feature mapping
%  sample_size: number of anchors
%  sigma1_scl, sigma2_scl: scaling factor of each view
%  maxiter: maximal number of iterations
%  max_sample_num: the maximal number of samples could be dealt
%
%output:
% model:
%  A1, A2: anchors
%  W1, W2: hash functions
%  squared_sigma1, squared_sigma2: kernel widthes
%  use_kernel: used in test phase
%  mean_vec1, mean_vec2: data mean of each view
%  

tic;

N = size(X1,2);
assert(N==size(X2,2));
assert(N==size(Y,2));
Y = sparse(Y);
K = opts.K;

if(~isfield(opts,'max_sample_num'))
    opts.max_sample_num = 10000;%30000
end
max_sample_num = min(opts.max_sample_num, N);
sel = randsample(N, max_sample_num);
X1 = X1(:, sel);
X2 = X2(:, sel);
Y = Y(:, sel);
N = max_sample_num;

%kernel feature mapping
if opts.use_kernel
    if(isfield(opts,'A1'))
        A1 = opts.A1;
    else
        ind = randsample(N, opts.sample_size);
        A1 = X1(:,ind);
    end
    D2 = Euclid2(A1, A1, 'col', 0);
    squared_sigma1 = mean(D2(:))*opts.sigma1_scl;
    D2 = Euclid2(A1, X1, 'col', 0);
    X1 = exp(-D2/2/squared_sigma1);
    model.A1 = A1;
    model.squared_sigma1 = squared_sigma1;
    
    if(isfield(opts,'A2'))
        A2 = opts.A2;
    else
        ind = randsample(N, opts.sample_size);
        A2 = X2(:,ind);
    end
    D2 = Euclid2(A2, A2, 'col', 0);
    squared_sigma2 = mean(D2(:))*opts.sigma2_scl;
    D2 = Euclid2(A2, X2, 'col', 0);
    X2 = exp(-D2/2/squared_sigma2);
    model.A2 = A2;
    model.squared_sigma2 = squared_sigma2;
    
    clear ind D2;
end
model.use_kernel = opts.use_kernel;

%remove data mean
mean_vec1 = mean(X1,2);
mean_vec2 = mean(X2,2);
X1 = bsxfun(@minus, X1, mean_vec1);
X2 = bsxfun(@minus, X2, mean_vec2);
model.mean_vec1 = mean_vec1;
model.mean_vec2 = mean_vec2;

%initialize B
W1 = ITQCCAtrain1(X1', Y', K);
B = sign(W1'*X1);
clear W1;

%compute P
Y = bsxfun(@rdivide, Y, sqrt(sum(Y.^2,1))+eps);
P = Y'*Y;
P = P - diag(diag(P));
P = P/sum(P(:));

%optimize binary codes using LBFGS
B = call_lbfgs(B(:)', P, opts.alpha/numel(B), opts.maxiter);
B = sign(reshape(B, K, N));
B = (B+1)/2;

%fit logistic regression model on B
W1 = fit_mul_logistic_fun(X1', B');
W2 = fit_mul_logistic_fun(X2', B');

model.W1 = W1;
model.W2 = W2;
model.time = toc;

end

function out = call_lbfgs(init, P, alpha, maxiter)
%call LBFGS

f = @(x) fun_grad_SePH(x, P, alpha);
out = lbfgsb(f, -inf(numel(init),1), inf(numel(init),1), struct('maxIts', maxiter, 'x0', init(:), 'printEvery', 5));

end



