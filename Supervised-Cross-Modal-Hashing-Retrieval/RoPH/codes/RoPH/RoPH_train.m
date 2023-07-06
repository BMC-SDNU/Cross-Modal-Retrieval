function model = RoPH_train(X1, X2, triplets, sims, opts)
%implementation of the proposed method RoPH
%input:
% X1: data of the first view, columns as samples
% X2: data of the second view
% triplets: all triplets
% sims: indicators of positive or negative triplet
% opts: training options
%  lambda: regularization parameter for ||H-B||_F^2
%  eta1,eta2: weight for ||W1^T\phi(X1)-B||_F^2 and ||W2^T\phi(X2)-B||_F^2
%  K: the number of bits
%  max_iter: maximal number of iterations
%  n_block: number of blocks
%
%output:
% model:
%  A1, A2: anchors
%  W1, W2: hash functions
%  squared_sigma1, squared_sigma2: kernel widthes
%  use_kernel: used in test phase
%  mean_vec1, mean_vec2: data mean of each view
%  B: binary codes {-1,+1}^{K*N} of the final iteration
%  time: training time

% This code is written by Kun Ding (kding@nlpr.ia.ac.cn).

tic;

N = size(X1,2);
assert(N==size(X2,2));
T = size(triplets,1)/N;
assert(T==round(T));

%%%%%%%%%%%%%%%check inputs and set default parameters%%%%%%%%%%%%%%%%%%%
%the range of sims should be [-1,1]
if(min(sims)<-1||max(sims)>1)
    error('the range of sims should be [-1,1]\n');
end

%margin
if(~isfield(opts,'margin'))
    opts.margin = opts.K;
end

%lambda
if(~isfield(opts,'lambda'))
    opts.lambda = 1;
end

%eta1 and eta2
if(~isfield(opts,'eta1'))
    opts.eta1 = 1e-5;
end
if(~isfield(opts,'eta2'))
    opts.eta2 = 1e-5;
end

%maximal iterations
if(~isfield(opts,'max_iter'))
    opts.max_iter = 5;
end

%if we want to know the objectives, set it as 1
if(~isfield(opts,'compute_obj'))
    opts.compute_obj = 0;
end

%number of blocks to use in parallel GraphCut
if(~isfield(opts,'n_block'))
    opts.n_block = 10;
end

%kernel feature mapping
if(opts.use_kernel)
    opts.sample_size = 500;
    opts.sigma1_scl = 1;
    opts.sigma2_scl = 1;
end

%%%%%%%%%%%%%%%%%%show parameter settings%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('number of bits, K=%d\n',opts.K);
fprintf('margin=%4.4f\n',opts.margin);
fprintf('regularization parameter for ||H-B||_F^2, lambda=%4.4f\n',opts.lambda);
fprintf('weight for ||W1^T*phi(X1)-B||_F^2 and ||W2^T*phi(X2)-B||_F^2, eta1=%e, eta2=%e\n',opts.eta1,opts.eta2);
fprintf('maximal iterations, max_iter=%d\n',opts.max_iter);
if(opts.use_kernel)
    fprintf('use_kernel=%d, number of anchors=%d, sigma1_scl=%4.4f, sigma2_scl=%4.4f\n',opts.use_kernel,opts.sample_size,opts.sigma1_scl,opts.sigma2_scl);
else
    fprintf('use_kernel=%d\n',opts.use_kernel);
end

%%%%%%%%%%%%%%%%%%%kernel feature mapping%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%extract triplet information%%%%%%%%%%%%%%%%%%%%%%%%%%
E = sparse(triplets, repmat((1:N*T)',1,2),repmat([1,-1],N*T,1),N,N*T);
Esum = compute_Esum(E, 500000);
row_sum = sum(Esum,1);
S = reshape(sims, T, N);
clear sims;

%%%%%%%%%%%%%%%%%%%initialization%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(isfield(opts,'B0'))
    B = B0;
else
    B = 2*(rand(opts.K,N)>=0.5)-1;
end
P1 = (X1*X1'+1e-5*eye(size(X1,1)))\eye(size(X1,1));
P2 = (X2*X2'+1e-5*eye(size(X2,1)))\eye(size(X2,1));
W1 = P1*(X1*B');
W2 = P2*(X2*B');
H = B;
M = 2*opts.K*ones(T,N);%margins

%%%%%%%%%%%%%%%%%%%main loop%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(N>300000)
    mkdir('./tmp');
else
    HEs = cell(1,opts.K);
end
objs = [];
lambda0 = opts.lambda;
for iter = 1:opts.max_iter
    
    %B step
    F1 = W1'*X1;
    F2 = W2'*X2;
    Stid = S.*M;
    for k = 1:opts.K
        if(N>300000)
            HEs_k = reshape(H(k,triplets(:,1))-H(k,triplets(:,2)),T,N);
            Stid = Stid - bsxfun(@times, HEs_k, B(k,:));
            save(['./tmp/HEs_',num2str(k)],'HEs_k');
        else
            HEs{k} = reshape(H(k,triplets(:,1))-H(k,triplets(:,2)),T,N);
            Stid = Stid - bsxfun(@times, HEs{k}, B(k,:));
        end
        fprintf('%d...',k);
        if(mod(k,10)==0)
            fprintf('\n');
        end
    end
    fprintf('\n');
    for tt = 1:3
        for k = 1:opts.K
            if(N>300000)
                load(['./tmp/HEs_',num2str(k)],'HEs_k');
                [B(k,:), Stid] = update_bk(Stid, HEs_k, B(k,:), H(k,:), F1(k,:), F2(k,:), opts);
            else
                [B(k,:), Stid] = update_bk(Stid, HEs{k}, B(k,:), H(k,:), F1(k,:), F2(k,:), opts);
            end
        end
    end
    clear HEs HEs_k;
    
    %H step
    for tt = 1:1
        for k = 1:opts.K
            [H(k,:), Stid] = update_hk(Stid, Esum, row_sum, E, B(k,:), H(k,:), opts);
            fprintf('%d...',k);
            if(mod(k,10)==0)
                fprintf('\n');
            end
        end
        fprintf('\n');
    end
    
    %M step
    Bcb = compactbit(B>0);
    Shat = compute_Shat(uint32(triplets-1), Bcb);
    M = max(double(Shat).*S,opts.margin);
    
    %W1 and W2 step
    W1 = P1*(X1*B');
    W2 = P2*(X2*B');
    
    %increase lambda?
    opts.lambda = opts.lambda*1;
    
    if(opts.compute_obj)
        objs(iter) = compute_obj(B, H, double(Shat), S, M, W1, W2, X1, X2, lambda0, opts.eta1, opts.eta2);
        fprintf('iter=%d, obj=%4.4f\n',iter,objs(iter));
    else
        fprintf('iter=%d\n',iter);
    end
    
end

model.B = B;
model.W1 = W1;
model.W2 = W2;
model.objs = objs;
model.time = toc;

if(N>300000)
    rmdir('./tmp','s');
end

end

function obj = compute_obj(B, H, Shat, S, M, W1, W2, X1, X2, lambda, eta1, eta2)

obj = sum(sum((Shat-S.*M).^2))...
    +eta1*sum(sum((W1'*X1-B).^2))...
    +eta2*sum(sum((W2'*X2-B).^2))...;
+lambda*sum(sum((H-B).^2));

end

function Esum = compute_Esum(E, bs)

N = size(E,2);
nb = ceil(N/bs);
Esum = E(:,1:min(bs,N))*E(:,1:min(bs,N))';
for i = 2:nb
    ind = (i-1)*bs+1:min(i*bs,N);
    Esum = Esum + E(:,ind)*E(:,ind)';
end

end

function [bk, Stid] = update_bk(Stid, HEs_k, bk, hk, f1k, f2k, opts)
%update the k-bit of B

Stid = Stid + bsxfun(@times, HEs_k, bk);
tmp = sum(HEs_k.*Stid,1) + opts.lambda*hk + opts.eta1*f1k + opts.eta2*f2k;
bk = sign_(tmp, bk);

%update Stid
Stid = Stid - bsxfun(@times, HEs_k, bk);

end

function [hk, Stid] = update_hk(Stid, Esum, row_sum, E, bk, hk, opts)

[T, N] = size(Stid);
Sold = reshape(hk*E,T,N);
Stid = Stid + bsxfun(@times, Sold, bk);
v = compute_rhs(Stid, E, bk, opts.lambda);
hk = call_GCmex(Esum, v, row_sum, hk, opts.n_block);%submodualr BQP, call Graphcut to solve

%update Stid
Snew = reshape(hk*E,T,N);
Stid = Stid - bsxfun(@times, Snew, bk);

end

function tmp = compute_rhs(Stid, E, bk, lambda)
%compute the coefficient f in BQP problem

Stid = bsxfun(@times, Stid, bk);
tmp = E*reshape(Stid,1,[])';
tmp = tmp' + lambda*bk;

end

function h = call_GCmex(Q, v, row_sum, h_init, n_block)
%Q: lower triangle matrix
%v: coefficient of -2*x*v'
%row_sum: row sum of Q
%h_init: inital h

labelcost = [0 0;0 1];
h_init = (h_init+1)/2;%convert to {0,1} range
if(n_block==1)
    unary = [zeros(1,length(v));(diag(Q)'-row_sum-v)/2];%convert to {0,1} GC problem, GCMex only use the bottom triangle of Q, so we divide 2
    [h,energy,energy_after] =  GCMex(h_init, single(unary), Q, single(labelcost), 0);%single-threaded
else
    [h,energy,energy_after] = GCMexPar(h_init, -(row_sum+v), Q, single(labelcost), 0, n_block, 1);%multi-threaded
end
h = 2*h'-1;

end
