function K = RBF_fast(Xq,Xr)
% RBF_fast: generate the RBF kernel
%
%
D = EuDist2(Xq',Xr',1);
sigma = mean(D(:));
K = exp(-D/(2*sigma^2));