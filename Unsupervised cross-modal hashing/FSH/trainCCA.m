function [ Wx, Wy, eigv ] = trainCCA(Cxx, Cyy, Cxy, nDim)
% [Wx, Wy, eigv] = trainCCA(Cxx, Cyy, Cxy, nDim)
% Computes the Canonical Correlation Analysis projections on two data
% matrices X and Y.
%
% Input:
% Cxx - covariance matrix of X-X
% Cyy - covariance matrix of Y-Y
% Cxy - covariance matrix of X-Y
% nDim - number of CCA dimensions
%
% Output:
% Wx - CCA projection matrix for modality X
% Wy - CCA projection matrix for modality Y
% eigv - corresponding eigenvalues for returned projections

option = struct('disp', 0);
A = Cxy/Cyy*Cxy'; B = Cxx;

% Use eig() instead of eigs() for stable performance

% [eigvectors, eigv] = eigs(A, B, nDim, 'lr', option);
% eigv = real(eigv); eigvectors = real(eigvectors);

% if (~exist('eigv', 'var') || sum(diag(eigv)==0)~=0 || min(diag(eigv))<=0)
%    disp('Use eig() instead');
    [eigvectors, eigv] = eig(A, B);
    [SD, SI] = sort(diag(eigv), 'descend');
    eigv = diag(SD(1:nDim));
    eigvectors = eigvectors(:, SI(1:nDim));
% end

Wx = eigvectors;
Wy = Cyy\Cxy'*eigvectors/sqrt(eigv);
