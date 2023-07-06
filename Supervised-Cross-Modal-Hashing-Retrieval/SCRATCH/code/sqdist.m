function d=sqdist(a,b)
% SQDIST - computes squared Euclidean distance matrix
%          computes a rectangular matrix of pairwise distances
% between points in A (given in columns) and points in B

% NB: very fast implementation taken from Roland Bunschoten

aa = sum(a.^2,1); bb = sum(b.^2,1); ab = a'*b; 
d = (repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

