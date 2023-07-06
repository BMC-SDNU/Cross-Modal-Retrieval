function w = traceProject(w,nRows,tau)
p = length(w);
W = reshape(w,nRows,p/nRows);
[U,S,V] = svd(W);
s = projectRandom2C(diag(S),tau);
W = U*setdiag(S,s)*V';
w = W(:);
