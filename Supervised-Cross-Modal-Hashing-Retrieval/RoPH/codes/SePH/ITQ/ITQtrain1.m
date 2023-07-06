function W = ITQtrain1(X, nbit)
%X: each row of X is a zero-centered sample
%nbit: number of hashing bit

% PCA, unsupervised
[W, ~] = eigs(cov(X),nbit);
X = X * W;

%ITQ
[~, R] = ITQ1(X, 200);%compute ratation matrix

%total transformation matrix
W = real(W*R);

end
