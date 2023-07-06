function W = ITQCCAtrain1(X, Y, nbit)
%X: each row of X is a zero-centered sample
%Y: label matrix
%nbit: number of hashing bit

%CCA, supervised
[W,r] = cca1(X, Y, 1e-4); % this computes CCA projections
W = W(:,1:nbit)*diag(r(1:nbit)); % this performs a scaling using eigenvalues
X = X*W; % final projection to obtain embedding E

%ITQ
[~, R] = ITQ1(X, 50);%compute ratation matrix

%total transformation matrix
W = real(W*R);

end
