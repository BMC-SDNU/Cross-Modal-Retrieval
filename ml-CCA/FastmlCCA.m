function [Xcca Ycca A,B,test] = FastmlCCA(X, Y, numeigs,kapa,Xte,Yte,cattr,labelsimilaritysigma,few)
%%--------------------------------------------------------------------------
% Implements Fast ml-CCA, written by Viresh Ranjan
% X is training data in first modality,size n * dx, where n is number of training samples, 
% Y is training data in first modality,size n * dy, where n is number of training samples, 

%numeigs = dimensionality of common subspace to be used for projection
% kapa = regularization constant
% Xte= test data in first modality, size ntest * dx, where ntest is number of test samples, 
% Yte = test data in second modality, size ntest * dy, where ntest is number of test samples,
%labelsimilaritysigma = scaling parameter
% few = number of pairs to be introduced
% cattr = multi-label vectors corresponding to X & Y, size n * dz, where n is number of training samples,
%test.Xcca, test.Ycca = projected test samples
% Xcca, Ycca = projected training samples
% A, B = projection vectors for first and second modality respectively
% 
if ~exist('kapa')
    kapa = 0.25;
end
[nx, dx] = size(X);
[ny, dy] = size(Y);
% Compute overall mean
labelsimilarity = 'l2exp';

mX = mean(X,1);
mY = mean(Y,1);
X = X - repmat(mX, [nx 1]);
Y = Y - repmat(mY, [ny 1]);
[IDX,D] = knnsearch(cattr,cattr,'NSMethod','kdtree','k',few,'Distance','euclidean');


%------------------------compute covariance matrices------------------------------
%---------------------------------------------------------------------------------
cxx = zeros(size(X,2),size(X,2));
cyy = zeros(size(Y,2),size(Y,2));
cxy = zeros(size(X,2),size(Y,2));

for i = 1:size(X,1)
	
	for j = 1:few
		if(strcmp(labelsimilarity,'l2exp'))
			dtemp = exp(-1*(pdist2(cattr(i,:),cattr(IDX(i,j),:),'euclidean'))/labelsimilaritysigma);
		end
		if(isnan(dtemp) | isinf(dtemp))
			dtemp = 10^-10;
		end
		cxy = cxy + (dtemp*X(i,:)'*Y(IDX(i,j),:));
		cxx = cxx + (dtemp*X(i,:)'*X(i,:));
		cyy = cyy + (dtemp*Y(IDX(i,j),:)'*Y(IDX(i,j),:));
	end
end
cxx = cxx/(size(X,1)*few);
cxy = cxy/(size(X,1)*few);
cyy = cyy/(size(X,1)*few);
%----------------------------------------------------------------------------------


ikx = (cxx + kapa * mean(diag(cxx)) * eye(dx)) \ cxy;
iky = (cyy + kapa * mean(diag(cyy)) * eye(dy)) \ cxy';
sstate = rng;
%opts.tol = 1e-3;
[A , ex, flag] = eigs(ikx * iky, [], numeigs, 'lr');
rng(sstate);
[s sid] = sort(diag(ex), 'descend');
ex = ex(sid, sid);
A = A(:, sid);
B = (iky * A) ./ (ones(dy, numeigs) * ex);
B = B ./ repmat(sqrt(diag(B'*B)'), size(B, 1), 1);
 Xcca = X*A;
 Ycca = Y*B;
if nargin>4,
	assert(size(Xte,1) == size(Yte,1),'X_test and Y_test have different number of samples');
	n=size(Xte,1);

	scaled_Xte = ((Xte - repmat(mX,n,1)));
	scaled_Xte(isnan(scaled_Xte)) = 0;
	test.Xcca = scaled_Xte * A;

	scaled_Yte = ((Yte - repmat(mY,n,1)));
	scaled_Yte(isnan(scaled_Yte)) = 0;
	test.Ycca = scaled_Yte * B;
    else
	test=0;
    end;