function [Xcca,Ycca, A, B, test] = cca3(X,Y,Xte,Yte)
%CCA3 computes canonical correlation matrices
%   Returns the rotation matrices A and B for modalities
%   X and Y respectively. As well as the CCA version of 
%   X and Y, Xcca and Ycca respectively.
%
%   [Xcca,Ycca, A, B, test] = cca3(X,Y,Xte,Yte)
%
%   If test samples of X and Y are provided, it also 
%   returns their cca version in the struct test:
%   	test.Xcca
%   	test.Ycca
%
    
    vX = sqrt(var(X,1));
    vY = sqrt(var(Y,1));
    mX = mean(X,1);
    mY = mean(Y,1);
    
    X = (X - repmat(mX,size(X,1),1))./repmat(vX,size(X,1),1);
    Y = (Y - repmat(mY,size(Y,1),1))./repmat(vY,size(Y,1),1);
    
    X(find(isnan(X))) = 0;
    Y(find(isnan(Y))) = 0;
    
    [A,B] = canoncorr(X,Y);
    
    Xcca = X*A;
    Ycca = Y*B;

    if nargin>2,
	assert(size(Xte,1) == size(Yte,1),'X_test and Y_test have different number of samples');
	n=size(Xte,1);

	scaled_Xte = ((Xte - repmat(mX,n,1)))./repmat(vX,n,1);
	scaled_Xte(isnan(scaled_Xte)) = 0;
	test.Xcca = scaled_Xte * A;

	scaled_Yte = ((Yte - repmat(mY,n,1)))./repmat(vY,n,1);
	scaled_Yte(isnan(scaled_Yte)) = 0;
	test.Ycca = scaled_Yte * B;
    else
	test=0;
    end;

