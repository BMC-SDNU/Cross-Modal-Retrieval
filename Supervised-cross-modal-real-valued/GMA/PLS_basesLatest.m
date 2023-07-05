%% FILE INFORMATION
% Version 1 : Makes it 3 times faster than the previous version and adding
% mean centering and normalization option
% Date - 04/11/2012
% Author - Abhishek Sharma
% E-mail - bhokaal@cs.umd.edu
% Copyright Information -- Feel free to use the way u want

%% USES OF LEARNED MATRICES W,P,Z AND B
% So, if you follow the convention as given in the code i.e.
% matrix X (Dx by N) and Y(Dy by N) should contain one subject's
% image in a column then you should do the following to get the
% projection once you have obtained W (Dx by Nfactors) and
% Z(Dy by Nfactor).

% COFx = W'*X;
% COFy = Z\Y;

% Here, COFx would be (Nfactor by N) and
% COFy will be (Nfactor by N) matrix
% and now each column in COFx and COFy is like a latent
% representation of the
% images in Intermediate subspace which can be compared.

% Dx = dimension of image in one pose
% Dy = dimension of image in other pose
% N = Number of subjects used for training.
% Nfactor = PLS factors used.

%% THE MAIN FUNCTION PART

function [W,Z,M,V] = PLS_basesLatest(X,Y,nfactor,centerFlag,normalizeFlag)

% INPUT PARAMETERS
% X and Y both are supplied in a form where each column contains one sample
% nfactor - # desired PLS fatcors
    % OPTIONAL INPUT PARAMS
    % centerFlag = 1 -> make the input samples mean centered else
    %            = 0 -> dont modify (default)
    % normalizeFlag = 1 -> make the features have unit variance
    %               = 0 -> dont modify (default)


% OUTPUT PARAMETERS
% W - the projection directions for X
% Z - Projection directions for Y
    % OPTIONAL OUTPUT PARAMS
    % M - 2 by 1 cell with M{1,1} =  mean of X; M{2,1} = mean of Y (meaningful only when centerFlag  1)
    % V - 2 by 1 cell with V{1,1} = variance of X features; V{2,1} = variance of Y features

XD = size(X,1); % X-dimension
YD = size(Y,1); % Y- dimension

% Input check
if nargin < 3
    print ('Not enough input arguments probably missing nfactor')
    return;
end

% Initialization of default params
if nargin < 5
    normalizeFlag = 0;
    if nargin < 4
        centerFlag = 0;
    end
end

% Number of samples
nx = size(X,2);
ny = size(Y,2);
if nx == ny
    n = nx;
else
    print ('The number of samples in X and Y are different');
    return;
end

% Centralize the data
% IMPORTANT -- IT WAS FOUND THAT SOMETIMES CENTRALIZING THE DATA HELPS AND
% SOMETIMES NOT SO CHOOSE YOUR OPTION BY UNCOMMENTING THE BELOW LINES

if ~centerFlag
    if nargout > 2
        M = cell(2,1);
        M{1,1} = mean(X,2);
        M{2,1} = mean(Y,2);
        X = bsxfun(@minus,X,M{1,1});
        Y = bsxfun(@minus,Y,M{2,1});
    else
        X = bsxfun(@minus,X,mean(X,2));
        Y = bsxfun(@minus,Y,mean(Y,2));
    end
end

% NORMALIZE THE DATA
% IMPORTANT -- IT WAS FOUND THAT SOMETIMES NORMALIZING THE DATA HELPS AND
% SOMETIMES NOT SO CHOOSE YOUR OPTION.

if ~normalizeFlag
    if nargout > 3
        V = cell(2,1);
        V{1,1} = std(X,1,2);
        V{2,1} = std(Y,1,2);
        tmp = (V{1,1} == 0);
        V{1,1}(tmp) = 1;
        tmp = (V{2,1} == 0);
        V{2,1}(tmp) = 1;
        V{1,1} = V{1,1}.^(0.5);
        V{2,1} = V{2,1}.^(0.5);
        X = bsxfun(@rdivide,X,V{1,1});
        Y = bsxfun(@rdivide,Y,V{2,1});
    else
        X = bsxfun(@rdivide,X,std(X,1,2).^(0.5));
        Y = bsxfun(@rdivide,X,std(Y,1,2).^(0.5));
    end
end

% make them as row vectors now


X = X';
Y = Y';

% Initialisation of some matrices

W = zeros(XD,nfactor);
A = X'*Y;
M_ = X'*X;
C = eye(XD);
P = zeros(XD,nfactor);
Z = zeros(YD,nfactor);

for i = 1:nfactor

    [dumm d q] = svds(A,1);
    w = C*(A*q);
    w = w/norm(w);
    W(:,i) = w;
    p = M_*w;
    c = w'*p;
    p = p/c;
    P(:,i) = p;
    q = A'*(w/c);
    Z(:,i) = q;
    A = A - (c*p)*q';
    M_ = M_ - (c*p)*p';
    C = C - w*p';
end
