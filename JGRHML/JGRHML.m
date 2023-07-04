function [mapIA, mapIT, mapTA, mapTI] = JGRHML(I_tr, T_tr, I_te, T_te, trainCat, testCat, alpha, beta, k)

% *************************************************************************
% *************************************************************************
% Parameters:
% I_tr: the feature matrix of image instances for training
%              dimension : tr_n * d_i
% T_tr: the feature matrix of text instances for training
%              dimension : tr_n * d_t
% I_te: the feature matrix of image instances for test
%              dimension : te_n * d_i
% T_te: the feature matrix of text instances for test
%              dimension : te_n * d_t
% trainCat: the category list of data for training
%              dimension : tr_n * 1
% testCat: the category list of data for test
%              dimension : te_n * 1
% alpha: parameter, default: 0.1
% beta: parameter, default: 1
% k: kNN parameter, default: 90
% *************************************************************************
% *************************************************************************

tr_n = size(I_tr, 1);
te_n = size(I_te, 1);
catNum = max(trainCat);
alphaI = alpha;
betaI = beta;
knnI = k;

[I_te_red, T_te_red, I_tr_red, T_tr_red, L_cat] = HMLInit(I_tr, I_te, T_tr, T_te, trainCat);

[I_tr_red I_te_red] = znorm(I_tr_red,I_te_red);
[T_tr_red T_te_red] = znorm(T_tr_red,T_te_red);

D = pdist([I_tr_red; I_te_red; T_tr_red; T_te_red],'euclidean');
Z = squareform(D);
Z = -Z;
Z = 1./(1+exp(-Z));
WI_red = Z;

L_cat = L_cat(1:tr_n, 1:catNum);
L_cat_n = 1 - L_cat;
L_cat = L_cat./repmat(sum(L_cat), size(L_cat,1), 1);
L_cat_n = L_cat_n./repmat(sum(L_cat_n), size(L_cat_n,1), 1);
L_cat = L_cat - L_cat_n;
L_cat = [L_cat; zeros(te_n, catNum)];
L_cat = [L_cat;L_cat];

I_high = ssl_knn(WI_red, L_cat, knnI, alphaI, betaI);
I_te_red = I_high(tr_n+1:tr_n+te_n,:);
T_te_red = I_high(2*tr_n+te_n+1:2*tr_n+2*te_n,:);

Z =  I_te_red * T_te_red';
W = Z;

W_II = I_te_red * I_te_red';
W_TT = T_te_red * T_te_red';
W_IT = W;
W = [W_II,W_IT;...
    W_IT',W_TT];

WI = W(1:te_n,:);
WT = W(te_n+1:te_n+te_n,:);

mapIT = evaluateMAPPR( W_IT, testCat, testCat);
mapTI = evaluateMAPPR( W_IT', testCat, testCat);
mapIA = evaluateMAPPR(WI, testCat, [testCat; testCat]);
mapTA = evaluateMAPPR(WT, testCat, [testCat; testCat]);