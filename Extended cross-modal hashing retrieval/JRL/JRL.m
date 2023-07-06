function [mapIA, mapIT, mapTA, mapTI] = JRL(I_tr, T_tr, I_te, T_te, trainCat, testCat, gamma, sigma, lamda, miu, k)
 
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
% gamma: sparse regularization parameter, default: 1000
% sigma: mapping regularization parameter, default: 1000
% lambda: graph regularization parameter, default: 1
% miu: high level regularization parameter, default: 1
% k: kNN parameter, default: 100
% *************************************************************************
% *************************************************************************

img_Dim = size(I_tr, 2);
txt_Dim = size(T_tr, 2);
cat_Num = max(trainCat);

[I_tr_n, I_te_n] = znorm(I_tr,I_te);
[T_tr_n, T_te_n] = znorm(T_tr,T_te);

tr_n = size(I_tr,1);
te_n = size(I_te,1);
L_cat = zeros(tr_n + te_n, cat_Num);
for i = 1:tr_n
    L_cat(i,trainCat(i)) = 1;
end
L_cat_img = L_cat;
L_cat_txt = L_cat;
trImgCat = trainCat;
teImgCat = testCat;
trTxtCat = trainCat;
teTxtCat = testCat;

tr_n_I = tr_n;
te_n_I = te_n;
tr_n_T = tr_n;
te_n_T = te_n;

pair = zeros(tr_n,2);
pair(:,1) = 1:tr_n;
pair(:,2) = 1:tr_n;

param.iteration = 5;
param.gamma = gamma;
param.sigma = sigma;
param.lamda = lamda;
param.miu = miu;

[I_tr_red,I_te_red,T_tr_red,T_te_red] = learn_projection(I_tr, I_te, T_tr, T_te, pair, img_Dim, txt_Dim, tr_n_I, tr_n_T, L_cat_img, L_cat_txt, cat_Num, param, k);

W_II = unifyKnnKernel(I_tr_red, I_te_red, I_tr_red, I_te_red, tr_n_I, te_n_I, tr_n_I, te_n_I, trImgCat, trImgCat, k);
W_IT = unifyKnnKernel(I_tr_red, I_te_red, T_tr_red, T_te_red, tr_n_I, te_n_I, tr_n_T, te_n_T, trImgCat, trTxtCat, k);
W_TT = unifyKnnKernel(T_tr_red, T_te_red, T_tr_red, T_te_red, tr_n_T, te_n_T, tr_n_T, te_n_T, trTxtCat, trTxtCat, k);

W_II = W_II(1:te_n_I, 1:te_n_I);
W_TT = W_TT(1:te_n_T, 1:te_n_T);
W_IT = W_IT(1:te_n_I, te_n_I+1:te_n_I+te_n_T);
W = [W_II,W_IT;...
    W_IT',W_TT];

for i = 1:length(W)
    W(i,i) = 9999;
end

WI = W(1:te_n_I,:);
WT = W(te_n_I+1:te_n_I+te_n_T,:);
disp(['image query all... ']);
[ mapIA, ~, ~] = evaluateMAPPR(WI, teImgCat, [teImgCat;teTxtCat]);
disp(['text query all... ']);
[ mapTA, ~, ~] = evaluateMAPPR(WT, teTxtCat, [teImgCat;teTxtCat]);
disp(['image query text... ']);
[ mapIT, ~, ~] = evaluateMAPPR(W_IT, teImgCat, teImgCat);
disp(['text query image... ']);
[ mapTI, ~, ~] = evaluateMAPPR(W_IT', teTxtCat, teTxtCat);