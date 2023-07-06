clc;
close all;
clear all;
addpath(genpath('./'));

%load data
load('./data/wiki.mat');%a mini dataset
split = data.split;

%% RoPH
opts = [];
opts.sample_size = 500;%change this for different datasets
opts.K = 32;
opts.max_iter = 5;
opts.lambda = 1;
opts.use_kernel = 1;%use kernel
opts.sigma1_scl = 0.8;
opts.sigma2_scl = 0.8;
opts.n_block = 10;%change this for different datasets
data = get_data(data, split);
[triplets, sims] = gen_triplets_ml(data.Y_train, 100);
model = RoPH_train(data.X1_train, data.X2_train, triplets(:,2:3), sims, opts);

B1_train = compactbit(model.B>0);
B1_test = RoPH_test(data.X1_test, model, 'uint8', 1);
B2_train = B1_train;
B2_test = RoPH_test(data.X2_test, model, 'uint8', 2);

%text as query
Dhat = hammDist_mex(B1_train, B2_test);
[~,IX] = sort(Dhat,1,'ascend');
NDCG_tx = NDCG_k(data.Y_train, data.Y_test, IX, 100);

%image as query
Dhat = hammDist_mex(B2_train, B1_test);
[~,IX] = sort(Dhat,1,'ascend');
NDCG_im = NDCG_k(data.Y_train, data.Y_test, IX, 100);

NDCG_tx, NDCG_im

%% SePH
opts = [];
opts.K = 32;
opts.alpha = 0.01;
opts.use_kernel = 1;
opts.sample_size = 500;
opts.sigma1_scl = 0.5;%0.5 is better than 0.1
opts.sigma2_scl = 0.5;
opts.maxiter = 100;
data = get_data(data, split);
model = SePH_train(data.X1_train, data.X2_train, data.Y_train, opts);

B1_train = SePH_test_fuse(data.X1_train, data.X2_train, model, 'uint8');%view 1
B1_test = SePH_test(data.X1_test, model, 'uint8', 1);
B2_train = SePH_test_fuse(data.X1_train, data.X2_train, model, 'uint8');%view 2
B2_test = SePH_test(data.X2_test, model, 'uint8', 2);

%text as query
Dhat = hammDist_mex(B1_train, B2_test);
[~,IX] = sort(Dhat,1,'ascend');
NDCG_tx = NDCG_k(data.Y_train, data.Y_test, IX, 100);

%image as query
Dhat = hammDist_mex(B2_train, B1_test);
[~,IX] = sort(Dhat,1,'ascend');
NDCG_im = NDCG_k(data.Y_train, data.Y_test, IX, 100);

NDCG_tx, NDCG_im
