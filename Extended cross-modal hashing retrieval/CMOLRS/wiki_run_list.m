% clc;
% clear;

% set parameters
params = struct();
% params.pos = 1; % corresponding
params.pos = 2; % the same class

params.list_num = 2e5; % number of training list (a list contains many triplets with the same query)
params.list_size = 4; % list size

params.dir = 1; % image to text
% params.dir = 2; % text to image

params.rank = 32;
params.base_margin = 20; % beta in Eqn(4)
params.alph = 0.5; % alpha in Eqn(4)
params.step_size = 10;

%% load data
load('data/wiki/data_cnnfc7_zj_zmnor.mat');
label_tr = scale2vec(label_tr); % change the representation of training labels
pca = 1;
if pca == 1
    options_=[];
    options_.PCARatio = 0.95;
    [eigvector,eigvalue] = myPCA(img_tr,options_);
    img_tr = img_tr*eigvector;
    img_te = img_te*eigvector;
    [eigvector,eigvalue] = myPCA(txt_tr,options_);
    txt_tr = txt_tr*eigvector;
    txt_te = txt_te*eigvector;
end

data_train = gen_tr_list_mc(label_tr, params);

[N_te, di] = size(img_te);
[N_tr, dt] = size(txt_tr);

%%
A = randn(di,params.rank);
B = randn(dt,params.rank);
batches = 10;
params.batch_size = params.list_num / batches;
time_used = zeros(1,batches);
map_i2t = zeros(1,batches);
map_t2i = zeros(1,batches);
begin_time = clock;
fout = fopen('record_wiki.txt', 'a');
fprintf(fout, '--------------------------------------------------\n');
for i = 1 : batches
    [model,parms] = loreta_similarity_list_triplet(A, B, img_tr',txt_tr', label_tr', (i-1)*params.batch_size, data_train, params);
    A = model.A;
    B = model.B;
    smatrix = img_te * A * B' * txt_te';
    fprintf('[%s] img search txt:\n', datestr(now,31));
    mapk = test_s_map(smatrix, label_te, label_te, fout);
    map_i2t(i) = mapk(end);
    fprintf('[%s] txt search img:\n', datestr(now,31));
    mapk = test_s_map(smatrix', label_te, label_te, fout);
    map_t2i(i) = mapk(end);
    time_used(i) = etime(clock,begin_time);
    fprintf('iteration %d out of %d, time elapsed %g sec\n',i,...
        batches, etime(clock,begin_time));
end
fclose(fout);
