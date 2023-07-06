function DLFH_demo()
addpath(genpath(fullfile('utils/')));
seed = 0;
rng('default');
rng(seed);
param.seed = seed;
dataname = 'flickr-25k';
%% parameters setting
% basie information
param.dataname = dataname;
param.method = 'DLFH';

% method parameters
bits = [8, 16, 32, 64];
nb = numel(bits);

param.bits = bits;
param.maxIter = 30;
param.gamma = 1e-6;
param.lambda = 8;

%% load dataset
dataset = load_data(dataname);

%% run algorithm
for i = 1: nb
    fprintf('...method: %s\n', param.method);
    fprintf('...bit: %d\n', bits(i));
    
    param.bit = bits(i);
    param.num_samples = param.bit;
    result = DLFH_algo(dataset, param);
    disp(['...mAP(i->t): ' num2str(result.hri2t.map, '%.4f') ', mAP(t->i): ' num2str(result.hrt2i.map, '%.4f')]);
    save(['log/DLFH-' dataname '-' num2str(param.bit) 'bits-' num2str(seed) '.mat'], 'result', 'param');
end
end

