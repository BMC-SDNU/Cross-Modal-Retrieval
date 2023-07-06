function ASCSH_demo()
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
bits = [8];
nb = numel(bits);

param.bits = bits;
param.maxIter = 25;
param.gamma = 1e-6;
param.lambda = 8; %At 64 bits, it is suggested that this parameter be set to 10-12

%flickr
param.lambda_c = 1e-3;
param.alpha1 = 5e-2;
param.alpha2 = param.alpha1;
param.mu = 1e-4;
param.eta = 0.005;

%nus
% param.lambda_c = 1e-3;
% param.alpha1 = 5e-3;
% param.alpha2 = param.alpha1;
% param.mu = 1e-4;
% param.eta = 1e-2;
% param.sc = 5000;

%iapr
% param.lambda_c = 1e-5;
% param.alpha1 = 5e-3;
% param.alpha2 = param.alpha1;
% param.mu = 1e-6;
% param.eta = 1e-4;
%% load dataset
dataset = load_data(dataname);
n_anchors = 1500;
rbf2;

%% run algorithm
for i = 1: nb
    fprintf('...method: %s\n', param.method);
    fprintf('...bit: %d\n', bits(i)); 
    param.bit = bits(i);
    param.num_samples = 2 * param.bit; %At 64 bits, it is suggested that this parameter be set to £¨2.5 * param.bit£©
    trainL = dataset.databaseL;
    ASCSH(trainL, param, dataset);
end
end

