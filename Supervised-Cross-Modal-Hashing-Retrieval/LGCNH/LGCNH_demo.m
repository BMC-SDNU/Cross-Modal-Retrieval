%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% IAPR
clear,clc;
addpath(fullfile('./utils/'));
addpath(fullfile('./netstructure/'));
addpath(fullfile('./methods/'));

%% load data
dataname = 'IAPR-TC12';
load ./data/IAPR-TC12/YAll.mat  
load ./data/IAPR-TC12/IAll.mat
load ./data/IAPR-TC12/LAll.mat
load ./data/IAPR-TC12/param.mat

bits = [16:16:64];
for i = 1:numel(bits)
    bit = bits(i);
    [ resultLGCNH ] = run_methods_LGCNH( YAll, LAll, IAll, param, dataname, bit );
    [ resultDCMH ] = run_methods_DCMH( YAll, LAll, IAll, param, dataname, bit );
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NUS-WIDE
clear,clc;
addpath(fullfile('./utils/'));
addpath(fullfile('./netstructure/'));
addpath(fullfile('./methods/'));

%% load data
dataname = 'NUS-WIDE';
load ./data/NUS-WIDE-TC21/YAll.mat 
% load ./data/NUS-WIDE-TC21/IAll.mat
load ./data/NUS-WIDE-TC21/nus-wide-tc21-iall.mat
load ./data/NUS-WIDE-TC21/LAll.mat
load ./data/NUS-WIDE-TC21/param.mat
findZeroTag = find(sum(YAll,2)==0); 
YAll(findZeroTag,:) = [];
IAll(:,:,:,findZeroTag) = [];
LAll(findZeroTag,:) = [];

bits = [16:16:64];
for i = 1:numel(bits)
    bit = bits(i);
    [ resultLGCNH ] = run_methods_LGCNH( YAll, LAll, IAll, param, dataname, bit );
    [ resultDCMH ] = run_methods_DCMH( YAll, LAll, IAll, param, dataname, bit );
end

