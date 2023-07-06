function [B, time] = RoPH_test(X, model, type, view)
%encode the input data to (compact) binary representation
%input:
% X: data, each column is a sample
% model: the trained model
% type: 'binary' for 0-1 binary encoding or 'uint8' for compact
%       binary encoding
% view: data view, 1 or 2
%
%output:
% B: the coding matrix, each column corresponds a sample
% time: time to encode the data

% This code is written by Kun Ding (kding@nlpr.ia.ac.cn).

tic;

assert(view==1||view==2);
view = num2str(view);

%kernel feature mapping
if model.use_kernel
    D2 = Euclid2(getfield(model,strcat('A',view)), X, 'col', 0);
    X = exp(-D2/2/getfield(model,strcat('squared_sigma',view)));
end

%remove mean value
mean_vec = getfield(model, strcat('mean_vec',view));
X = bsxfun(@minus, X, mean_vec);

%coding
W = getfield(model, strcat('W',view));
if(strcmp(type,'binary'))
    B = W'*X>0;
elseif(strcmp(type,'uint8'))
    B = compactbit(W'*X>0);
else
    error('Bad Type!');
end

time = toc;

end
