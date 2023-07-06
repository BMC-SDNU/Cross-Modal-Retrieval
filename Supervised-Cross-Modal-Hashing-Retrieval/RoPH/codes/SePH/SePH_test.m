function [B, time] = SePH_test(X, model, type, view)
%encode the test data to (compact) binary representation
%
%input:
% X: data, each column is a sample
% model: the trained model
% type: set 'binary' for 0-1 binary encoding or set 'uint8' for compact
%       binary encoding
% view: which data view, 1 for image view and 2 for text view
%
%output:
% B: the encoding, each column corresponds a sample

tic;

assert(view==1||view==2);
view = num2str(view);

if model.use_kernel
    D2 = Euclid2(getfield(model,strcat('A',view)), X, 'col', 0);
    X = exp(-D2/2/getfield(model,strcat('squared_sigma',view)));
end

mean_vec = getfield(model, strcat('mean_vec',view));
X = bsxfun(@minus, X, mean_vec);

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
