function [ result ] = run_methods_DCMH( YAll, LAll, IAll, param, dataname, bits)
%% parameter setting
param.maxIter = 500;
param.lr_txt = logspace(-1.5,-3,param.maxIter);
param.lr_img = logspace(-1.5,-3,param.maxIter);
param.gamma = 1;
param.eta = 1;
param.batch_size = 128;
param.dataname = dataname;

dataset.L01 = LAll;
dataset.Y = normZeroMean(YAll);
dataset.X = IAll;

nb = numel(bits);
%% training and evaluating DCMH
evaluation = cell(1, nb);
for i = 1: nb
    param.bit = bits(i);
    fprintf('...................................\n');
    fprintf('...dataset: %s\n', dataname);    
    fprintf('...bit: %d\n', param.bit);   
    param.method = 'DCMH'; result = process_DCMH_fix(dataset, param);
    fprintf('...method: %s\n', param.method);
    evaluation{i} = result;
    evaluation{i}.bit = param.bit;
    fprintf('...bit: %d\n ...i2tMAP: %f ...t2iMAP: %f', param.bit, result.hri2t.map, result.hrt2i.map);
    fprintf('...bit: %d\n ...i2tMAP: %f ...t2iMAP: %f', param.bit, result.hri2t_top500.topkMap, result.hrt2i_top500.topkMap);
end
save(['result\' param.method '_', dataname , '_' num2str(param.bit) '.mat'],'evaluation','param');

end

