function [ result ] = run_methods_LGCNH( YAll, LAll, IAll, param, dataname, bits)
%% parameter setting
param.maxIter = 100;
param.lr_txt = logspace(-1.5,-3,param.maxIter);
param.lr_img = logspace(-1.5,-3,param.maxIter);
param.gamma = 10;
param.eta = 1;
param.batch_size = 128; 
param.dataname = dataname;
if strcmp(param.dataname,'IAPR-TC12')
    param.lr_img_gcn = logspace(-1.5,-2,3); 
    param.lr_txt_gcn = logspace(-1.5,-2,3);           
else
    param.lr_img_gcn = logspace(-2.1,-2.1,3); 
    param.lr_txt_gcn = logspace(-2.1,-2.1,3); 
end

dataset.L01 = LAll;                     clear LAll
dataset.Y = YAll;                       clear YAll
dataset.X = IAll;                       clear IAll

nb = numel(bits);


%% training and evaluating LGCNH
evaluation = cell(1, nb);
for i = 1: nb
    param.bit = bits(i);
    fprintf('...................................\n');
    fprintf('...dataset: %s\n', dataname);    
    fprintf('...bit: %d\n', param.bit);   
    param.method = 'LGCNH'; result = process_MGCN_CMH_f2(dataset, param);
    fprintf('...method: %s\n', param.method);
    evaluation{i} = result;
    evaluation{i}.bit = param.bit;
    fprintf('...bit: %d\n ...i2tMAP: %f ...t2iMAP: %f', param.bit, result.hri2t.map, result.hrt2i.map);
    fprintf('...bit: %d\n ...i2tMAP: %f ...t2iMAP: %f', param.bit, result.hri2t_top500.topkMap, result.hrt2i_top500.topkMap);
end
save(['result\' param.method '_', dataname , '_' num2str(param.bit) '.mat'],'evaluation','param');


end

