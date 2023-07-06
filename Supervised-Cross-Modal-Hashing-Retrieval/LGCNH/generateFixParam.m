% fix all the random parameters for a fair comparison
net = load('data/imagenet-vgg-f.mat');
fileName = 'fixRandomParam';
addpath(fullfile('./netstructure/'));

for bit = [16:16:64]
       
    dataname = 'IAPR-TC12';
    dy = 2912;    
    txt_net = net_structure_txt(dy, bit);
    save([ fileName '\' dataname '_txt_net_', num2str(bit) '.mat'],'txt_net');
    clear dataname dy txt_net
    
    dataname = 'NUS-WIDE';
    dy = 1000;    
    txt_net = net_structure_txt(dy, bit);
    save([ fileName '\' dataname '_txt_net_', num2str(bit) '.mat'],'txt_net');
    clear dataname dy txt_net
    
    img_net = net_structure_img(net, bit);
    save([ fileName '\'  'img_net_', num2str(bit) '.mat'],'img_net');
    clear img_net
    
    gcn_txt_net = gcn_net(bit, bit);
    save([ fileName '\'  'gcn_txt_net_', num2str(bit) '.mat'],'gcn_txt_net');
    clear gcn_txt_net
    
    gcn_img_net = gcn_net(bit, bit);
    save([ fileName '\'  'gcn_img_net_', num2str(bit) '.mat'],'gcn_img_net');
    clear gcn_img_net
    
    dataname = 'NUS-WIDE';
    num_train = 10500;
    for rr = 1:100
        randvector(rr,:) = randperm(num_train);
    end
    save([ fileName '\'  'randVectorImg_' dataname '_' num2str(bit) '.mat'],'randvector');
    clear  num_train rr randvector dataname
    
    dataname = 'NUS-WIDE';
    num_train = 10500;
    for rr = 1:100
        randvector(rr,:) = randperm(num_train);
    end
    save([ fileName '\'  'randVectorTxt_' dataname '_' num2str(bit) '.mat'],'randvector');
    clear  num_train rr randvector dataname
    
    dataname = 'IAPR-TC12';
    num_train = 10000;
    for rr = 1:100
        randvector(rr,:) = randperm(num_train);
    end
    save([ fileName '\'  'randVectorImg_' dataname '_' num2str(bit) '.mat'],'randvector');
    clear  num_train rr randvector dataname
    
    dataname = 'IAPR-TC12';
    num_train = 10000;
    for rr = 1:100
        randvector(rr,:) = randperm(num_train);
    end
    save([ fileName '\'  'randVectorTxt_' dataname '_' num2str(bit) '.mat'],'randvector');
    clear  num_train rr randvector dataname

     
end