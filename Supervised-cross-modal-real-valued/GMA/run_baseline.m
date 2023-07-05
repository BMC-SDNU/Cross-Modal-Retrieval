clear;
addpath('code/');

dataset='wikipedia';
%dataset='flickr30k';
%dataset='twitter100k';

text=importdata(['../../../feature/' dataset '/lda.mat']);
%text=importdata(['../../../feature/' dataset '/text_word2vec_bow.mat']); 
%text_word2vec_bow.mat is WE-BoW text feature 
text=double(text);
disp('text size:')
size(text)
train_num=2173;
%for wikipedia, train_num=2173;
%for flickr30k, train_num=75000;
%for twitter100k, train_num=50000;
train_txt=text(1:train_num,:);
text=[];

train_im=importdata(['../../../feature/' dataset '/train_image.mat']);
train_im=double(train_im);

train_lab=(1:1:train_num).';

disp('image size:')
size(train_im)
disp('text size:')
size(train_txt)
disp('label size:')
size(train_lab)

path=['../../../result/Wout/' dataset '/lda/'];
%path=['../../../result/Wout/' dataset '/WE/'];
if exist(path)==0
    mkdir(path);
end

%dataCell
dataCell=cell(2,1);
%text
dataCell{1,1}.label=train_lab;
dataCell{1,1}.data=train_txt;
%image
dataCell{2,1}.label=train_lab;
dataCell{2,1}.data=train_im;
baselines={'pls','cca','blm','mfa'};
for index=1:length(baselines)
    baseline=baselines{index};
    options.method=baseline;
    disp(baseline)
    tic;
    switch lower(baseline)
        case 'pls'
            options.Factor=15;
            options.Lamda = 10;
            options.ReguAlpha =0.01;
        case 'cca'
            options.Factor=15;
            options.Lamda = 10;
            options.ReguAlpha =0.01;
        case 'blm'
            options.Factor=15;
            options.Lamda = 500;
        case 'lpp'
            options.Factor=15;
            options.Lamda =500;
            options.Mult = [1 1];
            options.ReguAlpha =0.01;
        case 'mfa'
            options.Factor=15;
            options.Lamda = 10;
            options.Mult = [1 1];
            options.meanMinus = 1;
            options.Dev = 0;
            options.ReguAlpha = 1e-6;
    end
    Wout = Newgma(dataCell,options);
    toc;
    savepath=[path baseline '/'];
    if exist(savepath)==0
        mkdir(savepath);
    end
    disp('save Wout!')
    W_T=Wout{1}.Bases;
    W_I=Wout{2}.Bases;
    save([savepath 'Wout.mat'],'W_T','W_I');
end 
