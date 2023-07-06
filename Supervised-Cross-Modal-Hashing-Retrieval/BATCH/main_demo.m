close all; clear; clc;
addpath(genpath('./utils/'));
addpath(genpath('./codes/'));

result_URL = './results/';
if ~isdir(result_URL)
    mkdir(result_URL);
end

db = {'MIRFLICKR0','NUSWIDE10'};
hashmethods = {'BATCH'};
loopnbits = [8 16 24 32 64 96 128];

param.top_K = 2000;

for dbi = 1     :length(db)
    db_name = db{dbi}; param.db_name = db_name;
    
    diary(['./results/conv_',db_name,'_result.txt']);
    diary on;
    
    %% load dataset
    load(['./datasets/',db_name,'.mat']);
    result_name = [result_URL 'final_' db_name '_result' '.mat'];

    if strcmp(db_name, 'MIRFLICKR0')
        X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
        R = randperm(size(L,1));
        sampleInds = R(2001:end);
        queryInds = R(1:2000);
        XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
        XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
        clear X Y L

    elseif strcmp(db_name, 'NUSWIDE10')
        X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
        R = randperm(size(L,1));
        %sampleInds = R(2001:end);
        sampleInds = R(2001:2000+10000);
        queryInds = R(1:2000);
        XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
        XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
        clear X Y L
    end
    clear I_tr I_te L_tr L_te
    
    %% Kernel representation
    param.nXanchors = 500; param.nYanchors = 1000;
    if 1
        anchor_idx = randsample(size(XTrain,1), param.nXanchors);
        XAnchors = XTrain(anchor_idx,:);
        anchor_idx = randsample(size(YTrain,1), param.nYanchors);
        YAnchors = YTrain(anchor_idx,:);
    else
        [~, XAnchors] = litekmeans(XTrain, param.nXanchors, 'MaxIter', 30);
        [~, YAnchors] = litekmeans(YTrain, param.nYanchors, 'MaxIter', 30);
    end
    
    [XKTrain,XKTest]=Kernel_Feature(XTrain,XTest,XAnchors);
    [YKTrain,YKTest]=Kernel_Feature(YTrain,YTest,YAnchors);

    
    %% Label Format
    if isvector(LTrain)
        LTrain = sparse(1:length(LTrain), double(LTrain), 1); LTrain = full(LTrain);
        LTest = sparse(1:length(LTest), double(LTest), 1); LTest = full(LTest);
    end
    
    
    %% Methods
    eva_info = cell(length(hashmethods),length(loopnbits));
    
    for ii =1:length(loopnbits)
        fprintf('======%s: start %d bits encoding======\n\n',db_name,loopnbits(ii));
        param.nbits = loopnbits(ii);
        
        for jj = 1:length(hashmethods)
            switch(hashmethods{jj})
                case 'BATCH'
                    fprintf('......%s start...... \n\n', 'BATCH');
                    BATCHparam = param;
                    BATCHparam.eta1 = 0.05; BATCHparam.eta2 = 0.05; BATCHparam.eta0 = 0.9;
                    BATCHparam.omega = 0.01; BATCHparam.xi = 0.01; BATCHparam.max_iter = 6;
                    eva_info_ = evaluate_BATCH(XKTrain,YKTrain,LTrain,XKTest,YKTest,LTest,BATCHparam);
            end
            eva_info{jj,ii} = eva_info_;
            clear eva_info_
        end
    end
    
    
    %% Results
    for ii = 1:length(loopnbits)
        for jj = 1:length(hashmethods)
            % MAP
            Image_VS_Text_MAP{jj,ii} = eva_info{jj,ii}.Image_VS_Text_MAP;
            Text_VS_Image_MAP{jj,ii} = eva_info{jj,ii}.Text_VS_Image_MAP;

            % Precision VS Recall
            Image_VS_Text_recall{jj,ii,:}    = eva_info{jj,ii}.Image_VS_Text_recall';
            Image_VS_Text_precision{jj,ii,:} = eva_info{jj,ii}.Image_VS_Text_precision';
            Text_VS_Image_recall{jj,ii,:}    = eva_info{jj,ii}.Text_VS_Image_recall';
            Text_VS_Image_precision{jj,ii,:} = eva_info{jj,ii}.Text_VS_Image_precision';

            % Top number Precision
            Image_To_Text_Precision{jj,ii,:} = eva_info{jj,ii}.Image_To_Text_Precision;
            Text_To_Image_Precision{jj,ii,:} = eva_info{jj,ii}.Text_To_Image_Precision;
            
            % Time
            trainT{jj,ii} = eva_info{jj,ii}.trainT;
        end
    end

    save(result_name,'eva_info','BATCHparam','loopnbits','hashmethods','sampleInds','queryInds',...
        'trainT','Image_VS_Text_MAP','Text_VS_Image_MAP','Image_VS_Text_recall','Image_VS_Text_precision',...
        'Text_VS_Image_recall','Text_VS_Image_precision','Image_To_Text_Precision','Text_To_Image_Precision','-v7.3');
    
    diary off;
end
