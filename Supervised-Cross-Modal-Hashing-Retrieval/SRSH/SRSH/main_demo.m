close all; clear; clc;
addpath(genpath('./utils/'));

db = {'WIKI','NUSWIDE10','IAPRTC-12','MIRFLICKR'};
hashmethods = {'SRSH'};

loopnbits = [16 32 64 96 128];
param.top_K = 2000;

nhmethods = length(hashmethods);
for index =2
    for dbi = 4
        db_name = db{dbi}; param.db_name = db_name;

        %diary(['./results/conv_',db_name,'_result.txt']);
        %diary on;

        % load dataset
        load(['./datasets/',db_name,'.mat']);

        if strcmp(db_name, 'WIKI')
            X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
            R = randperm(size(L,1));
            sampleInds = R(1:size(L_tr,1));
            queryInds = R(size(L_tr,1)+1:end);
            XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
            XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
            clear X Y L
            param.nAnchors = 500;

        elseif strcmp(db_name, 'MIRFLICKR2')
            X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
            R = randperm(size(L,1));
            sampleInds = R(2001:end);
            queryInds = R(1:2000);
            XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
            XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
            clear X Y L
            param.nAnchors = 1000;

        elseif strcmp(db_name, 'NUSWIDE10')
            X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
            R = randperm(size(L,1));
            sampleInds = R(2001:186577);
            %sampleInds = R(2001:end);
            queryInds = R(1:2000);
            XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
            XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
            clear X Y L
            param.nAnchors = 1500;
        elseif strcmp(db_name, 'IAPRTC-12')
            clear V_tr V_te
            X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
            R = randperm(size(L,1));
            queryInds = R(1:2000);
            sampleInds = R(2001:2000+5000);
            %sampleInds = R(2001:end);
            XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
            XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);

            param.nAnchors = 1000;

        elseif strcmp(db_name, 'MIRFLICKR')
            R = randperm(size(LAll,1));
            queryInds = R(1:2000);
            sampleInds = R(2001:16738);
            XTrain = XAll(sampleInds, :); YTrain = YAll(sampleInds, :); LTrain = LAll(sampleInds, :);
            XTest = XAll(queryInds, :); YTest = YAll(queryInds, :); LTest = LAll(queryInds, :);
            clear XAll YAll LAll

            param.nAnchors = 1000;

        end
        clear I_tr I_te L_tr L_te T_tr T_te

        if isvector(LTrain)
            LTrain = sparse(1:length(LTrain), double(LTrain), 1); LTrain = full(LTrain);
            LTest = sparse(1:length(LTest), double(LTest), 1); LTest = full(LTest);
        end


        %% Methods
        eva_info = cell(length(hashmethods),length(loopnbits));

        fprintf('\n\n\n======start %s======\n\n\n', db_name);
        for ii =1:length(loopnbits)
            fprintf('======start %d bits encoding======\n\n', loopnbits(ii));
            param.nbits = loopnbits(ii);
            fprintf('......%s start...... \n\n', 'SRSH');
            ct=cputime;
            SRSHparam = param;
            eva_info_ =evaluate_SRSH(XTrain,YTrain,LTrain,XTest,YTest,LTest,SRSHparam);
        end


    end

end
