function [] = demo_SCRATCH(bits, dataname)
    data_dir = '../../Data';

    if ~exist(data_dir, 'dir')
        error('No such dir(%s)', fullfile(pwd, data_dir))
    end

    if ~exist('../result', 'dir')
        mkdir('../result')
    end

    addpath(data_dir);
    bits = str2num(bits);

    if strcmp(dataname, 'flickr')
        load('mir_cnn.mat');
    elseif strcmp(dataname, 'nuswide')
        load('nus_cnn.mat');
    elseif strcmp(dataname, 'coco')
        load('coco_cnn.mat');
    else
        fprintf('ERROR dataname!');
    end

    XTrain = I_tr; YTrain = T_tr; LTrain = double(L_tr);
    clear I_tr T_tr L_tr;
    XTest = I_te; YTest = T_te; LTest = double(L_te);
    clear I_te T_te L_te;
    XDb = I_db; YDb = T_db; LDb = double(L_db);
    clear I_db T_db L_db;

    param.iter = 20; %20

    %% centralization
    %fprintf('centralizing data...\n');
    XTest = bsxfun(@minus, XTest, mean(XTrain, 1));
    XDb = bsxfun(@minus, XDb, mean(XTrain, 1));
    XTrain = bsxfun(@minus, XTrain, mean(XTrain, 1));

    YTest = bsxfun(@minus, YTest, mean(YTrain, 1));
    YDb = bsxfun(@minus, YDb, mean(YTrain, 1));
    YTrain = bsxfun(@minus, YTrain, mean(YTrain, 1));

    %% kernelization
    %fprintf('kernelizing...\n\n');
    [XKTrain, XKTest, XKDb] = Kernelize(XTrain, XTest, XDb);
    [YKTrain, YKTest, YKDb] = Kernelize(YTrain, YTest, YDb);

    XKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1));
    XKDb = bsxfun(@minus, XKDb, mean(XKTrain, 1));
    XKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));

    YKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1));
    YKDb = bsxfun(@minus, YKDb, mean(YKTrain, 1));
    YKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));

    param.lambdaX = 0.5; %0.5
    param.alpha = 500; %500
    param.Xmu = 1000; %1000
    param.gamma = 5; %5
    param.iter = 20; %20
    param.nbits = bits;

    [Wx, Wy, R, B] = train(XKTrain, YKTrain, param, LTrain);
    % train time

    BxTest = compactbit(XKTest * Wx' * R' >= 0);
    BxDb = compactbit(XKDb * Wx' * R' >= 0);
    ByTest = compactbit(YKTest * Wy' * R' >= 0);
    ByDb = compactbit(YKDb * Wy' * R' >= 0);

    hamm_T2I = hammingDist(ByTest, BxDb)';
    MAP_T2I = perf_metric4Label(LDb, LTest, hamm_T2I);

    hamm_I2T = hammingDist(BxTest, ByDb)';
    MAP_I2T = perf_metric4Label(LDb, LTest, hamm_I2T);

    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits, MAP_I2T, MAP_T2I);
    fclose(fid);

end
