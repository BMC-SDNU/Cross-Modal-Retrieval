function result = DLFH_algo(dataset, param)
%% training procedure
trainTime = tic;
trainL = dataset.databaseL;

[BX_opt, BY_opt] = DLFH(trainL, param);

fprintf('...training finishes\n');

XTrain = dataset.XDatabase;
YTrain = dataset.YDatabase;

Wx = (XTrain' * XTrain + param.gamma * eye(size(XTrain, 2))) \ ...
    XTrain' * BX_opt;
Wy = (YTrain' * YTrain + param.gamma * eye(size(YTrain, 2))) \ ...
    YTrain' * BY_opt;

XTest = dataset.XTest;
YTest = dataset.YTest;

testL = dataset.testL;
databaseL = dataset.databaseL;

tBX = compactbit(XTest * Wx > 0);
tBY = compactbit(YTest * Wy > 0);

dBX = compactbit(BX_opt > 0);
dBY = compactbit(BY_opt > 0);
fprintf('...generating codes for query set and compressing codes finish\n');
trainTime = toc(trainTime);
fprintf('...training time: %3.3f\n', trainTime);
result.trainTime = trainTime;

fprintf('...encoding finishes\n');

%% evaluation procedure
% hamming ranking
topk = 100;
hri2t = calcMapTopkMapTopkPreTopkRecLabel(testL, databaseL, tBX, dBY, topk);
hrt2i = calcMapTopkMapTopkPreTopkRecLabel(testL, databaseL, tBY, dBX, topk);

hli2t = calcPreRecRadiusLabel(testL, databaseL, tBX, dBY);
hlt2i = calcPreRecRadiusLabel(testL, databaseL, tBY, dBX);

result.hri2t = hri2t;
result.hrt2i = hrt2i;
result.hli2t = hli2t;
result.hlt2i = hlt2i;
result.topk = topk;

fprintf('...evaluating finishes\n');
end