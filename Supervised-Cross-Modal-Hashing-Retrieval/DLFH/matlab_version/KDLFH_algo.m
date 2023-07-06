function result = KDLFH_algo(dataset, param)
%% training procedure
trainTime = tic;
trainL = dataset.databaseL;

[BX_opt, BY_opt] = DLFH(trainL, param);

fprintf('...training finishes\n');

XTrain = dataset.XDatabase;
YTrain = dataset.YDatabase;

testL = dataset.testL;
databaseL = dataset.databaseL;

XSample = XTrain(1:5000, :);
bit = param.bit;
numTrain = size(XTrain, 1);
numSample = size(XSample, 1);
z = XSample * XSample';
z = repmat(diag(z), 1, numSample)  + repmat(diag(z)', numSample, 1) - 2 * z;
k1 = {};
k1.type = 0;
k1.param = mean(z(:));                                  %  $\sigma^2$ for RBF kernel in image view

YSample = YTrain(1:5000, :);
numSample = size(YSample, 1);
z = YSample * YSample';
z = repmat(diag(z), 1, numSample)  + repmat(diag(z)', numSample, 1) - 2 * z;
k2 = {};
k2.type = 0;
k2.param = mean(z(:));                                  %  $\sigma^2$ for RBF kernel in text view    

% Kernel Logistic Regression (KLR) Developed by Mark Schimidt     
kernelSampleNum = param.numKernel;



sampleType = param.sampleType;
if strcmp(sampleType,'Random')
    % Random Sampling for Learning KLR
    kernelSamples = sort(randperm(numTrain, kernelSampleNum));
    kernelXs{1} = XTrain(kernelSamples, :);
    kernelXs{2} = YTrain(kernelSamples, :);
elseif strcmp(sampleType,'Kernel')
    % Kmeans Sampling for Learning KLR
    opts = statset('Display', 'off', 'MaxIter', 100);
    [~, C] = kmeans(XTrain, kernelSampleNum, 'Start', 'sample', 'EmptyAction', 'singleton', 'Options', opts, 'OnlinePhase', 'off');
    kernelXs{1} = C;

    [~, C] = kmeans(YTrain, kernelSampleNum, 'Start', 'sample', 'EmptyAction', 'singleton', 'Options', opts, 'OnlinePhase', 'off');
    kernelXs{2} = C;
end

% Kernel Matrices
K01 = kernelMatrix(kernelXs{1}, kernelXs{1}, k1);
K02 = kernelMatrix(kernelXs{2}, kernelXs{2}, k2);
trainK1 = kernelMatrix(XTrain, kernelXs{1}, k1);
trainK2 = kernelMatrix(YTrain, kernelXs{2}, k2);
% RetrK1 = kernelMatrix(dataset.X(para.indexRetrieval,:), kernelXs{1}, k1);
% RetrK2 = kernelMatrix(dataset.Y(para.indexRetrieval,:), kernelXs{2}, k2);
queryK1 = kernelMatrix(dataset.XTest, kernelXs{1}, k1);
queryK2 = kernelMatrix(dataset.YTest, kernelXs{2}, k2);        

% Hash Codes for Retrieval Set and Query Set
tBX = zeros(size(dataset.XTest, 1), bit);
tBY = zeros(size(dataset.XTest, 1), bit);
options.Display = 'final';        
C = param.eta;                                                   % Weight for Regularization. 1e-2 is Good Enough.

R = 1:numTrain;
if param.small == true
    R = 1:5000;
end

% KLR for Each Bit
for b = 1 : bit
    tH = BX_opt(R, b);

    funObj = @(u)LogisticLoss(u, trainK1(R,:), tH);
    w = minFunc(@penalizedKernelL2, zeros(size(K01, 1),1), options, K01, funObj, C);  
    tBX(:, b) = sign(queryK1 * w);
end        

for b = 1 : bit 
    tH = BY_opt(R, b);
    funObj = @(u)LogisticLoss(u, trainK2(R,:), tH);
    w = minFunc(@penalizedKernelL2, zeros(size(K02, 1),1), options, K02, funObj, C); 
    tBY(:, b) = sign(queryK2 * w);
end

tBX = compactbit(tBX > 0);
tBY = compactbit(tBY > 0);

dBX = compactbit(BX_opt > 0);
dBY = compactbit(BY_opt > 0);

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


fprintf('...evaluating finishes\n');
end