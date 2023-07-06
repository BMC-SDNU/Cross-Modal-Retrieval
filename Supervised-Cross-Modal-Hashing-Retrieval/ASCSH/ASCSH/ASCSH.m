function [B] = ASCSH(trainLabel, param, dataset)

X1 = dataset.XDatabase';
X2 = dataset.YDatabase';
% I_te = dataset.XTest;
% T_te = dataset.YTest;
% L_te = dataset.testL;
% L_tr = dataset.databaseL;
XTest = dataset.XTest;
YTest = dataset.YTest;
testL = dataset.testL;
databaseL = dataset.databaseL;


[d,~] = size(X1);
bit = param.bit;
maxIter = param.maxIter;
sampleColumn = param.num_samples;
lambda = param.lambda;
alpha1 = param.alpha1;
alpha2 = param.alpha2;
gamma = param.gamma;
mu = param.mu;
lambda_c = param.lambda_c;
eta = param.eta;

numTrain = size(trainLabel, 1);

V_opt = ones(numTrain, bit);
V_opt(randn(numTrain, bit) < 0) = -1;
% V = V_opt';

B = ones(numTrain, bit);
B(randn(numTrain, bit) < 0) = -1;

K = zeros(bit,d);
Y = zeros(bit,d);
C = zeros(bit,d);
D1 = randn(bit,d);
D2 = randn(bit,d);
c = size(trainLabel,2);
G = randn(bit, c);
L = trainLabel';
% S = trainLabel * trainLabel' > 0;  
for epoch = 1:maxIter
   %% for test instance
   %% sample Sc
    Sc = randperm(numTrain, sampleColumn);
    % update BX
    SX = trainLabel * trainLabel(Sc, :)' > 0;    
    V_opt = updateColumnV(V_opt, B, SX, Sc, bit, lambda, sampleColumn, alpha1, alpha2, C', D1', D2', X1', X2', G', L', eta);
    V = V_opt';
    % update BY
    SY = trainLabel(Sc, :) * trainLabel' > 0;
    B = updateColumnB(B, V_opt, SY, Sc, bit, lambda, sampleColumn); 
    %G
     G = V*L' / (0.01*eye(c) + L*L');

    %C
    A0 = K-Y/mu; 
    A1 = V-D1*X1; 
    A2 = V-D2*X2;
    C = (mu*A0 + alpha1*A1*X1' + alpha2*A2*X2') / (mu*eye(d) + alpha1*X1*X1' +alpha2*X2*X2');
    
        
    %K
    [U1,S1,V1] = svd(C + Y./mu,'econ');
    a = diag(S1)-lambda_c/mu;
    a(a<0)=0; 
    T = diag(a);
    K = U1*T*V1';   
    
    %D1,D2
    M1 = V - C*X1;
    M2 = V - C*X2;
    
    D1 = alpha1*M1*X1' / (gamma*eye(d) + alpha1*X1*X1');
    D2 = alpha2*M2*X2' / (gamma*eye(d) + alpha1*X2*X2');
    
        
    
    mu = 1.01*mu;
    Y = Y + mu*(C-K);

   %real-time evaluation
    tBX = sign(XTest * (C+D1)');
    tBY = sign(YTest * (C+D2)');
    sim_ti = B * tBX';
    sim_it = B * tBY';
    R = size(B,1);
    ImgToTxt = mAP(sim_ti,databaseL,testL,R);
    TxtToImg = mAP(sim_it,databaseL,testL,R);

    fprintf('...iter:%d,   i2t:%.4f,   t2i:%.4f\n',epoch, ImgToTxt, TxtToImg)

end

end

% function U = updateColumnU(U, B, S, Sc, bit, lambda, sampleColumn)
% m = sampleColumn;
% n = size(U, 1);
% for k = 1: bit
%     TX = lambda * U * B(Sc, :)' / bit;
%     AX = 1 ./ (1 + exp(-TX));
%     Bjk = B(Sc, k)';
%     p = lambda * ((S - AX) .* repmat(Bjk, n, 1)) * ones(m, 1) / bit + m * lambda^2 * U(:, k) / (4 * bit^2);
%     U_opt = ones(n, 1);
%     U_opt(p < 0) = -1;
%     U(:, k) = U_opt;
% end
% end

function U = updateColumnV(U, B, S, Sc, bit, lambda, sampleColumn, alpha1, alpha2, C, D1, D2, X1, X2, R, L, eta)
m = sampleColumn;
n = size(U, 1);
for k = 1: bit
    TX = lambda * U * B(Sc, :)' / bit;
    AX = 1 ./ (1 + exp(-TX));
    Bjk = B(Sc, k)';
    aaa = (alpha1 + alpha2 + eta);
    p = lambda * ((S - AX) .* repmat(Bjk, n, 1)) * ones(m, 1) / bit...
        + (m * lambda^2 + 8*bit^2 *aaa)* U(:, k) / (4 * bit^2) + ...
        + 2*alpha1*( U(:,k)- X1*(C(:,k)+D1(:,k)) ) + 2*alpha2*( U(:,k)- X2*(C(:,k)+D2(:,k)) ) + 2*eta*(U(:,k) - L*R(:,k));
    U_opt = ones(n, 1);
    U_opt(p < 0) = -1;
    U(:, k) = U_opt;
end
end


function B = updateColumnB(B, U, S, Sc, bit, lambda, sampleColumn)
m = sampleColumn;
n = size(U, 1);
for k = 1: bit
    TX1 = lambda * U(Sc, :) * B' / bit;
    AX1 = 1 ./ (1 + exp(-TX1));
    Ujk = U(Sc, k)';  %1*8
    p = lambda * ((S' - AX1') .* repmat(Ujk, n, 1)) * ones(m, 1)  / bit + m * lambda^2 * B(:, k) / (4 * bit^2);
    B_opt = ones(n, 1);
    B_opt(p < 0) = -1;
    B(:, k) = B_opt;
end
end


