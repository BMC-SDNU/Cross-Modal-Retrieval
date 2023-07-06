function [BX_opt, BY_opt] = DLFH(trainLabel, param)
bit = param.bit;
maxIter = param.maxIter;
sampleColumn = param.num_samples;
lambda = param.lambda;

numTrain = size(trainLabel, 1);

U = ones(numTrain, bit);
U(randn(numTrain, bit) < 0) = -1;

V = ones(numTrain, bit);
V(randn(numTrain, bit) < 0) = -1;

for epoch = 1:maxIter
    % sample Sc
    Sc = randperm(numTrain, sampleColumn);
    % update BX
    SX = trainLabel * trainLabel(Sc, :)' > 0;
    U = updateColumnU(U, V, SX, Sc, bit, lambda, sampleColumn);

    % update BY
    SY = trainLabel(Sc, :) * trainLabel' > 0;
    V = updateColumnV(V, U, SY, Sc, bit, lambda, sampleColumn);
end

BX_opt = U;
BY_opt = V;

end

function U = updateColumnU(U, V, S, Sc, bit, lambda, sampleColumn)
m = sampleColumn;
n = size(U, 1);
for k = 1: bit
    TX = lambda * U * V(Sc, :)' / bit;
    AX = 1 ./ (1 + exp(-TX));
    Vjk = V(Sc, k)';
    p = lambda * ((S - AX) .* repmat(Vjk, n, 1)) * ones(m, 1) / bit + m * lambda^2 * U(:, k) / (4 * bit^2);
    U_opt = ones(n, 1);
    U_opt(p < 0) = -1;
    U(:, k) = U_opt;
end
end

function V = updateColumnV(V, U, S, Sc, bit, lambda, sampleColumn)
m = sampleColumn;
n = size(U, 1);
for k = 1: bit
    TX = lambda * U(Sc, :) * V' / bit;
    AX = 1 ./ (1 + exp(-TX));
    Ujk = U(Sc, k)';
    p = lambda * ((S' - AX') .* repmat(Ujk, n, 1)) * ones(m, 1)  / bit + m * lambda^2 * V(:, k) / (4 * bit^2);
    V_opt = ones(n, 1);
    V_opt(p < 0) = -1;
    V(:, k) = V_opt;
end
end
