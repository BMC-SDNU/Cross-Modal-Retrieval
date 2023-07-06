function [P1, P2, D, obj] = solveMDBE( X1, X2, L, lambda, beta, gamma, bits, maxIter )
%
%   minimize_{P1, P2, W1, W2, D}    
%   lambda *{ ||L - W1 * P1 * X1||^2 + beta * ||P1 * X1 - D * L||^2  +  gamma * ||W1||^2} 
%   (1 - lambda) *{ ||L - W2 * P2 * X2||^2 + beta * ||P2 * X2 - D * L||^2 +  gamma * ||W2||^2}
%
% Notation:
% X1: data matrix of View1, each column is a sample vector
% X2: data matrix of View2, each column is a sample vector
% lambda: trade off between different views
% beta: trade off between classification errors and lable consistent
% gamma: parameter to control the model complexity
% 
% Written by Di Wang
%
% Reference:
% Di Wang, Xinbo Gao, Xiumei Wang, Lihuo He, and Bo Yuan.
% "Multimodal Discriminative Binary Embedding for Large-Scale Cross-Modal Retrieval"
% IEEE TRANSACTIONS ON IMAGE PROCESSING (TIP),vol.25, no.10, pp.4540¨C4554, 2016.
% wangdi@xidian.edu.cn

row = size(X1,1);
rowt = size(X2,1);
P1 = rand(bits, row);
P2 = rand(bits, rowt);
threshold = 0.01;
lastF = 99999999;
iter = 1;
obj = zeros(maxIter, 1);

LL = L*L';
XX1 = X1*X1';
XX2 = X2*X2';

while (true)
    
    % update D 
    D = (lambda * P1 * X1 * L' + (1-lambda) * P2 * X2 * L')/LL;
    
    % update W1 and W2
    W1 = L * X1' * P1' / (P1 * XX1 * P1' + gamma * eye(bits));
    W2 = L * X2' * P2' / (P2 * XX2 * P2' + gamma * eye(bits));
    
    %update P1 and P2
    P1 = (lambda * W1' * W1 + (lambda*beta) * eye(bits)) \ (lambda * W1' * L * X1' + (lambda*beta) * D * L * X1');
    P2 = ((1-lambda) * W2' * W2 + ((1-lambda)*beta) * eye(bits)) \ ((1-lambda) * W2' * L * X2' + ((1-lambda)*beta) * D * L * X2');
    P1 = P1 / (XX1 + gamma * eye(row)); P2 = P2 / (XX2 + gamma * eye(rowt));
    
    % compute objective function
    norm1 = lambda * ( norm(L - W1 * P1 * X1, 'fro') + beta * norm(D * L - P1 * X1, 'fro') + gamma * norm(W1, 'fro'));
    norm2 = (1 - lambda) * ( norm(L - W2 * P2 * X2, 'fro') + beta * norm(D * L - P2 * X2, 'fro') + gamma * norm(W2, 'fro'));
    currentF = norm1 + norm2;
    obj(iter) = currentF;
    
    fprintf('\nobj at iteration %d: %.4f\n classification error for image: %.4f,\n classification error for text: %.4f\n\n', iter, currentF, norm1, norm2);
    if lastF - currentF < threshold
        fprintf('algorithm converges...\n');
        fprintf('final obj: %.4f\n classification error for image: %.4f,\n classification error for text: %.4f\n\n', currentF, norm1, norm2);
        return;
    end
    if iter>=maxIter
        return
    end
    iter = iter + 1;
    lastF = currentF;
end

end

