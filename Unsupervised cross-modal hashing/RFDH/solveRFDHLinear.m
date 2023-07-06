function [B, modelI, modelT, obj] = solveRFDHLinear( X1, X2, bits, lambda, gamma,  maxIter )
%
%   minimize_{B, P1, P2, F1, F2}    
%   £¨lambda1£©^lambda * ||X1 - F1 * B||2,1 + £¨lambda2£©^lambda * ||X2 - F2 * B||2,1  
%    +  gamma * { ||F1||^2 + ||F2||^2}
% 
% Notation:
% X1: data matrix of View1, each column is a sample vector
% X2: data matrix of View2, each column is a sample vector
% lambda: trade off between different views
% gamma: parameter to control the model complexity
% 
% Written by Di Wang
%
%

%% Initialization
N = size(X1,2);
% initialize B
B = rand(bits,N) > 0.5;
B = double(B);
% initialize F1 and F2
F1 = X1 * B' / (B * B' + gamma * eye(bits));
F2 = X2 * B' / (B * B' + gamma * eye(bits));
% initialize D1 and D2
E1 = X1 - F1*B;
EE = sqrt(sum(E1.*E1, 1) + eps);
temp = (1./EE);
D1 = sparse(1:N,1:N,temp);
E2 = X2 - F2*B;
EE = sqrt(sum(E2.*E2, 1) + eps);
temp = (1./EE);
D2 = sparse(1:N,1:N,temp);

iter = 1;
obj = zeros(maxIter, 1);
C = round(N/2);
lam = 1/(lambda-1);

%% compute iteratively
while (true)
    
    % update lambda1 and lambda2
    H1 = (lambda*trace(E1*D1*E1'))^lam;
    H2 = (lambda*trace(E2*D2*E2'))^lam;
    H12 = H1 + H2;
    lambda1 = H1 / H12;
    lambda2 = H2 / H12;
    lambda1 = lambda1^lambda;   
    lambda2 = lambda2^lambda;   
    % update B
    Q1 = D1*X1'*F1;
    Q2 = D2*X2'*F2;
    for time = 1:5
        B0 = B;
        for k = 1:bits
            Bt = B; Bt(k,:) = [];
            F1t = F1; F1t(:,k) = []; F2t = F2; F2t(:,k) = [];
            V1 = F1(:,k); V2 = F2(:,k);
            vv1 = lambda1*(V1'*F1t*Bt*D1+0.5*V1'*V1*diag(D1)'-Q1(:,k)');
            vv2 = lambda2*(V2'*F2t*Bt*D2+0.5*V2'*V2*diag(D2)'-Q2(:,k)');
            [~,ind] = sort(vv1 + vv2);
            B(k,ind(1:C)) = 1;
            B(k,ind(C+1:end)) = 0;
        end
        
        if norm(B-B0,'fro') < 1e-6 * norm(B0,'fro')
           break;
        end       
    end
    
    % update F1 and F2
    F1 = X1 * D1 * B' / (B * D1 * B' + (gamma/lambda1) * eye(bits));
    F2 = X2 * D2 * B' / (B * D2 * B' + (gamma/lambda2) * eye(bits));
        
    % update D1 and D2
    E1 = X1 - F1*B;
    EE = sqrt(sum(E1.*E1, 1) + eps);
    temp = (1./EE);
    D1 = sparse(1:N,1:N,temp);
    E2 = X2 - F2*B;
    EE = sqrt(sum(E2.*E2, 1) + eps);
    temp = (1./EE);
    D2 = sparse(1:N,1:N,temp);
    % compute objective function
    E1 = X1 - F1*B;
    E2 = X2 - F2*B;
    H1 = lambda1*trace(E1*D1*E1');
    H2 = lambda2*trace(E2*D2*E2');
    norm1 = H1 + H2;
    norm2 = gamma * (norm(F1,'fro') + norm(F2,'fro'));
    currentF = norm1 + norm2;
    obj(iter) = currentF;
    
    fprintf('\nobj at iteration %d: %.4f\n reconstruction error: %.4f,\n regularization error: %.4f\n\n', iter, currentF, norm1, norm2);
    if iter>=maxIter
        break;
    end
    iter = iter + 1;
    lastF = currentF;
end

eta = 10;
R = randn(bits,bits);
T = randn(size(B));

XX1 = X1 * X1' + gamma * eye(size(X1,1));
XX2 = X2 * X2' + gamma * eye(size(X2,1));
II = ones(1,N);
u1 = randn(bits,1);
u2 = randn(bits,1);
B = 2*B - 1;
for i = 1:5
    % update P1 and P2
    P1 = (T+u1*II) * X1' / XX1;
    P2 = (T+u2*II) * X2' / XX2;
    %update T
    T = (eta*R'*R + 2*eye(bits))\(P1*X1 - u1*II + P2*X2 - u2*II + eta*R'*B);
    % update R
    R = T * B' / (B * B' + gamma * eye(bits));
    % update u1 and u2
    u1 = mean(P1*X1-T,2);
    u2 = mean(P2*X2-T,2);
end
modelI = {};
modelI.P = P1;
modelI.R = R;
modelI.u = u1;

modelT = {};
modelT.P = P2;
modelT.R = R;
modelT.u = u2;
end

