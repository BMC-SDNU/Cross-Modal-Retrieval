function [W1, W2] = solveFOMH(phi_x1, phi_x2, L_tr, param, j)
% phi_x1t & phi_x2t    the nonlinearly transformed representation of the new chunk at current round t
% mu1t & mu2t          the weight at current round t
% W1 & W2              the projection matrix
% Lt                   the label matrix of the new chunk at current t
% Bt                   the hash code of the new chunk at current round t
% Dt                   the continuous substitution of Bt
% Zb & G               the parameters of ALM

%% parameter setting
[p1, num] = size(phi_x1);
[p2, ~] = size(phi_x2);
L = L_tr';

bits  = param.bits;
batch = param.batch;
alpha = param.alpha;
beta  = param.beta;
gamma = param.gamma;
rho   = param.rho;

mu1t = 0.5;
mu2t = 0.5;
lastF = 10000000000;
t = 1;
round = 1;

%% matrix initialization 
Bt = sign(-1+(1-(-1))*rand(bits, batch));
Zb = sign(-1+(1-(-1))*rand(bits, batch));
G = Bt - Zb;
Dt = randn(bits, batch);

BX1p = 0; 
BX2p = 0;
XX1p = 0; 
XX2p = 0;

%% iterative algorithm 
fprintf('\n==================================================== Run %d ====================================================\n', j);
while((t+batch-1) <= num)
    phi_x1t = phi_x1(:, t:(t+batch-1));
    phi_x2t = phi_x2(:, t:(t+batch-1));
    Lt = L(:,t:(t+batch-1));

    for iter=1:10
        % Update W1 W2
        BX1t = Bt*phi_x1t';
        BX1 = BX1p + BX1t;
        XX1t = phi_x1t*phi_x1t';
        XX1 = XX1p + XX1t;
        W1 = ((1/mu1t)*BX1) / ((1/mu1t)*XX1 + gamma*eye(p1));
        BX2t = Bt*phi_x2t';
        BX2 = BX2p + BX2t;
        XX2t = phi_x2t*phi_x2t';
        XX2 = XX2p + XX2t;
        W2 = ((1/mu2t)*BX2) / ((1/mu2t)*XX2 + gamma*eye(p2));

        % Update Bt
        Bt = sign(2*(1/mu1t)*W1*phi_x1t + 2*(1/mu2t)*W2*phi_x2t + 2*alpha*bits*(2*Dt*Lt'*Lt-Dt*ones(batch,1)*ones(batch,1)') - alpha*Dt*Dt'*Zb + 2*beta*Dt + rho*Zb - G);

        % Update Dt
        Dt = (alpha*Bt*Bt' + beta*eye(bits)) \ (alpha*bits*(2*Bt*Lt'*Lt-Bt*ones(batch,1)*ones(batch,1)') + beta*Bt);

        % Update ALM parameters
        Zb = sign(-alpha*Dt*Dt'*Bt + rho*Bt + G);
        G = G+ rho*(Bt-Zb);

        % Update \mu
        mu1t = (sqrt(sum(sum((Bt-W1*phi_x1t).^2)))) / (sqrt(sum(sum((Bt-W1*phi_x1t).^2)))+sqrt(sum(sum((Bt-W2*phi_x2t).^2))));
        mu2t = (sqrt(sum(sum((Bt-W2*phi_x2t).^2)))) / (sqrt(sum(sum((Bt-W1*phi_x1t).^2)))+sqrt(sum(sum((Bt-W2*phi_x2t).^2))));

        % Objective function
        norm1 = (1/mu1t)*sum(sum((Bt-W1*phi_x1t).^2)) + (1/mu2t)*sum(sum((Bt-W2*phi_x2t).^2));
        norm2 = sum(sum((Bt'*Dt-bits*(2*Lt'*Lt - ones(batch,1)*ones(batch,1)')).^2));
        norm3 = sum(sum((Bt-Dt).^2));
        norm4 = sum(sum((W1).^2)) + sum(sum((W2).^2));
        currentF = norm1 + alpha*norm2 + beta*norm3 + gamma*norm4;
        fprintf('CurrentF at iteration %d: %.4f; obj: %.4f\n', iter, currentF, lastF - currentF);
        lastF = currentF;
    end
    fprintf('-------------------------- Round %d: the %dth to %dth training samples. --------------------------\n', round, t, t+batch-1);

    t = t+batch;
    round = round+1;
    BX1p = BX1; 
    BX2p = BX2;
    XX1p = XX1; 
    XX2p = XX2;
end