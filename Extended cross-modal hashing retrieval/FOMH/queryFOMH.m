function [B_test, B_db] = queryFOMH(phi_db1, phi_db2, phi_test1, phi_test2, W1, W2, param)
%% Generate hash codes for test set
% parameter setting
[~, Nte] = size(phi_test1);
mu1t = 0.5;
mu2t = 0.5;
t = 1;
batch = param.batch;

% matrix initialization
Bp = [];
B = [];

% iterative algorithm
while(t<=Nte)
    if ((t+batch-1)<=Nte)
        phi_x1t = phi_test1(:, t:(t+batch-1));
        phi_x2t = phi_test2(:, t:(t+batch-1));
    else
        phi_x1t = phi_test1(:, t:Nte);
        phi_x2t = phi_test2(:, t:Nte);
    end
    
    for iter=1:5
        % Update hash codes for test set
        Bt = sign(2*mu1t*W1*phi_x1t + 2*mu2t*W2*phi_x2t);      
        % Update weight for test set
        mu1t = 1 / (2*sqrt(sum(sum((Bt-W1*phi_x1t).^2))));
        mu2t = 1 / (2*sqrt(sum(sum((Bt-W2*phi_x2t).^2))));
    end
    
    B = [Bp, Bt];
    t = t+batch;
    Bp=B;
end

B_test = sign(B');

%% Generate hash codes for retrieval set
% parameter setting
[~, Ndb] = size(phi_db1);
mu1t = 0.5;
mu2t = 0.5;
t = 1;
batch = param.batch;

% matrix initialization
Bp = [];
B = [];

% iterative algorithm
while(t<=Ndb)
    if ((t+batch-1)<=Ndb)
        phi_x1t = phi_db1(:, t:(t+batch-1));
        phi_x2t = phi_db2(:, t:(t+batch-1));
    else
        phi_x1t = phi_db1(:, t:Ndb);
        phi_x2t = phi_db2(:, t:Ndb);
    end
    for iter=1:5
        % Update hash codes for retrieval set
        Bt = sign(2*mu1t*W1*phi_x1t + 2*mu2t*W2*phi_x2t);      
        % Update weight for retrieval set
        mu1t = 1 / (2*sqrt(sum(sum((Bt-W1*phi_x1t).^2))));
        mu2t = 1 / (2*sqrt(sum(sum((Bt-W2*phi_x2t).^2))));
    end
    
    B = [Bp, Bt];
    t = t+batch;
    Bp=B;
end

B_db = sign(B');

end