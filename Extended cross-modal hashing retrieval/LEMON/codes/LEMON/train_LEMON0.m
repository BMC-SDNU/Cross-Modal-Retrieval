function [BB,XW,YW,HH] = train_LEMON0(XTrain_new,YTrain_new,LTrain_new,GTrain_new,param)
    
    max_iter = 5; param.epsilon = 1e-5;
    
    % parameters
    alpha = param.alpha; beta = param.beta;
    gamma = param.gamma; xi = param.xi;
    
    nbits = param.nbits;
    
    n2 = size(LTrain_new,1);
    
    %initization
    B_new = sign(randn(n2, nbits)); B_new(B_new==0) = -1;
    
    V_new = randn(n2, nbits);
    
    R = randn(nbits, nbits);
    [U11, ~, ~] = svd(R);
    R = U11(:, 1:nbits);
    
    
    for i = 1:max_iter
        %fprintf('iteration %3d\n', i);

        % update P
        P_new = ((V_new'*V_new)+gamma*eye(nbits)) \ V_new'*LTrain_new;
        
        % update V_new
        Z = B_new*R'+beta*LTrain_new*P_new'...
            +2*alpha*nbits*GTrain_new*(GTrain_new'*B_new)...
            -alpha*nbits*ones(n2,1)*(ones(1,n2)*B_new);
        
        Temp = Z'*Z-1/n2*(Z'*ones(n2,1)*(ones(1,n2)*Z));
        [~,Lmd,QQ] = svd(Temp); clear Temp
        idx = (diag(Lmd)>1e-6);
        Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
        P = (Z-1/n2*ones(n2,1)*(ones(1,n2)*Z)) *  (Q / (sqrt(Lmd(idx,idx))));
        P_ = orth(randn(n2,nbits-length(find(idx==1))));
        V_new = sqrt(n2)*[P P_]*[Q Q_]';
        
        
        % update R
        [UB, ~, UA] = svd(B_new'*V_new);
        R = UA * UB';
        
        % update B
        B_new = sign(2*alpha*nbits*GTrain_new*(GTrain_new'*V_new)...
            -alpha*nbits*ones(n2,1)*(ones(1,n2)*V_new)+V_new*R);
    end
    
    H1_new = V_new'*V_new;
    H2_new = V_new'*LTrain_new;
    H3_new = GTrain_new'*B_new;
    H4_new = ones(1,n2)*B_new;
    H5_new = B_new'*V_new;
    H6_new = GTrain_new'*V_new;
    H7_new = ones(1,n2)*V_new;
    H8_new = XTrain_new'*XTrain_new;
    H9_new = YTrain_new'*YTrain_new;
    H10_new = XTrain_new'*B_new;
    H11_new = YTrain_new'*B_new;
    
    HH{1,1} = H1_new;
    HH{1,2} = H2_new;
    HH{1,3} = H3_new;
    HH{1,4} = H4_new;
    HH{1,5} = H5_new;
    HH{1,6} = H6_new;
    HH{1,7} = H7_new;
    HH{1,8} = H8_new;
    HH{1,9} = H9_new;
    HH{1,10} = H10_new;
    HH{1,11} = H11_new;
    BB{1,1} = B_new;
    
    XW = (H8_new+xi*eye(size(XTrain_new,2))) \ H10_new;
    YW = (H9_new+xi*eye(size(YTrain_new,2))) \ H11_new;

end