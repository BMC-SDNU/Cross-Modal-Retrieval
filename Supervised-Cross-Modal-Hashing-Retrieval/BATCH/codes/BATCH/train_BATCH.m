function final_B = train_BATCH(GTrain,XKTrain,YKTrain,LTrain,param)

    % parameters
    nbits = param.nbits;
    n = size(LTrain,1);
    
    % initization
    B = sign(randn(n, nbits)); B(B==0) = -1;
    V = randn(n, nbits);
    
    for i = 1:param.max_iter
        fprintf('iteration %3d\n', i);
        
        % update U
        temp = V'*V; temp2 = eye(nbits);
        Ux = ((temp+1e-3*temp2) \ (V'*XKTrain));
        Uy = ((temp+1e-3*temp2) \ (V'*YKTrain));
        Ul = ((temp+1e-3*temp2) \ (V'*LTrain));
        clear temp temp2
        
        % update V
        Z = nbits*(GTrain*(GTrain'*B))+param.omega*B+param.eta1*(XKTrain*Ux')...
            +param.eta2*(YKTrain*Uy')+param.eta0*(LTrain*Ul');
        
        %Temp = Z - 1/n*ones(n,1)*(ones(1,n)*Z);
        %[P,Lmd,Q] = svd(Temp);
        %idx = (diag(Lmd)>1e-6);
        %Q = Q(:,idx); Q_ = orth(Q(:,~idx));
        %P = P(:,idx); P_ = orth(P(:,~idx));
        %V = sqrt(n)*[P P_]*[Q Q_]';
        
        Temp = Z'*Z-1/n*(Z'*ones(n,1)*(ones(1,n)*Z));
        [~,Lmd,QQ] = svd(Temp); clear Temp
        idx = (diag(Lmd)>1e-6);
        Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
        P = (Z-1/n*ones(n,1)*(ones(1,n)*Z)) *  (Q / (sqrt(Lmd(idx,idx))));
        P_ = orth(randn(n,nbits-length(find(idx==1))));
        V = sqrt(n)*[P P_]*[Q Q_]';
        
        % update B
        B = sign(nbits*(GTrain*(GTrain'*V))+param.omega*V);
    end

    final_B = sign(B);
    final_B(final_B==0) = -1;

end