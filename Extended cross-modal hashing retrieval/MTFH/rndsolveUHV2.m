function [ U, V, U_, V_, H1, H2 ] = rndsolveUHV2( alpha, beta, q1, q2, S, U, U_, V, V_, H1, H2 )
%SOLVEUHV 
% Reference:
% Xin Liu, Zhikai Hu, Haibin Ling, and Yiu-ming Cheung
% "MTFH: A Matrix Tri-Factorization Hashing Framework for Efficient Cross-Modal Retrieval"
% arXiv:1805.01963
% created by Zhikai Hu, Version1.0 -- May/2018

 
    iter = 1;
    times = 10;
    threshold = 1;
    lastF = realmax;
    run = 3;% ensemble rounds
    
    while(true)
        tic
        %% update H1, H2
        H1 = U_' * V / (V' * V + 0.1 * eye(q2));
        H2 = (U' * U + 0.1 * eye(q1))^-1 * U' * V_;
        
        %% update U
        P1 = alpha / q1 * U_' * S' + beta * H2 * V_';
        for test = 1: run
            for t = 1: times
                U0 = U;
                rnd = size(U, 2);
                for i = 1: randperm(rnd)
                    A = U;      A(:, i) = [];
                    B = U_;     b = B(:, i);    B(:, i) = [];
                    D = H2;     d = H2(i, :);   D(i, :) = [];
                    p1 = P1(i, :);
                    U(:, i) = sign(p1 - alpha / q1^2 * b' * B * A' - beta * d * D' * A');
                end
                if norm(U0 - U, 'fro') < 1e-6 * norm(U0, 'fro')
                    break;
                end
            end
            temp{test} = U;
        end
        U = sign(temp{1} + temp{2} + temp{3});
        clear temp
        
        %% update U_
        P2 = alpha / q1 * U' * S + beta * H1 * V';
        for test = 1: run
            for t = 1: times
                U_0 = U_;
                rnd = size(U_, 2);
                for i = 1: randperm(rnd)
                    A = U;     a = A(:, i);    A(:, i) = [];
                    B = U_;     B(:, i) = [];
                    p2 = P2(i, :);
                    U_(:, i) = sign(p2 - alpha / q1^2 * a' * A * B'); 
                end
                if norm(U_0 - U_, 'fro') < 1e-6 * norm(U_0, 'fro')
                    break;
                end
            end
            temp{test} = U_;
        end
        U_ = sign(temp{1} + temp{2} + temp{3});
        clear temp
                 
        %% update V
        P3 = (1 - alpha) / q2 * V_' * S + beta * H1' * U_';
        for test = 1: run
            for t = 1: times
                V0 = V;
                rnd = size(V, 2);
                for i = 1: randperm(rnd)
                    E = V;      E(:, i) = [];
                    F = V_;     f = F(:, i);    F(:, i) = [];
                    G = H2;     g = G(:, i);    G(:, i) = [];
                    p3 = P3(i, :);
                    V(:, i) = sign(p3 - (1 - alpha) / q2^2 * f' * F * E' - beta * g' * G * E');
                end
                if norm(V0 - V, 'fro') < 1e-6 * norm(V0, 'fro')
                    break;
                end
            end
            temp{test} = V;
        end
        V = sign(temp{1} + temp{2} + temp{3});
        clear temp
        
        %% update V_
        P4 = (1 - alpha) / q2 * V' * S' + beta * H2' * U';
        for test = 1: run
            for t = 1:times
                V_0 = V_;
                rnd = size(V_, 2);
                for i = 1: randperm(rnd)
                    E = V;      e = E(:, i);    E(:, i) = [];
                    F = V_;    F(:, i) = [];
                    p4 = P4(i, :);
                    V_(:, i) = sign(p4 - (1 - alpha) / q2^2 * e' * E * F');
                end
                if norm(V_0 - V_, 'fro') < 1e-6 * norm(V_0, 'fro')
                    break;
                end
            end
            temp{test} = V_;
        end
        V_ = sign(temp{1} + temp{2} + temp{3});
        clear temp
        toc
        
        %% loss
        norm1 = alpha * norm(S - U * U_' / q1, 'fro')^2;
        norm2 = (1 - alpha) * norm(S - V_ * V' / q2, 'fro')^2;
        norm3 = beta * (norm(U_ - V * H1', 'fro')^2 + norm(V_ - U * H2, 'fro')^2);
        overall = norm1 + norm2 + norm3;
        fprintf('final obj: %.4f\n L1: %.4f,\n L2: %.4f,\n L3: %.4f,\n\n', overall, norm1, norm2, norm3);
        if (lastF - overall) < threshold
            fprintf('algorithm converges...\n');
            fprintf('final obj: %.4f\n L1: %.4f,\n L2: %.4f,\n L3: %.4f,\n\n', overall, norm1, norm2, norm3);
            return;
        end
        iter = iter + 1;
        lastF = overall;
    end
end

