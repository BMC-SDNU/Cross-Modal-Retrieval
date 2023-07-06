% demo of our CVPR paper
% X -- 1st modality dx * N
% Y -- 2nd modality dy * N
% bit -- the number of hash bit
% opt -- the parameter variable
% author : Lynn Liu
% Email : lynnliu.xmu@gmail.com

function [Wx, Wy, B] = trainFSH(X, Y, bit, opt)

lam = opt.lam;
Nsamp = opt.Nsamp;
iter = opt.iter;
lambda = opt.lambda;
k = opt.k;

yeta = 0.5;

N = size(X,2);
dx = size(X,1);
dy = size(Y,1);

if opt.km;
    
    rp = randperm(N);
    rp = rp(1:Nsamp);
    anchorX = X(:,rp); anchorY = Y(:,rp);
else
    N = size(X,2);
    nClass = Nsamp;
    
    fea = [X;Y];
    [~, center] = litekmeans(fea', nClass, 'MaxIter', 10);
    anchorX = center(:,1:size(X,1)); anchorX = anchorX';
    anchorY = center(:,size(X,1)+1:end); anchorY = anchorY';

end
%%
if k == 0
    A = EuDist2(X',anchorX',0);
    A = exp(-A/(2*1^2));
    B = EuDist2(Y',anchorY',0);
    B = exp(-B/(2*1^2));
    D = yeta * A + (1-yeta) * B; 
    clear A B;
else
    D1 = EuDist2(X',anchorX',1);
    [~,indx] = sort(D1,2);
    indx = indx(:,1:k);
    clear D1;
    D2 = EuDist2(Y',anchorY',1);
    [~,indy] = sort(D2,2);
    indy = indy(:,1:k);
    clear D2;
    D3 = EuDist2(anchorX',anchorX',1);
    [~,indax] = sort(D3,2);
    indax = indax(:,1:k);
    D4 = EuDist2(anchorY',anchorY',1);
    [~,inday] = sort(D4,2);
    inday = inday(:,1:k);
    D3 = exp(-D3/(2*1^2)); D4 = exp(-D4/(2*1^2));
    
    for m1 = 1:N
        for m2 = 1:Nsamp
            s1 = (1/k^2) * sum(sum(D3(indx(m1,:),inday(m2,:))));
            s2 = (1/k^2) * sum(sum(D4(indy(m1,:),indax(m2,:))));
            A1(m1,m2) = s1;
            A2(m1,m2) = s2;
        end
    end
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if opt.cca
    Cxx = X*X' + 1e-6*eye(dx);
    Cyy = Y*Y' + 1e-6*eye(dy);
    Cxy = X*Y';
    [Wx, Wy, ~] = trainCCA(Cxx, Cyy, Cxy, bit);
    B = (Wy' * Y);
    Bs = (Wy' * anchorY);
else
    Wx = randn(dx,bit);
    Wy = randn(dy,bit);
    B = (Wy' * Y);
    Bs = (Wy' * anchorY);
end
D = [];
for i = 1:iter
    D = yeta * A1 + (1-yeta) * A2;
    fprintf('iteration number : %d\n',i);
    D1 = Bs * D' + 2 * lambda * (Wx' * X + Wy' * Y);
    B = sign(D1);
    % D2 = Bs * Bs' + 2 * lambda * eye(bit,bit);
    % B = sign(D2\D1);
    Bs = sign(B * D);
    Wx = (X * X') \ X * B';
    Wy = (Y * Y') \ Y * B';
    

    gg = 1/(1-lam);
    P1 = trace(D3 - Bs'*B*A1); P2 = trace(D4 - Bs'*B*A2);
    yeta = (lam * P1)^(gg) / ( lam * P1 + lam * P2)^(gg);
 
end

end
