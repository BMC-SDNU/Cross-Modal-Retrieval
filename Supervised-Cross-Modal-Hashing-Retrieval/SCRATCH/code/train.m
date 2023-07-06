function [Wx, Wy, R, B] = train(X, Y, param, L)

% fprintf('training...\n');

%% set the parameters
nbits = param.nbits;
lambdaX = param.lambdaX;
lambdaY = 1-lambdaX;
alpha = param.alpha;
gamma = param.gamma;
Xmu = param.Xmu;
Ymu = Xmu;

%% get the dimensions
[n, dX] = size(X);
dY = size(Y,2);

%% transpose the matrices
X = X'; Y = Y'; L = L';

%% initialization
V = randn(nbits, n);
Wx = randn(nbits, dX);
Wy = randn(nbits, dY);
R = randn(nbits, nbits);
[U11, ~, ~] = svd(R);
R = U11(:,1:nbits);

%% iterative optimization
for iter = 1:param.iter

    % update B
    B = -1*ones(nbits,n);
    B((R*V)>=0) = 1;

    % update G
    Ux = lambdaX*(X*V')/(lambdaX*(V*V')+gamma*eye(nbits));
    Uy = lambdaY*(Y*V')/(lambdaY*(V*V')+gamma*eye(nbits));
    G = alpha*(L*V')/(alpha*(V*V')+gamma*eye(nbits));

    % update W
    Wx = Xmu*(V*X')/(Xmu*(X*X')+gamma*eye(dX));
    Wy = Ymu*(V*Y')/(Ymu*(Y*Y')+gamma*eye(dY));

    % update V
    V = (lambdaX*(Ux'*Ux)+lambdaY*(Uy'*Uy)+alpha*(G'*G)+(R'*R)+(Xmu+Ymu+gamma)*eye(nbits))\(lambdaX*(Ux'*X)+lambdaY*(Uy'*Y)+Xmu*(Wx*X)+Ymu*(Wy*Y)+alpha*(G'*L)+(R'*B));

    % update R
    [S1, ~, S2] = svd(B*V');
    R = S1*S2';

    %objective function
    J = lambdaX*norm(X-Ux*V,'fro')^2+lambdaY*norm(Y-Uy*V,'fro')^2+alpha*norm(L-G*V,'fro')^2+norm(B-R*V,'fro')^2+Xmu*norm(V-Wx*X,'fro')^2+Ymu*norm(V-Wy*Y,'fro')^2;
    %fprintf('objective function value @ iter %d: %f\n', iter, J);
end
