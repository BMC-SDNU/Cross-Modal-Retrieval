function [model,params] = loreta_similarity_margin(A,B,X,Y,L,start_iter,data_tri,params)
%
% Input:
% -- A                  init A (W = A*B') (dx x k) matrix, where k is the model rank.
% -- B                  init B (W = A*B') (dy x k) matrix, where k is the model rank.
% -- X                  (dx x N) data matrix of the first modality, where N is the number of samples.
% -- Y                  (dy x N) data matrix of the second modality, where N is the number of samples.
% -- L                  (c x N) label matrix, where N is the number of samples.
% -- start_iter         starting point in training triplets
% -- data_tri           training triplets
%
% Parameters:
% -- params.step_size   step size
% -- params.batch_size  batch size
% -- params.rank        rank of model
% -- params.dir         direction
% -- params.base_margin beta in Eqn(4)
% -- params.alph        alpha in Eqn(4)
%
% Output:
% -- model              model.A, model.B
% -- params

% Initialize pseudo inverses - assumes A and B are of rank k
Apinv = (A'*A)\A';
Bpinv = (B'*B)\B';
%---------

num_steps = params.batch_size;
alph = params.alph;
loss = zeros(1,num_steps);

for i = start_iter+1 : start_iter+num_steps
    % get a query/positive/negative triplet
    q_ind = data_tri(i, 1);
    p_ind = data_tri(i, 2);
    n_ind = data_tri(i, 3);
    %---------------
    if params.dir == 1
        x1 = X(:,q_ind);
        x2 = Y(:,p_ind)-Y(:,n_ind);
        margin_f = sqrt((Y(:,p_ind)-Y(:,n_ind))'*(Y(:,p_ind)-Y(:,n_ind)));
        margin_l = sum(abs(L(:,p_ind)-L(:,n_ind)));
        margin = params.base_margin*(alph*margin_f + (1-alph)*margin_l);
    elseif params.dir == 2
        x1 = X(:,p_ind)-X(:,n_ind);
        x2 = Y(:,q_ind);
        margin_f = sqrt((X(:,p_ind)-X(:,n_ind))'*(X(:,p_ind)-X(:,n_ind)));
        margin_l = sum(abs(L(:,p_ind)-L(:,n_ind)));
        margin = params.base_margin*(alph*margin_f + (1-alph)*margin_l);
    end
    
    pred = (x1'*A)*(B'*x2);
    if pred<margin
        loss(i-start_iter) = 1;
        [A,B,Apinv,Bpinv] = online_step_loreta1(A,B,Apinv,Bpinv,...
            x1,x2,1,params.step_size);
    end
end

% Set output
model.A = A;
model.B = B;
model.loss = loss;

end

function [Z1,Z2,Apinv,Bpinv] = online_step_loreta1(A,B,Apinv,Bpinv,x1,x2,y,t)
% [Z1,Z2,Apinv,Bpinv] = online_step_loreta1(A,B,Apinv,Bpinv,x1,x2,y,t)
%
% This function implements Loreta-1 from the paper:
% "Online Learning in the Manifold of Low-rank Matrices", Uri Shalit,
% Daphna Weinshall and Gal Chechik, NIPS 2010
%
%
%
% Do retraction step given the gradient: Z1*Z2' = R_{AB}(t*y*A*B')
% The gradient is represented as y*x1*x2' (a matrix)
%
% Inputs:
% A,B          - The low-rank factors of the current model. A is (n X k), B is (m X k)
% Apinv, Bpinv - The pseudo-inverses of A and B
% x1, x2       - The factors of the rank-1 gradient matrix. x1 is (n X 1), x2 is (m X 1)
% t            - The step size
% y            - The sign of the step, either -1 or 1
%
% Outputs:
% Z1, Z2       - The low-rank factors of the model after the retractions step
% Apinv, Bpinv - The pseudo-inverses of Z1, Z2 respectively

x2temp = t*y*x2;
temp1 = x2temp'*(Bpinv'*(Apinv*x1)); %this is a scalar if x1, x2 are rank-1

UUTx1 = A*(Apinv*x1);
cA =  UUTx1*(-(1/2)+(3/8)*(temp1) ) + x1*(1-(1/2)*temp1) ; %nx1
dA =  Bpinv*x2temp; %kx1
Z1 = A + cA*dA';

x2VVT = (x2temp'*B)*Bpinv;
cB = Apinv*x1; %kx1
dB = ((-1/2)+(3/8)*temp1)*x2VVT' + (1-(1/2)*temp1)*x2temp; %nx1
Z2 = B + dB*cB';

Apinv = rank1_pinv_update(A,Apinv,cA,dA);
Bpinv = rank1_pinv_update(B,Bpinv,dB,cB);

end

function [Apinv_new] = rank1_pinv_update(A,Apinv,c,d)
%
% A is nxk, c is nx1 and d is kx1
% Apinv_new is the pseudoinverse of A+c*d'
% ref.: Carl D Mayer, "Generalized inversion of modified matrices"

beta = 1+d'*(Apinv*c);
v = Apinv*c;
n = Apinv'*d;
w = c-A*(Apinv*c);
%m = d-A'*(Apinv'*d);   we deal only with full column rank matrices, therefore norm_m
%should be always zero

norm_w = w'*w;
%norm_m = m'*m;
norm_m = 0; %we deal only with full column rank matrices, therefore norm_m should be always zero
norm_v = v'*v;
norm_n = n'*n;

if abs(beta)>eps && norm_m<eps
    G = (Apinv*n)/beta;
    G = G*w';
    scalar = (beta/(norm_w*norm_n+beta^2));
    temp = (norm_w/beta)*(Apinv*n);
    temp = scalar*(temp+v);
    temp = temp*(norm_n*w/beta + n)';
    G = G-temp;
elseif norm_w>eps &&  abs(beta)<eps && norm_m<eps
    % G = -(Apinv*n)*n'/norm_n-v*w'/norm_w;
    G = -Apinv*(n/norm_n);
    G = G*n';
    temp = v*(w'/norm_w);
    G = G-temp;
elseif norm_w>eps && norm_m>eps
    G = -v*w'/norm_w-m*n'/norm(m)+beta*m*w'/(norm_w*norm_m);
elseif norm_w<eps && norm_m>eps && abs(beta)<eps
    G = -v*(v'*Apinv)/norm_v-m*n'/norm(m);
elseif norm_w<eps && abs(beta)>eps
    G = m*(v'*Apinv)/beta - (beta/(norm_v*norm_m+beta^2))* (norm_v*m/beta+v)*( (norm_m/beta)*(Apinv'*v)+n)';
elseif norm_w<eps && norm_m<eps && abs(beta)<eps
    G = -v*(v'*Apinv)/norm_v - n*(n'*Apinv)/norm_n + ((v'*(Apinv*n))/(norm_v*norm_n))*v*n';
else
    error('something is wrong: norm_w=%g, norm_v=%g, norm_m=%g, norm_n=%g, and beta=%g',norm_w,norm_v,norm_m,norm_n,beta);
end

Apinv_new = Apinv+G;

end
