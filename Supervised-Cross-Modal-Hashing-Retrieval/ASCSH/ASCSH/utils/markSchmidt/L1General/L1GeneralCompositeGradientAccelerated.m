function [w,fEvals] = L1GeneralProjectedSubGradient(gradFunc,w,lambda,params,varargin)

% Process input options
[verbose,maxIter,optTol,L] = ...
    myProcessOptions(params,'verbose',1,'maxIter',500,...
    'optTol',1e-6,'L',[]);

% Start log
if verbose
    fprintf('%10s %10s %15s %15s %15s %8s\n','Iteration','FunEvals','Step Length','Function Val','Opt Cond','Non-Zero');
end

p = length(w);

% Compute Evaluate Function
[f,g] = gradFunc(w,varargin{:});
funEvals = 1;

if isempty(L)
    alpha_max = 1;
    alpha = 1;
else
    alpha_max = L;
end
alpha = alpha_max;

a = 1;
z = w;
for i = 1:maxIter
    w_old = w;
    
    w_new = softThreshold(w-alpha*g,alpha,lambda);
    [f_new,g_new] = gradFunc(w_new,varargin{:});
    funEvals = funEvals+1;
    
    phi_T = f_new + sum(lambda.*(abs(w_new)));
    mL = f + g'*(w_new-w) + (w_new-w)'*(w_new-w)/(2*alpha) + sum(lambda.*(abs(w_new)));
    
    if phi_T > mL
        if verbose
        fprintf('Decreasing Lipschitz estimate\n');
        end
        alpha = alpha/2;
        
        w_new = softThreshold(w-alpha*g,alpha,lambda);
        [f_new,g_new] = gradFunc(w_new,varargin{:});
        funEvals = funEvals+1;
        
        phi_T = f_new + sum(lambda.*(abs(w_new)));
        mL = f + g'*(w_new-w) + (w_new-w)'*(w_new-w)/(2*alpha) + sum(lambda.*(abs(w_new)));
    end
    
    % Extrapolation step
    z_new = w_new;
    a_new = (1 + sqrt(1+4*a*a))/2;
    w_new = z_new + ((a-1)/a_new)*(z_new - z);
    [f_new,g_new] = gradFunc(w_new,varargin{:});
    funEvals = funEvals+1;
    
    a = a_new;
    z = z_new;
    w = w_new;
    f = f_new;
    g = g_new;
        
    if verbose
        fprintf('%6d %6d %8.5e %8.5e %8.5e %6d\n',i,funEvals,alpha,f+sum(lambda.*abs(w)),max(abs(w-softThreshold(w-g,1,lambda))),nnz(w));
    end
    
    if sum(abs(w-w_old)) < optTol
        break;
    end
    
end
end
%% Soft-threshold
function [w] = softThreshold(w,t,lambda)
w = sign(w).*max(0,abs(w)-lambda*t);
end


