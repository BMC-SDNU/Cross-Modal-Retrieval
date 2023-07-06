function [w] = L1General2_AS(funObj,w,lambda,options)

%% Process Options
if nargin < 4
    options = [];
end

[verbose,optTol,progTol,maxIter,L] = ...
    myProcessOptions(options,'verbose',1,'optTol',1e-5,'progTol',1e-9,...
    'maxIter',500,'L',[]);

if verbose
    fprintf('%6s %6s %12s %12s %12s %6s\n','Iter','fEvals','stepLen','fVal','optCond','nnz');
end

%% Evaluate Initial Point
p = length(w);
w = [w.*(w>0);-w.*(w<0)];
[f,g] = nonNegGrad(funObj,w,p,lambda);
funEvals = 1;

optCond = max(abs(nonNegProject(w - g)-w));
if optCond < optTol
    if verbose
        fprintf('First-order optimality satisfied at initial point\n');
    end
    return;
end

%% Main loop

mu = 0;
gamma = 1;
alphap = 1;

if isempty(L)
    t = min(1,1/sum(abs(g)));
    L = 1/t;
end
y = w;
f_y = f;
g_y = g;
for i = 0:maxIter
    
    while 1
        b=-gamma+ mu;
        alpha= (b+ sqrt(b*b + 4* L * gamma)) / (2*L);
        beta= (gamma - gamma* alphap) / (alphap * gamma + alphap* L * alpha);
        
        if i > 0
            y = w + beta*(w - w_old);
            [f_y,g_y] = nonNegGrad(funObj,y,p,lambda);
            funEvals = funEvals+1;
        end
        
        w_new = nonNegProject(y - g_y/L);
        [f_new,g_new] = nonNegGrad(funObj,w_new,p,lambda);
        funEvals = funEvals+1;
        
        % Backtrack if not below Lipschitz constant
        l_sum = f_new - f_y - g_y'*(w_new-y);
        r_sum = (1/2)*(w_new-y)'*(w_new-y);
        if l_sum <= r_sum*L
            break;
        else
            if verbose
                fprintf('Decreasing Step Size\n');
            end
            L = max(2*L,l_sum/r_sum);
            
            if max(abs(g_y/L)) < progTol
               if verbose 
                   fprintf('Line search is dead\n');
               end
               w = w(1:p)-w(p+1:end);
               return
            end
        end
    end
    
    % Take step
    f_old = f;
    g_old = g;
    w_old = w;
    w = w_new;
    f = f_new;
    g = g_new;
    
    optCond = max(abs(nonNegProject(w - g)-w));
    
    % Output Log
    if verbose
        fprintf('%6d %6d %8.5e %8.5e %8.5e %6d\n',i+1,funEvals,alpha,f,optCond,nnz(w(1:p)-w(p+1:end)));
    end
    
    if optCond < optTol
        if verbose
            fprintf('First-order optimality below optTol\n');
        end
        break;
    end
    
    % Check for lack of progress
    if max(abs(w-w_old)) < progTol || abs(f-f_old) < progTol
        if verbose
            fprintf('Progress in parameters or objective below progTol\n');
        end
        break;
    end
    
    % Check for iteration limit
    if funEvals >= maxIter
        if verbose
            fprintf('Function evaluations reached maxIter\n');
        end
        break;
    end
    
    gamma=L* alpha* alpha;    alphap=alpha;
    ratio=L / (l_sum/ r_sum);
    
    if (ratio > 5)
        if verbose
            fprintf('Increasing step size\n');
        end
        L=L*0.8;
    end
end
w = w(1:p)-w(p+1:end);

end

%% Psuedo-gradient calculation
function [f,g,H] = nonNegGrad(funObj,w,p,lambda)

[f,g] = funObj(w(1:p)-w(p+1:end));

f = f + sum(lambda.*w(1:p)) + sum(lambda.*w(p+1:end));

g = [g;-g] + [lambda.*ones(p,1);lambda.*ones(p,1)];
end

function [w] = nonNegProject(w)
w(w < 0) = 0;
end
