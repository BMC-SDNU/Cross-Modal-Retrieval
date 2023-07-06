function [w] = L1General2_DSST(funObj,w,lambda,options)

%% Process Options
if nargin < 4
    options = [];
end

[verbose,optTol,progTol,maxIter,suffDec,quadraticInit] = ...
    myProcessOptions(options,'verbose',1,'optTol',1e-5,'progTol',1e-9,...
    'maxIter',500,'suffDec',1e-4,'quadraticInit',0);

if verbose
    fprintf('%6s %6s %12s %12s %12s %6s\n','Iter','fEvals','stepLen','fVal','optCond','nnz');
end

%% Evaluate Initial Point
p = length(w);
[f,g,D] = funObj(w);
f = f + sum(lambda.*abs(w));
funEvals = 1;

% Check optimality
optCond = max(abs(w-softThreshold(w-g,1,lambda)));
if optCond < optTol
    if verbose
        fprintf('First-order optimality satisfied at initial point\n');
    end
    return;
end

%% Main loop
for i = 1:maxIter
    
    % Compute direction
    w_new = softThreshold(w-g./D,1,lambda./D);
    d = w_new-w;
    
    % Compute directional derivative, check that we can make progress
    pg = pseudoGrad(w,g,lambda);
    gtd = pg'*d;
    if gtd > -progTol
        if verbose
            fprintf('Directional derivative below progTol\n');
        end
        break;
    end
    
    t = 1;
    if quadraticInit
        if i > 1
            t = min(1,2*(f-f_prev)/gtd);
            w_new = w+t*d;
        end
        f_prev = f;
    end
    
    % Compute objective at new point
    [f_new,g_new,D_new] = funObj(w_new);
    f_new = f_new + sum(lambda.*abs(w_new));
    funEvals = funEvals+1;
    
    % Line search along projection arc
    f_old = f;
    while f_new > f + suffDec*pg'*(w_new-w) || ~isLegal(f_new)
        t_old = t;
        
        % Backtracking
        if verbose
            fprintf('Backtracking...\n');
        end
        if ~isLegal(f_new)
            if verbose
                fprintf('Halving Step Size\n');
            end
            t = .5*t;
        else
            pg = pseudoGrad(w_new,g_new,lambda);
            t = polyinterp([0 f gtd; t f_new pg'*d]);
        end
        
        t = .5*t;
        
        % Check whether step has become too small
        if max(abs(t*d)) < progTol
            if verbose
                fprintf('Step too small in line search\n');
            end
            t = 0;
            w_new = w;
            f_new = f;
            g_new = g;
            break;
        end
        
        % Compute soft-thresholded point
        w_new = w+t*d;
        [f_new,g_new,D_new] = funObj(w_new);
        f_new = f_new + sum(lambda.*abs(w_new));
        funEvals = funEvals+1;
    end
    
    % Take step
    w = w_new;
    f = f_new;
    g = g_new;
    D = D_new;
    
    % Output Log
    if verbose
        fprintf('%6d %6d %8.5e %8.5e %8.5e %6d\n',i,funEvals,t,f,max(abs(w-softThreshold(w-g,1,lambda))),nnz(w));
    end
    
    % Check Optimality
    optCond = max(abs(w-softThreshold(w-g,1,lambda)));
    if optCond < optTol
        if verbose
            fprintf('First-order optimality below optTol\n');
        end
        break;
    end
    
    % Check for lack of progress
    if max(abs(t*d)) < progTol || abs(f-f_old) < progTol
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
end

end

%% Soft-threshold
function [w] = softThreshold(w,t,lambda)
w = sign(w).*max(0,abs(w)-lambda*t);
end

%% Pseud-gradient
function [pGrad] = pseudoGrad(w,g,lambda)
pGrad = zeros(size(g));
pGrad(g < -lambda) = g(g < -lambda) + lambda(g < -lambda);
pGrad(g > lambda) = g(g > lambda) - lambda(g > lambda);
nonZero = w~=0 | lambda==0;
pGrad(nonZero) = g(nonZero) + lambda(nonZero).*sign(w(nonZero));
end
