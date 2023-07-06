function [w] = L1General2_SpaRSA(funObj,w,lambda,options)

%% Process Options
if nargin < 4
    options = [];
end

[verbose,optTol,progTol,maxIter,suffDec,memory] = ...
    myProcessOptions(options,'verbose',1,'optTol',1e-5,'progTol',1e-9,...
    'maxIter',500,'suffDec',1e-4,'memory',10);

if verbose
    fprintf('%6s %6s %12s %12s %12s %6s\n','Iter','fEvals','stepLen','fVal','optCond','nnz');
end

%% Evaluate Initial Point
p = length(w);
[f,g] = funObj(w);
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
    if i == 1
        t = min(1,1/sum(abs(g)));
        old_fvals = repmat(-inf,[memory 1]);
        old_fvals(1) = f;
        fr = f;
        alpha = 1;
    else
        y = g-g_old;
        s = w-w_old;

        alpha = (y'*s)/(y'*y);
        if alpha <= 1e-10 || alpha > 1e10
            fprintf('BB update is having some trouble, implement fix!\n');
            pause;
        end
        t = 1;

        if i <= memory
            old_fvals(i) = f;
        else
            old_fvals = [old_fvals(2:end);f];
        end
        fr = max(old_fvals);
    end
    f_old = f;
    g_old = g;
    w_old = w;
    
    d = softThreshold(w-alpha*g,alpha,lambda)-w;
    w_new = w + t*d;
    
    % Compute directional derivative, check that we can make progress
    pg = pseudoGrad(w,g,lambda);
    gtd = pg'*d;
    if gtd > -progTol
        if verbose
            fprintf('Directional derivative below progTol\n');
        end
        break;
    end
    
    % Compute soft-thresholded point
    [f_new,g_new] = funObj(w_new);
    f_new = f_new + sum(lambda.*abs(w_new));
    funEvals = funEvals+1;
    
    % Line search along projection arc
    while f_new > fr + suffDec*pg'*(w_new-w) || ~isLegal(f_new)
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
        [f_new,g_new] = funObj(w_new);
        f_new = f_new + sum(lambda.*abs(w_new));
        funEvals = funEvals+1;
    end
    
    % Take step
    w = w_new;
    f = f_new;
    g = g_new;
    
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