function [w] = L1General2_AS(funObj,w,lambda,options)

%% Process Options
if nargin < 4
    options = [];
end

[verbose,optTol,progTol,maxIter,suffDec,corrections] = ...
    myProcessOptions(options,'verbose',1,'optTol',1e-5,'progTol',1e-9,...
    'maxIter',500,'suffDec',1e-4,'corrections',100);

if verbose
    fprintf('%6s %6s %12s %12s %12s %6s\n','Iter','fEvals','stepLen','fVal','optCond','nnz');
end

%% Evaluate Initial Point
p = length(w);
w = [w.*(w>0);-w.*(w<0)];
[f,g] = nonNegGrad(funObj,w,p,lambda);
funEvals = 1;

% Compute working set and check optimality
W = (w~=0) | (g < 0);
optCond = max(abs(g(W)));
if optCond < optTol
    if verbose
        fprintf('First-order optimality satisfied at initial point\n');
    end
    w = w(1:p)-w(p+1:end);
    return;
end

%% Main loop
for i = 1:maxIter
    
    % Compute direction
    d = zeros(2*p,1);
    if i == 1
        d(W) = -g(W);
        Y = zeros(2*p,0);
        S = zeros(2*p,0);
        sigma = 1;
        t = min(1,1/sum(abs(g)));
    else
        y = g-g_old;
        s = w-w_old;
        
        correctionsStored = size(Y,2);
        if correctionsStored < corrections
            Y(:,correctionsStored+1) = y;
            S(:,correctionsStored+1) = s;
        else
            Y = [Y(:,2:corrections) y];
            S = [S(:,2:corrections) s];
        end
        
        ys = y'*s;
        if ys > 1e-10
            sigma = ys/(y'*y);
        end
        
        curvSat = sum(Y(W,:).*S(W,:)) > 1e-10;
        
        d(W) = lbfgsC(-g(W),S(W,curvSat),Y(W,curvSat),sigma);
        t = 1;
    end
    f_old = f;
    g_old = g;
    w_old = w;
    
    % Compute directional derivative, check that we can make progress
    gtd = g'*d;
    if gtd > -progTol
        if verbose
            fprintf('Directional derivative below progTol\n');
        end
        break;
    end
    
    % Compute projected point
    w_new = nonNegProject(w+t*d);
    [f_new,g_new] = nonNegGrad(funObj,w_new,p,lambda);
    funEvals = funEvals+1;
    
    % Line search along projection arc
    while f_new > f + suffDec*g'*(w_new-w) || ~isLegal(f_new)
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
            t = polyinterp([0 f gtd; t f_new g_new'*d]);
        end
        
        % Adjust if interpolated value near boundary
        if t < t_old*1e-3
            if verbose == 3
                fprintf('Interpolated value too small, Adjusting\n');
            end
            t = t_old*1e-3;
        elseif t > t_old*0.6
            if verbose == 3
                fprintf('Interpolated value too large, Adjusting\n');
            end
            t = t_old*0.6;
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
        
        % Compute projected point
        w_new = nonNegProject(w+t*d);
        [f_new,g_new] = nonNegGrad(funObj,w_new,p,lambda);
        funEvals = funEvals+1;
    end
    
    % Take step
    w = w_new;
    f = f_new;
    g = g_new;
    
    % Compute new working set
    W = (w~=0) | (g < 0);
    
    % Output Log
    if verbose
        fprintf('%6d %6d %8.5e %8.5e %8.5e %6d\n',i,funEvals,t,f,max(abs(g(W))),nnz(w(1:p)-w(p+1:end)));
    end
    
    % Check Optimality
    optCond = max(abs(g(W)));
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
