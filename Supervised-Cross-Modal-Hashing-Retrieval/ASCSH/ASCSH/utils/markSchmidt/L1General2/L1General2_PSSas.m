function [w] = L1General2_AS(funObj,w,lambda,options)

%% Process Options
if nargin < 4
    options = [];
end

[verbose,optTol,progTol,maxIter,suffDec,corrections,K] = ...
    myProcessOptions(options,'verbose',1,'optTol',1e-5,'progTol',1e-9,...
    'maxIter',500,'suffDec',1e-4,'corrections',100,'K',[]);

if verbose
    fprintf('%6s %6s %12s %12s %12s %6s %6s\n','Iter','fEvals','stepLen','fVal','optCond','nnz','addStp');
end

%% Evaluate Initial Point
p = length(w);
[f,g] = pseudoGrad(funObj,w,lambda);
funEvals = 1;

if isempty(K)
    K = p;
end

% Check optimality
optCond = max(abs(g));
if optCond < optTol
    if verbose
        fprintf('First-order optimality satisfied at initial point\n');
    end
    return;
end

%% Main loop
for i = 1:maxIter
    
    % Compute preliminary working set (non-zero and unregularized vars)
    W = lambda == 0 | w ~=0;
    
    % Compute direction
    d = zeros(p,1);
    adds = 0;
    if i == 1
        % On first iteration Hessian approximation is identity, 
        % so will have all k largest variables satisfying sign condition
        [sorted sortedInd] = sort(abs(g),'descend');
        adds = sum(W(sortedInd(1:K)) == 0 & g(sortedInd(1:K)) ~= 0);
        W(sortedInd(1:K)) = 1;
        
        d(W) = -g(W);
        Y = zeros(p,0);
        S = zeros(p,0);
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
        
        % The current d is fine, now do a binary search for a value  
        % k <= K such that the sign condition is satisfied for k but not for
        % k+1
        [sorted sortedInd] = sort(abs(g),'descend');
        LB = 0;
        UB = K+1;
        while UB-LB ~= 1
           k = ceil((UB+LB)/2);
           if g(sortedInd(k)) == 0
              %fprintf('Variable should not move away from zero\n');
              UB = k;
           else
               W_new = W;
               W_new(sortedInd(1:k)) = 1;
               d_new = zeros(p,1);
               curvSat = sum(Y(W_new,:).*S(W_new,:)) > 1e-10;
               d_new(W_new) = lbfgsC(-g(W_new),S(W_new,curvSat),Y(W_new,curvSat),sigma);
               if any(sign(d_new(w==0)).*sign(-g(w==0)) == -1)
                   %fprintf('Sign condition violated\n');
                   UB = k;
               else
                   %fprintf('Sign condition satisfied\n');
                   d = d_new;
                   adds = sum(W~=W_new);
                   LB = k;
               end
               
           end
        end
    end
    f_old = f;
    g_old = g;
    w_old = w;
    
    % Compute desired orthant
    xi = sign(w);
    xi(w==0) = sign(-g(w==0));
    
    % Compute directional derivative, check that we can make progress
    gtd = g'*d;
    if gtd > -progTol
        if verbose
            fprintf('Directional derivative below progTol\n');
        end
        break;
    end
    
    % Compute projected point
    w_new = orthantProject(w+t*d,xi);
    [f_new,g_new] = pseudoGrad(funObj,w_new,lambda);
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
        w_new = orthantProject(w+t*d,xi);
        [f_new,g_new] = pseudoGrad(funObj,w_new,lambda);
        funEvals = funEvals+1;
    end
    
    % Take step
    w = w_new;
    f = f_new;
    g = g_new;
    
    % Output Log
    if verbose
        fprintf('%6d %6d %8.5e %8.5e %8.5e %6d %6d\n',i,funEvals,t,f,max(abs(g)),nnz(w),adds);
    end
    
    % Check Optimality
    optCond = max(abs(g));
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

%% Psuedo-gradient calculation
function [f,pGrad,H] = pseudoGrad(funObj,w,lambda)

[f,g] = funObj(w);

f = f + sum(lambda.*abs(w));

pGrad = zeros(size(g));
pGrad(g < -lambda) = g(g < -lambda) + lambda(g < -lambda);
pGrad(g > lambda) = g(g > lambda) - lambda(g > lambda);
nonZero = w~=0 | lambda==0;
pGrad(nonZero) = g(nonZero) + lambda(nonZero).*sign(w(nonZero));

end

function [w] = orthantProject(w,xi)
w(sign(w) ~= xi) = 0;
end
