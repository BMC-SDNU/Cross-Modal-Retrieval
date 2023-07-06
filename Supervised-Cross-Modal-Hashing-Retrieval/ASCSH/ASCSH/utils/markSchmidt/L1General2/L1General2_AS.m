function [w] = L1General2_AS(funObj,w,lambda,options)

%% Process Options
if nargin < 4
    options = [];
end

[verbose,optTol,progTol,maxIter,suffDec,corrections] = ...
    myProcessOptions(options,'verbose',1,'optTol',1e-5,'progTol',1e-9,...
    'maxIter',500,'suffDec',1e-4,'corrections',100);

if verbose
    fprintf('%6s %6s %12s %12s %12s %6s %6s %12s\n','Iter','fEvals','stepLen','fVal','optCond','nnz','addStp','wokingOpt');
end

%% Evaluate Initial Point
p = length(w);
[f,g] = pseudoGrad(funObj,w,lambda);
funEvals = 1;

% Check optimality
optCond = max(abs(g));
if optCond < optTol
    if verbose
        fprintf('First-order optimality satisfied at initial point\n');
    end
    return;
end

%% Main loop
i = 1;
W = lambda == 0 | w ~= 0; % Initial working set
addStep = 0;
oldAdd = [0 0];
while 1
    optCond = max(abs(g(W)));
    while optCond >= optTol
        
        % Compute direction
        d = zeros(p,1);
        if i == 1
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
            if verbose == 2
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
        
        % Update working set
        W = lambda == 0 | w ~= 0;

        % Check Optimality of working set
        optCond = max(abs(g(W)));
        
        % Output Log
        if verbose
            fprintf('%6d %6d %8.5e %8.5e %8.5e %6d %6d %8.5e\n',i,funEvals,t,f,max(abs(g)),nnz(w),addStep,optCond);
            addStep = 0;
        end
        i = i + 1;
        
        if optCond < optTol
            if verbose == 2
                fprintf('First-order optimality of working set below optTol\n');
            end
            break;
        end

        % Check for lack of progress
        if max(abs(t*d)) < progTol || abs(f-f_old) < progTol
            if verbose == 2
                fprintf('Progress in parameters or objective below progTol\n');
            end
            break;
        end
    end

    % Check for iteration limit
    if funEvals >= maxIter
        if verbose
            fprintf('Function evaluations reached maxIter\n');
        end
        break;
    end
    
    % Check Optimality
    optCond = max(abs(g));
    if optCond < optTol
        if verbose
            fprintf('First-order optimality below optTol\n');
        end
        break;
    end

    % Update working set
    [maxVal,maxInd] = max(abs(g));
    W = w ~=0;
    addStep = W(maxInd)==0;
    W(maxInd) = 1;
    
    if maxInd == oldAdd(2)
        if verbose
            fprintf('Not making any more progress by adding most violating variable\n');
        end
        break;
    end
    oldAdd = [maxInd oldAdd(1)];
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
