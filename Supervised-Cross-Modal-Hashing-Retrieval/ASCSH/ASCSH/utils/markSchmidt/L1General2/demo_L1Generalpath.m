%% Generate Some Synthetic Data
clear all

nInstances = 2000;
nVars = 10000;
sparsityFactor = .1;
flipFactor = .1;
X = [ones(nInstances,1) randn(nInstances,nVars-1)];
w = randn(nVars,1).*(rand(nVars,1) < sparsityFactor);
y = sign(X*w);
flipPos = rand(nInstances,1) < flipFactor;
y(flipPos) = -y(flipPos);
        
%% Optimize un-regularized variables with regularized variables fixed at 0
% (for logistic regression, only the bias is un-regularized and it has a
% closed-form solution)
w = zeros(nVars,1);

% Solve for bias
w(1) = log(sum(y==1)/sum(y==-1));

% If the solution for the un-regularized variables does not have a closed-form solution,
% you can do this step numerically as follows:
%w(lambda==0) = minFunc(@LogisticLoss,0,[],X(:,lambda==0),y);

fprintf('Optimal bias with no features: %f\n',w(1));

%% Compute gradient with unregularized variables set to their optimal
% value, to get the value of lambda that makes all other variables zero
[f,g] = LogisticLoss(w,X,y);
lambdaMax = max(abs(g));

fprintf('Maximum value of lambda: %f\n',lambdaMax);

%% Choose the sequence of lambda values we will solve for

lambdaValues = lambdaMax*[1:-.05:0];
nLambdaValues = length(lambdaValues);

%% Now solve the problems, using warm-starting and an active-set method
W = zeros(nVars,nLambdaValues);
lambdaVect = [0;ones(nVars-1,1)]; % Pattern of the lambda vector
for i = 1:nLambdaValues
    lambda = lambdaValues(i);
    fprintf('lambda = %f\n',lambda);
    
    % Compute free variables (uses w and g from above)
    free = lambdaVect == 0 | w ~= 0 | (w == 0 & (abs(g) > lambda));
        
    while 1
        % Solve with respect to free variables
        funObj = @(w)LogisticLoss(w,X(:,free),y);
        w(free) = L1General2_PSSgb(funObj,w(free),lambda*lambdaVect(free));
        
        % Compute new set of free variables
        [f,g] = LogisticLoss(w,X,y);
        free_new = lambdaVect == 0 | w ~= 0 | (w == 0 & (abs(g) > lambda));
        
        if any(free_new == 1 & free == 0)
            % We added a new free variable, re-optimize
            free = free_new;
        else
            % Optimal solution found
            break;
        end
    end
    
    % Record solution
    W(:,i) = w;
end

% Plot regularization path
fprintf('Regularization path calculation done, plotting...\n');
figure;
plot(lambdaValues,W);
legend('Bias');
ylim([-.1 .1]);
title('Regularization Path');
figure;
plot(lambdaValues,W);
xlim([lambdaValues(end) lambdaValues(1)]);
ylim([-.1 .1]);
title('Regularization Path (close to zero)');
