%% Generate Some Synthetic Data
clear all

nInstances = 200;
nVars = 250;
nTargets = 10;
sparsityFactor = .5;
flipFactor = .1;
X = [ones(nInstances,1) randn(nInstances,nVars-1)];
W = diag([0;rand(nVars-1,1) < sparsityFactor])*randn(nVars,nTargets);
Y = sign(X*W);
flipPos = rand(nInstances*nTargets,1) < flipFactor;
Y(flipPos) = -Y(flipPos);

% Set up groups
% (group 0 is not regularzed, we make the bias variables belong to group 0)
groups = repmat([0:nVars-1]',1,nTargets);
groups = groups(:);
nGroups = max(groups);

% Make colors for plotting
colors = rand(nVars,3);

%% Optimize un-regularized variables with regularized variables fixed at 0
% (for multi-task logistic regression, the biases are un-regularized and have a
% closed-form solution)
W = zeros(nVars,nTargets);

% Solve for bias
for t = 1:nTargets
    W(1,t) = log(sum(Y(:,t)==1)/sum(Y(:,t)==-1));
end

% If the solution for the un-regularized variables does not have a closed-form solution,
% you can do this step numerically:
W(1,:) = minFunc(@SimultaneousLogisticLoss,zeros(nTargets,1),[],X(:,1),Y);

fprintf('Optimal biases with no features:\n');
W(1,:)

%% Compute gradient with unregularized variables set to their optimal
% value, to get the value of lambda that makes all other variables zero
% (for the L2-group norm, this is the maximum L2-norm
% among the gradients of the groups)
[f,g] = SimultaneousLogisticLoss(W(:),X,Y);
gradNorms = sqrt(accumarray(groups(groups~=0),g(groups~=0).^2));
lambdaMax = max(gradNorms);

fprintf('Maximum value of lambda: %f\n',lambdaMax);

%% Choose the sequence of lambda values we will solve for

lambdaValues = lambdaMax*[1:-.05:0];
nLambdaValues = length(lambdaValues);

%% Now solve the problems, using warm-starting and an active-set method
Wpath = zeros(nVars,nTargets,nLambdaValues);

options.method = 'bbst';
for i = 1:nLambdaValues
    lambda = lambdaValues(i);
    fprintf('lambda = %f\n',lambda);
    
    % Compute free variables
    grpNorms = sqrt(accumarray(groups(groups~=0),W(groups~=0).^2));
    freeGroups = false(nGroups,1);
    freeGroups(grpNorms ~= 0) = 1;
    freeGroups(gradNorms > lambda) = 1;
    freeVars = repmat([true;freeGroups],[1 nTargets]);
    nSubGroups = sum(freeGroups);
    subGroups = repmat([0:nSubGroups]',1,nTargets);
    subGroups = subGroups(:);
    
    if i == 1
        % Code doesn't handle case of no groups, so we just use the
        % previously computed solution in this case
        Wpath(:,:,i) = W;
    else
        while 1
            % Solve with respect to free variables
            funObj = @(W)SimultaneousLogisticLoss(W,X(:,[true;freeGroups]),Y);
            W(freeVars) = L1GeneralGroup_SoftThresh(funObj,W(freeVars),lambda*ones(nSubGroups,1),subGroups,options);
            
            freeVars_old = freeVars;
            [f,g] = SimultaneousLogisticLoss(W(:),X,Y);
            gradNorms = sqrt(accumarray(groups(groups~=0),g(groups~=0).^2));
            
            % Compute free variables
            grpNorms = sqrt(accumarray(groups(groups~=0),W(groups~=0).^2));
            freeGroups = false(nGroups,1);
            freeGroups(grpNorms ~= 0) = 1;
            freeGroups(gradNorms > lambda) = 1;
            freeVars = repmat([true;freeGroups],[1 nTargets]);
            nSubGroups = sum(freeGroups);
            subGroups = repmat([0:nSubGroups]',1,nTargets);
            subGroups = subGroups(:);
            
            if ~any(freeVars == 1 & freeVars_old == 0)
                % Optimal solution found
                break;
            end
        end
        
        % Record solution
        Wpath(:,:,i) = W;
    end
end

%% Plot regularization path
fprintf('Regularization path calculation done, plotting...\n');
figure;
plot(lambdaValues,squeeze(Wpath(1,:,:)),'color',colors(1,:));
legend('Biases');
xlim([lambdaValues(end) lambdaValues(1)]);
title('Regularization Path (L2 Group Norm)');
hold on
for i = 2:nVars
    plot(lambdaValues,squeeze(Wpath(i,:,:)),'color',colors(i,:));
end
figure;
plot(lambdaValues,squeeze(Wpath(1,:,:)),'color',colors(1,:));
legend('Biases');
xlim([lambdaValues(end) lambdaValues(1)]);
ylim([-.1 .1]);
title('Regularization Path (L2 Group Norm, close to zero)');
hold on
for i = 2:nVars
    plot(lambdaValues,squeeze(Wpath(i,:,:)),'color',colors(i,:));
end
pause

%% ***************************************************
%
%
% Now do it all over again with the Linf group norm
%
%
% ****************************************************

%% Optimize un-regularized variables with regularized variables fixed at 0
% (for multi-task logistic regression, the biases are un-regularized and have a
% closed-form solution)
W = zeros(nVars,nTargets);

% Solve for bias
for t = 1:nTargets
    W(1,t) = log(sum(Y(:,t)==1)/sum(Y(:,t)==-1));
end

%% Compute gradient with unregularized variables set to their optimal
% value, to get the value of lambda that makes all other variables zero
% (for the Linf-group norm, this is the maximum L1-norm
% among the gradients of the groups)
[f,g] = SimultaneousLogisticLoss(W(:),X,Y);
for grp = 1:nGroups
    gradNorms(grp) = sum(abs(g(groups==grp)));
end
lambdaMax = max(gradNorms);

fprintf('Maximum value of lambda: %f\n',lambdaMax);

%% Choose the sequence of lambda values we will solve for

lambdaValues = lambdaMax*[1:-.05:0];
nLambdaValues = length(lambdaValues);

%% Now solve the problems, using warm-starting and an active-set method
Wpath = zeros(nVars,nTargets,nLambdaValues);

options.norm = inf;
options.testOpt = 0; % Reduce number of soft-threshold calculations (since Linf soft-threshold is not efficiently implemented)
for i = 1:nLambdaValues
    lambda = lambdaValues(i);
    fprintf('lambda = %f\n',lambda);
    
    % Compute free variables
    for grp = 1:nGroups
        grpNorms(grp) = max(abs(g(groups==grp)));
    end
    freeGroups = false(nGroups,1);
    freeGroups(grpNorms ~= 0) = 1;
    freeGroups(gradNorms > lambda) = 1;
    freeVars = repmat([true;freeGroups],[1 nTargets]);
    nSubGroups = sum(freeGroups);
    subGroups = repmat([0:nSubGroups]',1,nTargets);
    subGroups = subGroups(:);
    
    if i == 1
        % Code doesn't handle case of no groups, so we just use the
        % previously computed solution in this case
        Wpath(:,:,i) = W;
    else
        while 1
            % Solve with respect to free variables
            funObj = @(W)SimultaneousLogisticLoss(W,X(:,[true;freeGroups]),Y);
            W(freeVars) = L1GeneralGroup_SoftThresh(funObj,W(freeVars),lambda*ones(nSubGroups,1),subGroups,options);
            
            freeVars_old = freeVars;
            [f,g] = SimultaneousLogisticLoss(W(:),X,Y);
            for grp = 1:nGroups
                gradNorms(grp) = sum(abs(g(groups==grp)));
            end
            
            % Compute free variables
            for grp = 1:nGroups
                grpNorms(grp) = max(abs(g(groups==grp)));
            end
            freeGroups = false(nGroups,1);
            freeGroups(grpNorms ~= 0) = 1;
            freeGroups(gradNorms > lambda) = 1;
            freeVars = repmat([true;freeGroups],[1 nTargets]);
            nSubGroups = sum(freeGroups);
            subGroups = repmat([0:nSubGroups]',1,nTargets);
            subGroups = subGroups(:);
            
            if ~any(freeVars == 1 & freeVars_old == 0)
                % Optimal solution found
                break;
            end
        end
        
        % Record solution
        Wpath(:,:,i) = W;
    end
end

%% Plot regularization path
fprintf('Regularization path calculation done, plotting...\n');
figure;
plot(lambdaValues,squeeze(Wpath(1,:,:)),'color',colors(1,:));
legend('Biases');
xlim([lambdaValues(end) lambdaValues(1)]);
title('Regularization Path (Linf Group Norm)');
hold on
for i = 2:nVars
    plot(lambdaValues,squeeze(Wpath(i,:,:)),'color',colors(i,:));
end
figure;
plot(lambdaValues,squeeze(Wpath(1,:,:)),'color',colors(1,:));
legend('Biases');
xlim([lambdaValues(end) lambdaValues(1)]);
ylim([-.1 .1]);
title('Regularization Path (Linf Group Norm, close to zero)');
hold on
for i = 2:nVars
    plot(lambdaValues,squeeze(Wpath(i,:,:)),'color',colors(i,:));
end