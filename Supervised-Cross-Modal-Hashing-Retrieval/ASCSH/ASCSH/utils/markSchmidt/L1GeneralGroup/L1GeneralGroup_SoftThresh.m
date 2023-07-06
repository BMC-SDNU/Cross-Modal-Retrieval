function [w] = L1GeneralGroup_BBST(funObj,w,lambda,groups,options)

[norm,method] = myProcessOptions(options,'norm',2,'method','bbst');
p = length(w);

%% Make regularizer function
if norm == 2
    funReg = @(w)groupL12regularizer(w,lambda,groups);
elseif norm == inf
    funReg = @(w)groupL1infregularizer(w,lambda,groups);
elseif norm == 0
    funReg = @(w)groupL1Tregularizer(w,lambda,groups);
end

%% Make threshold function
if norm == 2
    funThresh = @(w,t)groupSoftThreshold(w,t,lambda,groups);
elseif norm == inf
    funThresh = @(w,t)groupInfSoftThreshold(w,t,lambda,groups);
elseif norm == 0
   funThresh = @(w,t)groupTraceSoftThreshold(w,t,lambda,groups); 
end
%% Solve
if strcmp(method,'bbst')
    w = minConf_BBST(funObj,funReg,w,funThresh,options);
else
    w = minConf_QNST(funObj,funReg,w,funThresh,options);
end

%% Truncate to original variables
w = w(1:p);
