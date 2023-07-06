function w = L1GeneralGroup_Auxiliary(funObj,w,lambda,groups,options)
% w = L1GeneralGroup_Auxiliary(funObj,w,lambda,groups,options)

[norm,method] = myProcessOptions(options,'norm',2,'method','spg');
p = length(w);

%% Introduce auxiliary variables
w = [w;sqrt(accumarray(groups(groups~=0),w(groups~=0).^2))];

%% Make augmented objective function
auxObj = @(w)auxGroupLoss(w,groups,lambda,funObj);

%% Make projection function
[groupStart,groupPtr] = groupl1_makeGroupPointers(groups);
if norm==2
    %funProj = @(w)auxGroupL2Project(w,p,groupStart,groupPtr);
    groupStart = int32(groupStart-1);
    groupPtr = int32(groupPtr-1);
    funProj = @(w)auxGroupL2ProjectC(w,groupStart,groupPtr);
elseif norm == inf
    %funProj = @(w)auxGroupLinfProject(w,p,groupStart,groupPtr);
    groupStart = int32(groupStart-1);
    groupPtr = int32(groupPtr-1);
    funProj = @(w)auxGroupLinfProjectC(w,groupStart,groupPtr);
elseif norm == 0
    funProj = @(w)auxGroupTraceProject(w,p,groupStart,groupPtr);
end

%% Solve
if strcmp(method,'spg')
w = minConf_SPG(auxObj,w,funProj,options);
elseif strcmp(method,'opg')
w = minConf_OPG(auxObj,w,funProj,options);
else
w = minConf_PQN(auxObj,w,funProj,options);
end

%% Truncate to original variables
w = w(1:p);
