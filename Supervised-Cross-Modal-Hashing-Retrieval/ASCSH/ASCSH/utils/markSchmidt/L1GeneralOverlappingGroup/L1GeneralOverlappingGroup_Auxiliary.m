function w = L1GeneralOverlappingGroup_Auxiliary(funObj,w,lambda,varGroupMatrix,options)
% w = L1GeneralOverlappingGroup_Auxiliary(funObj,w,lambda,varGroupMatrix,options)

if nargin < 5
	options = [];
end

[method] = myProcessOptions(options,'method','spg');
[p,nGroups] = size(varGroupMatrix);

%% Introduce auxiliary variables
alpha = zeros(nGroups,1);
for g = 1:nGroups
    alpha(g) = norm(w(find(varGroupMatrix(:,g))));
end
w = [w;alpha];

%% Make augmented objective function
auxObj = @(w)ndAuxGroupLoss(w,varGroupMatrix,lambda,funObj);

%% Make projection function
[groupStart,groupPtr] = NDgroupl1_makeGroupPointers(varGroupMatrix);
funProj = @(wAlpha)projectNDgroup_DykstraFastC(wAlpha,groupStart-1,groupPtr-1,2);

%% Solve
options.testOpt = 0;
if strcmp(method,'spg')
w = minConf_SPG(auxObj,w,funProj,options);
elseif strcmp(method,'opg')
w = minConf_OPG(auxObj,w,funProj,options);
else
w = minConf_PQN(auxObj,w,funProj,options);
end

%% Truncate to original variables
w = w(1:p);
