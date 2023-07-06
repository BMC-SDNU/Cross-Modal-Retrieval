function [model] = LLM_trainFull(X,options,model)

[lambda,param,regType,order,useMex,infer] = myProcessOptions(options,'lambda',1,'param','I','regType','2','order',3,'useMex',1,'infer','exact');

[nSamples,nNodes] = size(X);
nStates = max(X(:));

%% Choice of canonical parameterization
model.param = param;
switch param
    case 'C1'
        param = 'C';
    case 'C2'
        X = 1+mod(X,2);
        param = 'C';
    case 'CR'
        canon = 1 + (rand(nNodes,1) > .5);
        model.canon = canon;
        X(:,canon==2) = 1+mod(X(:,canon==2),2);
        param = 'C';
end

%% Initialize Edges
nEdges2 = 0;
nEdges3 = 0;
nEdges4 = 0;
nEdges5 = 0;
nEdges6 = 0;
nEdges7 = 0;
edges2 = zeros(0,2);
edges3 = zeros(0,3);
edges4 = zeros(0,4);
edges5 = zeros(0,5);
edges6 = zeros(0,6);
edges7 = zeros(0,7);
if order >= 2
	edges2 = zeros(nchoosek(nNodes,2),2);
	if order >= 3
		edges3 = zeros(nchoosek(nNodes,3),3);
		if order >= 4
			edges4 = zeros(nchoosek(nNodes,4),4);
			if order >= 5
				edges5 = zeros(nchoosek(nNodes,5),5);
				if order >= 6
					edges6 = zeros(nchoosek(nNodes,6),6);
					if order >= 7
						edges7 = zeros(nchoosek(nNodes,7),7);
					end
				end
			end
		end
	end
end
for n1 = 1:nNodes
	if order >= 2
		for n2 = n1+1:nNodes
			nEdges2 = nEdges2+1;
			edges2(nEdges2,:) = [n1 n2];
			if order >= 3
				for n3 = n2+1:nNodes
					nEdges3 = nEdges3+1;
					edges3(nEdges3,:) = [n1 n2 n3];
					if order >= 4
						for n4 = n3+1:nNodes
							nEdges4 = nEdges4+1;
							edges4(nEdges4,:) = [n1 n2 n3 n4];
							if order >= 5
								for n5 = n4+1:nNodes
									nEdges5 = nEdges5+1;
									edges5(nEdges5,:) = [n1 n2 n3 n4 n5];
									if order >= 6
										for n6 = n5+1:nNodes
											nEdges6 = nEdges6+1;
											edges6(nEdges6,:) = [n1 n2 n3 n4 n5 n6];
											if order >= 7
												for n7 = n6+1:nNodes
													nEdges7 = nEdges7+1;
													edges7(nEdges7,:) = [n1 n2 n3 n4 n5 n6 n7];
												end
											end
										end
									end
								end
							end
						end
					end
				end
			end
		end
	end
end

%% Initialize Weights
[w1,w2,w3,w4,w5,w6,w7] = LLM_initWeights(param,nNodes,nStates,edges2,edges3,edges4,edges5,edges6,edges7);
if nargin < 3
    % Cold-start with all parameters zero
    w = [w1(:);w2(:);w3(:);w4(:);w5(:);w6(:)];
else
    % Warm-start at previous model
    w = model.w;
end

%% Convert everything to int32
X = int32(X);
nStates = int32(nStates);
[edges2,edges3,edges4,edges5,edges6,edges7] = deal(int32(edges2),int32(edges3),int32(edges4),int32(edges5),int32(edges6),int32(edges7));

if strcmp(infer,'exact')
    %% Compute sufficient statistics of data
    if useMex
        [ss1,ss2,ss3,ss4,ss5,ss6,ss7] = LLM_suffStatC(param,X,nStates,edges2,edges3,edges4,edges5,edges6,edges7);
    else
        [ss1,ss2,ss3,ss4,ss5,ss6,ss7] = LLM_suffStat(param,X,nStates,edges2,edges3,edges4,edges5,edges6,edges7);
    end
    
    %% Loss function
    funObj = @(w)LLM_NLL(w,param,nSamples,ss1,ss2,ss3,ss4,ss5,ss6,ss7,edges2,edges3,edges4,edges5,edges6,edges7,useMex);
else
	[Xunique,Xreps] = LLM_unique(X);
	funObj = @(w)LLM_pseudo(w,param,Xunique,Xreps,nStates,edges2,edges3,edges4,edges5,edges6,edges7,useMex);
end

%% Optimize
options.Display = 0;
options.verbose = 0;
options.corr = 10;
options.corrections = 10;
switch regType
    case '2' % L2-regularization
        lambdaVect = lambda*[zeros(numel(w1),1);ones(numel(w2),1);2*ones(numel(w3),1);4*ones(numel(w4),1);8*ones(numel(w5),1);16*ones(numel(w6),1);32*ones(numel(w7),1)];
        regFunObj = @(w)penalizedL2(w,funObj,lambdaVect);
        w = minFunc(regFunObj,w,options);
    case '1' % L1-regularization
        lambdaVect = lambda*[zeros(numel(w1),1);ones(numel(w2),1);2*ones(numel(w3),1);4*ones(numel(w4),1);8*ones(numel(w5),1);16*ones(numel(w6),1);32*ones(numel(w7),1)];
        w = L1General2_PSSgb(funObj,w,lambdaVect,options);
    case 'G' % Group L1-regularization
        g1 = zeros(size(w1));
        g2 = zeros(size(w2));
        g3 = zeros(size(w3));
        g4 = zeros(size(w4));
        g5 = zeros(size(w5));
        g6 = zeros(size(w6));
        g7 = zeros(size(w7));
        groups = makeGroups(param,order,g1,g2,g3,g4,g5,g6,g7,nEdges2,nEdges3,nEdges4,nEdges5,nEdges6,nEdges7);
        lambdaVect = lambda*[ones(nEdges2,1);2*ones(nEdges3,1);4*ones(nEdges4,1);8*ones(nEdges5,1);16*ones(nEdges6,1);32*ones(nEdges7,1)];
        w = L1GeneralGroup_Auxiliary(funObj,w,lambdaVect,groups,options);
    case 'H' % Hierarchical Group L1-Regularization
        varGroupMatrix = LLM_makeVarGroupMatrix(param,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7);
        [groupStart,groupPtr] = NDgroupl1_makeGroupPointers(varGroupMatrix);
        lambdaVect = lambda*[32*ones(nEdges7,1);16*ones(nEdges6,1);8*ones(nEdges5,1);4*ones(nEdges4,1);2*ones(nEdges3,1);ones(nEdges2,1)];
        w = L1GeneralOverlappingGroup_Auxiliary(funObj,w,lambdaVect,varGroupMatrix,options);
end

%% Make model struct
model.useMex = useMex;
model.nStates = nStates;
model.w = w;
model.edges2 = edges2;
model.edges3 = edges3;
model.edges4 = edges4;
model.edges5 = edges5;
model.edges6 = edges6;
model.edges7 = edges7;
model.nll = @nll;
model.infer = infer;

end

%% Make groups for Group L1-Regularization
function groups = makeGroups(param,order,g1,g2,g3,g4,g5,g6,g7,nEdges2,nEdges3,nEdges4,nEdges5,nEdges6,nEdges7)
if order >= 2
    for e = 1:nEdges2
        switch param
            case {'C','I','S'}
                g2(e) = e;
            case 'P'
                g2(:,e) = e;
            case 'F'
                g2(:,:,e) = e;
        end
    end
    if order >= 3
        offset = nEdges2;
        for e = 1:nEdges3
            switch param
                case {'C','I','S'}
                    g3(e) = e + offset;
                case 'P'
                    g3(:,e) = e + offset;
                case 'F'
                    g3(:,:,:,e) = e + offset;
            end
        end
        if order >= 4
            offset = offset+nEdges3;
            for e = 1:nEdges4
                switch param
                    case {'C','I','S'}
                        g4(e) = e + offset;
                    case 'P'
                        g4(:,e) = e + offset;
                    case 'F'
                        g4(:,:,:,:,e) = e + offset;
                end
            end
            if order >= 5
                offset = offset+nEdges4;
                for e = 1:nEdges5
                    switch param
                        case {'C','I','S'}
                            g5(e) = e + offset;
                        case 'P'
                            g5(:,e) = e + offset;
                        case 'F'
                            g5(:,:,:,:,:,e) = e + offset;
                    end
                end
                if order >= 6
                    offset = offset+nEdges5;
                    for e = 1:nEdges6
                        switch param
                            case {'C','I','S'}
                                g6(e) = e + offset;
                            case 'P'
                                g6(:,e) = e + offset;
                            case 'F'
                                g6(:,:,:,:,:,:,e) = e + offset;
                        end
                    end
                    if order >= 7
                        offset = offset+nEdges6;
                        for e = 1:nEdges7
                            switch param
                                case {'C','I','S'}
                                    g7(e) = e + offset;
                                case 'P'
                                    g7(:,e) = e + offset;
                                case 'F'
                                    g7(:,:,:,:,:,:,:,e) = e + offset;
                            end
                        end
                    end
                end
            end
        end
    end
end
groups = [g1(:);g2(:);g3(:);g4(:);g5(:);g6(:);g7(:)];
end
function f = nll(model,X,testInfer)

%% Convert everything to int32
nSamples = size(X,1);
X = int32(X);

% Choice of canonical parameterization
param = model.param;
switch param
    case 'C1'
        param = 'C';
    case 'C2'
        X = 1+mod(X,2);
        param = 'C';
    case 'CR'
        canon = model.canon;
        X(:,canon==2) = 1+mod(X(:,canon==2),2);
        param = 'C';
end

if strcmp(testInfer,'exact')
% Compute sufficient statistics of data
if model.useMex
    [ss1,ss2,ss3,ss4,ss5,ss6,ss7] = LLM_suffStatC(param,X,model.nStates,model.edges2,model.edges3,model.edges4,model.edges5,model.edges6,model.edges7);
else
    [ss1,ss2,ss3,ss4,ss5,ss6,ss7] = LLM_suffStat(param,X,model.nStates,model.edges2,model.edges3,model.edges4,model.edges5,model.edges6,model.edges7);
end
f = LLM_NLL(model.w,param,nSamples,ss1,ss2,ss3,ss4,ss5,ss6,ss7,model.edges2,model.edges3,model.edges4,model.edges5,model.edges6,model.edges7,model.useMex);
else
	[Xunique,Xreps] = LLM_unique(X);
	f = LLM_pseudo(model.w,param,Xunique,Xreps,model.nStates,model.edges2,model.edges3,model.edges4,model.edges5,model.edges6,model.edges7,model.useMex);
end
end
