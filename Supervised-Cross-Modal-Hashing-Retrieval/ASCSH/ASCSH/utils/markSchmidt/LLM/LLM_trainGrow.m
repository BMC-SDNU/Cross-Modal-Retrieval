function [model] = LLM_trainGrow(X,options,model)

[lambda,param,regType,order,useMex,verbose,infer] = myProcessOptions(options,'lambda',1,'param','F','regType','H','order',3,'useMex',1,'verbose',1,'infer','exact');

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


%% Initialize Edges and weights
if nargin < 3
    edges2 = zeros(0,2);
    edges3 = zeros(0,3);
    edges4 = zeros(0,4);
    edges5 = zeros(0,5);
    edges6 = zeros(0,6);
    edges7 = zeros(0,7);
    [w1,w2,w3,w4,w5,w6,w7] = LLM_initWeights(param,nNodes,nStates,edges2,edges3,edges4,edges5,edges6,edges7);
else
    edges2 = model.edges2;
    edges3 = model.edges3;
    edges4 = model.edges4;
    edges5 = model.edges5;
    edges6 = model.edges6;
    edges7 = model.edges7;
    [w1,w2,w3,w4,w5,w6,w7] = LLM_splitWeights(model.w,param,nNodes,nStates,edges2,edges3,edges4,edges5,edges6,edges7);
end

%% Convert everything to int32
X = int32(X);
nStates = int32(nStates);
[edges2,edges3,edges4,edges5,edges6,edges7] = deal(int32(edges2),int32(edges3),int32(edges4),int32(edges5),int32(edges6),int32(edges7));

for i = 1:10
    
    % Solve with current active set
    [w1,w2,w3,w4,w5,w6,w7] = optimize(param,X,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7,lambda,useMex,regType,infer);
    
    edges2_old = edges2;
    edges3_old = edges3;
    edges4_old = edges4;
    edges5_old = edges5;
    edges6_old = edges6;
    edges7_old = edges7;
    
    % Add Edges on boundary
    [edges2,edges3,edges4,edges5,edges6,edges7,w1,w2,w3,w4,w5,w6,w7] = ...
        updateEdges(param,order,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7);
    
    % Prune Edges on boundary that don't give local improvement
    [edges2,edges3,edges4,edges5,edges6,edges7,w1,w2,w3,w4,w5,w6,w7] = ...
        pruneEdges(param,X,lambda,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7,verbose,useMex,regType,infer);    
    
    if numel(edges2)==numel(edges2_old) && all(edges2(:) == edges2_old(:)) && ...
            numel(edges3)==numel(edges3_old) && all(edges3(:) == edges3_old(:)) && ...
            numel(edges4)==numel(edges4_old) && all(edges4(:) == edges4_old(:)) && ...
            numel(edges5)==numel(edges5_old) && all(edges5(:) == edges5_old(:)) && ...
            numel(edges6)==numel(edges6_old) && all(edges6(:) == edges6_old(:)) && ...
            numel(edges7)==numel(edges7_old) && all(edges7(:) == edges7_old(:))
        %fprintf('Done\n');
        %pause
        break;
    end
end
model.w = [w1(:);w2(:);w3(:);w4(:);w5(:);w6(:);w7(:)];
model.useMex = useMex;
model.nStates = nStates;
model.edges2 = edges2;
model.edges3 = edges3;
model.edges4 = edges4;
model.edges5 = edges5;
model.edges6 = edges6;
model.edges7 = edges7;
model.nll = @nll;
model.infer = infer;
end

function [edges2,edges3,edges4,edges5,edges6,edges7,w1,w2,w3,w4,w5,w6,w7] = updateEdges(param,order,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7)

nNodes = size(w1,1);
nStates = size(w1,2)+1;
nEdges2 = size(edges2,1);
nEdges3 = size(edges3,1);
nEdges4 = size(edges4,1);
nEdges5 = size(edges5,1);
nEdges6 = size(edges6,1);
nEdges7 = size(edges7,1);

% Store weights
weightsHash = java.util.Hashtable;
for e = 1:nEdges2
    switch param
        case {'I','C','S'}
            weightsHash.put(num2str(edges2(e,:)),w2(e));
        case 'P'
            weightsHash.put(num2str(edges2(e,:)),w2(:,e));
        case 'F'
            weightsHash.put(num2str(edges2(e,:)),w2(:,:,e));
    end
end
for e = 1:nEdges3
    switch param
        case {'I','C','S'}
            weightsHash.put(num2str(edges3(e,:)),w3(e));
        case 'P'
            weightsHash.put(num2str(edges3(e,:)),w3(:,e));
        case 'F'
            weightsHash.put(num2str(edges3(e,:)),w3(:,:,:,e));
    end
end
for e = 1:nEdges4
    switch param
        case {'I','C','S'}
            weightsHash.put(num2str(edges4(e,:)),w4(e));
        case 'P'
            weightsHash.put(num2str(edges4(e,:)),w4(:,e));
        case 'F'
            weightsHash.put(num2str(edges4(e,:)),w4(:,:,:,:,e));
    end
end
for e = 1:nEdges5
    switch param
        case {'I','C','S'}
            weightsHash.put(num2str(edges5(e,:)),w5(e));
        case 'P'
            weightsHash.put(num2str(edges5(e,:)),w5(:,e));
        case 'F'
            weightsHash.put(num2str(edges5(e,:)),w5(:,:,:,:,:,e));
    end
end
for e = 1:nEdges6
    switch param
        case {'I','C','S'}
            weightsHash.put(num2str(edges6(e,:)),w6(e));
        case 'P'
            weightsHash.put(num2str(edges6(e,:)),w6(:,e));
        case 'F'
            weightsHash.put(num2str(edges6(e,:)),w6(:,:,:,:,:,:,e));
    end
end
for e = 1:nEdges7
    switch param
        case {'I','C','S'}
            weightsHash.put(num2str(edges7(e,:)),w7(e));
        case 'P'
            weightsHash.put(num2str(edges7(e,:)),w7(:,e));
        case 'F'
            weightsHash.put(num2str(edges7(e,:)),w7(:,:,:,:,:,:,:,e));
    end
end

% Find non-zero edges
edges2_new = zeros(0,2);
edges3_new = zeros(0,3);
edges4_new = zeros(0,4);
edges5_new = zeros(0,5);
edges6_new = zeros(0,6);
edges7_new = zeros(0,7);

for e = 1:nEdges2
    switch param
        case {'I','C','S'}
            add = w2(e) ~= 0;
        case 'P'
            add = any(w2(:,e) ~= 0);
        case 'F'
            add = any(w2(:,:,e) ~= 0);
    end
    if add
        edges2_new(end+1,:) = edges2(e,:);
    end
end
if order >= 3
    for e = 1:nEdges3
        switch param
            case {'I','C','S'}
                add = w3(e) ~= 0;
            case 'P'
                add = any(w3(:,e) ~= 0);
            case 'F'
                add = any(w3(:,:,:,e) ~= 0);
        end
        if add
            edges3_new(end+1,:) = edges3(e,:);
        end
    end
    if order >= 4
        for e = 1:nEdges4
            switch param
                case {'I','C','S'}
                    add = w4(e) ~= 0;
                case 'P'
                    add = any(w4(:,e) ~= 0);
                case 'F'
                    add = any(w4(:,:,:,:,e) ~= 0);
            end
            if add
                edges4_new(end+1,:) = edges4(e,:);
            end
        end
        if order >= 5
            for e = 1:nEdges5
                switch param
                    case {'I','C','S'}
                        add = w5(e) ~= 0;
                    case 'P'
                        add = any(w5(:,e) ~= 0);
                    case 'F'
                        add = any(w5(:,:,:,:,:,e) ~= 0);
                end
                if add
                    edges5_new(end+1,:) = edges5(e,:);
                end
            end
            if order >= 6
                for e = 1:nEdges6
                    switch param
                        case {'I','C','S'}
                            add = w6(e) ~= 0;
                        case 'P'
                            add = any(w6(:,e) ~= 0);
                        case 'F'
                            add = any(w6(:,:,:,:,:,:,e) ~= 0);
                    end
                    if add
                        edges6_new(end+1,:) = edges6(e,:);
                    end
                end
                if order >= 7
                    for e = 1:nEdges7
                        switch param
                            case {'I','C','S'}
                                add = w7(e) ~= 0;
                            case 'P'
                                add = any(w7(:,e) ~= 0);
                            case 'F'
                                add = any(w7(:,:,:,:,:,:,e) ~= 0);
                        end
                        if add
                            edges7_new(end+1,:) = edges7(e,:);
                        end
                    end
                end
            end
        end
    end
end

% Find candidate groups
edges2_cand = zeros(0,2);
edges3_cand = zeros(0,3);
edges4_cand = zeros(0,4);
edges5_cand = zeros(0,5);
edges6_cand = zeros(0,6);
edges7_cand = zeros(0,7);
for n1 = 1:nNodes
    for n2 = n1+1:nNodes
        edges2_cand(end+1,:) = [n1 n2];
    end
end
if order >= 3
    hash = java.util.Hashtable;
    for e = 1:size(edges2_new,1)
        for n = 1:nNodes
            if all(edges2_new(e,:) ~= n)
                nodes = sort([edges2_new(e,:) n]);
                key = num2str(nodes);
                if hash.containsKey(key)
                    hash.put(key,hash.get(key)+1);
                    if hash.get(key) == 3
                        edges3_cand(end+1,:) = nodes;
                    end
                else
                    hash.put(key,1);
                end
                hash.get(key);
            end
        end
    end
    
    if order >= 4
        hash = java.util.Hashtable;
        for e = 1:size(edges3_new,1)
            for n = 1:nNodes
                if all(edges3_new(e,:) ~= n)
                    nodes = sort([edges3_new(e,:) n]);
                    key = num2str(nodes);
                    if hash.containsKey(key)
                        hash.put(key,hash.get(key)+1);
                        if hash.get(key) == 4
                            edges4_cand(end+1,:) = nodes;
                        end
                    else
                        hash.put(key,1);
                    end
                end
            end
        end
        
        if order >= 5
            hash = java.util.Hashtable;
            for e = 1:size(edges4_new,1)
                for n = 1:nNodes
                    if all(edges4_new(e,:) ~= n)
                        nodes = sort([edges4_new(e,:) n]);
                        key = num2str(nodes);
                        if hash.containsKey(key)
                            hash.put(key,hash.get(key)+1);
                            if hash.get(key) == 5
                                edges5_cand(end+1,:) = nodes;
                            end
                        else
                            hash.put(key,1);
                        end
                    end
                end
            end
            
            if order >= 6
                hash = java.util.Hashtable;
                for e = 1:size(edges5_new,1)
                    for n = 1:nNodes
                        if all(edges5_new(e,:) ~= n)
                            nodes = sort([edges5_new(e,:) n]);
                            key = num2str(nodes);
                            if hash.containsKey(key)
                                hash.put(key,hash.get(key)+1);
                                if hash.get(key) == 6
                                    edges6_cand(end+1,:) = nodes;
                                end
                            else
                                hash.put(key,1);
                            end
                        end
                    end
                end
                
                if order >= 7
                    hash = java.util.Hashtable;
                    for e = 1:size(edges6_new,1)
                        for n = 1:nNodes
                            if all(edges6_new(e,:) ~= n)
                                nodes = sort([edges6_new(e,:) n]);
                                key = num2str(nodes);
                                if hash.containsKey(key)
                                    hash.put(key,hash.get(key)+1);
                                    if hash.get(key) == 7
                                        edges7_cand(end+1,:) = nodes;
                                    end
                                else
                                    hash.put(key,1);
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
edges2 = unique([edges2_new;edges2_cand],'rows');
edges3 = unique([edges3_new;edges3_cand],'rows');
edges4 = unique([edges4_new;edges4_cand],'rows');
edges5 = unique([edges5_new;edges5_cand],'rows');
edges6 = unique([edges6_new;edges6_cand],'rows');
edges7 = unique([edges7_new;edges7_cand],'rows');

nEdges2 = size(edges2,1);
nEdges3 = size(edges3,1);
nEdges4 = size(edges4,1);
nEdges5 = size(edges5,1);
nEdges6 = size(edges6,1);
nEdges7 = size(edges7,1);

[junk,w2,w3,w4,w5,w6,w7] = LLM_initWeights(param,nNodes,nStates,edges2,edges3,edges4,edges5,edges6,edges7);
for e = 1:nEdges2
    key = num2str(edges2(e,:));
    if weightsHash.containsKey(key)
        switch param
            case {'I','C','S'}
                w2(e) = weightsHash.get(key);
            case 'P'
                w2(:,e) = weightsHash.get(key);
            case 'F'
                w2(:,:,e) = weightsHash.get(key);
        end
    end
end
for e = 1:nEdges3
    key = num2str(edges3(e,:));
    if weightsHash.containsKey(key)
        switch param
            case {'I','C','S'}
                w3(e) = weightsHash.get(key);
            case 'P'
                w3(:,e) = weightsHash.get(key);
            case 'F'
                w3(:,:,:,e) = weightsHash.get(key);
        end
    end
end
for e = 1:nEdges4
    key = num2str(edges4(e,:));
    if weightsHash.containsKey(key)
        switch param
            case {'I','C','S'}
                w4(e) = weightsHash.get(key);
            case 'P'
                w4(:,e) = weightsHash.get(key);
            case 'F'
                w4(:,:,:,:,e) = weightsHash.get(key);
        end
    end
end
for e = 1:nEdges5
    key = num2str(edges5(e,:));
    if weightsHash.containsKey(key)
        switch param
            case {'I','C','S'}
                w5(e) = weightsHash.get(key);
            case 'P'
                w5(:,e) = weightsHash.get(key);
            case 'F'
                w5(:,:,:,:,:,e) = weightsHash.get(key);
        end
    end
end
for e = 1:nEdges6
    key = num2str(edges6(e,:));
    if weightsHash.containsKey(key)
        switch param
            case {'I','C','S'}
                w6(e) = weightsHash.get(key);
            case 'P'
                w6(:,e) = weightsHash.get(key);
            case 'F'
                w6(:,:,:,:,:,:,e) = weightsHash.get(key);
        end
    end
end
for e = 1:nEdges7
    key = num2str(edges7(e,:));
    if weightsHash.containsKey(key)
        switch param
            case {'I','C','S'}
                w7(e) = weightsHash.get(key);
            case 'P'
                w7(:,e) = weightsHash.get(key);
            case 'F'
                w7(:,:,:,:,:,:,:,e) = weightsHash.get(key);
        end
    end
end
end

%%
function [edges2,edges3,edges4,edges5,edges6,edges7,w1,w2,w3,w4,w5,w6,w7] = pruneEdges(param,X,lambda,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7,verbose,useMex,regType,infer)

[nSamples,nNodes] = size(X);
nStates = size(w1,2)+1;
nEdges2 = size(edges2,1);
nEdges3 = size(edges3,1);
nEdges4 = size(edges4,1);
nEdges5 = size(edges5,1);
nEdges6 = size(edges6,1);
nEdges7 = size(edges7,1);

% Convert everything to int32
X = int32(X);
nStates = int32(nStates);
[edges2,edges3,edges4,edges5,edges6,edges7] = deal(int32(edges2),int32(edges3),int32(edges4),int32(edges5),int32(edges6),int32(edges7));

if strcmp(infer,'exact')
	% Compute sufficient statistics
	if useMex
		[ss1,ss2,ss3,ss4,ss5,ss6,ss7] = LLM_suffStatC(param,X,nStates,edges2,edges3,edges4,edges5,edges6,edges7);
	else
		[ss1,ss2,ss3,ss4,ss5,ss6,ss7] = LLM_suffStat(param,X,nStates,edges2,edges3,edges4,edges5,edges6,edges7);
	end
	
	% Evaluate gradient
	[f,g] = LLM_NLL([w1(:);w2(:);w3(:);w4(:);w5(:);w6(:);w7(:)],param,nSamples,ss1,ss2,ss3,ss4,ss5,ss6,ss7,edges2,edges3,edges4,edges5,edges6,edges7,useMex);
else
	[Xunique,Xreps] = LLM_unique(X);
	[f,g] = LLM_pseudo([w1(:);w2(:);w3(:);w4(:);w5(:);w6(:);w7(:)],param,Xunique,Xreps,nStates,edges2,edges3,edges4,edges5,edges6,edges7,useMex);
end
[g1,g2,g3,g4,g5,g6,g7] = LLM_splitWeights(g,param,nNodes,nStates,edges2,edges3,edges4,edges5,edges6,edges7);

killedEdges = zeros(0,1);
for e = 1:nEdges2
    switch param
        case {'I','C','S'}
            w = w2(e);
        case 'P'
            w = w2(:,e);
        case 'F'
            w = w2(:,:,e);
    end
    if all(w(:) == 0)
        switch param
            case {'I','C','S'}
                g = g2(e);
            case 'P'
                g = g2(:,e);
            case 'F'
                g = g2(:,:,e);
        end
        if regType == '1'
            ng = max(abs(g(:)));
        else
            ng = norm(g(:));
        end
        if ng <= lambda
            killedEdges(end+1,1) = e;
        elseif verbose
            fprintf('Trying to add edge (%d,%d)\n',edges2(e,:));
        end
    end
end
edges2(killedEdges,:) = [];
switch param
    case {'I','C','S'}
        w2(killedEdges) = [];
    case 'P'
        w2(:,killedEdges) = [];
    case 'F'
        w2(:,:,killedEdges) = [];
end
killedEdges = zeros(0,1);
for e = 1:nEdges3
    switch param
        case {'I','C','S'}
            w = w3(e);
        case 'P'
            w = w3(:,e);
        case 'F'
            w = w3(:,:,:,e);
    end
    if all(w(:) == 0)
        switch param
            case {'I','C','S'}
                g = g3(e);
            case 'P'
                g = g3(:,e);
            case 'F'
                g = g3(:,:,:,e);
        end
        if regType == '1'
            ng = max(abs(g(:)));
        else
            ng = norm(g(:));
        end
        if ng <= 2*lambda
            killedEdges(end+1,1) = e;
        elseif verbose
            fprintf('Trying to add edge (%d,%d,%d)\n',edges3(e,:));
        end
    end
end
edges3(killedEdges,:) = [];
switch param
    case {'I','C','S'}
        w3(killedEdges) = [];
    case 'P'
        w3(:,killedEdges) = [];
    case 'F'
        w3(:,:,:,killedEdges) = [];
end
killedEdges = zeros(0,1);
for e = 1:nEdges4
    switch param
        case {'I','C','S'}
            w = w4(e);
        case 'P'
            w = w4(:,e);
        case 'F'
            w = w4(:,:,:,:,e);
    end
    if all(w(:) == 0)
        switch param
            case {'I','C','S'}
                g = g4(e);
            case 'P'
                g = g4(:,e);
            case 'F'
                g = g4(:,:,:,:,e);
        end
        if regType == '1'
            ng = max(abs(g(:)));
        else
            ng = norm(g(:));
        end
        if ng <= 4*lambda
            killedEdges(end+1,1) = e;
        elseif verbose
            fprintf('Trying to add edge (%d,%d,%d,%d)\n',edges4(e,:));
        end
    end
end
edges4(killedEdges,:) = [];
switch param
    case {'I','C','S'}
        w4(killedEdges) = [];
    case 'P'
        w4(:,killedEdges) = [];
    case 'F'
        w4(:,:,:,:,killedEdges) = [];
end
killedEdges = zeros(0,1);
for e = 1:nEdges5
    switch param
        case {'I','C','S'}
            w = w5(e);
        case 'P'
            w = w5(:,e);
        case 'F'
            w = w5(:,:,:,:,:,e);
    end
    if all(w(:) == 0)
        switch param
            case {'I','C','S'}
                g = g5(e);
            case 'P'
                g = g5(:,e);
            case 'F'
                g = g5(:,:,:,:,:,e);
        end
        if regType == '1'
            ng = max(abs(g(:)));
        else
            ng = norm(g(:));
        end
        if ng <= 8*lambda
            killedEdges(end+1,1) = e;
        elseif verbose
            fprintf('Trying to add edge (%d,%d,%d,%d,%d)\n',edges5(e,:));
        end
    end
end
edges5(killedEdges,:) = [];
switch param
    case {'I','C','S'}
        w5(killedEdges) = [];
    case 'P'
        w5(:,killedEdges) = [];
    case 'F'
        w5(:,:,:,:,:,killedEdges) = [];
end
killedEdges = zeros(0,1);
for e = 1:nEdges6
    switch param
        case {'I','C','S'}
            w = w6(e);
        case 'P'
            w = w6(:,e);
        case 'F'
            w = w6(:,:,:,:,:,:,e);
    end
    if all(w(:) == 0)
        switch param
            case {'I','C','S'}
                g = g6(e);
            case 'P'
                g = g6(:,e);
            case 'F'
                g = g6(:,:,:,:,:,:,e);
        end
        if regType == '1'
            ng = max(abs(g(:)));
        else
            ng = norm(g(:));
        end
        if ng <= 16*lambda
            killedEdges(end+1,1) = e;
        elseif verbose
            fprintf('Trying to add edge (%d,%d,%d,%d,%d,%d)\n',edges6(e,:));
        end
    end
end
edges6(killedEdges,:) = [];
switch param
    case {'I','C','S'}
        w6(killedEdges) = [];
    case 'P'
        w6(:,killedEdges) = [];
    case 'F'
        w6(:,:,:,:,:,:,killedEdges) = [];
end
killedEdges = zeros(0,1);
for e = 1:nEdges7
    switch param
        case {'I','C','S'}
            w = w7(e);
        case 'P'
            w = w7(:,e);
        case 'F'
            w = w7(:,:,:,:,:,:,:,e);
    end
    if all(w(:) == 0)
        switch param
            case {'I','C','S'}
                g = g7(e);
            case 'P'
                g = g7(:,e);
            case 'F'
                g = g7(:,:,:,:,:,:,:,e);
        end
        if regType == '1'
            ng = max(abs(g(:)));
        else
            ng = norm(g(:));
        end
        if ng <= 32*lambda
            killedEdges(end+1,1) = e;
        elseif verbose
            fprintf('Trying to add edge (%d,%d,%d,%d,%d,%d,%d)\n',edges7(e,:));
        end
    end
end
edges7(killedEdges,:) = [];
switch param
    case {'I','C','S'}
        w7(killedEdges) = [];
    case 'P'
        w7(:,killedEdges) = [];
    case 'F'
        w7(:,:,:,:,:,:,:,killedEdges) = [];
end

end

%%
function [w1,w2,w3,w4,w5,w6,w7] = optimize(param,X,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7,lambda,useMex,regType,infer)

useMex = 1;
[nSamples,nNodes] = size(X);
nStates = size(w1,2)+1;
nEdges2 = size(edges2,1);
nEdges3 = size(edges3,1);
nEdges4 = size(edges4,1);
nEdges5 = size(edges5,1);
nEdges6 = size(edges6,1);
nEdges7 = size(edges7,1);

w = [w1(:);w2(:);w3(:);w4(:);w5(:);w6(:);w7(:)];
nVars = length(w);

% Convert everything to int32
X = int32(X);
nStates = int32(nStates);
[edges2,edges3,edges4,edges5,edges6,edges7] = deal(int32(edges2),int32(edges3),int32(edges4),int32(edges5),int32(edges6),int32(edges7));

if strcmp(infer,'exact')
	% Compute sufficient statistics
	if useMex
		[ss1,ss2,ss3,ss4,ss5,ss6,ss7] = LLM_suffStatC(param,X,nStates,edges2,edges3,edges4,edges5,edges6,edges7);
	else
		[ss1,ss2,ss3,ss4,ss5,ss6,ss7] = LLM_suffStat(param,X,nStates,edges2,edges3,edges4,edges5,edges6,edges7);
	end
	
	% Set up objective function
	funObj = @(w)LLM_NLL(w,param,nSamples,ss1,ss2,ss3,ss4,ss5,ss6,ss7,edges2,edges3,edges4,edges5,edges6,edges7,useMex);
else
	[Xunique,Xreps] = LLM_unique(X);
	funObj = @(w)LLM_pseudo(w,param,Xunique,Xreps,nStates,edges2,edges3,edges4,edges5,edges6,edges7,useMex);
end

% Solve optimization
options.verbose = 0;
options.corr = 10;
options.corrections = 10;
switch regType
    case '1'
        lambdaVect = lambda*[zeros(numel(w1),1);ones(numel(w2),1);2*ones(numel(w3),1);4*ones(numel(w4),1);8*ones(numel(w5),1);16*ones(numel(w6),1);32*ones(numel(w7),1)];
        w = L1General2_PSSgb(funObj,w,lambdaVect,options);
    case 'G'
        lambdaVect = lambda*[ones(nEdges2,1);2*ones(nEdges3,1);4*ones(nEdges4,1);8*ones(nEdges5,1);16*ones(nEdges6,1);32*ones(nEdges7,1)];
        g1 = zeros(size(w1));
        g2 = zeros(size(w2));
        g3 = zeros(size(w3));
        g4 = zeros(size(w4));
        g5 = zeros(size(w5));
        g6 = zeros(size(w6));
        g7 = zeros(size(w7));
        groups = makeGroups(param,g1,g2,g3,g4,g5,g6,g7,nEdges2,nEdges3,nEdges4,nEdges5,nEdges6,nEdges7);
        w = L1GeneralGroup_Auxiliary(funObj,w,lambdaVect,groups,options);
    case 'H'
        lambdaVect = lambda*[32*ones(nEdges7,1);16*ones(nEdges6,1);8*ones(nEdges5,1);4*ones(nEdges4,1);2*ones(nEdges3,1);ones(nEdges2,1)];
        %save temp.mat
		varGroupMatrix = LLM_makeVarGroupMatrix_fast(param,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7);
        %varGroupMatrix = LLM_makeVarGroupMatrix(param,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7);
        w = L1GeneralOverlappingGroup_Auxiliary(funObj,w,lambdaVect,varGroupMatrix,options);
end
[w1,w2,w3,w4,w5,w6,w7] = LLM_splitWeights(w,param,nNodes,nStates,edges2,edges3,edges4,edges5,edges6,edges7);
end

%% Make groups for Group L1-Regularization
function groups = makeGroups(param,g1,g2,g3,g4,g5,g6,g7,nEdges2,nEdges3,nEdges4,nEdges5,nEdges6,nEdges7)
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
groups = [g1(:);g2(:);g3(:);g4(:);g5(:);g6(:);g7(:)];
end

%% Test function
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
