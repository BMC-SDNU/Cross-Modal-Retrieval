function [X,edges2,edges3,edges4,edges5,edges6,edges7] = LLM_generate(nSamples,nNodes,nStates,edgeProbs,param,useMex)

%% Make some edges
edges2 = zeros(0,2);
for n1 = 1:nNodes
	for n2 = n1+1:nNodes
		if rand < edgeProbs(1)
			edges2(end+1,:) = [n1 n2];
		end
	end
end
edges3 = zeros(0,3);
for n1 = 1:nNodes
	for n2 = n1+1:nNodes
		for n3 = n2+1:nNodes
			if rand < edgeProbs(2)
				if 1
					if all(ismember([n1 n2;n1 n3;n2 n3],edges2,'rows'))
						%fprintf('Hierarchical edge\n');
						%pause
					else
						%continue
					end
				end
				edges3(end+1,:) = [n1 n2 n3];
			end
		end
	end
end
edges4 = zeros(0,4);
for n1 = 1:nNodes
	for n2 = n1+1:nNodes
		for n3 = n2+1:nNodes
			for n4 = n3+1:nNodes
				if rand < edgeProbs(3)
					edges4(end+1,:) = [n1 n2 n3 n4];
				end
			end
		end
	end
end
edges5 = zeros(0,5);
for n1 = 1:nNodes
	for n2 = n1+1:nNodes
		for n3 = n2+1:nNodes
			for n4 = n3+1:nNodes
				for n5 = n4+1:nNodes
					if rand < edgeProbs(4)
						edges5(end+1,:) = [n1 n2 n3 n4 n5];
					end
				end
			end
		end
	end
end
edges6 = zeros(0,6);
for n1 = 1:nNodes
	for n2 = n1+1:nNodes
		for n3 = n2+1:nNodes
			for n4 = n3+1:nNodes
				for n5 = n4+1:nNodes
					for n6 = n5+1:nNodes
						if rand < edgeProbs(5)
							edges6(end+1,:) = [n1 n2 n3 n4 n5 n6];
						end
					end
				end
			end
		end
	end
end
edges7 = zeros(0,7);
for n1 = 1:nNodes
	for n2 = n1+1:nNodes
		for n3 = n2+1:nNodes
			for n4 = n3+1:nNodes
				for n5 = n4+1:nNodes
					for n6 = n5+1:nNodes
						for n7 = n6+1:nNodes
							if rand < edgeProbs(6)
								edges7(end+1,:) = [n1 n2 n3 n4 n5 n6 n7];
							end
						end
					end
				end
			end
		end
	end
end
%% Convert edges, nStates to int32
[edges2,edges3,edges4,edges5,edges6,edges7] = deal(int32(edges2),int32(edges3),int32(edges4),int32(edges5),int32(edges6),int32(edges7));

%% Make log-potentials
[w1,w2,w3,w4,w5,w6,w7] = LLM_initWeights(param,nNodes,nStates,edges2,edges3,edges4,edges5,edges6,edges7);
[w1,w2,w3,w4,w5,w6,w7] = deal(randn(size(w1)),randn(size(w2)),randn(size(w3)),randn(size(w4)),randn(size(w5)),randn(size(w6)),randn(size(w7)));

if useMex
    Z = LLM_inferC(param,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7);
else
    Z = LLM_infer(param,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7);
end
X = LLM_sample(param,Z,nSamples,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7);

%% Convert samples to int32
X = int32(X);