function [pseudoNLL,g] = LLM_pseudo(w,param,Y,Yr,nStates,edges2,edges3,edges4,edges5,edges6,edges7,useMex)

[nInstances,nNodes] = size(Y);
nEdges2 = size(edges2,1);
nEdges3 = size(edges3,1);
nEdges4 = size(edges4,1);
nEdges5 = size(edges5,1);
nEdges6 = size(edges6,1);
nEdges7 = size(edges7,1);

%% Split Weights
[w1,w2,w3,w4,w5,w6,w7] = LLM_splitWeights(w,param,nNodes,nStates,edges2,edges3,edges4,edges5,edges6,edges7);

pseudoNLL = 0;
g1 = zeros(size(w1));
g2 = zeros(size(w2));
g3 = zeros(size(w3));
g4 = zeros(size(w4));
g5 = zeros(size(w5));
g6 = zeros(size(w6));
g7 = zeros(size(w7));

%% Compute pseudo-likelihood

if useMex
	pseudoNLL = LLM_pseudoC(param,Y-1,Yr,g1,g2,g3,g4,g5,g6,g7,edges2-1,edges3-1,edges4-1,edges5-1,edges6-1,edges7-1,w1,w2,w3,w4,w5,w6,w7);
else
	for i = 1:nInstances
		
		% Compute conditional potential of each node being in each state
		logpot = zeros(nStates,nNodes);
		for n = 1:nNodes
			for s = 1:nStates-1
				logpot(s,n) = logpot(s,n) + w1(n,s);
			end
		end
		for e = 1:nEdges2
			n1 = edges2(e,1);
			n2 = edges2(e,2);
			
			y1 = Y(i,n1);
			y2 = Y(i,n2);
			
			switch param
				case 'C'
					if y2 == 1
						logpot(1,n1) = logpot(1,n1) + w2(e);
					end
					if y1 == 1
						logpot(1,n2) = logpot(1,n2) + w2(e);
					end
				case 'I'
					logpot(y2,n1) = logpot(y2,n1) + w2(e);
					logpot(y1,n2) = logpot(y1,n2) + w2(e);
				case 'P'
					logpot(y2,n1) = logpot(y2,n1) + w2(y2,e);
					logpot(y1,n2) = logpot(y1,n2) + w2(y1,e);
				case 'S'
					logpot(mod(y2,2)+1,n1) = logpot(mod(y2,2)+1,n1) + w2(e);
					logpot(mod(y1,2)+1,n2) = logpot(mod(y1,2)+1,n2) + w2(e);
				case 'F'
					for s = 1:nStates
						logpot(s,n1) = logpot(s,n1) + w2(s,y2,e);
						logpot(s,n2) = logpot(s,n2) + w2(y1,s,e);
					end
			end
		end
		for e = 1:nEdges3
			n1 = edges3(e,1);
			n2 = edges3(e,2);
			n3 = edges3(e,3);
			
			y1 = Y(i,n1);
			y2 = Y(i,n2);
			y3 = Y(i,n3);
			
			switch param
				case 'C'
					if y2==1 && y3==1
						logpot(1,n1) = logpot(1,n1) + w3(e);
					end
					if y1==1 && y3==1
						logpot(1,n2) = logpot(1,n2) + w3(e);
					end
					if y1==1 && y2==1
						logpot(1,n3) = logpot(1,n3) + w3(e);
					end
				case 'I'
					if y2==y3
						logpot(y2,n1) = logpot(y2,n1) + w3(e);
					end
					if y1==y3
						logpot(y1,n2) = logpot(y1,n2) + w3(e);
					end
					if y1==y2
						logpot(y1,n3) = logpot(y1,n3) + w3(e);
					end
				case 'P'
					if y2==y3
						logpot(y2,n1) = logpot(y2,n1) + w3(y2,e);
					end
					if y1==y3
						logpot(y1,n2) = logpot(y1,n2) + w3(y1,e);
					end
					if y1==y2
						logpot(y1,n3) = logpot(y1,n3) + w3(y1,e);
					end
				case 'S'
					logpot(mod(y2+y3,2)+1,n1) = logpot(mod(y2+y3,2)+1,n1) + w3(e);
					logpot(mod(y1+y3,2)+1,n2) = logpot(mod(y1+y3,2)+1,n2) + w3(e);
					logpot(mod(y1+y2,2)+1,n3) = logpot(mod(y1+y2,2)+1,n3) + w3(e);
				case 'F'
					for s = 1:nStates
						logpot(s,n1) = logpot(s,n1) + w3(s,y2,y3,e);
						logpot(s,n2) = logpot(s,n2) + w3(y1,s,y3,e);
						logpot(s,n3) = logpot(s,n3) + w3(y1,y2,s,e);
					end
			end
		end
		for e = 1:nEdges4
			n1 = edges4(e,1);
			n2 = edges4(e,2);
			n3 = edges4(e,3);
			n4 = edges4(e,4);
			
			y1 = Y(i,n1);
			y2 = Y(i,n2);
			y3 = Y(i,n3);
			y4 = Y(i,n4);
			
			switch param
				case 'C'
					if y2==1 && y3==1 && y4==1
						logpot(1,n1) = logpot(1,n1) + w4(e);
					end
					if y1==1 && y3==1 && y4==1
						logpot(1,n2) = logpot(1,n2) + w4(e);
					end
					if y1==1 && y2==1 && y4==1
						logpot(1,n3) = logpot(1,n3) + w4(e);
					end
					if y1==1 && y2==1 && y3==1
						logpot(1,n4) = logpot(1,n4) + w4(e);
					end
				case 'I'
					if y2==y3 && y3==y4
						logpot(y2,n1) = logpot(y2,n1) + w4(e);
					end
					if y1==y3 && y3==y4
						logpot(y1,n2) = logpot(y1,n2) + w4(e);
					end
					if y1==y2 && y2==y4
						logpot(y1,n3) = logpot(y1,n3) + w4(e);
					end
					if y1==y2 && y2==y3
						logpot(y1,n4) = logpot(y1,n4) + w4(e);
					end
				case 'P'
					if y2==y3 && y3==y4
						logpot(y2,n1) = logpot(y2,n1) + w4(y2,e);
					end
					if y1==y3 && y3==y4
						logpot(y1,n2) = logpot(y1,n2) + w4(y1,e);
					end
					if y1==y2 && y2==y4
						logpot(y1,n3) = logpot(y1,n3) + w4(y1,e);
					end
					if y1==y2 && y2==y3
						logpot(y1,n4) = logpot(y1,n4) + w4(y1,e);
					end
				case 'S'
					logpot(mod(y2+y3+y4,2)+1,n1) = logpot(mod(y2+y3+y4,2)+1,n1) + w4(e);
					logpot(mod(y1+y3+y4,2)+1,n2) = logpot(mod(y1+y3+y4,2)+1,n2) + w4(e);
					logpot(mod(y1+y2+y4,2)+1,n3) = logpot(mod(y1+y2+y4,2)+1,n3) + w4(e);
					logpot(mod(y1+y2+y3,2)+1,n4) = logpot(mod(y1+y2+y3,2)+1,n4) + w4(e);
				case 'F'
					for s = 1:nStates
						logpot(s,n1) = logpot(s,n1) + w4(s,y2,y3,y4,e);
						logpot(s,n2) = logpot(s,n2) + w4(y1,s,y3,y4,e);
						logpot(s,n3) = logpot(s,n3) + w4(y1,y2,s,y4,e);
						logpot(s,n4) = logpot(s,n4) + w4(y1,y2,y3,s,e);
					end
			end
		end
		for e = 1:nEdges5
			n1 = edges5(e,1);
			n2 = edges5(e,2);
			n3 = edges5(e,3);
			n4 = edges5(e,4);
			n5 = edges5(e,5);
			
			y1 = Y(i,n1);
			y2 = Y(i,n2);
			y3 = Y(i,n3);
			y4 = Y(i,n4);
			y5 = Y(i,n5);
			
			switch param
				case 'C'
					if y2==1 && y3==1 && y4==1 && y5==1
						logpot(1,n1) = logpot(1,n1) + w5(e);
					end
					if y1==1 && y3==1 && y4==1 && y5==1
						logpot(1,n2) = logpot(1,n2) + w5(e);
					end
					if y1==1 && y2==1 && y4==1 && y5==1
						logpot(1,n3) = logpot(1,n3) + w5(e);
					end
					if y1==1 && y2==1 && y3==1 && y5==1
						logpot(1,n4) = logpot(1,n4) + w5(e);
					end
					if y1==1 && y2==1 && y3==1 && y4==1
						logpot(1,n5) = logpot(1,n5) + w5(e);
					end
				case 'I'
					if y2==y3 && y3==y4 && y4==y5
						logpot(y2,n1) = logpot(y2,n1) + w5(e);
					end
					if y1==y3 && y3==y4 && y4==y5
						logpot(y1,n2) = logpot(y1,n2) + w5(e);
					end
					if y1==y2 && y2==y4 && y4==y5
						logpot(y1,n3) = logpot(y1,n3) + w5(e);
					end
					if y1==y2 && y2==y3 && y3==y5
						logpot(y1,n4) = logpot(y1,n4) + w5(e);
					end
					if y1==y2 && y2==y3 && y3==y4
						logpot(y1,n5) = logpot(y1,n5) + w5(e);
					end
				case 'P'
					if y2==y3 && y3==y4 && y4==y5
						logpot(y2,n1) = logpot(y2,n1) + w5(y2,e);
					end
					if y1==y3 && y3==y4 && y4==y5
						logpot(y1,n2) = logpot(y1,n2) + w5(y1,e);
					end
					if y1==y2 && y2==y4 && y4==y5
						logpot(y1,n3) = logpot(y1,n3) + w5(y1,e);
					end
					if y1==y2 && y2==y3 && y3==y5
						logpot(y1,n4) = logpot(y1,n4) + w5(y1,e);
					end
					if y1==y2 && y2==y3 && y3==y4
						logpot(y1,n5) = logpot(y1,n5) + w5(y1,e);
					end
				case 'S'
					logpot(mod(y2+y3+y4+y5,2)+1,n1) = logpot(mod(y2+y3+y4+y5,2)+1,n1) + w5(e);
					logpot(mod(y1+y3+y4+y5,2)+1,n2) = logpot(mod(y1+y3+y4+y5,2)+1,n2) + w5(e);
					logpot(mod(y1+y2+y4+y5,2)+1,n3) = logpot(mod(y1+y2+y4+y5,2)+1,n3) + w5(e);
					logpot(mod(y1+y2+y3+y5,2)+1,n4) = logpot(mod(y1+y2+y3+y5,2)+1,n4) + w5(e);
					logpot(mod(y1+y2+y3+y4,2)+1,n5) = logpot(mod(y1+y2+y3+y4,2)+1,n5) + w5(e);
					case 'F'
					for s = 1:nStates
						logpot(s,n1) = logpot(s,n1) + w5(s,y2,y3,y4,y5,e);
						logpot(s,n2) = logpot(s,n2) + w5(y1,s,y3,y4,y5,e);
						logpot(s,n3) = logpot(s,n3) + w5(y1,y2,s,y4,y5,e);
						logpot(s,n4) = logpot(s,n4) + w5(y1,y2,y3,s,y5,e);
						logpot(s,n5) = logpot(s,n5) + w5(y1,y2,y3,y4,s,e);
					end
			end
		end
		for e = 1:nEdges6
			n1 = edges6(e,1);
			n2 = edges6(e,2);
			n3 = edges6(e,3);
			n4 = edges6(e,4);
			n5 = edges6(e,5);
			n6 = edges6(e,6);
			
			y1 = Y(i,n1);
			y2 = Y(i,n2);
			y3 = Y(i,n3);
			y4 = Y(i,n4);
			y5 = Y(i,n5);
			y6 = Y(i,n6);
			
			switch param
				case 'C'
					if y2==1 && y3==1 && y4==1 && y5==1 && y6==1
						logpot(1,n1) = logpot(1,n1) + w6(e);
					end
					if y1==1 && y3==1 && y4==1 && y5==1 && y6==1
						logpot(1,n2) = logpot(1,n2) + w6(e);
					end
					if y1==1 && y2==1 && y4==1 && y5==1 && y6==1
						logpot(1,n3) = logpot(1,n3) + w6(e);
					end
					if y1==1 && y2==1 && y3==1 && y5==1 && y6==1
						logpot(1,n4) = logpot(1,n4) + w6(e);
					end
					if y1==1 && y2==1 && y3==1 && y4==1 && y6==1
						logpot(1,n5) = logpot(1,n5) + w6(e);
					end
					if y1==1 && y2==1 && y3==1 && y4==1 && y5==1
						logpot(1,n6) = logpot(1,n6) + w6(e);
					end
				case 'I'
					if y2==y3 && y3==y4 && y4==y5 && y5==y6
						logpot(y2,n1) = logpot(y2,n1) + w6(e);
					end
					if y1==y3 && y3==y4 && y4==y5 && y5==y6
						logpot(y1,n2) = logpot(y1,n2) + w6(e);
					end
					if y1==y2 && y2==y4 && y4==y5 && y5==y6
						logpot(y1,n3) = logpot(y1,n3) + w6(e);
					end
					if y1==y2 && y2==y3 && y3==y5 && y5==y6
						logpot(y1,n4) = logpot(y1,n4) + w6(e);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y6
						logpot(y1,n5) = logpot(y1,n5) + w6(e);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y5
						logpot(y1,n6) = logpot(y1,n6) + w6(e);
					end
				case 'P'
					if y2==y3 && y3==y4 && y4==y5 && y5==y6
						logpot(y2,n1) = logpot(y2,n1) + w6(y2,e);
					end
					if y1==y3 && y3==y4 && y4==y5 && y5==y6
						logpot(y1,n2) = logpot(y1,n2) + w6(y1,e);
					end
					if y1==y2 && y2==y4 && y4==y5 && y5==y6
						logpot(y1,n3) = logpot(y1,n3) + w6(y1,e);
					end
					if y1==y2 && y2==y3 && y3==y5 && y5==y6
						logpot(y1,n4) = logpot(y1,n4) + w6(y1,e);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y6
						logpot(y1,n5) = logpot(y1,n5) + w6(y1,e);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y5
						logpot(y1,n6) = logpot(y1,n6) + w6(y1,e);
					end
				case 'S'
					logpot(mod(y2+y3+y4+y5+y6,2)+1,n1) = logpot(mod(y2+y3+y4+y5+y6,2)+1,n1) + w6(e);
					logpot(mod(y1+y3+y4+y5+y6,2)+1,n2) = logpot(mod(y1+y3+y4+y5+y6,2)+1,n2) + w6(e);
					logpot(mod(y1+y2+y4+y5+y6,2)+1,n3) = logpot(mod(y1+y2+y4+y5+y6,2)+1,n3) + w6(e);
					logpot(mod(y1+y2+y3+y5+y6,2)+1,n4) = logpot(mod(y1+y2+y3+y5+y6,2)+1,n4) + w6(e);
					logpot(mod(y1+y2+y3+y4+y6,2)+1,n5) = logpot(mod(y1+y2+y3+y4+y6,2)+1,n5) + w6(e);
					logpot(mod(y1+y2+y3+y4+y5,2)+1,n6) = logpot(mod(y1+y2+y3+y4+y5,2)+1,n6) + w6(e);
					case 'F'
					for s = 1:nStates
						logpot(s,n1) = logpot(s,n1) + w6(s,y2,y3,y4,y5,y6,e);
						logpot(s,n2) = logpot(s,n2) + w6(y1,s,y3,y4,y5,y6,e);
						logpot(s,n3) = logpot(s,n3) + w6(y1,y2,s,y4,y5,y6,e);
						logpot(s,n4) = logpot(s,n4) + w6(y1,y2,y3,s,y5,y6,e);
						logpot(s,n5) = logpot(s,n5) + w6(y1,y2,y3,y4,s,y6,e);
						logpot(s,n6) = logpot(s,n6) + w6(y1,y2,y3,y4,y5,s,e);
					end
			end
		end
		for e = 1:nEdges7
			n1 = edges7(e,1);
			n2 = edges7(e,2);
			n3 = edges7(e,3);
			n4 = edges7(e,4);
			n5 = edges7(e,5);
			n6 = edges7(e,6);
			n7 = edges7(e,7);
			
			y1 = Y(i,n1);
			y2 = Y(i,n2);
			y3 = Y(i,n3);
			y4 = Y(i,n4);
			y5 = Y(i,n5);
			y6 = Y(i,n6);
			y7 = Y(i,n7);
			
			switch param
				case 'C'
					if y2==1 && y3==1 && y4==1 && y5==1 && y6==1 && y7==1
						logpot(1,n1) = logpot(1,n1) + w7(e);
					end
					if y1==1 && y3==1 && y4==1 && y5==1 && y6==1 && y7==1
						logpot(1,n2) = logpot(1,n2) + w7(e);
					end
					if y1==1 && y2==1 && y4==1 && y5==1 && y6==1 && y7==1
						logpot(1,n3) = logpot(1,n3) + w7(e);
					end
					if y1==1 && y2==1 && y3==1 && y5==1 && y6==1 && y7==1
						logpot(1,n4) = logpot(1,n4) + w7(e);
					end
					if y1==1 && y2==1 && y3==1 && y4==1 && y6==1 && y7==1
						logpot(1,n5) = logpot(1,n5) + w7(e);
					end
					if y1==1 && y2==1 && y3==1 && y4==1 && y5==1 && y7==1
						logpot(1,n6) = logpot(1,n6) + w7(e);
					end
					if y1==1 && y2==1 && y3==1 && y4==1 && y5==1 && y6==1
						logpot(1,n7) = logpot(1,n7) + w7(e);
					end
				case 'I'
					if y2==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7
						logpot(y2,n1) = logpot(y2,n1) + w7(e);
					end
					if y1==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7
						logpot(y1,n2) = logpot(y1,n2) + w7(e);
					end
					if y1==y2 && y2==y4 && y4==y5 && y5==y6 && y6==y7
						logpot(y1,n3) = logpot(y1,n3) + w7(e);
					end
					if y1==y2 && y2==y3 && y3==y5 && y5==y6 && y6==y7
						logpot(y1,n4) = logpot(y1,n4) + w7(e);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y6 && y6==y7
						logpot(y1,n5) = logpot(y1,n5) + w7(e);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y7
						logpot(y1,n6) = logpot(y1,n6) + w7(e);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6
						logpot(y1,n7) = logpot(y1,n7) + w7(e);
					end
				case 'P'
					if y2==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7
						logpot(y2,n1) = logpot(y2,n1) + w7(y2,e);
					end
					if y1==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7
						logpot(y1,n2) = logpot(y1,n2) + w7(y1,e);
					end
					if y1==y2 && y2==y4 && y4==y5 && y5==y6 && y6==y7
						logpot(y1,n3) = logpot(y1,n3) + w7(y1,e);
					end
					if y1==y2 && y2==y3 && y3==y5 && y5==y6 && y6==y7
						logpot(y1,n4) = logpot(y1,n4) + w7(y1,e);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y6 && y6==y7
						logpot(y1,n5) = logpot(y1,n5) + w7(y1,e);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y7
						logpot(y1,n6) = logpot(y1,n6) + w7(y1,e);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6
						logpot(y1,n7) = logpot(y1,n7) + w7(y1,e);
					end
				case 'S'
					logpot(mod(y2+y3+y4+y5+y6+y7,2)+1,n1) = logpot(mod(y2+y3+y4+y5+y6+y7,2)+1,n1) + w7(e);
					logpot(mod(y1+y3+y4+y5+y6+y7,2)+1,n2) = logpot(mod(y1+y3+y4+y5+y6+y7,2)+1,n2) + w7(e);
					logpot(mod(y1+y2+y4+y5+y6+y7,2)+1,n3) = logpot(mod(y1+y2+y4+y5+y6+y7,2)+1,n3) + w7(e);
					logpot(mod(y1+y2+y3+y5+y6+y7,2)+1,n4) = logpot(mod(y1+y2+y3+y5+y6+y7,2)+1,n4) + w7(e);
					logpot(mod(y1+y2+y3+y4+y6+y7,2)+1,n5) = logpot(mod(y1+y2+y3+y4+y6+y7,2)+1,n5) + w7(e);
					logpot(mod(y1+y2+y3+y4+y5+y7,2)+1,n6) = logpot(mod(y1+y2+y3+y4+y5+y7,2)+1,n6) + w7(e);
					logpot(mod(y1+y2+y3+y4+y5+y6,2)+1,n7) = logpot(mod(y1+y2+y3+y4+y5+y6,2)+1,n7) + w7(e);
					case 'F'
					for s = 1:nStates
						logpot(s,n1) = logpot(s,n1) + w7(s,y2,y3,y4,y5,y6,y7,e);
						logpot(s,n2) = logpot(s,n2) + w7(y1,s,y3,y4,y5,y6,y7,e);
						logpot(s,n3) = logpot(s,n3) + w7(y1,y2,s,y4,y5,y6,y7,e);
						logpot(s,n4) = logpot(s,n4) + w7(y1,y2,y3,s,y5,y6,y7,e);
						logpot(s,n5) = logpot(s,n5) + w7(y1,y2,y3,y4,s,y6,y7,e);
						logpot(s,n6) = logpot(s,n6) + w7(y1,y2,y3,y4,y5,s,y7,e);
						logpot(s,n7) = logpot(s,n7) + w7(y1,y2,y3,y4,y5,y6,s,e);
					end
			end
		end
		
		% Compute conditional normalizing constant and update objective
		logZ = mylogsumexp(logpot');
		for n = 1:nNodes
			pseudoNLL = pseudoNLL - Yr(i)*logpot(Y(i,n),n) + Yr(i)*logZ(n);
		end
		
		% Update gradient
		nodeBel = exp(logpot - repmat(logZ',[nStates 1]));
		for n = 1:nNodes
			y1 = Y(i,n);
			if y1 < nStates
				g1(n,y1) = g1(n,y1) - Yr(i)*1;
			end
			g1(n,:) = g1(n,:) + Yr(i)*nodeBel(1:end-1,n)';
		end
		for e = 1:nEdges2
			n1 = edges2(e,1);
			n2 = edges2(e,2);
			
			y1 = Y(i,n1);
			y2 = Y(i,n2);
			
			switch param
				case 'C'
					if y1==1 && y2 == 1
						g2(e) = g2(e) - Yr(i)*2;
					end
					if y2 == 1
						g2(e) = g2(e) + Yr(i)*nodeBel(1,n1);
					end
					if y1 == 1
						g2(e) = g2(e) + Yr(i)*nodeBel(1,n2);
					end
				case 'I'
					if y1==y2
						g2(e) = g2(e) - Yr(i)*2;
					end
					g2(e) = g2(e) + Yr(i)*nodeBel(y2,n1);
					g2(e) = g2(e) + Yr(i)*nodeBel(y1,n2);
				case 'P'
					if y1==y2
						g2(y1,e) = g2(y1,e) - Yr(i)*2;
					end
					g2(y2,e) = g2(y2,e) + Yr(i)*nodeBel(y2,n1);
					g2(y1,e) = g2(y1,e) + Yr(i)*nodeBel(y1,n2);
				case 'S'
					if mod(y1+y2,2)
						g2(e) = g2(e) - Yr(i)*2;
					end
					g2(e) = g2(e) + Yr(i)*nodeBel(mod(y2,2)+1,n1);
					g2(e) = g2(e) + Yr(i)*nodeBel(mod(y1,2)+1,n2);
				case 'F'
					g2(y1,y2,e) = g2(y1,y2,e) - Yr(i)*2;
					for s = 1:nStates
						g2(s,y2,e) = g2(s,y2,e) + Yr(i)*nodeBel(s,n1);
						g2(y1,s,e) = g2(y1,s,e) + Yr(i)*nodeBel(s,n2);
					end
			end
		end
		for e = 1:nEdges3
			n1 = edges3(e,1);
			n2 = edges3(e,2);
			n3 = edges3(e,3);
			
			y1 = Y(i,n1);
			y2 = Y(i,n2);
			y3 = Y(i,n3);
			
			switch param
				case 'C'
					if y1==1 && y2==1 && y3==1
						g3(e) = g3(e) - Yr(i)*3;
					end
					
					if y2==1 && y3==1
						g3(e) = g3(e) + Yr(i)*nodeBel(1,n1);
					end
					if y1==1 && y3==1
						g3(e) = g3(e) + Yr(i)*nodeBel(1,n2);
					end
					if y1==1 && y2==1
						g3(e) = g3(e) + Yr(i)*nodeBel(1,n3);
					end
				case 'I'
					if y1==y2 && y2==y3
						g3(e) = g3(e) - Yr(i)*3;
					end
					
					if y2==y3
						g3(e) = g3(e) + Yr(i)*nodeBel(y2,n1);
					end
					if y1==y3
						g3(e) = g3(e) + Yr(i)*nodeBel(y1,n2);
					end
					if y1==y2
						g3(e) = g3(e) + Yr(i)*nodeBel(y1,n3);
					end
				case 'P'
					if y1==y2 && y2==y3
						g3(y1,e) = g3(y1,e) - Yr(i)*3;
					end
					
					if y2==y3
						g3(y2,e) = g3(y2,e) + Yr(i)*nodeBel(y2,n1);
					end
					if y1==y3
						g3(y1,e) = g3(y1,e) + Yr(i)*nodeBel(y1,n2);
					end
					if y1==y2
						g3(y1,e) = g3(y1,e) + Yr(i)*nodeBel(y1,n3);
					end
				case 'S'
					if mod(y1+y2+y3,2)
						g3(e) = g3(e) - Yr(i)*3;
					end
					g3(e) = g3(e) + Yr(i)*nodeBel(mod(y2+y3,2)+1,n1);
					g3(e) = g3(e) + Yr(i)*nodeBel(mod(y1+y3,2)+1,n2);
					g3(e) = g3(e) + Yr(i)*nodeBel(mod(y1+y2,2)+1,n3);
				case 'F'
					g3(y1,y2,y3,e) = g3(y1,y2,y3,e) - Yr(i)*3;
					for s = 1:nStates
						g3(s,y2,y3,e) = g3(s,y2,y3,e) + Yr(i)*nodeBel(s,n1);
						g3(y1,s,y3,e) = g3(y1,s,y3,e) + Yr(i)*nodeBel(s,n2);
						g3(y1,y2,s,e) = g3(y1,y2,s,e) + Yr(i)*nodeBel(s,n3);
					end
			end
		end
		for e = 1:nEdges4
			n1 = edges4(e,1);
			n2 = edges4(e,2);
			n3 = edges4(e,3);
			n4 = edges4(e,4);
			
			y1 = Y(i,n1);
			y2 = Y(i,n2);
			y3 = Y(i,n3);
			y4 = Y(i,n4);
			
			switch param
				case 'C'
					if y1==1 && y2==1 && y3==1 && y4==1
						g4(e) = g4(e) - Yr(i)*4;
					end
					
					if y2==1 && y3==1 && y4==1
						g4(e) = g4(e) + Yr(i)*nodeBel(1,n1);
					end
					if y1==1 && y3==1 && y4==1
						g4(e) = g4(e) + Yr(i)*nodeBel(1,n2);
					end
					if y1==1 && y2==1 && y4==1
						g4(e) = g4(e) + Yr(i)*nodeBel(1,n3);
					end
					if y1==1 && y2==1 && y3==1
						g4(e) = g4(e) + Yr(i)*nodeBel(1,n4);
					end
				case 'I'
					if y1==y2 && y2==y3 && y3==y4
						g4(e) = g4(e) - Yr(i)*4;
					end
					
					if y2==y3 && y3==y4
						g4(e) = g4(e) + Yr(i)*nodeBel(y2,n1);
					end
					if y1==y3 && y3==y4
						g4(e) = g4(e) + Yr(i)*nodeBel(y1,n2);
					end
					if y1==y2 && y2==y4
						g4(e) = g4(e) + Yr(i)*nodeBel(y1,n3);
					end
					if y1==y2 && y2==y3
						g4(e) = g4(e) + Yr(i)*nodeBel(y1,n4);
					end
				case 'P'
					if y1==y2 && y2==y3 && y3==y4
						g4(y1,e) = g4(y1,e) - Yr(i)*4;
					end
					
					if y2==y3 && y3==y4
						g4(y2,e) = g4(y2,e) + Yr(i)*nodeBel(y2,n1);
					end
					if y1==y3 && y3==y4
						g4(y1,e) = g4(y1,e) + Yr(i)*nodeBel(y1,n2);
					end
					if y1==y2 && y2==y4
						g4(y1,e) = g4(y1,e) + Yr(i)*nodeBel(y1,n3);
					end
					if y1==y2 && y2==y3
						g4(y1,e) = g4(y1,e) + Yr(i)*nodeBel(y1,n4);
					end
				case 'S'
					if mod(y1+y2+y3+y4,2)
						g4(e) = g4(e) - Yr(i)*4;
					end
					g4(e) = g4(e) + Yr(i)*nodeBel(mod(y2+y3+y4,2)+1,n1);
					g4(e) = g4(e) + Yr(i)*nodeBel(mod(y1+y3+y4,2)+1,n2);
					g4(e) = g4(e) + Yr(i)*nodeBel(mod(y1+y2+y4,2)+1,n3);
					g4(e) = g4(e) + Yr(i)*nodeBel(mod(y1+y2+y3,2)+1,n4);
					case 'F'
					g4(y1,y2,y3,y4,e) = g4(y1,y2,y3,y4,e) - Yr(i)*4;
					for s = 1:nStates
						g4(s,y2,y3,y4,e) = g4(s,y2,y3,y4,e) + Yr(i)*nodeBel(s,n1);
						g4(y1,s,y3,y4,e) = g4(y1,s,y3,y4,e) + Yr(i)*nodeBel(s,n2);
						g4(y1,y2,s,y4,e) = g4(y1,y2,s,y4,e) + Yr(i)*nodeBel(s,n3);
						g4(y1,y2,y3,s,e) = g4(y1,y2,y3,s,e) + Yr(i)*nodeBel(s,n4);
					end
			end
		end
		for e = 1:nEdges5
			n1 = edges5(e,1);
			n2 = edges5(e,2);
			n3 = edges5(e,3);
			n4 = edges5(e,4);
			n5 = edges5(e,5);
			
			y1 = Y(i,n1);
			y2 = Y(i,n2);
			y3 = Y(i,n3);
			y4 = Y(i,n4);
			y5 = Y(i,n5);
			
			switch param
				case 'C'
					if y1==1 && y2==1 && y3==1 && y4==1 && y5==1
						g5(e) = g5(e) - Yr(i)*5;
					end
					
					if y2==1 && y3==1 && y4==1 && y5==1
						g5(e) = g5(e) + Yr(i)*nodeBel(1,n1);
					end
					if y1==1 && y3==1 && y4==1 && y5==1
						g5(e) = g5(e) + Yr(i)*nodeBel(1,n2);
					end
					if y1==1 && y2==1 && y4==1 && y5==1
						g5(e) = g5(e) + Yr(i)*nodeBel(1,n3);
					end
					if y1==1 && y2==1 && y3==1 && y5==1
						g5(e) = g5(e) + Yr(i)*nodeBel(1,n4);
					end
					if y1==1 && y2==1 && y3==1 && y4==1
						g5(e) = g5(e) + Yr(i)*nodeBel(1,n5);
					end
				case 'I'
					if y1==y2 && y2==y3 && y3==y4 && y4==y5
						g5(e) = g5(e) - Yr(i)*5;
					end
					
					if y2==y3 && y3==y4 && y4==y5
						g5(e) = g5(e) + Yr(i)*nodeBel(y2,n1);
					end
					if y1==y3 && y3==y4 && y4==y5
						g5(e) = g5(e) + Yr(i)*nodeBel(y1,n2);
					end
					if y1==y2 && y2==y4 && y4==y5
						g5(e) = g5(e) + Yr(i)*nodeBel(y1,n3);
					end
					if y1==y2 && y2==y3 && y3==y5
						g5(e) = g5(e) + Yr(i)*nodeBel(y1,n4);
					end
					if y1==y2 && y2==y3 && y3==y4
						g5(e) = g5(e) + Yr(i)*nodeBel(y1,n5);
					end
				case 'P'
					if y1==y2 && y2==y3 && y3==y4 && y4==y5
						g5(y1,e) = g5(y1,e) - Yr(i)*5;
					end
					
					if y2==y3 && y3==y4 && y4==y5
						g5(y2,e) = g5(y2,e) + Yr(i)*nodeBel(y2,n1);
					end
					if y1==y3 && y3==y4 && y4==y5
						g5(y1,e) = g5(y1,e) + Yr(i)*nodeBel(y1,n2);
					end
					if y1==y2 && y2==y4 && y4==y5
						g5(y1,e) = g5(y1,e) + Yr(i)*nodeBel(y1,n3);
					end
					if y1==y2 && y2==y3 && y3==y5
						g5(y1,e) = g5(y1,e) + Yr(i)*nodeBel(y1,n4);
					end
					if y1==y2 && y2==y3 && y3==y4
						g5(y1,e) = g5(y1,e) + Yr(i)*nodeBel(y1,n5);
					end
				case 'S'
					if mod(y1+y2+y3+y4+y5,2)
						g5(e) = g5(e) - Yr(i)*5;
					end
					g5(e) = g5(e) + Yr(i)*nodeBel(mod(y2+y3+y4+y5,2)+1,n1);
					g5(e) = g5(e) + Yr(i)*nodeBel(mod(y1+y3+y4+y5,2)+1,n2);
					g5(e) = g5(e) + Yr(i)*nodeBel(mod(y1+y2+y4+y5,2)+1,n3);
					g5(e) = g5(e) + Yr(i)*nodeBel(mod(y1+y2+y3+y5,2)+1,n4);
					g5(e) = g5(e) + Yr(i)*nodeBel(mod(y1+y2+y3+y4,2)+1,n5);
					case 'F'
					g5(y1,y2,y3,y4,y5,e) = g5(y1,y2,y3,y4,y5,e) - Yr(i)*5;
					for s = 1:nStates
						g5(s,y2,y3,y4,y5,e) = g5(s,y2,y3,y4,y5,e) + Yr(i)*nodeBel(s,n1);
						g5(y1,s,y3,y4,y5,e) = g5(y1,s,y3,y4,y5,e) + Yr(i)*nodeBel(s,n2);
						g5(y1,y2,s,y4,y5,e) = g5(y1,y2,s,y4,y5,e) + Yr(i)*nodeBel(s,n3);
						g5(y1,y2,y3,s,y5,e) = g5(y1,y2,y3,s,y5,e) + Yr(i)*nodeBel(s,n4);
						g5(y1,y2,y3,y4,s,e) = g5(y1,y2,y3,y4,s,e) + Yr(i)*nodeBel(s,n5);
					end
			end
		end
		for e = 1:nEdges6
			n1 = edges6(e,1);
			n2 = edges6(e,2);
			n3 = edges6(e,3);
			n4 = edges6(e,4);
			n5 = edges6(e,5);
			n6 = edges6(e,6);
			
			y1 = Y(i,n1);
			y2 = Y(i,n2);
			y3 = Y(i,n3);
			y4 = Y(i,n4);
			y5 = Y(i,n5);
			y6 = Y(i,n6);
			
			switch param
				case 'C'
					if y1==1 && y2==1 && y3==1 && y4==1 && y5==1 && y6==1
						g6(e) = g6(e) - Yr(i)*6;
					end
					
					if y2==1 && y3==1 && y4==1 && y5==1 && y6==1
						g6(e) = g6(e) + Yr(i)*nodeBel(1,n1);
					end
					if y1==1 && y3==1 && y4==1 && y5==1 && y6==1
						g6(e) = g6(e) + Yr(i)*nodeBel(1,n2);
					end
					if y1==1 && y2==1 && y4==1 && y5==1 && y6==1
						g6(e) = g6(e) + Yr(i)*nodeBel(1,n3);
					end
					if y1==1 && y2==1 && y3==1 && y5==1 && y6==1
						g6(e) = g6(e) + Yr(i)*nodeBel(1,n4);
					end
					if y1==1 && y2==1 && y3==1 && y4==1 && y6==1
						g6(e) = g6(e) + Yr(i)*nodeBel(1,n5);
					end
					if y1==1 && y2==1 && y3==1 && y4==1 && y5==1
						g6(e) = g6(e) + Yr(i)*nodeBel(1,n6);
					end
				case 'I'
					if y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6
						g6(e) = g6(e) - Yr(i)*6;
					end
					
					if y2==y3 && y3==y4 && y4==y5 && y5==y6
						g6(e) = g6(e) + Yr(i)*nodeBel(y2,n1);
					end
					if y1==y3 && y3==y4 && y4==y5 && y5==y6
						g6(e) = g6(e) + Yr(i)*nodeBel(y1,n2);
					end
					if y1==y2 && y2==y4 && y4==y5 && y5==y6
						g6(e) = g6(e) + Yr(i)*nodeBel(y1,n3);
					end
					if y1==y2 && y2==y3 && y3==y5 && y5==y6
						g6(e) = g6(e) + Yr(i)*nodeBel(y1,n4);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y6
						g6(e) = g6(e) + Yr(i)*nodeBel(y1,n5);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y5
						g6(e) = g6(e) + Yr(i)*nodeBel(y1,n6);
					end
				case 'P'
					if y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6
						g6(y1,e) = g6(y1,e) - Yr(i)*6;
					end
					
					if y2==y3 && y3==y4 && y4==y5 && y5==y6
						g6(y2,e) = g6(y2,e) + Yr(i)*nodeBel(y2,n1);
					end
					if y1==y3 && y3==y4 && y4==y5 && y5==y6
						g6(y1,e) = g6(y1,e) + Yr(i)*nodeBel(y1,n2);
					end
					if y1==y2 && y2==y4 && y4==y5 && y5==y6
						g6(y1,e) = g6(y1,e) + Yr(i)*nodeBel(y1,n3);
					end
					if y1==y2 && y2==y3 && y3==y5 && y5==y6
						g6(y1,e) = g6(y1,e) + Yr(i)*nodeBel(y1,n4);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y6
						g6(y1,e) = g6(y1,e) + Yr(i)*nodeBel(y1,n5);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y5
						g6(y1,e) = g6(y1,e) + Yr(i)*nodeBel(y1,n6);
					end
				case 'S'
					if mod(y1+y2+y3+y4+y5+y6,2)
						g6(e) = g6(e) - Yr(i)*6;
					end
					g6(e) = g6(e) + Yr(i)*nodeBel(mod(y2+y3+y4+y5+y6,2)+1,n1);
					g6(e) = g6(e) + Yr(i)*nodeBel(mod(y1+y3+y4+y5+y6,2)+1,n2);
					g6(e) = g6(e) + Yr(i)*nodeBel(mod(y1+y2+y4+y5+y6,2)+1,n3);
					g6(e) = g6(e) + Yr(i)*nodeBel(mod(y1+y2+y3+y5+y6,2)+1,n4);
					g6(e) = g6(e) + Yr(i)*nodeBel(mod(y1+y2+y3+y4+y6,2)+1,n5);
					g6(e) = g6(e) + Yr(i)*nodeBel(mod(y1+y2+y3+y4+y5,2)+1,n6);
					case 'F'
					g6(y1,y2,y3,y4,y5,y6,e) = g6(y1,y2,y3,y4,y5,y6,e) - Yr(i)*6;
					for s = 1:nStates
						g6(s,y2,y3,y4,y5,y6,e) = g6(s,y2,y3,y4,y5,y6,e) + Yr(i)*nodeBel(s,n1);
						g6(y1,s,y3,y4,y5,y6,e) = g6(y1,s,y3,y4,y5,y6,e) + Yr(i)*nodeBel(s,n2);
						g6(y1,y2,s,y4,y5,y6,e) = g6(y1,y2,s,y4,y5,y6,e) + Yr(i)*nodeBel(s,n3);
						g6(y1,y2,y3,s,y5,y6,e) = g6(y1,y2,y3,s,y5,y6,e) + Yr(i)*nodeBel(s,n4);
						g6(y1,y2,y3,y4,s,y6,e) = g6(y1,y2,y3,y4,s,y6,e) + Yr(i)*nodeBel(s,n5);
						g6(y1,y2,y3,y4,y5,s,e) = g6(y1,y2,y3,y4,y5,s,e) + Yr(i)*nodeBel(s,n6);
					end
			end
		end
		for e = 1:nEdges7
			n1 = edges7(e,1);
			n2 = edges7(e,2);
			n3 = edges7(e,3);
			n4 = edges7(e,4);
			n5 = edges7(e,5);
			n6 = edges7(e,6);
			n7 = edges7(e,7);
			
			y1 = Y(i,n1);
			y2 = Y(i,n2);
			y3 = Y(i,n3);
			y4 = Y(i,n4);
			y5 = Y(i,n5);
			y6 = Y(i,n6);
			y7 = Y(i,n7);
			
			switch param
				case 'C'
					if y1==1 && y2==1 && y3==1 && y4==1 && y5==1 && y6==1 && y7==1
						g7(e) = g7(e) - Yr(i)*7;
					end
					
					if y2==1 && y3==1 && y4==1 && y5==1 && y6==1 && y7==1
						g7(e) = g7(e) + Yr(i)*nodeBel(1,n1);
					end
					if y1==1 && y3==1 && y4==1 && y5==1 && y6==1 && y7==1
						g7(e) = g7(e) + Yr(i)*nodeBel(1,n2);
					end
					if y1==1 && y2==1 && y4==1 && y5==1 && y6==1 && y7==1
						g7(e) = g7(e) + Yr(i)*nodeBel(1,n3);
					end
					if y1==1 && y2==1 && y3==1 && y5==1 && y6==1 && y7==1
						g7(e) = g7(e) + Yr(i)*nodeBel(1,n4);
					end
					if y1==1 && y2==1 && y3==1 && y4==1 && y6==1 && y7==1
						g7(e) = g7(e) + Yr(i)*nodeBel(1,n5);
					end
					if y1==1 && y2==1 && y3==1 && y4==1 && y5==1 && y7==1
						g7(e) = g7(e) + Yr(i)*nodeBel(1,n6);
					end
					if y1==1 && y2==1 && y3==1 && y4==1 && y5==1 && y6==1
						g7(e) = g7(e) + Yr(i)*nodeBel(1,n7);
					end
				case 'I'
					if y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7
						g7(e) = g7(e) - Yr(i)*7;
					end
					
					if y2==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7
						g7(e) = g7(e) + Yr(i)*nodeBel(y2,n1);
					end
					if y1==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7
						g7(e) = g7(e) + Yr(i)*nodeBel(y1,n2);
					end
					if y1==y2 && y2==y4 && y4==y5 && y5==y6 && y6==y7
						g7(e) = g7(e) + Yr(i)*nodeBel(y1,n3);
					end
					if y1==y2 && y2==y3 && y3==y5 && y5==y6 && y6==y7
						g7(e) = g7(e) + Yr(i)*nodeBel(y1,n4);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y6 && y6==y7
						g7(e) = g7(e) + Yr(i)*nodeBel(y1,n5);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y7
						g7(e) = g7(e) + Yr(i)*nodeBel(y1,n6);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6
						g7(e) = g7(e) + Yr(i)*nodeBel(y1,n7);
					end
				case 'P'
					if y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7
						g7(y1,e) = g7(y1,e) - Yr(i)*7;
					end
					
					if y2==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7
						g7(y2,e) = g7(y2,e) + Yr(i)*nodeBel(y2,n1);
					end
					if y1==y3 && y3==y4 && y4==y5 && y5==y6 && y6==y7
						g7(y1,e) = g7(y1,e) + Yr(i)*nodeBel(y1,n2);
					end
					if y1==y2 && y2==y4 && y4==y5 && y5==y6 && y6==y7
						g7(y1,e) = g7(y1,e) + Yr(i)*nodeBel(y1,n3);
					end
					if y1==y2 && y2==y3 && y3==y5 && y5==y6 && y6==y7
						g7(y1,e) = g7(y1,e) + Yr(i)*nodeBel(y1,n4);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y6 && y6==y7
						g7(y1,e) = g7(y1,e) + Yr(i)*nodeBel(y1,n5);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y7
						g7(y1,e) = g7(y1,e) + Yr(i)*nodeBel(y1,n6);
					end
					if y1==y2 && y2==y3 && y3==y4 && y4==y5 && y5==y6
						g7(y1,e) = g7(y1,e) + Yr(i)*nodeBel(y1,n7);
					end
				case 'S'
					if mod(y1+y2+y3+y4+y5+y6+y7,2)
						g7(e) = g7(e) - Yr(i)*7;
					end
					g7(e) = g7(e) + Yr(i)*nodeBel(mod(y2+y3+y4+y5+y6+y7,2)+1,n1);
					g7(e) = g7(e) + Yr(i)*nodeBel(mod(y1+y3+y4+y5+y6+y7,2)+1,n2);
					g7(e) = g7(e) + Yr(i)*nodeBel(mod(y1+y2+y4+y5+y6+y7,2)+1,n3);
					g7(e) = g7(e) + Yr(i)*nodeBel(mod(y1+y2+y3+y5+y6+y7,2)+1,n4);
					g7(e) = g7(e) + Yr(i)*nodeBel(mod(y1+y2+y3+y4+y6+y7,2)+1,n5);
					g7(e) = g7(e) + Yr(i)*nodeBel(mod(y1+y2+y3+y4+y5+y7,2)+1,n6);
					g7(e) = g7(e) + Yr(i)*nodeBel(mod(y1+y2+y3+y4+y5+y6,2)+1,n7);
				case 'F'
					g7(y1,y2,y3,y4,y5,y6,y7,e) = g7(y1,y2,y3,y4,y5,y6,y7,e) - Yr(i)*7;
					for s = 1:nStates
						g7(s,y2,y3,y4,y5,y6,y7,e) = g7(s,y2,y3,y4,y5,y6,y7,e) + Yr(i)*nodeBel(s,n1);
						g7(y1,s,y3,y4,y5,y6,y7,e) = g7(y1,s,y3,y4,y5,y6,y7,e) + Yr(i)*nodeBel(s,n2);
						g7(y1,y2,s,y4,y5,y6,y7,e) = g7(y1,y2,s,y4,y5,y6,y7,e) + Yr(i)*nodeBel(s,n3);
						g7(y1,y2,y3,s,y5,y6,y7,e) = g7(y1,y2,y3,s,y5,y6,y7,e) + Yr(i)*nodeBel(s,n4);
						g7(y1,y2,y3,y4,s,y6,y7,e) = g7(y1,y2,y3,y4,s,y6,y7,e) + Yr(i)*nodeBel(s,n5);
						g7(y1,y2,y3,y4,y5,s,y7,e) = g7(y1,y2,y3,y4,y5,s,y7,e) + Yr(i)*nodeBel(s,n6);
						g7(y1,y2,y3,y4,y5,y6,s,e) = g7(y1,y2,y3,y4,y5,y6,s,e) + Yr(i)*nodeBel(s,n7);
					end
			end
		end
	end
end
g = [g1(:);g2(:);g3(:);g4(:);g5(:);g6(:);g7(:)];