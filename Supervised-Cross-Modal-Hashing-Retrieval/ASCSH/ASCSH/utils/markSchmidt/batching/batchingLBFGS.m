function [w,output] = batchingLBFGS(funObj,w,nInstances,options)

debug = 0;

if nargin < 4
	options = [];
end

[verbose,trace,gamma,delta,corr,maxIter,optTol,progTol,scaleStep,t0,strongly,scaleObj,wolfe,fullObj] = ...
	myProcessOptions(options,...
	'verbose',1,'trace',0,'gamma',1.1,'delta',1,'corr',10,'maxIter',500,...
	'optTol',1e-5,'progTol',1e-9,...
	'scaleStep',1,'t0',1e-8,'strongly',0,'scaleObj',0,'wolfe',0,'fullObj',@(w)funObj(w,1:nInstances));

if trace
	fEval = zeros(0,1);
	fVal = zeros(0,1);
	optConds = zeros(0,1);
	bSizes = zeros(0,1);
end

funEvals = 0;
nVars = length(w);
perm = randperm(nInstances);
S = zeros(nVars,0);
Y = zeros(nVars,0);
Hdiag = t0;
while funEvals < nInstances*maxIter
	
	% Choose batch
	if funEvals == 0;
		bs = 1;
	else
		bs = min(nInstances,bs*gamma+delta);
	end
	batchSize = floor(bs);
	batch = perm(1:batchSize);
	
	
	if debug && (batchSize ~= nInstances)
		pause
	end
	
	if funEvals == 0
		[f,g] = funObj(w,batch);
		funEvals = batchSize;
		
		if scaleObj
			f = f*(nInstances/batchSize);
			g = g*(nInstances/batchSize);
		end
		
		d = -g;
		t = t0;
	else
		%tBeforeDir = t
		newExamples = setdiff(batch,oldBatch);
		if ~isempty(newExamples)
			[fsub,gsub] = funObj(w,newExamples);
			funEvals = funEvals+length(newExamples);
			
			if scaleObj
				f = (f*(length(oldBatch)/nInstances) + fsub)*(nInstances/batchSize);
				g = (g*(length(oldBatch)/nInstances) + gsub)*(nInstances/batchSize);
			else
				f = f + fsub;
				g = g + gsub;
			end
		end
		if isempty(S)
			if verbose
				fprintf('No L-BFGS corrections yet\n');
			end
			d = -Hdiag*g;
		else
			if issparse(g)
				d = lbfgs(-g,S,Y,Hdiag);
			else
				d = lbfgsC(-g,S,Y,Hdiag);
			end
		end
		if scaleStep
			t = length(oldBatch)/length(batch);
		else
			t = 1;
		end
		%tAfterDir = t
	end
	oldBatch = batch;
	
	if max(abs(g)) < optTol
		fprintf('Batch is optimal\n');
		continue
	end
	
	if scaleStep == 2
		t = length(batch)/nInstances;
	end
	
	gtd = g'*d;
	if batchSize == nInstances && -gtd < progTol
		if verbose
			fprintf('At full batch and directional derivative below progTol\n');
		end
		break;
	end
	
	c1 = 1e-4;
	c2 = .9;
	f_old = f;
	if wolfe % Line for strong Wolfe conditions (requires scaleObj = 0)
		%tBeforeSearch = t
		%pause
		[t,f_new,g_new,batchEvals] = WolfeLineSearch(w,t,d,f,g,gtd,c1,c2,2,0,maxIter,progTol,1,0,0,funObj,batch);
		w_new = w + t*d;
		%tAfterSearch = t
		funEvals = funEvals+batchSize*batchEvals;
	else % Simple Armijo backtracking
		w_new = w + t*d;
		if batchSize == nInstances
			[f_new,g_new] = fullObj(w_new);
		else
			[f_new,g_new] = funObj(w_new,batch);
		end
		funEvals = funEvals+batchSize;
		
		if scaleObj
			f_new = f_new*(nInstances/batchSize);
			g_new = g_new*(nInstances/batchSize);
		end
		
		while ~isLegal(f_new) || f_new > f + c1*t*gtd
			t_old = t;
			if verbose
				fprintf('Backtracking\n');
			end
			if ~isLegal(f_new)
				if verbose
					fprintf('Halving Step Size\n');
				end
				t = t/2;
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
			w_new = w + t*d;
			if batchSize == nInstances
				[f_new,g_new] = fullObj(w_new);
			else
				[f_new,g_new] = funObj(w_new,batch);
			end
			funEvals = funEvals+batchSize;
			
			if scaleObj
				f_new = f_new*(nInstances/batchSize);
				g_new = g_new*(nInstances/batchSize);
			end
			
			if max(abs(t*d)) <= progTol
				if verbose
					fprintf('Norm of step below progTol in line search\n');
				end
				t = 0;
				f_new = f;
				g_new = g;
				break;
			end
		end
	end
	
	if trace
		fEval(end+1,1) = funEvals;
		if batchSize ~= nInstances
			[fFull gFull] = funObj(w,1:nInstances);
			fVal(end+1,1) = fFull;
			optConds(end+1,1) = max(abs(gFull));
		else
			fVal(end+1,1) = f_new;
			optConds(end+1,1) = max(abs(g_new));
		end
		bSizes(end+1,1) = batchSize;
	end
	
	if verbose
		if trace
				fprintf('FunEvals = %.2f of %d (batchSize = %d, t = %e, fsub = %e, f = %e, opt = %e)\n',funEvals/nInstances,maxIter,batchSize,t,(nInstances/batchSize)*f_new,fVal(end),optConds(end));
		else
			if batchSize == nInstances
				fprintf('FunEvals = %.2f of %d (batchSize = %d, t = %e, fsub = %e, opt = %e)\n',funEvals/nInstances,maxIter,batchSize,t,(nInstances/batchSize)*f_new,max(abs(g_new)));
			else
				fprintf('FunEvals = %.2f of %d (batchSize = %d, t = %e, fsub = %e)\n',funEvals/nInstances,maxIter,batchSize,t,(nInstances/batchSize)*f_new);
			end
		end
	end
	
	% Update L-BFGS vectors
	y = g_new-g;
	s = t*d;
	ys = y'*s;
	if strongly || ys > 1e-10
		nCorrects = size(S,2);
		if nCorrects < corr
			S(:,nCorrects+1) = s;
			Y(:,nCorrects+1) = y;
		else
			S = [S(:,2:corr) s];
			Y = [Y(:,2:corr) y];
		end
		Hdiag = ys/(y'*y);
	else
		if verbose
			fprintf('Skipping Update\n');
		end
	end
	
	w = w_new;
	f = f_new;
	g = g_new;
	
	if batchSize == nInstances
		if max(abs(g)) < optTol
			if verbose
				fprintf('At full batch and optimality tolerance below optTol\n');
			end
			break;
		end
		if max(abs(t*d)) < progTol
			if verbose
				fprintf('At full batch and norm of step below progTol\n');
			end
			break;
		end
		if abs(f-f_old) < progTol
			if verbose
				fprintf('At full batch and change in function value below progTol\n');
			end
			break;
		end
	end
	
end
output.f = f_new;
if trace
	output.fEval = fEval;
	output.fVal = fVal;
	output.optCond = optConds;
	output.bSizes = bSizes;
end