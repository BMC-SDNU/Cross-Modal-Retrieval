function [w,output] = batchingL2SVM(At,b,lambda,w,options)

debug = 0;

if nargin < 4
	options = [];
end

[verbose,trace,gamma,delta,corr,maxIter,optTol,progTol,scaleStep,t0,boundType] = ...
	myProcessOptions(options,...
	'verbose',1,'trace',0,'gamma',1.1,'delta',1,'corr',10,'maxIter',500,...
	'optTol',1e-5,'progTol',1e-9,...
	'scaleStep',1,'t0',1e-8,'boundType',0);

if trace
	fEval = zeros(0,1);
	fVal = zeros(0,1);
	optConds = zeros(0,1);
	bSizes = zeros(0,1);
end

[nVars,nInstances] = size(At);
Anrms = sqrt(sum(At.^2))';
lowerBound = -inf(nInstances,1);

funEvals = 0;
perm = randperm(nInstances);
S = zeros(nVars,0);
Y = zeros(nVars,0);
Hdiag = t0;
if boundType == 1
maxBoundIter = 0;
boundIter = zeros(nInstances,1);
w_diffs = zeros(nVars,0);
end
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
		[f,g,bAw,evals] = funObj(w,At,b,lambda,batch,lowerBound);
		funEvals = evals;
		
		% Set initial lower bound for the batch
		lowerBound(batch) = bAw;
				
		d = -g;
		t = t0;
	else
		%tBeforeDir = t
		newExamples = setdiff(batch,oldBatch);
		if ~isempty(newExamples)
			[fsub,gsub,bAw,evals] = funObj(w,At,b,lambda,newExamples,lowerBound);
			funEvals = funEvals+evals;
			
			% Set initial lower bound for the new examples in the batch
			lowerBound(newExamples) = bAw;
						
			f = f + fsub;
			g = g + gsub;
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
	
	% Compute bound for step
	dnrm = norm(d);
	if boundType == 0
		LB = lowerBound - t*dnrm*Anrms;
	else
		LB = lowerBound;
		ind = boundIter == 0;
		LB(ind) = lowerBound(ind) - t*Anrms(ind)*dnrm;
		for j = 1:maxBoundIter
			ind = boundIter==j;
			LB(ind) = lowerBound(ind) - Anrms(ind)*norm(t*d + w_diffs(:,j));
		end
	end
	
	c1 = 1e-4;
	c2 = .9;
	f_old = f;
	% Simple Armijo backtracking
	w_new = w + t*d;
	if batchSize == nInstances
		[f_new,g_new,bAw,evals,updated] = fullObj(w_new,At,b,lambda,LB);
	else
		[f_new,g_new,bAw,evals,updated] = funObj(w_new,At,b,lambda,batch,LB);
	end
	funEvals = funEvals+evals;
	
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
		
		% Update bound for step
		if boundType == 0
			LB = lowerBound - t*dnrm*Anrms;
		else
			LB = lowerBound;
			ind = boundIter == 0;
			LB(ind) = lowerBound(ind) - t*Anrms(ind)*dnrm;
			for j = 1:maxBoundIter
				ind = boundIter==j;
				LB(ind) = lowerBound(ind) - Anrms(ind)*norm(t*d + w_diffs(:,j));
			end
		end
		
		w_new = w + t*d;
		if batchSize == nInstances
			[f_new,g_new,bAw,evals,updated] = fullObj(w_new,At,b,lambda,LB);
		else
			[f_new,g_new,bAw,evals,updated] = funObj(w_new,At,b,lambda,batch,LB);
		end
		funEvals = funEvals+evals;
		
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
	
	if trace
		fEval(end+1,1) = funEvals;
		if batchSize ~= nInstances
			[fFull gFull] = fullObj(w_new,At,b,lambda,lowerBound - t*dnrm*Anrms);
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
	if lambda > 0 || ys > 1e-10
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
	
	
	% Update the bound for examples that we looked at
	if batchSize == nInstances
		lowerBound(updated) = bAw;
	else
		lowerBound(batch(updated)) = bAw;
	end
	
	% Update the bound for examples that we didn't look at
	if boundType == 0
		if batchSize == nInstances
			lowerBound(~updated) = lowerBound(~updated) - t*dnrm*Anrms(~updated);
		else
			lowerBound(batch(~updated)) = lowerBound(batch(~updated)) - t*dnrm*Anrms(batch(~updated));
		end
	else
		if batchSize == nInstances
			boundIter(updated) = 0;
			boundIter(~updated) = boundIter(~updated)+1;
		else
			boundIter(batch(updated)) = 0;
			boundIter(batch(~updated)) = boundIter(batch(~updated))+1;
		end
				
		% Update the set of differences, and truncate to the relevant ones
		maxBoundIter = max(boundIter)
		if maxBoundIter > 0
			for j = maxBoundIter:-1:2
				w_diffs(:,j) = w_diffs(:,j-1)+w_new-w;
			end
			w_diffs(:,1) = w_new-w;
		end
		
	end
		
	w = w_new;
	f = f_new;
	g = g_new;
	
	
	if 0 % Check that function and gradient wrt batch are correct
		[f2,g2] = penalizedL2(w,@SSVMLossIncremental_Xtranspose,lambda*length(batch)/nInstances,At,b,batch);
		f-f2
		norm(g-g2,'inf')
	end
	
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

	function [f,g,yXw,evals,updated] = fullObj(w,Xt,y,lambda,lowerBound)
		
		
		% Objective
		updated = lowerBound < 1;
		ind = find(updated);
		Xw = (w'*Xt(:,ind))';
		yXw = y(ind).*Xw;
		err = 1-yXw;
		viol = find(err>=0);
		f = sum(err(viol).^2) + sum(lambda.*(w.^2));
		
		% Gradient
		if nargout > 1
			if isempty(viol)
				g = 2*lambda.*w;
			else
				g = -2*(Xt(:,ind(viol))*(err(viol).*y(ind(viol)))) + 2*lambda.*w;
			end
		end
		evals = length(ind);
		
		%saved = size(Xt,2)-evals
	end

	function [f,g,yXw,evals,updated] = funObj(w,Xt,y,lambda,batch,lowerBound)

		% Objective
		updated = lowerBound(batch) < 1;
		ind = find(updated);
		Xw = (w'*Xt(:,batch(ind)))';
		yXw = y(batch(ind)).*Xw;
		err = 1-yXw;
		viol = find(err>=0);
		f = sum(err(viol).^2) + (length(batch)/nInstances)*sum(lambda.*(w.^2));
		
		% Gradient
		if nargout > 1
			if isempty(viol)
				g = (2*length(batch)/nInstances)*lambda.*w;
			else
				g = -2*(Xt(:,batch(ind(viol)))*(err(viol).*y(batch(ind(viol))))) + (2*length(batch)/nInstances)*lambda.*w;
			end
		end
		evals = length(batch(ind));
		
		%saved = length(batch)-evals
	end
end
