function [pr, all] = getprfrompn(pn, mode);
% mode 0: interpolated

[n c] = size(pn);
pnround = round(pn .* repmat([1:c], n, 1));
all = [];
pr = [];

if (mode == 0)
	for i = 1:n
	  l =  pn(i,:);
	  p = pnround(i,:);
	  curr = l(end);
	  for j=[numel(l):-1:1]
	    if(l(j) < curr) 
	      l(j) = curr;
	    else
	      curr = l(j);
	    end
	  end
	  idx = 1;
	  up = [];
	  upid = [];
	  for j = max(p):-1:1
	    f = find(p == j);
	    up(idx) = l(f(1));
	    up(idx+1) = l(f(1));
	    upid(idx) = j;
	    upid(idx+1) = j - 0.99999;
	    idx = idx+2;
	  end
	  pr(i,:) = interp1(upid/max(p), up, [0:0.1:1], 'nearest', 'extrap');
	  all(i,:) = l;
	end
else
	for i = 1:n
	  p = pnround(i,:);
	  up = [];
	  upid = [];
	  for j = 1:max(p)
	    f = find(p == j);
	    up(j) = j;
	    upid(j) = f(1);
	  end
	  prec = up./upid;
	  reca = up./max(up);
	  pr(i,:) = interp1(reca, prec, [0:0.01:1], 'linear');
	  f = find(isnan(pr(i,:)) == 0);
	  pr(i, isnan(pr(i,:))) = pr(i,f(1));
	end
end

