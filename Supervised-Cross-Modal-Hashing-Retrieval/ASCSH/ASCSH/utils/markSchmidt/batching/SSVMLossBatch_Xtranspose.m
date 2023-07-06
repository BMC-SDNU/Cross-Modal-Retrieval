function [f,g] = SSVMLossIncremental_Xtranspose(w,Xt,y,batch)
% w(feature,1)
% X(instance,feature)
% y(instance,1)

n = length(batch);
p = size(Xt,1);

Xw = (w'*Xt(:,batch))';
yXw = y(batch).*Xw;

err = 1-yXw;
viol = find(err>=0);
f = sum(err(viol).^2);

if nargout > 1
	if isempty(viol)
		g = zeros(p,1);
	else
		g = -2*(Xt(:,batch(viol))*(err(viol).*y(batch(viol))));
	end
end