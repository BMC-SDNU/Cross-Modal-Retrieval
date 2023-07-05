function dist=norm_corr(P,Q)

n=length(P);

if size(P,1)>size(P,2),
	x=P';
else
	x=P;
end

if size(Q,1)>size(Q,2),
	y=Q;
else
	y=Q';
end

dist = -x*y/ (norm(x)*norm(y));

