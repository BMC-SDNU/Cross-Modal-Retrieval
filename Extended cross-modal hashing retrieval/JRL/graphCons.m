function L = graphCons(X, k)
 
D = pdist(X,'euclidean');
Z = squareform(D);
Z = -Z;
Z = 1./(1+exp(-Z));
K = Z;
N=size(K,1);

Kn=zeros(N,N);
for i=1:N
    [Ki,indx]=sort(K(i,:),'descend');
    ind=indx(2:k+1);
    
    Kn(i,ind)=K(i,ind);     
end;
Kn=(Kn'+Kn)/2;

Kn(Kn~=0) = 1;

D=sum(Kn,2); D=1./sqrt(D); 
D=sparse(diag(D));
L = eye(N)-D*K*D;
clear K;