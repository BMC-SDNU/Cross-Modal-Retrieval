function prob=ssl_knn(K,init_prob,k,alpha,beta)

N=size(K,1);
if beta ~= 1
    K=exp(log(K)/beta); 
end
Kn=zeros(N,N);
for i=1:N
    [Ki,indx]=sort(K(i,:),'descend');
    ind=indx(2:k+1);
    Kn(i,ind)=K(i,ind);     
end;
Kn=(Kn'+Kn)/2;
clear K;
D=sum(Kn,2); D=1./sqrt(D); 
D=sparse(diag(D));
W=inv(eye(N)-alpha*D*sparse(Kn)*D);
clear D Kn;
prob=(1-alpha)*W*init_prob;   
   
