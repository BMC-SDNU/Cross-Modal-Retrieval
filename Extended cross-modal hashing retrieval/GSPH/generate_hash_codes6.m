function [A,B,f] = generate_hash_codes6(S,m,n,q)

%% TRYING to implement massive parallelizations 
% using gradient descent search for this case

a = -1; b = 1;
A = (b-a)*rand(m,q,'double') + a;
A = sign(A);
B = (b-a)*rand(n,q,'double') + a;
B = sign(B);

niter = 100;
tolerance = 1e-6;
for t=1:niter    
    
    if t==1
        f_prev = 1e100;
    else
        f_prev = f(t-1);
    end
    
    % Fix B and update A 
    % Do for all rows of A
    R1 = (B*A.' - q*S.')-B(:,1)*A(:,1).';
    H_a = sum(B.*B,1);
    
    for l=1:q
        % update all a_il for all i together parallely        
        z = R1.*repmat(B(:,l),1,size(R1,2));
        z = sum(z,1);
        A(:,l) = -z./H_a(l);        
        A(A>1)=1; A(A<-1)=-1;
        % update the R1 lookup table 
        if l~=q
            R1 = R1 - B(:,l+1)*A(:,l+1).' + B(:,l)*A(:,l).';
        end
    end
    
    % Fix A and update B 
    % Do for all rows of B
    R2 = (A*B.' - q*S)-A(:,1)*B(:,1).';
    H_b = sum(A.*A,1);
    
    for l=1:q
        % update all a_il for all i together parallely        
        z = R2.*repmat(A(:,l),1,size(R2,2));
        z = sum(z,1);
        B(:,l) = -z./H_b(l);        
        B(B>1)=1; B(B<-1)=-1;
        % update the R2 lookup table 
        if l~=q
            R2 = R2 - A(:,l+1)*B(:,l+1).' + A(:,l)*B(:,l).';
        end
    end
    
    % Compute the function value 
    f(t) = norm(q*S-A*B.','fro');
    
    if (f_prev-f(t))/f_prev <= tolerance
        break;
    end
    
    if mod(t,1)==0
    fprintf('The iteration is : %i and f val is : %f \n', t, f(t));
    end
end
A = sign(A); B = sign(B);