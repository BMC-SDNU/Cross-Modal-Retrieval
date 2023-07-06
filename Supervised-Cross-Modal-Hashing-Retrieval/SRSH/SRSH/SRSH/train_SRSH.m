function [XW,YW,F,B,R] = train_SRSH(XTrain, YTrain,Label,B,gmap,Fmap,tol,maxItr,debug,alpha,p,S)
    % ---------- Argument defaults ----------
    if ~exist('debug','var') || isempty(debug)
        debug=1;
    end
    if ~exist('tol','var') || isempty(tol)
        tol=1e-5;
    end
    if ~exist('maxItr','var') || isempty(maxItr)
        maxItr=1000;
    end

    % ---------- End ----------
    
    % label matrix N x c
    if isvector(Label)
         Y = sparse(1:length(y), double(y), 1); Y = full(Y);
    else
       Y = Label;
    end
    

    bit = size(B,2);
    R = randn(bit,bit);
    [U11 S2 V2] = svd(R);
    R = U11(:,1:bit);

    F=B;
    D=B;
    XW=randn(size(XTrain,2),size(B,2));
    YW=randn(size(YTrain,2),size(B,2));
    j=0;
    while j < maxItr    
        j=j+1;  
        
        B=ex_sign(S*F,B);       
        XT=F-XTrain*XW;
        YT=F-YTrain*YW;
        XD=[];
        YD=[];
        for i=1:size(XTrain,1)  
            XD=[XD 1/(2/p*(norm(XT(i,:),2)^(2-p)))];
            YD=[YD 1/(2/p*(norm(YT(i,:),2)^(2-p)))];   
        end
        XD=diag(XD);
        YD=diag(YD);
        %����D        
        

        J=Fmap.Xnu*XD+Fmap.Ynu*YD;
        K=B'*B;
        L=-1*(bit*S*B+Fmap.Xnu*XD*XTrain*XW+Fmap.Ynu*YD*YTrain*YW);
        F=lyap(J,K,L);


        %
        XW=inv(XTrain'*XD*XTrain+0.05*eye(size(XTrain,2)))*XTrain'*XD*F;
        YW=inv(YTrain'*YD*YTrain+0.05*eye(size(YTrain,2)))*YTrain'*YD*F;
        %
  
        B=ex_sign(S*F,B);
    
    end   
    
end

function B=ex_sign(a,b)
         idx=find(abs(a)<1e-5);
         a(idx)=b(idx);
         B=(a>0)*2-1;
end