function [KTrain, KTest, KDb] = Kernelize(Train,Test,Db)
    [n,~]=size(Train);
    [nT,~]=size(Test);
    [nD,~]=size(Db);
    n_anchor=500;
    
    anchor=Train(randsample(n,n_anchor),:);
   
    KTrain = sqdist(Train',anchor');
    sigma = mean(mean(KTrain,2));
    KTrain = exp(-KTrain/(2*sigma));  
    mvec = mean(KTrain);
    KTrain = KTrain-repmat(mvec,n,1);
    
    KTest = sqdist(Test',anchor');
    KTest = exp(-KTest/(2*sigma));
    KTest = KTest-repmat(mvec,nT,1);

    KDb = sqdist(Db',anchor');
    KDb = exp(-KDb/(2*sigma));
    KDb = KDb-repmat(mvec,nD,1);
end
