function [XKTrain,XKTest]=Kernel_Feature(XTrain,XTest,Anchors)
    
    [nX,Xdim]=size(XTrain);

    [nXT,XTdim]=size(XTest);


    XKTrain = sqdist(XTrain',Anchors');
    Xsigma = mean(mean(XKTrain,2));
    XKTrain = exp(-XKTrain/(2*Xsigma));
    Xmvec = mean(XKTrain);
    XKTrain = XKTrain-repmat(Xmvec,nX,1);
    
    XKTest = sqdist(XTest',Anchors');
    XKTest = exp(-XKTest/(2*Xsigma));
    XKTest = XKTest-repmat(Xmvec,nXT,1);
end