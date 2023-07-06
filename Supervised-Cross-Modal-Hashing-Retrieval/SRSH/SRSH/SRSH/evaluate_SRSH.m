function evaluation_info=evaluate_SRSH(XTrain,YTrain,LTrain,XTest,YTest,LTest,SRSHparam)
    
    maxItr=2;
    gmap.lambda=1;
    gmap.loss='L2';
    Fmap.type='RBF';
    Fmap.Xnu=0.7;
    % Ynu=1;
    Fmap.Ynu=0.3;
    Fmap.lambda=1e-2;
    alpha=0.03;
    p=[1.2];
    S=(LTrain*LTrain'>=1);
    S=2*S-1;
    
    nAnchors = SRSHparam.nAnchors;
    n=size(XTrain,1);
    
    anchor_idx = randsample(n, nAnchors);
    XAnchors = XTrain(anchor_idx,:); YAnchors = YTrain(anchor_idx,:);
    
    [XKTrain,XKTest]=Kernel_Feature(XTrain,XTest,XAnchors);
    [YKTrain,YKTest]=Kernel_Feature(YTrain,YTest,YAnchors);
    
    XTrain = XKTrain; XTest = XKTest;
    YTrain = YKTrain; YTest = YKTest;

    nbits = SRSHparam.nbits;
    Ntrain=size(XTrain,1);
    % Init Z
    randn('seed',3);
    Zinit=sign(randn(Ntrain,nbits));

    tic;
    [XW,YW,F,B,R] = train_SRSH(XTrain,YTrain,LTrain,Zinit,gmap,Fmap,[],maxItr,0,alpha,p,S);
    traintime=toc;  % Training Time
    evaluation_info.trainT=traintime;
    
    tic;
    
    BxTest = compactbit(XTest*XW>0);
    ByTrain = compactbit(B>0);
    DHamm = hammingDist(BxTest, ByTrain);
    [~, orderH] = sort(DHamm, 2);
     evaluation_info.Image_VS_Text_MAP = mAP(orderH', LTrain, LTest);
    [evaluation_info.Image_VS_Text_precision, evaluation_info.Image_VS_Text_recall] = precision_recall(orderH', LTrain, LTest);
    evaluation_info.Image_To_Text_Precision = precision_at_k(orderH', LTrain, LTest,SRSHparam.top_K);
    
    evaluation_info.Image_VS_Text_orderH = orderH';
    
    
    ByTest = compactbit(YTest*YW> 0);
    ByTrain = compactbit(B>0);
    DHamm = hammingDist(ByTest, ByTrain);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
    [evaluation_info.Text_VS_Image_precision,evaluation_info.Text_VS_Image_recall] = precision_recall(orderH', LTrain, LTest);
    evaluation_info.Text_To_Image_Precision = precision_at_k(orderH', LTrain, LTest,SRSHparam.top_K);
    
    evaluation_info.Text_VS_Image_orderH = orderH';
    
    compressiontime=toc;
    evaluation_info.compressT=compressiontime;
end
