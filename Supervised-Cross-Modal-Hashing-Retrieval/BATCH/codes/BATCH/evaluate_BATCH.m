function evaluation_info=evaluate_BATCH(XKTrain,YKTrain,LTrain,XKTest,YKTest,LTest,param)
    
    GTrain = NormalizeFea(LTrain,1);
    
    tic;
    
    % Hash codes learning
    B = train_BATCH(GTrain,XKTrain,YKTrain,LTrain,param);
    
    % Hash functions learning
    XW = (XKTrain'*XKTrain+param.xi*eye(size(XKTrain,2)))    \    (XKTrain'*B);
    YW = (YKTrain'*YKTrain+param.xi*eye(size(YKTrain,2)))    \    (YKTrain'*B);
    
    traintime=toc;
    evaluation_info.trainT=traintime;
    
    tic;
    
    % Cross-Modal Retrieval
    BxTest = compactbit(XKTest*XW>0);
    BxTrain = compactbit(B>0);
    DHamm = hammingDist(BxTest, BxTrain);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Image_VS_Text_MAP = mAP(orderH', LTrain, LTest);
    [evaluation_info.Image_VS_Text_precision, evaluation_info.Image_VS_Text_recall] = precision_recall(orderH', LTrain, LTest);
    evaluation_info.Image_To_Text_Precision = precision_at_k(orderH', LTrain, LTest,param.top_K);
    
    ByTest = compactbit(YKTest*YW> 0);
    ByTrain = compactbit(B>0); % ByTrain = BxTrain;
    DHamm = hammingDist(ByTest, ByTrain);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
    [evaluation_info.Text_VS_Image_precision,evaluation_info.Text_VS_Image_recall] = precision_recall(orderH', LTrain, LTest);
    evaluation_info.Text_To_Image_Precision = precision_at_k(orderH', LTrain, LTest,param.top_K);
    compressiontime=toc;
    
    evaluation_info.compressT=compressiontime;
    %evaluation_info.BxTrain = BxTrain;
    %evaluation_info.ByTrain = ByTrain;
    %evaluation_info.B = B;

end
