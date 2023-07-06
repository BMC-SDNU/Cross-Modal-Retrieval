function result = process_MGCN_CMH_f2(dataset, param)
    %%
    %min||FG-S||+gamma*(||F-B||+||G-B||+||Fgcn-B||+||Ggcn-B||)
    %+alpha*(||GgcnG-S||+||FFgcn-S||)  
    
    %% Generate train data
    XTrain = dataset.X(:, :, :, param.indexTrain);
    YTrain = dataset.Y(param.indexTrain, :);
    trainLabel = dataset.L01(param.indexTrain,:);

    %% Construct similarity matrix
    S = trainLabel * trainLabel' > 0;
    Sn = (1 - S) * (-1);
    S = S + Sn;    

    %% Initialize parameter
    bit = param.bit;
    gamma = param.gamma;
    eta = param.eta;
    lr_img = param.lr_img;
    lr_txt = param.lr_txt;
    lr_img_gcn = param.lr_img_gcn;
    lr_txt_gcn = param.lr_txt_gcn;
    
    [num_train, dy] = size(YTrain);
    maxIter = param.maxIter;
    F = zeros(bit, num_train);
    G = zeros(bit, num_train);
    Fgcn = zeros(bit, num_train);
    Ggcn = zeros(bit, num_train);
    Y = YTrain';
    net = load('data/imagenet-vgg-f.mat');
%     txt_net = net_structure_txt(dy, bit);
%     img_net = net_structure_img(net, bit);
%     gcn_txt_net = gcn_net(bit, bit);
%     gcn_img_net = gcn_net(bit, bit);
    eval(['load fixRandomParam/' param.dataname '_txt_net_' num2str(bit) ';']);
    eval(['load fixRandomParam/img_net_' num2str(bit) ';']);
    eval(['load fixRandomParam/gcn_txt_net_' num2str(bit) ';']);
    eval(['load fixRandomParam/gcn_img_net_' num2str(bit) ';']);
    loss = zeros(1, maxIter);
    Sloss = zeros(1, maxIter);
    PlossI = zeros(1, maxIter);
    PlossT = zeros(1, maxIter);
    PlossSum = zeros(1, maxIter);
    W_img = zeros(size(S));
    W_txt = zeros(size(S));
    batch_size = param.batch_size;
    train_time = 0;
    FS = F * S; FF = F * F';
    
    %% Start iteration
    for epoch = 1: maxIter
        epoch_time = tic;
        % Update B
        B = e_sign(F + G + Fgcn + Ggcn);
        % Update G
        GgGg = Ggcn * Ggcn'; GgS = Ggcn * S;
        randvector = [];
        eval(['load fixRandomParam/randVectorImg_' param.dataname '_' num2str(bit) ';']);
        ix = randvector(epoch,:);
%         ix = randperm(num_train);
        batch_size = param.batch_size;
        for ii = 0:ceil(num_train/batch_size)-1
            index = ix((1+ii*batch_size):min((ii+1)*batch_size, num_train));
            y = Y(:, index);
            y = gpuArray(single(reshape(y,[1,size(y,1),1,size(y,2)])));
            res = vl_simplenn(txt_net,y);
            output = gather(squeeze(res(end).x));
            G(:,index) = output;            
            dJdInnerlossG = 2*FF*G(:,index) - 2*FS(:,index) + (2*GgGg*G(:,index) - 2*GgS(:,index)); 
            G1 = G*ones(num_train,1);
            dJdGB = 2*gamma*(G(:,index)-B(:,index))+2*eta*repmat(G1,1,numel(index));
            dJdGb = dJdInnerlossG + dJdGB;
            dJdGb = single(gpuArray(reshape(dJdGb,[1,1,size(dJdGb,1),size(dJdGb,2)])));
            res = vl_simplenn(txt_net,y,dJdGb);         
            n_layers = numel(txt_net.layers);
            txt_net = update_net(txt_net,res,lr_txt(epoch),num_train,n_layers,batch_size);
        end
        clear index y res output
        W_txt = constructLocalMatrix(G',64);
        W_txt = full(W_txt);
               
        % Update F       
        GG = G * G';  GS = G * S';
        FgFg = Fgcn * Fgcn'; FgS = Fgcn * S';
        batch_size = param.batch_size;
        randvector = [];
        eval(['load fixRandomParam/randVectorTxt_' param.dataname '_' num2str(bit) ';']);
        ix = randvector(epoch,:);
%         ix = randperm(num_train);
        for ii = 0:ceil(num_train/batch_size)-1
            index = ix((1+ii*batch_size):min((ii+1)*batch_size, num_train));
            img = single(XTrain(:,:,:,index));
            im_ = img - repmat(net.meta.normalization.averageImage,1,1,1,size(img,4));
            im_ = gpuArray(im_);        
            res = vl_simplenn(img_net,im_);
            n_layers = numel(img_net.layers);
            output = gather(squeeze(res(end).x));
            F(:,index) = output;        
            F1 = F*ones(num_train,1);
            dJdInnerlossF = 2*GG*F(:,index) - 2*GS(:,index) + (2*FgFg*F(:,index) - 2*FgS(:,index));  
            dJdB = 2*gamma*(F(:,index)-B(:,index)) + 2*eta*repmat(F1,1,numel(index));        
            dJdFb = dJdInnerlossF + dJdB;
            dJdFb = reshape(dJdFb,[1,1,size(dJdFb,1),size(dJdFb,2)]);
            dJdFb = gpuArray(single(dJdFb));        
            res = vl_simplenn(img_net,im_, dJdFb);
            img_net = update_net(img_net,res,lr_img(epoch),num_train,n_layers,batch_size);
        end        
        clear index img im_ res output
        W_img = constructLocalMatrix(F',64);
        W_img = full(W_img);

        W_sum = (W_txt + W_img) > 0;
        D_sum = sum(W_sum,2);
        D_sum1 = D_sum.^(-0.5);
        W_sum = D_sum1 .* W_sum .* D_sum1';
        
        % update Ggcn  
        batch_size = num_train ;
        y = G;
        A = W_sum + eye(num_train);
        y = y * A;
        y = gpuArray(single(reshape(y,[1,1,size(y,1),size(y,2)])));            
            
        for gcni = 1:3
            res = vl_simplenn(gcn_txt_net,y);
            output = gather(squeeze(res(end).x));
            Ggcn = output * A;            
            dJdInnerlossGg = 2*GG*Ggcn - 2*GS; 
            Gg1 = Ggcn*ones(num_train,1);
            dJdGgB = 2*gamma*(Ggcn - B)+2*eta*repmat(Gg1,1,num_train);
            dJdGgb = (dJdInnerlossGg + dJdGgB) * A;
            dJdGgb = single(gpuArray(reshape(dJdGgb,[1,1,size(dJdGgb,1),size(dJdGgb,2)])));
            res = vl_simplenn(gcn_txt_net,y,dJdGgb);    
            n_layers = numel(gcn_txt_net.layers);
            gcn_txt_net = update_net(gcn_txt_net,res,lr_txt_gcn(gcni),num_train,n_layers,batch_size);
        end
        clear index y res output A
        
        % update Fgcn
        FS = F * S; FF = F * F';
        batch_size = num_train ;
        y = F;
        A = W_sum + eye(num_train);
        y = y * A;
        y = gpuArray(single(reshape(y,[1,1,size(y,1),size(y,2)])));            
            
        for gcni = 1:3
            res = vl_simplenn(gcn_img_net,y);
            output = gather(squeeze(res(end).x));
            Fgcn = output * A;            
            dJdInnerlossFg = 2*FF*Fgcn - 2*FS; 
            Fg1 = Fgcn*ones(num_train,1);
            dJdFgB = 2*gamma*(Fgcn - B)+2*eta*repmat(Fg1,1,num_train);
            dJdFgb = (dJdInnerlossFg + dJdFgB) * A;
            dJdFgb = single(gpuArray(reshape(dJdFgb,[1,1,size(dJdFgb,1),size(dJdFgb,2)])));
            res = vl_simplenn(gcn_img_net,y,dJdFgb);    
            n_layers = numel(gcn_img_net.layers);
            gcn_img_net = update_net(gcn_img_net,res,lr_img_gcn(gcni),num_train,n_layers,batch_size);
        end
        clear index y res output A
        
        epoch_time = toc(epoch_time);
        train_time = train_time + epoch_time / 60;        
        
        % Caculate loss
        if mod(epoch,10) == 0
            fprintf('...epoch: %3d/%d\t traintime:%3.3f\n',epoch,maxIter,...
                 train_time);  
        end         
        
        if strcmp(param.dataname,'IAPR-TC12') && (bit == 16)
            W_same = W_img .* W_txt;
            PlossI(epoch) = sum(W_same(:)) / sum(W_img(:));
            PlossT(epoch) = sum(W_same(:)) / sum(W_txt(:));
        end
        
    end
    fprintf('...training finishes\n');

    %% Evaluation
    XRetrieval = dataset.X(:,:,:,param.indexRetrieval);
    YRetrieval = dataset.Y(param.indexRetrieval,:);
    retrievalLabel = dataset.L01(param.indexRetrieval,:);

    XQuery = dataset.X(:,:,:,param.indexQuery);
    YQuery = dataset.Y(param.indexQuery,:);
    queryLabel = dataset.L01(param.indexQuery,:);

    [rBX] = generateImgCode(img_net,XRetrieval,bit);
    [qBX] = generateImgCode(img_net,XQuery,bit);
    [rBY] = generateTxtCode(txt_net,YRetrieval',bit);
    [qBY] = generateTxtCode(txt_net,YQuery',bit);

    rBX = compactbit(rBX > 0);
    rBY = compactbit(rBY > 0);
    qBX = compactbit(qBX > 0);
    qBY = compactbit(qBY > 0);
    fprintf('...encoding finishes\n');
    result.rBX = rBX;
    result.rBY = rBY;
    result.qBX = qBX;
    result.qBY = qBY;

    % hamming ranking
    result.hri2t = calcMapTopkMapTopkPreTopkRecLabel(queryLabel, retrievalLabel, qBX, rBY);
    result.hrt2i = calcMapTopkMapTopkPreTopkRecLabel(queryLabel, retrievalLabel, qBY, rBX);
    
    result.hri2t_top500 = calcMapTopkMapTopkPreTopkRecLabel(queryLabel, retrievalLabel, qBX, rBY, 500);
    result.hrt2i_top500 = calcMapTopkMapTopkPreTopkRecLabel(queryLabel, retrievalLabel, qBY, rBX, 500);
    
    % hash lookup
    result.hli2t = calcPreRecRadiusLabel(queryLabel, retrievalLabel, qBX, rBY);
    result.hlt2i = calcPreRecRadiusLabel(queryLabel, retrievalLabel, qBY, rBX);

    result.Sloss = Sloss;
    result.PlossI = PlossI;
    result.PlossT = PlossT;
    result.PlossSum = PlossSum;
    result.loss = loss;
    result.traintime = train_time;
end


function B = generateImgCode(img_net,images,bit)
    batch_size = 256;
    num = size(images,4);
    B = zeros(num,bit);
    for i = 1:ceil(num/batch_size)
        index = (i-1)*batch_size+1:min(i*batch_size,num);
        image = single(images(:,:,:,index));
        im_ = imresize(image,img_net.meta.normalization.imageSize(1:2));
        im_ = im_ - repmat(img_net.meta.normalization.averageImage,1,1,1,size(im_,4));        
        res = vl_simplenn(img_net,gpuArray(im_));
        output = gather(squeeze(res(end).x));
        B(index,:) = sign(output');
    end
end

function B = generateTxtCode(txt_net,text,bit)
    num = size(text,2);
    batch_size = 5000;
    B = zeros(num,bit);
    for i = 1:ceil(num/batch_size)
        index = (i-1)*batch_size+1:min(i*batch_size,num);
        y = text(:,index);
        y = gpuArray(single(reshape(y,[1,size(y,1),1,size(y,2)])));
        res = vl_simplenn(txt_net,y);
        output = gather(squeeze(res(end).x));
        B(index,:) = sign(output');
    end
end

function B = e_sign(A)
    B = zeros(size(A));
    B(A > 0) = 1;
    B(A < 0) = -1;
end