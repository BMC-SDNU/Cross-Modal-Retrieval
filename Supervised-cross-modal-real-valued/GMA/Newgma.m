%Written by 					Abhishek Sharma
%Final modification date 	11/03/2013
%Please cite "Generalized Multiview Analysis: A discriminative Latent Space",

%@inproceedings{gma-cvpr12,
%  author    = {Abhishek Sharma and Abhishek Kumar and Hal Daume III and David W Jacobs},
%  title     = {Generalized Multiview Analysis: A discriminative Latent Space},
%  booktitle = {CVPR},
%  year      = {2012},
%  pages     = {2160 -- 2167},
%}

function Wout = Newgma(dataCell,options)
% this is a general function which will give you the projection directions
% directly in the cell W.
% The function first uses the codes of Deng Cai for constructing matrices W
% and D for each modality for a given method. Then it makes the final
% matrix Wf and Df whos eigen-analysis is carried out to give the cell W
% for final projection.

%% Inputs
%       dataCell - It will be a cell containing data in different views each cell
% is a structure and will have two fileds
%       dataCell.data - data in each row
%       dataCell.label - column vector of label this should be needed whenever
% you are trying to make W or D using Supervised Mode.
%       options - This is the options structure which carries all the
%       options required to make matrices W and D.
% A complete list of all the fields and their functions realted to the construction of graph matrices is given in LGE.m
% At this place it requires these fields to decide which method to use
%        options.method -
%                                    lda - lda
%                                    lpp - lpp
%                                    npe - npe
%                                    mfa - mfa
%                                    isop - IsoProjection
% For these different options different my<METHOD> (e.g. myLPP, myLDA)are called to return
% the W and D matrices which are then arranged accordingly to get the
% final eigen-value solution.
%% OUTPUT
% The cell containing the projection directions for each views. Each
% column is a projection direction.

%% COMMON NOTES ON THE PROGRAM FLOW AND VARIABLES USED WITH PURPOSE
% Program Flow
% step 1 - initialize various containers for each views to hold learned outputs
% step 2 - Dimensionality reduction before GMA (optional)
% step 3 - Construct W and D matrices for each view
% step 4 - construct the matrix Z1 and Z2 with different pairing options (refer to the eqn7 in paper )
% step 5 - Arrange the matrices accordingly to get the Big matrices Wf and Df
% step 6 - Tune the parameters (optional) I WOULD STRONGLY RECOMMEND TUNING IT YOURSELF BECAUSE I USED
%          A SIMPLE HEURISTIC TO SET THEM, BUT PPL HAVE REPORTED FAR BETTER RESULTS WHEN THEY TUNE IT PROPERLY
% step 7 - generalized eigen-value solution

% Common Notes
% 1. IT IS IMPORTANT TO UNDERSTAND THAT PCA IS REQUIRED BECAUSE MORE THAN OFTEN THE RESULTING MATRIX Df WILL BE
% ILL-CONDITIONED/SINGULAR BECAUSE OF 'LARGE D AND SMALL N' PROBLEM. THEREFORE, EITHER DIMENSIONLITY REDUCTION
% BEFORE GMA OR ADDING A REGULARIZATION TERM TO Bf WILL HELP.

% 2. Tune the parameters \gamma and \alpha (refer to eqn7) properly to get good results. A rule of thumb is to keep
% \alpha > 5*(sum(eig(Z1*Z2'))sum(eig(A1))), but it can be varied accordingly as well. For the parameter \gamma
% I used \gamma = sum(eig(B1))/sum(eig(B2)), but ppl have suggested some tuning based on cross-validation results in superior performance.


%% Getting different W and D

nV = length(dataCell);
W = cell(nV,1);
D = cell(nV,1);

Dim = zeros(nV,1);
nS = zeros(nV,1);

for i = 1:nV
    [nS(i) Dim(i)] = size(dataCell{i,1}.data); % each data sample in one row
end
method = options.method;

% setting up arrays to hold eigen-vectors and means for different views if we want to carry out PCA before GMA
if (isfield(options, 'PCA') && options.PCA)
    evFin = cell(nV,1); % PCA eigen-vectors to project data
    mPCA = cell(nV,1); % PCA means to subtract before projection
end

for i = 1:nV % loop over different views to get corresponding W and D matrices
    
    % subtract the means in each view and putput the means in structure Wout.Mean
    if (isfield(options,'meanMinus') && options.meanMinus)
        Wout{i,1}.Mean = mean(dataCell{i,1}.data);
        dataCell{i,1}.data = dataCell{i,1}.data - repmat(Wout{i,1}.Mean,nS(i),1);
    end
    % make the data standard deviation equals 1 for each dimension and output the inverse of std in Wout.Dev
    if (isfield(options,'Dev') && options.Dev)
        Wout{i,1}.Dev = std(dataCell{i,1}.data,1,1);
        tmp = 1./(Wout{i,1}.Dev);
        dataCell{i,1}.data = bsxfun(@times,dataCell{i,1}.data,tmp);
    end
    % carry out PCA, project the data on eigen-vectors and output in dataCell{i,1}.data, save eigen-vectors (evFin) and mean of projected data in mPCA
    if (isfield(options, 'PCA') && options.PCA)
        [dataCell{i,1}.data evFin{i,1} mPCA{i,1}] = pcaIn(dataCell{i,1}.data,options);
        [nS(i) Dim(i)] = size(dataCell{i,1}.data); % The new data dimension
    end

    switch lower(method)
        case 'pca'
            W{i,1} = (dataCell{i,1}.data)'*(dataCell{i,1}.data); % GMPCA
            
            % ================= LDA STARTS ========
        case 'lda'
            %             options.gnd = dataCell{i,1}.label;
            %             [Wdumm  dumm] = constructW(dataCell{i,1}.data,options);
            options.gnd = dataCell{i,1}.label;
            [Wdumm  dumm] = constructW(dataCell{i,1}.data,options);
            D{i,1} = (dataCell{i,1}.data)'*(eye(size(Wdumm,1))-Wdumm)*(dataCell{i,1}.data);
            %D{i,1} = (dataCell{i,1}.data)'*(dataCell{i,1}.data);
            Wdumm = full(Wdumm);
            W{i,1} = (dataCell{i,1}.data)'*Wdumm*(dataCell{i,1}.data);
            %=============== LDA ENDS==============
            
            
            %=============== CCA STARTS ===========
        case 'cca'
            D{i,1} = (dataCell{i,1}.data)'*(dataCell{i,1}.data);
            W{i,1} = (dataCell{i,1}.data)'*(dataCell{i,1}.data);
            
            
            % =============== BLM STARTS ============
        case 'blm'
            D{i,1} = eye(Dim(i));
            W{i,1} = (dataCell{i,1}.data)'*(dataCell{i,1}.data);
            
            
            % =========== LPP STARTS ================
        case 'lpp'
            [Wdumm  dumm] = constructW(dataCell{i,1}.data,options);
            D{i,1} = sum(Wdumm,2);
            Wdumm = full(Wdumm);
            D{i,1} = (dataCell{i,1}.data)'*(diag(D{i,1})-Wdumm)*(dataCell{i,1}.data);
            W{i,1} = (dataCell{i,1}.data)'*Wdumm*(dataCell{i,1}.data);
            % =========== LPP ENDS =================

            
            
            % =============== NPE STARTS =============
        case 'npe'
            options.gnd = dataCell{i,1}.label;
            [Wdumm] = myNPE(options,dataCell{i,1}.data);
            Wdumm = full(Wdumm);
            W{i,1} = (dataCell{i,1}.data)'*Wdumm*(dataCell{i,1}.data);
            D{i,1} = (dataCell{i,1}.data)'*(dataCell{i,1}.data);
            % ===============NPE ENDS===============

            % ============== MFA STARTA ==============
        case 'mfa'
            options.gnd = dataCell{i,1}.label;
            gnd = dataCell{i,1}.label;
            [Wdumm,  Ddumm] = myMFA(gnd,options,dataCell{i,1}.data);
            Wdumm = full(Wdumm);
            Ddumm = full(Ddumm);
            W{i,1} = (dataCell{i,1}.data)'*Wdumm*(dataCell{i,1}.data);
            D{i,1} = (dataCell{i,1}.data)'*Ddumm*(dataCell{i,1}.data);

            % =============== MFA ENDS ===============


            % ================ ISO-PROECTION STARTS =====
        case 'isop'
            options.gnd = dataCell{i,1}.label;
            [Wdumm] = myIsoP(options,dataCell{i,1}.data);
            Wdumm = full(Wdumm);
            D{i,1} = (dataCell{i,1}.data)'*(dataCell{i,1}.data);
            W{i,1} = (dataCell{i,1}.data)'*Wdumm*(dataCell{i,1}.data);

            % =============== ISOP ENDS ==============
        case 'pls'
           % display('doing PLS');
        otherwise
            display('Error No method recognized !!!!!')
            return;
    end % end switch
end % end for loop

vLabel = cell(nV,1);
for i = 1:nV
    vLabel{i,1} = dataCell{i,1}.label;
end


%% Now making the full matrix

switch lower(method)
    case 'pca'

        for i = 1:nV
            if isfield(options,'PCA')
                Wout{i,1}.evs = evFin{i,1};
                Wout{i,1}.mPCA = mPCA{i,1};
            end
        end

        % ----------------------------- PLS STARTS ----------------------------------
    case 'pls'

        if (isfield(options,'AlignMode'))
            alignMode = options.AlignMode;
        else
            alignMode = 1; % Align all the samples
        end

        for i = 1:nV
            tmp = dataCell{i,1}.label;
            label = unique(tmp);
            Wout{i,1}.classMean = zeros(1,Dim(i));
            for c = 1 : length(label)
                fil = tmp == label(c);
                Wout{i,1}.classMean(c,:) =  mean(dataCell{i,1}.data(fil,:));
            end
        end


        switch alignMode
            case 1 % Align all samples
                for i = 1:nV
                    Wout{i,1}.alignCol = dataCell{i,1}.data;
                end
            case 2 % Align class centres
                for i = 1:nV
                    Wout{i,1}.alignCol = Wout{i,1}.classMean;
                end
            case 3 % Align after clustering
                k = options.NumCluster;
                for in = 1:nV
                    Wout{in,1}.classLabel = label;
                    Wout{in,1}.classID = cell(length(label),1);
                    Wout{in,1}.classCent = cell(length(label),1);
                    Wout{in,1}.alignCol = [];
                    if in == 1
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            [id cC] = kmeans(tmp,k,'emptyaction','drop');
                            Wout{in,1}.classID{c,1} = id;
                            Wout{in,1}.classCent{c,1} =  cC;
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; cC];
                        end
                    else
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            Wout{in,1}.classID{c,1} = Wout{1,1}.classID{c,1};
                            for inn = 1:k
                                fil1 = Wout{in,1}.classID{c,1} == inn;
                                Wout{in,1}.classCent{c,1}(inn,:) = mean(tmp(fil1,:));
                            end
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; Wout{in,1}.classCent{c,1}];
                        end
                    end
                end
            case 4
                if (isfield(options,'nPair') && options.nPair > 0)
                C1 = generateRandomPairs(vLabel,options.nPair);
                else
                    options.nPair = length(vLabel{1,1})*2;
                    C1 = generateRandomPairs(vLabel,options.nPair);
                end
                for i = 1:nV
                    Wout{i,1}.alignCol = dataCell{i,1}.data(C1(:,i),:);
                    Wout{i,1}.alignCol = [Wout{i,1}.alignCol; Wout{i,1}.classMean];
                end
        end
        X = (Wout{1,1}.alignCol)';
        Y = (Wout{2,1}.alignCol)';
        %此处我认为源代码有误
        %[Wpls,evals,Qpls,deviations] = PLS_basesLatest(X,Y,min(options.Factor,min(Dim)),0,0);
        [Wpls,Qpls,evals,deviations] = PLS_basesLatest(X,Y,min(options.Factor,min(Dim)),0,0);
        Wout{1,1}.Bases = Wpls;
        Wout{2,1}.Bases = Qpls;    
        
        for i = 1:nV
            Wout{i,1}.Evals = evals;
            if isfield(options,'PCA')
                Wout{i,1}.evs = evFin{i,1};
                Wout{i,1}.mPCA = mPCA{i,1};
            end
        end

        %%% ========================= LDA STARTS ==========================

    case 'lda'

        if (isfield(options,'AlignMode'))
            alignMode = options.AlignMode;
        else
            alignMode = 1; % Align all the samples
        end

        for i = 1:nV
            tmp = dataCell{i,1}.label;
            label = unique(tmp);
            Wout{i,1}.classMean = zeros(1,Dim(i));
            for c = 1 : length(label)
                fil = tmp == label(c);
                Wout{i,1}.classMean(c,:) =  mean(dataCell{i,1}.data(fil,:));
            end
        end


        switch alignMode
            case 1 % Align all samples
                for i = 1:nV
                    Wout{i,1}.alignCol = dataCell{i,1}.data;
                end
            case 2 % Align class centres
                for i = 1:nV
                    Wout{i,1}.alignCol = Wout{i,1}.classMean;
                end
            case 3 % Align after clustering
                k = options.NumCluster;
                for in = 1:nV
                    Wout{in,1}.classLabel = label;
                    Wout{in,1}.classID = cell(length(label),1);
                    Wout{in,1}.classCent = cell(length(label),1);
                    Wout{in,1}.alignCol = [];
                    if in == 1
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            [id cC] = kmeans(tmp,k,'emptyaction','drop');
                            Wout{in,1}.classID{c,1} = id;
                            Wout{in,1}.classCent{c,1} =  cC;
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; cC];
                        end
                    else
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            Wout{in,1}.classID{c,1} = Wout{1,1}.classID{c,1};
                            for inn = 1:k
                                fil1 = Wout{in,1}.classID{c,1} == inn;
                                Wout{in,1}.classCent{c,1}(inn,:) = mean(tmp(fil1,:));
                            end
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; Wout{in,1}.classCent{c,1}];
                        end
                    end
                end

            case 4
                if (isfield(options,'nPair') && options.nPair > 0)
                C1 = generateRandomPairs(vLabel,options.nPair);
                else
                    options.nPair = length(vLabel{1,1})*2;
                    C1 = generateRandomPairs(vLabel,options.nPair);
                end
                for i = 1:nV
                    Wout{i,1}.alignCol = dataCell{i,1}.data(C1(:,i),:);
                    Wout{i,1}.alignCol = [Wout{i,1}.alignCol; Wout{i,1}.classMean];
                end
        end
        Wf = zeros(sum(Dim),sum(Dim));
        Df = zeros(sum(Dim),sum(Dim));
        for r = 1:nV
            for c = r:nV
                rs = sum(Dim(1:r-1))+1;
                re = sum(Dim(1:r));
                cs = sum(Dim(1:c-1))+1;
                ce = sum(Dim(1:c));
                if r == c
                    Wf(rs:re,cs:ce) = W{r,1};
                    Df(rs:re,cs:ce) = D{r,1}*options.Mult(r);
                else
                    tmp = Wout{r,1}.alignCol'*Wout{c,1}.alignCol*options.Lamda;
                    Wf(rs:re,cs:ce) = tmp;
                    Wf(cs:ce,rs:re) = tmp';
                end
            end
        end


        if (isfield(options,'Autopara') && options.Autopara)
            [Wf Df] = tuneParameter(Wf,Df,options,Dim);
        else
            Df = Df + options.ReguAlpha*eye(sum(Dim));
            Df = (Df + Df')/2;
            Wf = (Wf + Wf')/2;
        end
        opts.disp = 0;
        [eigVec eigVal] = eigs(Wf,Df,min(options.Factor,min(Dim)),'LA',opts);

        for i = 1:nV
            Wout{i,1}.Bases = eigVec(sum(Dim(1:i-1))+1:sum(Dim(1:i)),:);
            Wout{i,1}.Evals = diag(eigVal);
            if (isfield(options,'PCA') && options.PCA)
                Wout{i,1}.evs = evFin{i,1};
                Wout{i,1}.mPCA = mPCA{i,1};
            end
        end
        %% ----------------------------------------------------- MFA STARTS ---------------------------------------------
    case 'mfa'
        if (isfield(options,'AlignMode'))
            alignMode = options.AlignMode;
        else
            alignMode = 1; % Align all the samples
        end

        for i = 1:nV
            tmp = dataCell{i,1}.label;
            label = unique(tmp);
            Wout{i,1}.classMean = zeros(1,Dim(i));
            for c = 1 : length(label)
                fil = tmp == label(c);
                Wout{i,1}.classMean(c,:) =  mean(dataCell{i,1}.data(fil,:));
            end
        end


        switch alignMode
            case 1 % Align all samples
                for i = 1:nV
                    Wout{i,1}.alignCol = dataCell{i,1}.data;
                end
            case 2 % Align class centres
                for i = 1:nV
                    Wout{i,1}.alignCol = Wout{i,1}.classMean;
                end
            case 3 % Align after clustering
                k = options.NumCluster;
                for in = 1:nV
                    Wout{in,1}.classLabel = label;
                    Wout{in,1}.classID = cell(length(label),1);
                    Wout{in,1}.classCent = cell(length(label),1);
                    Wout{in,1}.alignCol = [];
                    if in == 1
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            [id cC] = kmeans(tmp,k,'emptyaction','drop');
                            Wout{in,1}.classID{c,1} = id;
                            Wout{in,1}.classCent{c,1} =  cC;
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; cC];
                        end
                    else
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            Wout{in,1}.classID{c,1} = Wout{1,1}.classID{c,1};
                            for inn = 1:k
                                fil1 = Wout{in,1}.classID{c,1} == inn;
                                Wout{in,1}.classCent{c,1}(inn,:) = mean(tmp(fil1,:));
                            end
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; Wout{in,1}.classCent{c,1}];
                        end
                    end
                end

            case 4
                if (isfield(options,'nPair') && options.nPair > 0)
                C1 = generateRandomPairs(vLabel,options.nPair);
                else
                    options.nPair = length(vLabel{1,1})*2;
                    C1 = generateRandomPairs(vLabel,options.nPair);
                end
                    
                for i = 1:nV
                    Wout{i,1}.alignCol = dataCell{i,1}.data(C1(:,i),:);
                    Wout{i,1}.alignCol = [Wout{i,1}.alignCol; Wout{i,1}.classMean];
                end
        end
        Wf = zeros(sum(Dim),sum(Dim));
        Df = zeros(sum(Dim),sum(Dim));
        for r = 1:nV
            for c = r:nV
                rs = sum(Dim(1:r-1))+1;
                re = sum(Dim(1:r));
                cs = sum(Dim(1:c-1))+1;
                ce = sum(Dim(1:c));
                if r == c
                    Wf(rs:re,cs:ce) = W{r,1};
                    Df(rs:re,cs:ce) = D{r,1}*options.Mult(r) + options.ReguAlpha*eye(Dim(r));
                else
                    tmp = Wout{r,1}.alignCol'*Wout{c,1}.alignCol*options.Lamda;
                    Wf(rs:re,cs:ce) = tmp;
                    Wf(cs:ce,rs:re) = tmp';
                end
            end
        end

        if (isfield(options,'Autopara') && options.Autopara)
            [Wf Df] = tuneParameter(Wf,Df,options,Dim);
        else
            Df = Df + options.ReguAlpha*eye(sum(Dim));
            Df = (Df + Df')/2;
            Wf = (Wf + Wf')/2;
        end
        opts.disp = 0;
        [eigVec, eigVal] = eigs(Wf,Df,min(options.Factor,min(Dim)),'LA',opts);

        for i = 1:nV
            Wout{i,1}.Bases = eigVec(sum(Dim(1:i-1))+1:sum(Dim(1:i)),:);
            Wout{i,1}.Evals = diag(eigVal);
            if isfield(options,'PCA')
                Wout{i,1}.mPCA = mPCA{i,1};
                Wout{i,1}.evs = evFin{i,1};
            end
        end

        %% -------------------- LPP STARTS ---------------------------
    case 'lpp'
        if (isfield(options,'AlignMode'))
            alignMode = options.AlignMode;
        else
            alignMode = 1; % Align all the samples
        end

        for i = 1:nV
            tmp = dataCell{i,1}.label;
            label = unique(tmp);
            Wout{i,1}.classMean = zeros(1,Dim(i));
            for c = 1 : length(label)
                fil = tmp == label(c);
                Wout{i,1}.classMean(c,:) =  mean(dataCell{i,1}.data(fil,:));
            end
        end


        switch alignMode
            case 1 % Align all samples
                for i = 1:nV
                    Wout{i,1}.alignCol = dataCell{i,1}.data;
                end
            case 2 % Align class centres
                for i = 1:nV
                    Wout{i,1}.alignCol = Wout{i,1}.classMean;
                end
            case 3 % Align after clustering
                k = options.NumCluster;
                for in = 1:nV
                    Wout{in,1}.classLabel = label;
                    Wout{in,1}.classID = cell(length(label),1);
                    Wout{in,1}.classCent = cell(length(label),1);
                    Wout{in,1}.alignCol = [];
                    if in == 1
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            [id cC] = kmeans(tmp,k,'emptyaction','drop');
                            Wout{in,1}.classID{c,1} = id;
                            Wout{in,1}.classCent{c,1} =  cC;
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; cC];
                        end
                    else
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            Wout{in,1}.classID{c,1} = Wout{1,1}.classID{c,1};
                            for inn = 1:k
                                fil1 = Wout{in,1}.classID{c,1} == inn;
                                Wout{in,1}.classCent{c,1}(inn,:) = mean(tmp(fil1,:));
                            end
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; Wout{in,1}.classCent{c,1}];
                        end
                    end
                end

            case 4
                if (isfield(options,'nPair') && options.nPair > 0)
                C1 = generateRandomPairs(vLabel,options.nPair);
                else
                    options.nPair = length(vLabel{1,1})*2;
                    C1 = generateRandomPairs(vLabel,options.nPair);
                end
                for i = 1:nV
                    Wout{i,1}.alignCol = dataCell{i,1}.data(C1(:,i),:);
                    Wout{i,1}.alignCol = [Wout{i,1}.alignCol; Wout{i,1}.classMean];
                end
        end
        Wf = zeros(sum(Dim),sum(Dim));
        Df = zeros(sum(Dim),sum(Dim));
        for r = 1:nV
            for c = r:nV
                rs = sum(Dim(1:r-1))+1;
                re = sum(Dim(1:r));
                cs = sum(Dim(1:c-1))+1;
                ce = sum(Dim(1:c));
                if r == c
                    Wf(rs:re,cs:ce) = W{r,1};
                    Df(rs:re,cs:ce) = D{r,1}*options.Mult(r) + options.ReguAlpha*eye(Dim(r));
                else
                    tmp = Wout{r,1}.alignCol'*Wout{c,1}.alignCol*options.Lamda;
                    Wf(rs:re,cs:ce) = tmp;
                    Wf(cs:ce,rs:re) = tmp';
                end
            end
        end

        if (isfield(options,'Autopara') && options.Autopara)
            [Wf Df] = tuneParameter(Wf,Df,options,Dim);
        else
            Df = Df + options.ReguAlpha*eye(sum(Dim));
            Df = (Df + Df')/2;
            Wf = (Wf + Wf')/2;
        end
        opts.disp = 0;
        [eigVec eigVal] = eigs(Wf,Df,min(options.Factor,min(Dim)),'LA',opts);

        for i = 1:nV
            Wout{i,1}.Bases = eigVec(sum(Dim(1:i-1))+1:sum(Dim(1:i)),:);
            Wout{i,1}.Evals = diag(eigVal);
            if isfield(options,'PCA')
                Wout{i,1}.evs = evFin{i,1};
                Wout{i,1}.mPCA = mPCA{i,1};
            end
        end
        %% ------------------------ NPE STARTS --------------------------------------
    case 'npe'
        if (isfield(options,'AlignMode'))
            alignMode = options.AlignMode;
        else
            alignMode = 1; % Align all the samples
        end

        for i = 1:nV
            tmp = dataCell{i,1}.label;
            label = unique(tmp);
            Wout{i,1}.classMean = zeros(1,Dim(i));
            for c = 1 : length(label)
                fil = tmp == label(c);
                Wout{i,1}.classMean(c,:) =  mean(dataCell{i,1}.data(fil,:));
            end
        end


        switch alignMode
            case 1 % Align all samples
                for i = 1:nV
                    Wout{i,1}.alignCol = dataCell{i,1}.data;
                end
            case 2 % Align class centres
                for i = 1:nV
                    Wout{i,1}.alignCol = Wout{i,1}.classMean;
                end
            case 3 % Align after clustering
                k = options.NumCluster;
                for in = 1:nV
                    Wout{in,1}.classLabel = label;
                    Wout{in,1}.classID = cell(length(label),1);
                    Wout{in,1}.classCent = cell(length(label),1);
                    Wout{in,1}.alignCol = [];
                    if in == 1
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            [id cC] = kmeans(tmp,k,'emptyaction','drop');
                            Wout{in,1}.classID{c,1} = id;
                            Wout{in,1}.classCent{c,1} =  cC;
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; cC];
                        end
                    else
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            Wout{in,1}.classID{c,1} = Wout{1,1}.classID{c,1};
                            for inn = 1:k
                                fil1 = Wout{in,1}.classID{c,1} == inn;
                                Wout{in,1}.classCent{c,1}(inn,:) = mean(tmp(fil1,:));
                            end
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; Wout{in,1}.classCent{c,1}];
                        end
                    end
                end
        end
        Wf = zeros(sum(Dim),sum(Dim));
        Df = zeros(sum(Dim),sum(Dim));
        for r = 1:nV
            for c = r:nV
                rs = sum(Dim(1:r-1))+1;
                re = sum(Dim(1:r));
                cs = sum(Dim(1:c-1))+1;
                ce = sum(Dim(1:c));
                if r == c
                    Wf(rs:re,cs:ce) = W{r,1};
                    Df(rs:re,cs:ce) = D{r,1}*options.Mult(r) + options.ReguAlpha*eye(Dim(r));
                else
                    tmp = Wout{r,1}.alignCol'*Wout{c,1}.alignCol*options.Lamda;
                    Wf(rs:re,cs:ce) = tmp;
                    Wf(cs:ce,rs:re) = tmp';
                end
            end
        end
        if (isfield(options,'Autopara') && options.Autopara)
            [Wf Df] = tuneParameter(Wf,Df,options,Dim);
        else
            Df = Df + options.ReguAlpha*eye(sum(Dim));
            Df = (Df + Df')/2;
            Wf = (Wf + Wf')/2;
        end
        opts.disp = 0;
        [eigVec eigVal] = eigs(Wf,Df,min(options.Factor,min(Dim)),'LA',opts);

        for i = 1:nV
            Wout{i,1}.Bases = eigVec(sum(Dim(1:i-1))+1:sum(Dim(1:i)),:);
            Wout{i,1}.Evals = diag(eigVal);
            if isfield(options,'PCA')
                Wout{i,1}.evs = evFin{i,1};
                Wout{i,1}.mPCA = mPCA{i,1};
            end
        end

        %% ------------------------------- ISOP STARTS ------------------------------------
    case 'isop'
        if (isfield(options,'AlignMode'))
            alignMode = options.AlignMode;
        else
            alignMode = 1; % Align all the samples
        end

        for i = 1:nV
            tmp = dataCell{i,1}.label;
            label = unique(tmp);
            Wout{i,1}.classMean = zeros(1,Dim(i));
            for c = 1 : length(label)
                fil = tmp == label(c);
                Wout{i,1}.classMean(c,:) =  mean(dataCell{i,1}.data(fil,:));
            end
        end


        switch alignMode
            case 1 % Align all samples
                for i = 1:nV
                    Wout{i,1}.alignCol = dataCell{i,1}.data;
                end
            case 2 % Align class centres
                for i = 1:nV
                    Wout{i,1}.alignCol = Wout{i,1}.classMean;
                end
            case 3 % Align after clustering
                k = options.NumCluster;
                for in = 1:nV
                    Wout{in,1}.classLabel = label;
                    Wout{in,1}.classID = cell(length(label),1);
                    Wout{in,1}.classCent = cell(length(label),1);
                    Wout{in,1}.alignCol = [];
                    if in == 1
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            [id cC] = kmeans(tmp,k,'emptyaction','drop');
                            Wout{in,1}.classID{c,1} = id;
                            Wout{in,1}.classCent{c,1} =  cC;
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; cC];
                        end
                    else
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            Wout{in,1}.classID{c,1} = Wout{1,1}.classID{c,1};
                            for inn = 1:k
                                fil1 = Wout{in,1}.classID{c,1} == inn;
                                Wout{in,1}.classCent{c,1}(inn,:) = mean(tmp(fil1,:));
                            end
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; Wout{in,1}.classCent{c,1}];
                        end
                    end
                end
        end
        Wf = zeros(sum(Dim),sum(Dim));
        Df = zeros(sum(Dim),sum(Dim));
        for r = 1:nV
            for c = r:nV
                rs = sum(Dim(1:r-1))+1;
                re = sum(Dim(1:r));
                cs = sum(Dim(1:c-1))+1;
                ce = sum(Dim(1:c));
                if r == c
                    Wf(rs:re,cs:ce) = W{r,1};
                    Df(rs:re,cs:ce) = D{r,1}*options.Mult(r) + options.ReguAlpha*eye(Dim(r));
                else
                    tmp = Wout{r,1}.alignCol'*Wout{c,1}.alignCol*options.Lamda;
                    Wf(rs:re,cs:ce) = tmp;
                    Wf(cs:ce,rs:re) = tmp';
                end
            end
        end
        if (isfield(options,'Autopara') && options.Autopara)
            [Wf Df] = tuneParameter(Wf,Df,options,Dim);
        else
            Df = Df + options.ReguAlpha*eye(sum(Dim));
            Df = (Df + Df')/2;
            Wf = (Wf + Wf')/2;
        end
        opts.disp = 0;
        [eigVec eigVal] = eigs(Wf,Df,min(options.Factor,min(Dim)),'LA',opts);

        for i = 1:nV
            Wout{i,1}.Bases = eigVec(sum(Dim(1:i-1))+1:sum(Dim(1:i)),:);
            Wout{i,1}.Evals = diag(eigVal);
            if isfield(options,'PCA')
                Wout{i,1}.evs = evFin{i,1};
                Wout{i,1}.mPCA = mPCA{i,1};
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%% CCA starts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
    case 'cca'

        if (isfield(options,'AlignMode'))
            alignMode = options.AlignMode;
        else
            alignMode = 1; % Align all the samples
        end

        for i = 1:nV
            tmp = dataCell{i,1}.label;
            label = unique(tmp);
            Wout{i,1}.classMean = zeros(1,Dim(i));
            for c = 1 : length(label)
                fil = tmp == label(c);
                Wout{i,1}.classMean(c,:) =  mean(dataCell{i,1}.data(fil,:));
            end
        end


        switch alignMode
            case 1 % Align all samples
                for i = 1:nV
                    Wout{i,1}.alignCol = dataCell{i,1}.data;
                end
            case 2 % Align class centres
                for i = 1:nV
                    Wout{i,1}.alignCol = Wout{i,1}.classMean;
                end
            case 3 % Align after clustering
                k = options.NumCluster;
                for in = 1:nV
                    Wout{in,1}.classLabel = label;
                    Wout{in,1}.classID = cell(length(label),1);
                    Wout{in,1}.classCent = cell(length(label),1);
                    Wout{in,1}.alignCol = [];
                    if in == 1
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            [id cC] = kmeans(tmp,k,'emptyaction','drop');
                            Wout{in,1}.classID{c,1} = id;
                            Wout{in,1}.classCent{c,1} =  cC;
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; cC];
                        end
                    else
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            Wout{in,1}.classID{c,1} = Wout{1,1}.classID{c,1};
                            for inn = 1:k
                                fil1 = Wout{in,1}.classID{c,1} == inn;
                                Wout{in,1}.classCent{c,1}(inn,:) = mean(tmp(fil1,:));
                            end
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; Wout{in,1}.classCent{c,1}];
                        end
                    end
                end

            case 4
                if (isfield(options,'nPair') && options.nPair > 0)
                C1 = generateRandomPairs(vLabel,options.nPair);
                else
                    options.nPair = length(vLabel{1,1})*2;
                    C1 = generateRandomPairs(vLabel,options.nPair);
                end
                
                for i = 1:nV
                    Wout{i,1}.alignCol = dataCell{i,1}.data(C1(:,i),:);
                    Wout{i,1}.alignCol = [Wout{i,1}.alignCol; Wout{i,1}.classMean];
                end
        end
        Wf = zeros(sum(Dim),sum(Dim));
        Df = zeros(sum(Dim),sum(Dim));
        for r = 1:nV
            for c = r:nV
                rs = sum(Dim(1:r-1))+1;
                re = sum(Dim(1:r));
                cs = sum(Dim(1:c-1))+1;
                ce = sum(Dim(1:c));
                if r == c
                    Wf(rs:re,cs:ce) = W{r,1};
                    Df(rs:re,cs:ce) = D{r,1} + options.ReguAlpha*eye(Dim(r));
                else
                    tmp = Wout{r,1}.alignCol'*Wout{c,1}.alignCol*options.Lamda;
                    Wf(rs:re,cs:ce) = tmp;
                    Wf(cs:ce,rs:re) = tmp';
                end
            end
        end
        if (isfield(options,'Autopara') && options.Autopara)
            [Wf Df] = tuneParameter(Wf,Df,options,Dim);
        else
            Df = Df + options.ReguAlpha*eye(sum(Dim));
            Df = (Df + Df')/2;
            Wf = (Wf + Wf')/2;
        end
        opts.disp = 0;
        [eigVec eigVal] = eigs(Wf,Df,min(options.Factor,min(Dim)),'LA',opts);

        for i = 1:nV
            Wout{i,1}.Bases = eigVec(sum(Dim(1:i-1))+1:sum(Dim(1:i)),:);
            Wout{i,1}.Evals = diag(eigVal);
            if (isfield(options,'PCA') && options.PCA)
                Wout{i,1}.evs = evFin{i,1};
                Wout{i,1}.mPCA = mPCA{i,1};
            end
        end
        % ============================   BLM STARTS   ==========================================
    case 'blm'

        if (isfield(options,'AlignMode'))
            alignMode = options.AlignMode;
        else
            alignMode = 1; % Align all the samples
        end

        for i = 1:nV
            tmp = dataCell{i,1}.label;
            label = unique(tmp);
            Wout{i,1}.classMean = zeros(1,Dim(i));
            for c = 1 : length(label)
                fil = tmp == label(c);
                Wout{i,1}.classMean(c,:) =  mean(dataCell{i,1}.data(fil,:));
            end
        end


        switch alignMode
            case 1 % Align all samples
                for i = 1:nV
                    Wout{i,1}.alignCol = dataCell{i,1}.data;
                end
            case 2 % Align class centres
                for i = 1:nV
                    Wout{i,1}.alignCol = Wout{i,1}.classMean;
                end
            case 3 % Align after clustering
                k = options.NumCluster;
                for in = 1:nV
                    Wout{in,1}.classLabel = label;
                    Wout{in,1}.classID = cell(length(label),1);
                    Wout{in,1}.classCent = cell(length(label),1);
                    Wout{in,1}.alignCol = [];
                    if in == 1
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            [id cC] = kmeans(tmp,k,'emptyaction','drop');
                            Wout{in,1}.classID{c,1} = id;
                            Wout{in,1}.classCent{c,1} =  cC;
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; cC];
                        end
                    else
                        for c = 1:length(label)
                            fil = dataCell{in,1}.label == label(c);
                            tmp = dataCell{in,1}.data(fil,:);
                            Wout{in,1}.classID{c,1} = Wout{1,1}.classID{c,1};
                            for inn = 1:k
                                fil1 = Wout{in,1}.classID{c,1} == inn;
                                Wout{in,1}.classCent{c,1}(inn,:) = mean(tmp(fil1,:));
                            end
                            Wout{in,1}.alignCol = [ Wout{in,1}.alignCol ; Wout{in,1}.classCent{c,1}];
                        end
                    end
                end

            case 4
                if (isfield(options,'nPair') && options.nPair > 0)
                C1 = generateRandomPairs(vLabel,options.nPair);
                else
                    options.nPair = length(vLabel{1,1})*2;
                    C1 = generateRandomPairs(vLabel,options.nPair);
                end
                for i = 1:nV
                    Wout{i,1}.alignCol = dataCell{i,1}.data(C1(:,i),:);
                    Wout{i,1}.alignCol = [Wout{i,1}.alignCol; Wout{i,1}.classMean];
                end
        end
        Wf = zeros(sum(Dim),sum(Dim));

        for r = 1:nV
            for c = r:nV
                rs = sum(Dim(1:r-1))+1;
                re = sum(Dim(1:r));
                cs = sum(Dim(1:c-1))+1;
                ce = sum(Dim(1:c));
                if r == c
                    Wf(rs:re,cs:ce) = W{r,1};

                else
                    tmp = Wout{r,1}.alignCol'*Wout{c,1}.alignCol*options.Lamda;
                    Wf(rs:re,cs:ce) = tmp;
                    Wf(cs:ce,rs:re) = tmp';
                end
            end
        end

        Wf = (Wf + Wf')/2;
        opts.disp = 0;
        [eigVec eigVal] = eigs(Wf,min(options.Factor,sum(Dim)),'LA',opts);

        for i = 1:nV
            Wout{i,1}.Bases = eigVec(sum(Dim(1:i-1))+1:sum(Dim(1:i)),:);
            Wout{i,1}.Evals = diag(eigVal);
            if isfield(options,'PCA')
                Wout{i,1}.evs = evFin{i,1};
                Wout{i,1}.mPCA = mPCA{i,1};
            end
        end
end
end
%% ------------------------ FINISH -----------------------------------------------


function [W D] = tuneParameter(A,B,options,Dim)
W = (A+A')/2;
D = B;

nV = length(Dim);
%Wsc = zeros(nV,1);
Dsc = zeros(nV,1);
%Csc = zeros(nV,nV);

for r1 = 1:nV
    for c1 = r1:nV
        rs1 = sum(Dim(1:r1-1))+1;
        re1 = sum(Dim(1:r1));
        cs1 = sum(Dim(1:c1-1))+1;
        ce1 = sum(Dim(1:c1));
        if r1 == c1

            if r1 == 1
                % Wsc(r1) = trace(W(rs1:re1,cs1:ce1));
                D(rs1:re1,cs1:ce1) = D(rs1:re1,cs1:ce1)/options.Mult(r1);
                Dsc(r1) = trace(D(rs1:re1,cs1:ce1));

            else

                % Wsc(r1) = trace(W(rs1:re1,cs1:ce1));
                Dsc(r1) = trace(D(rs1:re1,cs1:ce1));
                %Wratio = Wsc(1)/Wsc(r1);
                Dratio = Dsc(1)/Dsc(r1);

                % W(rs1:re1,cs1:ce1) = Wratio*W(rs1:re1,cs1:ce1);
                D(rs1:re1,cs1:ce1) = D(rs1:re1,cs1:ce1)*Dratio;
            end

            %                         else
            %                             W(rs1:re1,cs1:ce1) = W(rs1:re1,cs1:ce1)/options.Lamda;
            %                             W(cs1:ce1,rs1:re1) = W(cs1:ce1,rs1:re1)/options.Lamda;
            %                             Csc(r1,c1) = norm(W(rs1:re1,cs1:ce1),'fro');
            %                             Cratio = Wsc(1)/Csc(r1,c1);
            %                             W(rs1:re1,cs1:ce1) = W(rs1:re1,cs1:ce1)*Cratio*options.Lamda;
            %                             W(cs1:ce1,rs1:re1) = W(cs1:ce1,rs1:re1)*Cratio*options.Lamda;
        end
    end
end

D = D + options.ReguAlpha*eye(sum(Dim));
%W = (W + W')/2;
D = (D+D')/2;

end


function [X ev meanPCA] = pcaIn(Xin,opts)
oTmp.disp = 0;
numSamples = size(Xin,1); % Xin is in the form where one sample is in one row

if isfield(opts,'meanMinus') && opts.meanMinus
    A = Xin*Xin';
else
    Xin = Xin - repmat(mean(Xin,1),numSamples,1);
    A = Xin*Xin';
end

dim = size(Xin,2);
[ev ed] = eigs(A,numSamples-1,'LA',oTmp);

if (~isfield(opts,'PCAthresh'))
    opts.PCAthresh = 0.95; % default
end

if opts.PCAthresh > 1
    ev = ev(:,1:opts.PCAthresh);
else
    counter = 0;
    ed1 = diag(ed);
    tot = sum(ed1);
    rat = 0;
    runSum = 0;
    while rat < opts.PCAthresh
        counter = counter +1;
        runSum = runSum + ed1(counter);
        rat = runSum/tot;
    end
    ev = ev(:,1:counter);
    ed = ed(1:counter,1:counter);
    ev = Xin'*ev;
    ev = ev*ed^(-0.5);
end
X = Xin*ev;
meanPCA = mean(X);
X = X - repmat(mean(X,1),numSamples,1);
end
