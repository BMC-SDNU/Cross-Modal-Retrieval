function [info] = retrieval(q,gt_q,retriev,gt_r,performancemeasure);
%
%       [info] = retrieval(quer,gt_q,retr,gt_r,opt)
%
%               q    : query set (one sample per row)
%               gt_q    : groundtruth for the query set
%
%               retriev    : retrieval set (one sample per row)
%               gt_r    : groundtruth for the retrieval set
%
%performancemeasure = 'PrecisionatK' / 'NDCGatK'

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = q;
Y = retriev;
% DIST =1-pdist2(X,Y,'euclidean');
% DIST = X*Y.';

DIST =1-pdist2(X,Y,'hamming');

DIST(isnan(DIST)) = -1000000000;

if(strcmp(performancemeasure,'PrecisionatK'))
    pk = [];
    tp = 0;
    fp = 0;
    K = 10;
    for query = 1:size(X,1)
        idx = gt_q(query,:);
        d = DIST(query,:);
        [c2,id2]=sort(d,'descend'); %descend as cosine similarity
        
        for i=1:K
            
            idy = gt_r(id2(i),:);
            
            for subquery =  1:size(idx,2)
                if(idy(1,subquery)>0)
                    if(idx(1,subquery)>0)
                        tp = tp + 1;
                    end
                    if(idx(1,subquery)==0)
                        fp = fp + 1;
                    end
                end
            end
            
            
            
        end
        
    end
    PK = tp/(tp+fp);
    info.result= PK;
end
if(strcmp(performancemeasure,'NDCGatK'))
    K = 30;
    catte1 = gt_r;
    NDCG = [];
    
    %***********************************************
    D = 1-pdist2(catte1,catte1,'cosine');
    %***********************************************
    
    
    for query = 1:size(X,1)
        idx = gt_q(query,:);
        if(nnz(idx)>0)
            d = DIST(query,:);
            [c2,id2]=sort(d,'descend'); %descend as cosine similarity
            
            
            %***********************************************
%             d1 = 1-pdist2(catte1(query,:),catte1,'cosine');
            d1 = D(query,:);
            %***********************************************
            
            
            
            d2 = sort(d1,'descend');
            dcg = d1(id2(1));
            idcg = d2(1);
            for i=2:K
                dcg = dcg+(d1(id2(i))/log2(i));
                idcg = idcg + (d2(i)/log2(i));
            end
            NDCG = vertcat(NDCG,dcg/idcg);
        end
    end
    info.result= mean(NDCG);
end
