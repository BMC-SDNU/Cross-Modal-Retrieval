%% Pre-processing data for cnn: substract mean and divided by std
%   inputMat: Mat file, each row is a data
%   val: returned mat
function val = pre_cnn(inputMat)
%     ori_data = load(inputMat);    
%     val = struct2array(ori_data);
    val = inputMat;
    
%     fprintf('Substract mean\n');
    mean_val = mean(val,1);
    mean_val = repmat(mean_val,[size(val,1),1]);
    val = val - mean_val;
    
%     fprintf('Divided by std\n');
    std_val = std(val);
    std_val(find(std_val==0))=1;
    std_val = repmat(std_val,[size(val,1),1]);
    val = val./std_val;
end