function dataset = load_data(dataname)
switch dataname
    case 'nus-vgg'
        load ./data/mynus_cnn.mat I_te I_tr T_te T_tr L_te L_tr;
        dataset.XTest = I_te;
        dataset.YTest = T_te;
        dataset.XDatabase = I_tr;
        dataset.YDatabase = T_tr;
        dataset.testL = L_te;
        dataset.databaseL = L_tr;
    case 'flickr-25k'
        load ./data/flickr-25k.mat XTest YTest XDatabase YDatabase testL databaseL;
        inx = randperm(size(databaseL,1),size(databaseL,1));
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XDatabase(inx,:);
        dataset.YDatabase = YDatabase(inx,:);
        dataset.testL = testL;
        dataset.databaseL = databaseL(inx,:);
    case 'flickr-25k-vgg'
        load ./data/flickr-25k.mat VTest YTest VDatabase YDatabase testL databaseL;
        inx = randperm(size(databaseL,1),10000);
        dataset.XTest = VTest;
        dataset.YTest = YTest;
        dataset.XDatabase = VDatabase(inx,:);
        dataset.YDatabase = YDatabase(inx,:);
        dataset.testL = testL;
        dataset.databaseL = databaseL(inx,:);
    case 'iapr-tc12'
        load ./data/iapr-tc12.mat XTest YTest XDatabase YDatabase testL databaseL;
        inx = randperm(size(databaseL,1),size(databaseL,1));
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XDatabase(inx,:);
        dataset.YDatabase = YDatabase(inx,:);
        dataset.testL = testL;
        dataset.databaseL = databaseL(inx,:);
    case 'iapr-tc12-vgg'
        load ./data/iapr-tc12.mat VTest YTest VDatabase YDatabase testL databaseL;
        inx = randperm(size(databaseL,1),10000);
        dataset.XTest = VTest;
        dataset.YTest = YTest;
        dataset.XDatabase = VDatabase;
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
    case 'nus-wide-tc10'
        load ./data/nus-wide-clear.mat XTest YTest XDatabase YDatabase testL databaseL;
        inx = randperm(size(databaseL,1),3000);
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XDatabase(inx,:);
        dataset.YDatabase = YDatabase(inx,:);
        dataset.testL = testL;
        dataset.databaseL = databaseL(inx,:);
    case 'nus-wide-tc10-vgg'
        load ./data/nus-wide.mat VTest YTest VDatabase YDatabase testL databaseL;
        inx = randperm(size(databaseL,1),15000);
        dataset.XTest = VTest;
        dataset.YTest = YTest;
        dataset.XDatabase = VDatabase(inx,:);
        dataset.YDatabase = YDatabase(inx,:);
        dataset.testL = testL;
        dataset.databaseL = databaseL(inx,:);
end
end

