function dataset = load_data(dataname)
switch dataname
    case 'flickr-25k'
        load ../data/flickr-25k.mat XTest YTest XDatabase YDatabase testL databaseL;
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XDatabase;
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
    case 'flickr-25k-vgg'
        load ../data/flickr-25k.mat VTest YTest VDatabase YDatabase testL databaseL;
        dataset.XTest = VTest;
        dataset.YTest = YTest;
        dataset.XDatabase = VDatabase;
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
    case 'iapr-tc12'
        load ../data/iapr-tc12.mat XTest YTest XDatabase YDatabase testL databaseL;
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XDatabase;
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
    case 'iapr-tc12-vgg'
        load ../data/iapr-tc12.mat VTest YTest VDatabase YDatabase testL databaseL;
        dataset.XTest = VTest;
        dataset.YTest = YTest;
        dataset.XDatabase = VDatabase;
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
    case 'nus-wide-tc10'
        load ../data/nus-wide-clear.mat XTest YTest XDatabase YDatabase testL databaseL;
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XDatabase;
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
    case 'nus-wide-tc10-vgg'
        load ../data/nus-wide.mat VTest YTest VDatabase YDatabase testL databaseL;
        dataset.XTest = VTest;
        dataset.YTest = YTest;
        dataset.XDatabase = VDatabase;
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
end
end

