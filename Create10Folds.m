function [TrainFolds, TestFolds] = Create10Folds(datastore, FORMAT)

TrainFolds = cell(1,10);
TestFolds = cell(1,10);


folds = 10;
imds = imageDatastore(datastore, 'IncludeSubfolders',true, 'LabelSource','foldernames');
[imd1 imd2 imd3 imd4 imd5 imd6 imd7 imd8 imd9 imd10] = splitEachLabel(imds,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1, 'randomize');

partStores{1} = imd1.Files ;
partStores{2} = imd2.Files ;
partStores{3} = imd3.Files ;
partStores{4} = imd4.Files ;
partStores{5} = imd5.Files ;

partStores{6} = imd6.Files ;
partStores{7} = imd7.Files ;
partStores{8} = imd8.Files ;
partStores{9} = imd9.Files ;
partStores{10} = imd10.Files; 
idx = crossvalind('Kfold', folds, folds);

for i = 1:folds
    test_idx = (idx == i);
    train_idx = ~test_idx;
    imdsValidation = imageDatastore(partStores{test_idx}, 'IncludeSubfolders', true,'FileExtensions',FORMAT, 'LabelSource', 'foldernames');
    imdsTrain = imageDatastore(cat(1, partStores{train_idx}), 'IncludeSubfolders', true,'FileExtensions',FORMAT, 'LabelSource', 'foldernames');
    
    TrainFolds{1,i} = imdsTrain;
    TestFolds{1,i} = imdsValidation;
  
end

end