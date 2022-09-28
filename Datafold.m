function [Train, Test] = Datafold(Train_dataset,Test_dataset,n_folds)
Train = {};
Test = {};

for i=1:n_folds
    
Train{i} = imageDatastore(cat(1,Train_dataset{1, 1}{1, i}.Files,Train_dataset{1, 2}{1, i}.Files,Train_dataset{1, 3}{1, i}.Files,Train_dataset{1, 4}{1, i}.Files));
Train{i}.Labels = cat(1,Train_dataset{1, 1}{1, i}.Labels,Train_dataset{1, 2}{1, i}.Labels,Train_dataset{1, 3}{1, i}.Labels,Train_dataset{1, 4}{1, i}.Labels);
Test{i} = imageDatastore(cat(1,Test_dataset{1, 1}{1, i}.Files,Test_dataset{1, 2}{1, i}.Files,Test_dataset{1, 3}{1, i}.Files,Test_dataset{1, 4}{1, i}.Files));
Test{i}.Labels = cat(1,Test_dataset{1, 1}{1, i}.Labels,Test_dataset{1, 2}{1, i}.Labels,Test_dataset{1, 3}{1, i}.Labels,Test_dataset{1, 4}{1, i}.Labels);

end