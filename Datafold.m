function [Train, Test] = Datafold(Paths_Train,Paths_Test,n_folds,column)


for i=1:n_folds
    
Train{i} = imageDatastore(cat(1,Paths_Train{1, 1}{1, i}(:,column),Paths_Train{1, 2}{1, i}(:,column),Paths_Train{1, 3}{1, i}(:,column),Paths_Train{1, 4}{1, i}(:,column)),"Labels", ...
    ...
    cat(1,Paths_Train{1, 1}{1, i}(:,2),Paths_Train{1, 2}{1, i}(:,2),Paths_Train{1, 3}{1, i}(:,2),Paths_Train{1, 4}{1, i}(:,2)));

Test{i} = imageDatastore(cat(1,Paths_Test{1, 1}{1, i}(:,column),Paths_Test{1, 2}{1, i}(:,column),Paths_Test{1, 3}{1, i}(:,column),Paths_Test{1, 4}{1, i}(:,column)),"Labels", ...
    ...
    cat(1,Paths_Test{1, 1}{1, i}(:,2),Paths_Test{1, 2}{1, i}(:,2),Paths_Test{1, 3}{1, i}(:,2),Paths_Test{1, 4}{1, i}(:,2)));

Train{i}.Labels = categorical(Train{i}.Labels);
Test{i}.Labels = categorical(Test{i}.Labels);
end