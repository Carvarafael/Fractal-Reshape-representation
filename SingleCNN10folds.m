function [origFeatures,YPred,probs, vLossOrig, time_training,flattingLayer1,flattingLayerTrain] = SingleCNN10folds(Train_dataset,Test_dataset, Neural,numClasses, ENVIRONMENT, ME,folds,learningRate)

%Hiper Parâmetros
MiniBatch = 32;
ILR = 0.01;
LRDP = 2;
LRDF = 0.75;
L2R = learningRate;

for i=1:folds
    %Ler folds a partir dos arquivos
    imdsValidationOrig = Test_dataset{1, i};
    imdsTrainOrig = Train_dataset{1, i};
    [imdsTrainOrig,imdsValidation] = splitEachLabel(imdsTrainOrig,0.7);
    %Número de amostras de treinamento
    trainingSamples = countEachLabel(imdsTrainOrig);
    trainingSamples = table2cell(trainingSamples);
%     t_num = 0;
%     for j = 1:numClasses
%         t_num = t_num + trainingSamples{j,2};
%     end

    %Número de amostras de validação
    validationSamples = countEachLabel(imdsValidation); %Number of validation samples
    validationSamples = table2cell(validationSamples);
%     v_num = 0;
%     for j = 1:numClasses
%         v_num = v_num + validationSamples{j,2};
%     end

%     imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[-20,20], ...
%     'RandXTranslation',[-3 3], ...
%     'RandYTranslation',[-3 3]);
   
    
    trainResizeOrig = augmentedImageDatastore(Neural.Layers(1, 1).InputSize, imdsTrainOrig);%, 'DataAugmentation', imageAugmenter);
    validationResizeOrig = augmentedImageDatastore(Neural.Layers(1, 1).InputSize, imdsValidation);%, 'DataAugmentation', imageAugmenter);
    validationResizeOrig2 = augmentedImageDatastore(Neural.Layers(1, 1).InputSize, imdsValidationOrig);%, 'DataAugmentation', imageAugmenter);
    
    


    %Configuração da CNN
    optionsOrig = trainingOptions(  'sgdm', 'MiniBatchSize',MiniBatch, 'MaxEpochs',20, 'InitialLearnRate',ILR,...
    'LearnRateSchedule','piecewise', 'LearnRateDropPeriod', LRDP, 'LearnRateDropFactor', LRDF, ...
'Shuffle','every-epoch', 'ValidationData',validationResizeOrig, 'ValidationFrequency',5,'Verbose'...
,false, 'Plots','None', 'ExecutionEnvironment',ENVIRONMENT, 'L2Regularization', L2R);



    %Now this is where the fun begins
    tic
    try
        [netOrig, infoResNet50Orig{1,i}] = trainNetwork(trainResizeOrig,Neural,optionsOrig);
    catch
        warning('Problem using GPU. Trying CPU this iteration');
        options = trainingOptions('sgdm', 'MiniBatchSize',MiniBatch, 'MaxEpochs',ME, 'InitialLearnRate',ILR, 'LearnRateSchedule','piecewise', 'LearnRateDropPeriod', LRDP, 'LearnRateDropFactor', LRDF, 'Shuffle','every-epoch', 'ValidationData',validationResizeOrig, 'ValidationFrequency',10, 'Verbose',false, 'Plots','training-progress', 'ExecutionEnvironment',ENVIRONMENT, 'L2Regularization', L2R);
        [netOrig, infoResNet50Orig{1,i}] = trainNetwork(trainResizeOrig,Neural,options);
    end
    
%     figure
%     x = 1:50;
%     y = infoResNet50Orig{1, 5}.ValidationAccuracy;
%     y2 = infoResNet50Orig{1, 5}.TrainingAccuracy;
%     xs = x(~isnan(y));
%     ys = y(~isnan(y));
%     yi = interp1(xs, ys, x, 'Linear');
%     plot(x,y2,x,yi)

    
    %Get softmax Layer
     softmaxLayer = 'Softmax';
   % softmaxLayer = 'fc1000_softmax';
    
    %flattingLayer = 'efficientnet-b0|model|blocks_0|se|GlobAvgPool';
   %flattingLayer = 'max_pooling2d_1';
   %  flattingLayer2 = 'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool';
   %flattingLayer2 = 'avg_pool';
    %Obter probabilidades da camada Softmax
    try
        featuresTestSoftmaxOrig = activations(netOrig,validationResizeOrig2,softmaxLayer,'OutputAs','rows');
        %featuresTestflattingLayer = activations(netOrig,validationResizeOrig2,flattingLayer,'OutputAs','rows');
        %featuresTestflattingLayer2 = activations(netOrig,validationResizeOrig2,flattingLayer2,'OutputAs','rows');
        %featuresTrainflattingLayer = activations(netOrig,trainResizeOrig,flattingLayer,'OutputAs','rows');
        %featuresTrainflattingLayer2 = activations(netOrig,trainResizeOrig,flattingLayer2,'OutputAs','rows');
        [YPred,probs] = classify(netOrig,validationResizeOrig2);
    catch
        warning('Problem using GPU. Trying CPU this iteration');
        featuresTestSoftmaxOrig = activations(netOrig,validationResizeOrig2,softmaxLayer,'OutputAs','rows', 'ExecutionEnvironment','cpu');
        %featuresTestflattingLayer = activations(netOrig,validationResizeOrig2,flattingLayer,'OutputAs','rows', 'ExecutionEnvironment','cpu');
        %featuresTestflattingLayer2 = activations(netOrig,validationResizeOrig2,flattingLayer2,'OutputAs','rows', 'ExecutionEnvironment','cpu');
        %featuresTrainflattingLayer = activations(netOrig,trainResizeOrig,flattingLayer,'OutputAs','rows', 'ExecutionEnvironment','cpu');
        %featuresTrainflattingLayer2 = activations(netOrig,trainResizeOrig,flattingLayer2,'OutputAs','rows', 'ExecutionEnvironment','cpu');
        
    end
    %ConfusionMatricesOrig{1, i} = confusionmat(confusionTrue, confusionPredOrig);
    origFeatures{1, i} = featuresTestSoftmaxOrig;
    flattingLayer1{1, i} = {};
    flattingLayerTrain = {};
    %flattingLayer1{1, i} = featuresTestflattingLayer;
    %flattingLayer1{2, i} = featuresTestflattingLayer2;
    %flattingLayerTrain{1, i} = featuresTrainflattingLayer;
    %flattingLayerTrain{2, i} = featuresTrainflattingLayer2;

    time_training(1,i) = toc;
end
%Info{1, 1} = ConfusionMatricesOrig;

vLossOrig = infoResNet50Orig;