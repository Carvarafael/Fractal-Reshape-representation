function [origFeatures,YPred,probs, vLossOrig, time_training] = SingleCNN10folds(Train_dataset,Test_dataset, Neural,numClasses, ENVIRONMENT, ME)
folds = 10;

%Hiper Parâmetros
MiniBatch = 32;
ILR = 0.01;
LRDP = 2;
LRDF = 0.75;
L2R = 0.0001;

for i=1:10
    %Ler folds a partir dos arquivos
    imdsValidationOrig = Test_dataset{1, i};
    imdsTrainOrig = Train_dataset{1, i};

    %Número de amostras de treinamento
    trainingSamples = countEachLabel(imdsTrainOrig);
    trainingSamples = table2cell(trainingSamples);
    t_num = 0;
    for j = 1:numClasses
        t_num = t_num + trainingSamples{j,2};
    end

    %Número de amostras de validação
    validationSamples = countEachLabel(imdsValidationOrig); %Number of validation samples
    validationSamples = table2cell(validationSamples);
    v_num = 0;
    for j = 1:numClasses
        v_num = v_num + validationSamples{j,2};
    end

%     imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[-20,20], ...
%     'RandXTranslation',[-3 3], ...
%     'RandYTranslation',[-3 3]);
   
    
    trainResizeOrig = augmentedImageDatastore(Neural.Layers(1, 1).InputSize, imdsTrainOrig);%, 'DataAugmentation', imageAugmenter);
    validationResizeOrig = augmentedImageDatastore(Neural.Layers(1, 1).InputSize, imdsValidationOrig);%, 'DataAugmentation', imageAugmenter);

    %Configuração da CNN
    optionsOrig = trainingOptions('sgdm', 'MiniBatchSize',MiniBatch, 'MaxEpochs',ME, 'InitialLearnRate',ILR, 'LearnRateSchedule','piecewise', 'LearnRateDropPeriod', LRDP, 'LearnRateDropFactor', LRDF, 'Shuffle','every-epoch', 'ValidationData',validationResizeOrig, 'ValidationFrequency',10,'Verbose',false, 'Plots','none', 'ExecutionEnvironment',ENVIRONMENT, 'L2Regularization', L2R);

    %Now this is where the fun begins
    tic
    try
        [netOrig, infoResNet50Orig{1,i}] = trainNetwork(trainResizeOrig,Neural,optionsOrig);
    catch
        warning('Problem using GPU. Trying CPU this iteration');
        options = trainingOptions('sgdm', 'MiniBatchSize',MiniBatch, 'MaxEpochs',ME, 'InitialLearnRate',ILR, 'LearnRateSchedule','piecewise', 'LearnRateDropPeriod', LRDP, 'LearnRateDropFactor', LRDF, 'Shuffle','every-epoch', 'ValidationData',validationResizeOrig, 'ValidationFrequency',10, 'Verbose',false, 'Plots','none', 'ExecutionEnvironment',ENVIRONMENT, 'L2Regularization', L2R);
        [netOrig, infoResNet50Orig{1,i}] = trainNetwork(trainResizeOrig,Neural,options);
    end

    %Get softmax Layer
    softmaxLayer = Neural.Layers(end-1).Name;
    
    %Obter probabilidades da camada Softmax
    try
        featuresTestSoftmaxOrig = activations(netOrig,validationResizeOrig,softmaxLayer,'OutputAs','rows');
        [YPred,probs] = classify(netOrig,validationResizeOrig);
    catch
        warning('Problem using GPU. Trying CPU this iteration');
        featuresTestSoftmaxOrig = activations(netOrig,validationResizeOrig,softmaxLayer,'OutputAs','rows', 'ExecutionEnvironment','cpu');
    end
    %ConfusionMatricesOrig{1, i} = confusionmat(confusionTrue, confusionPredOrig);
    origFeatures{1, i} = featuresTestSoftmaxOrig;

    time_training(1,i) = toc;
end
%Info{1, 1} = ConfusionMatricesOrig;

vLossOrig = 0;

for i = 1:folds
    vLossOrig = vLossOrig + infoResNet50Orig{1, i}.ValidationLoss(end);
end