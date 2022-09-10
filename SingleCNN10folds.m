function [Info, origFeatures, vLossOrig, time] = SingleCNN10folds(dataset, Neural, numClasses, size, ENVIRONMENT, tFilesOrig, vFilesOrig, ME)
folds = 10;

%%
%Informações das CNN
infoResNet50Orig = cell(1,folds);

time = zeros(1,10);

%%
%Hiper Parâmetros
MiniBatch = 32;
ILR = 0.01;
LRDP = 2;
LRDF = 0.75;
L2R = 0.0001;
%ME = 10;

%%
%Probabilidades Softmax
origFeatures = cell(1, folds);

Info = cell(1, 1);

%%
%Matrizes de confusão
ConfusionMatricesOrig = cell(1, folds);
%imageAugmenter = imageDataAugmenter('RandRotation',[0,360], 'RandXReflection', true, 'RandYReflection', true);

%%
%Principal
for i = 1:folds
    %Ler folds a partir dos arquivos
    imdsValidationOrig = vFilesOrig{1, i};
    imdsTrainOrig = tFilesOrig{1, i};
    
    
    
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
    
    %Matriz de confusão base (100% True)
    confusionTrue = zeros(1, v_num);
    for j = 1:v_num
        str = imdsValidationOrig.Files{j, 1};
        if dataset==1 %CR
            if contains(str, 'Benign')
                confusionTrue(1, j) = 1;
            else
                confusionTrue(1, j) = 2;
            end
        elseif dataset==2 %LA
            if contains (str, '\1\')
                confusionTrue(1, j) = 1;
            elseif contains (str, '\2\')
                confusionTrue(1, j) = 2;
            elseif contains (str, '\3\')
                confusionTrue(1, j) = 3;
            else
                confusionTrue(1, j) = 4;
            end
        elseif dataset==3 %LG
            if contains(str, '\1\')
                confusionTrue(1, j) = 1;
            else
                confusionTrue(1, j) = 2;
            end
        elseif dataset==4 %NHL
            if contains(str, '\CLL\')
                confusionTrue(1, j) = 1;
            elseif contains(str, '\FL\')
                confusionTrue(1, j) = 2;
            else
                confusionTrue(1, j) = 3;
            end
        elseif dataset==5 %UCSB
            if contains(str, '\Benign\')
                confusionTrue(1, j) = 1;
            else
                confusionTrue(1, j) = 2;
            end
        elseif dataset==6 %DIS
            if contains (str, 'Healthy')
                confusionTrue(1, j) = 1;
            elseif contains (str, 'Mild')
                confusionTrue(1, j) = 2;
            elseif contains (str, 'Moderate')
                confusionTrue(1, j) = 3;
            else
                confusionTrue(1, j) = 4;
            end
        elseif dataset ==7 %v7Labs
            if contains (str, 'Covid19')
                confusionTrue(1, j) = 1;
            else
                confusionTrue(1, j) = 2;
            end
        elseif dataset ==8 %DARK
            if contains (str, 'Covid')
                confusionTrue(1, j) = 1;
            elseif contains (str, 'NoFindings')
                confusionTrue(1, j) = 2;
            else
                confusionTrue(1, j) = 3;
            end
        end
    end
    
    %Matrizes de confusão das predições
    confusionPredOrig = zeros(1, v_num);
    
    %Ajustar tamanho das imagens para input da CNN
    %trainResizeOrig = augmentedImageDatastore(size, imdsTrainOrig, 'ColorPreprocessing', 'gray2rgb');%, 'DataAugmentation', imageAugmenter);
    %validationResizeOrig = augmentedImageDatastore(size, imdsValidationOrig, 'ColorPreprocessing', 'gray2rgb');
    
    trainResizeOrig = augmentedImageDatastore(size, imdsTrainOrig);%, 'DataAugmentation', imageAugmenter);
    validationResizeOrig = augmentedImageDatastore(size, imdsValidationOrig);
    
    
    %Configuração da CNN
    optionsOrig = trainingOptions('sgdm', 'MiniBatchSize',MiniBatch, 'MaxEpochs',ME, 'InitialLearnRate',ILR, 'LearnRateSchedule','piecewise', 'LearnRateDropPeriod', LRDP, 'LearnRateDropFactor', LRDF, 'Shuffle','every-epoch', 'ValidationData',validationResizeOrig, 'ValidationFrequency',10, 'Verbose',false, 'Plots','none', 'ExecutionEnvironment',ENVIRONMENT, 'L2Regularization', L2R);
    
    %Now this is where the fun begins
    tic
    try
    [netOrig, infoResNet50Orig{1,i}] = trainNetwork(trainResizeOrig,Neural,optionsOrig);
    catch
        warning('Problem using GPU. Trying CPU this iteration');
        options = trainingOptions('sgdm', 'MiniBatchSize',MiniBatch, 'MaxEpochs',ME, 'InitialLearnRate',ILR, 'LearnRateSchedule','piecewise', 'LearnRateDropPeriod', LRDP, 'LearnRateDropFactor', LRDF, 'Shuffle','every-epoch', 'ValidationData',validationResizeOrig, 'ValidationFrequency',10, 'Verbose',false, 'Plots','none', 'ExecutionEnvironment','cpu', 'L2Regularization', L2R);
        [netOrig, infoResNet50Orig{1,i}] = trainNetwork(trainResizeOrig,Neural,options);
    end
    
    
    %Obter probabilidades da camada Softmax
    try
        featuresTestSoftmaxOrig = activations(netOrig,validationResizeOrig,'softmax','OutputAs','rows');
    catch
        warning('Problem using GPU. Trying CPU this iteration');
        featuresTestSoftmaxOrig = activations(netOrig,validationResizeOrig,'softmax','OutputAs','rows', 'ExecutionEnvironment','cpu');
    end
    %Verificar número de acertos
    for j = 1:v_num
       [M, ICNNOrig] = max(featuresTestSoftmaxOrig(j, :));
       confusionPredOrig(1, j) = ICNNOrig;
    end
    
    ConfusionMatricesOrig{1, i} = confusionmat(confusionTrue, confusionPredOrig);
    origFeatures{1, i} = featuresTestSoftmaxOrig;
    
    time(1,i) = toc
end
Info{1, 1} = ConfusionMatricesOrig;

vLossOrig = 0;

for i = 1:folds
    vLossOrig = vLossOrig + infoResNet50Orig{1, i}.ValidationLoss(end);
end
end