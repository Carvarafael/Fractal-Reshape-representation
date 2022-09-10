function [Info, gadfFeatures, vLossGADF, scoreMaps, imageMaps, labelMaps] = SingleCNN10foldsGradCAM(dataset, Neural, numClasses, SIZE, ENVIRONMENT, tFilesGADF, vFilesGADF, ME)
folds = 10;

%%
%Informações das CNN
infoResNet50GADF = cell(1,folds);
scoreMaps = cell(1,10);
imageMaps = cell(1,10);
labelMaps = cell(1,10);


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
gadfFeatures = cell(1, folds);

Info = cell(1, 1);

%%
%Matrizes de confusão
ConfusionMatricesGADF = cell(1, folds);
%imageAugmenter = imageDataAugmenter('RandRotation',[0,360], 'RandXReflection', true, 'RandYReflection', true);

%%
%Principal
for i = 1:folds
    %Ler folds a partir dos arquivos
    imdsValidationGADF = vFilesGADF{1, i};
    imdsTrainGADF = tFilesGADF{1, i};
    
    
    
    %Número de amostras de treinamento
    trainingSamples = countEachLabel(imdsTrainGADF);
    trainingSamples = table2cell(trainingSamples);
    t_num = 0;
    for j = 1:numClasses
        t_num = t_num + trainingSamples{j,2};
    end
    
    %Número de amostras de validação
    validationSamples = countEachLabel(imdsValidationGADF); %Number of validation samples
    validationSamples = table2cell(validationSamples);
    v_num = 0;
    for j = 1:numClasses
        v_num = v_num + validationSamples{j,2};
    end
    
    scoreMapList = cell(1, v_num);
    imageMapList = cell(1, v_num);
    labelList = cell(1,v_num);
    
    %Matriz de confusão base (100% True)
    confusionTrue = zeros(1, v_num);
    for j = 1:v_num
        str = imdsValidationGADF.Files{j, 1};
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
        end
    end
    
    %Matrizes de confusão das predições
    confusionPredGADF = zeros(1, v_num);
    
    %Ajustar tamanho das imagens para input da CNN
    trainResizeGADF = augmentedImageDatastore(SIZE, imdsTrainGADF);%, 'DataAugmentation', imageAugmenter);
    validationResizeGADF = augmentedImageDatastore(SIZE, imdsValidationGADF);
    
    
    %Configuração da CNN
    optionsGADF = trainingOptions('sgdm', 'MiniBatchSize',MiniBatch, 'MaxEpochs',ME, 'InitialLearnRate',ILR, 'LearnRateSchedule','piecewise', 'LearnRateDropPeriod', LRDP, 'LearnRateDropFactor', LRDF, 'Shuffle','every-epoch', 'ValidationData',validationResizeGADF, 'ValidationFrequency',10, 'Verbose',false, 'Plots','none', 'ExecutionEnvironment',ENVIRONMENT, 'L2Regularization', L2R);
    
    %Now this is where the fun begins
    tic
    try
    [netGADF, infoResNet50GADF{1,i}] = trainNetwork(trainResizeGADF,Neural,optionsGADF);
    catch
        warning('Problem using GPU. Trying CPU this iteration');
        options = trainingOptions('sgdm', 'MiniBatchSize',MiniBatch, 'MaxEpochs',ME, 'InitialLearnRate',ILR, 'LearnRateSchedule','piecewise', 'LearnRateDropPeriod', LRDP, 'LearnRateDropFactor', LRDF, 'Shuffle','every-epoch', 'ValidationData',validationResizeGADF, 'ValidationFrequency',10, 'Verbose',false, 'Plots','none', 'ExecutionEnvironment','cpu', 'L2Regularization', L2R);
        [netGADF, infoResNet50GADF{1,i}] = trainNetwork(trainResizeGADF,Neural,options);
    end
    
    %ScoreMap
    ims = validationResizeGADF.read();
    nMaps = size(ims, 1);
    for j = 1:nMaps
        X = ims.input{j,1};
        label = classify(netGADF, X);
        scoreMapList{1,j} = gradCAM(netGADF, X, label);
        labelList{1,j} = label;
        imageMapList{1,j} = X;
    end
    
    scoreMaps{1,i} = scoreMapList;
    imageMaps{1,i} = imageMapList;
    labelMaps{1,i} = labelList;
    
    %Obter probabilidades da camada Softmax
    featuresTestSoftmaxGADF = activations(netGADF,validationResizeGADF,'softmax','OutputAs','rows');
    

    %Verificar número de acertos
    for j = 1:v_num
       [M, ICNNGADF] = max(featuresTestSoftmaxGADF(j, :));
       confusionPredGADF(1, j) = ICNNGADF;
    end
    
    ConfusionMatricesGADF{1, i} = confusionmat(confusionTrue, confusionPredGADF);
    gadfFeatures{1, i} = featuresTestSoftmaxGADF;
    
    toc
end
Info{1, 1} = ConfusionMatricesGADF;

vLossGADF = 0;

for i = 1:folds
    vLossGADF = vLossGADF + infoResNet50GADF{1, i}.ValidationLoss(end);
end
end