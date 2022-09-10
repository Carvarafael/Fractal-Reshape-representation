clear
clc
%% Local das imagens
% 1º Tipo
cd('D:\TCC\ImagemTeste\Imagens_n_Segmentadas\AumentoDados\resnet50\normal')
Normal = dir('*.png'); %get all png files from path
for i=1:length(Normal)
    Normal(i).class = 'normal'; %atribuir a classe
    Normal(i).imagem = strcat(Normal(i).folder,'\',Normal(i).name);
end

% 2º Tipo
cd('D:\TCC\ImagemTeste\Imagens_n_Segmentadas\AumentoDados\resnet50\leve')
Leve = dir('*.png');%get all png files from path
for i=1:length(Leve)
    Leve(i).class = 'Leve'; %atribuir a classe
    Leve(i).imagem = strcat(Leve(i).folder,'\',Leve(i).name);
end
% 3º Tipo
cd('D:\TCC\ImagemTeste\Imagens_n_Segmentadas\AumentoDados\resnet50\moderado')
moderado = dir('*.png');%get all png files from path
for i=1:length(moderado)
    moderado(i).class = 'moderado'; %atribuir a classe
    moderado(i).imagem = strcat(moderado(i).folder,'\',moderado(i).name);
end

% 4º Tipo
cd('D:\TCC\ImagemTeste\Imagens_n_Segmentadas\AumentoDados\resnet50\grave')
grave = dir('*.png');%get all png files from path
for i=1:length(grave)
    grave(i).class = 'grave'; %atribuir a classe
    grave(i).imagem = strcat(grave(i).folder,'\',grave(i).name);
end
%% Criação do ImageDatastore
Images = [Normal;Leve;moderado;grave]; %Jutando os em um estrutura
% Transformando em um vetor coluna
Set = {};
for i=1:length(Images)
    Set{i} = Images(i).imagem;
    Set = Set';
end

% Transformando em um vetor coluna
Set2 = {};
for i=1:length(Images)
    Set2{i} = Images(i).class;
    Set2 = Set2';
end

%Muda o tipo para categoria (Necessário para a rede)
Set2 = categorical(Set2);

%% Definição do tamanho do grupo de treinamento,teste e validação
%Define o Image Datastore (Imagem + classe)
Train = imageDatastore(Set,"Labels",Set2);

[Treinamento,Teste] = splitEachLabel(Train,0.7,'randomized'); % 70% para treinamento
[Treinamento,Validacao] = splitEachLabel(Treinamento,0.9,'randomized'); %10% para validação


%% Definir a Rede
net = resnet18;
%net = alexnet;
%net = vgg16;

%% Transferencia de Apendizado, altera a ultima camada e definição dos
%parâmetros da rede
if isa(net,'SeriesNetwork')
    lgraph = layerGraph(net.Layers);
else
    lgraph = layerGraph(net);
end

[learnableLayer,classLayer] = findLayersToReplace(lgraph);

numClasses = numel(categories(Treinamento.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer); % Altera as camadas para o novo grupo de saída

layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:3) = freezeWeights(layers(1:3));
lgraph = createLgraphUsingConnections(layers,connections);
miniBatchSize = 10; %Quantidade de lotes
valFrequency = floor(numel(Treinamento.Files)/miniBatchSize); % Define a quantidade e iterações por epoca
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',1, ... %Altera a quantidade e epocas
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',Validacao, ... %Defini o grupo de Validação
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Treina o modelo com os parâmetros definidos
net = trainNetwork(Treinamento,lgraph,options);


% Calcula a acurácia do modelo com o gropo de teste
acc2 = [];
for i=1:16
    teste2 = partition(Teste,16,i);
    [YPred,probs] = classify(net,teste2);
    accuracy = mean(YPred == teste2.Labels);
    acc2 = [acc2 accuracy];
end
mean(acc2)

% Salva os parâmetros da rede
save('Alex.mat','net')