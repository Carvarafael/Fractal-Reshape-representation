function [predicted, classProbs, file_txt] = Weka_Classification(Actualdir,Imagedir,OrigTrainFolds, OrigTestFolds,TipoClass,features,folds)

%Get 10fold features to weka classification
VTrain = {};
VTest = {};
NumTrain = [];
NumTest = [];
VTrain_final = [];
VTest_final = [];
for i =1:length(Imagedir)
    for j=1:folds
        for k=1:length(OrigTrainFolds{1, i}{1, j}.Files)
            Number = split(OrigTrainFolds{1, i}{1, j}.Files(k),"\");
            Number = str2double(cell2mat(regexp(Number{end},'\d+','Match')));
            NumTrain(k,1) = Number;
        end
        VTrain{1,i}{1,j} = features{1,i}(NumTrain,:);

        for k=1:length(OrigTestFolds{1, i}{1, j}.Files)
            Number = split(OrigTestFolds{1, i}{1, j}.Files(k),"\");
            Number = str2double(cell2mat(regexp(Number{end},'\d+','Match')));
            NumTest(k,1) = Number;
        end
        VTest{1,i}{1,j} = features{1,i}(NumTest,:);
    end
end

for i = 1:length(Imagedir)
   VTrain_final = [VTrain_final;VTrain{1, i}{1, 1}];
   VTest_final = [VTest_final;VTest{1, i}{1, 1}];
end

%Classes usadas para criação do .arff
classes1='1.000000e+00';
classes2='2.000000e+00';
classes3='3.000000e+00';
classes4='4.000000e+00';

for i=1:folds
    %Prepara para salvar a crossValidation e cria o arquivo .arff para a
    %classificação
    mkdir([Actualdir,'\WekaClassification'])
    cd ([Actualdir,'\WekaClassification'])
    num = num2str(i);
    CriaARFF_4Multi([Actualdir,'\WekaClassification\'], ['TrainFolds-',num2str(i)], VTrain_final,classes1,classes2,classes3,classes4);
    %disp(strcat('Criando Vetores Treinamento! ',num));
    CriaARFF_4Multi([Actualdir,'\WekaClassification\'], ['TestFolds-',num2str(i)], VTest_final,classes1,classes2,classes3,classes4);
    %disp(strcat('Criando Vetores Teste! ',num));
end    

%   teste se o weka esta ativo no sistema operacional
javaaddpath('C:\Program Files\Weka-3-8-6\weka.jar','-end');

i = wekaPathCheck();


%caminho para salvar os arquivos com resultado da classificação dos folds

for i = 1:folds
    num = num2str(i);
    %Nome do arquivo a ser carregado
    filename1 = strcat([Actualdir,'\WekaClassification\'],'TrainFolds-',num,'.arff');
    filename2 = strcat([Actualdir,'\WekaClassification\'],'TestFolds-',num,'.arff');
    %carrega os arquivos arff construídos
    train = loadARFF(filename1);
    test = loadARFF(filename2);
    %treina o classificador
    nb = trainWekaClassifier(train,TipoClass);
    %testa o classificador
    [predicted,classProbs] = wekaClassify(test,nb);
    % calcula a curva ROC
    file_txt = rocWeka(filename1,filename2,TipoClass);
    %salva arquivo no caminho especificado para cada fold
    
end
cd (Actualdir);
end