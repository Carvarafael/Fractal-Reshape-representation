%%
%Script para aplicar ADASYN em datasets com duas classes
%Todas as imagens devem ser RGB e de mesma resolução

    count = 0;
	dX = 522; %dimensão X
	dY = 775; %dimensão Y
    CR = zeros(165,dX,dY,3); %(Total de imagens no dataset, Dimensão X, Dimensão Y, Dimensão Z (RGB = 3))
    folder = strcat('D:\Guilherme Freire\FoldsADASYN\CR\Train\Malignant'); %Diretório de origem da classe com MAIOR número de casos
    files = dir(fullfile(folder, '*.png'));
    for i = 1:length(files)
        baseName = files(i).name;
        fullName = fullfile(folder, baseName);
        CR(i, :, :, :) = imread(fullName);
        count = count+1;
    end
    shift = count;
    folder = strcat('D:\Guilherme Freire\FoldsADASYN\CR\Train\Benign'); %Diretório de origem da classe com MENOR número de casos
    files = dir(fullfile(folder, '*.png'));
    for i = 1:length(files)
        baseName = files(i).name;
        fullName = fullfile(folder, baseName);
        CR(i+shift, :, :, :) = imread(fullName);
        count = count + 1;
    end
    labels = zeros(1,count);
    for i = 1:shift
        labels(1,i) = 0;
    end
    for i = shift+1:count
        labels(1,i) = 1;
    end
    Benign = CR(1:shift, :, :, :);
    Malginant = CR(shift+1:count, :, :, :);
    labels_b = zeros(1,shift);
    labels_m = ones(1,count-shift);
    labels_b = logical(labels_b);
    labels_m = logical(labels_m);
    BenignFlat = reshape(Benign, [shift, X*Y*3]);
    MalignantFlat = reshape(Malginant, [count-shift, X*Y*3]);
    labels_b = labels_b';
    labels_m = labels_m';
    adasyn_features                 = [BenignFlat; MalignantFlat];
    adasyn_labels                   = [labels_b  ; labels_m  ];
    adasyn_beta                     = [];   %let ADASYN choose default
    adasyn_kDensity                 = [];   %let ADASYN choose default
    adasyn_kSMOTE                   = [];   %let ADASYN choose default
    adasyn_featuresAreNormalized    = true;    %false lets ADASYN handle normalization
    [adasyn_featuresSyn, adasyn_labelsSyn] = ADASYN(adasyn_features, adasyn_labels, adasyn_beta, adasyn_kDensity, adasyn_kSMOTE, adasyn_featuresAreNormalized);
    if size(adasyn_featuresSyn, 1) == 0
        continue
    end
    disp(adasyn_labelsSyn);
    genIMGS = size(adasyn_labelsSyn, 1);
    reshaped = reshape(adasyn_featuresSyn, [genIMGS, dX, dY, 3]);
    clearvars -except reshaped genIMGS dX dY
    A = zeros(dX, dY,3);
    for f = 1:genIMGS
        for i = 1:dX
            for j = 1:dY
                for k = 1:3
                    A(i, j, k) = reshaped(f, i, j, k);
                end
            end
        end
        A = uint8(A);
        imwrite(A, strcat('D:\Guilherme Freire\FoldsADASYN\CR\Train\Benign\ADASYN', num2str(f), '.png')); %Diretório onde serão salvas as imagens geradas
    end

