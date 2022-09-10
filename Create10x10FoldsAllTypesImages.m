for k = 1:10
    %Criar Folds para imagens originais
    [CROrigTrainFolds, CROrigTestFolds] = Create10Folds('C:\Users\thais\Documents\Doutorado\UniboServer-Documenti\MATLAB\DatasetsFNN\Imagens Originais\CR', '.png');
    [LAOrigTrainFolds, LAOrigTestFolds] = Create10Folds('C:\Users\thais\Documents\Doutorado\UniboServer-Documenti\MATLAB\DatasetsFNN\Imagens Originais\LA', '.png');
    [LGOrigTrainFolds, LGOrigTestFolds] = Create10Folds('C:\Users\thais\Documents\Doutorado\UniboServer-Documenti\MATLAB\DatasetsFNN\Imagens Originais\LG', '.png');
    [NHLOrigTrainFolds, NHLOrigTestFolds] = Create10Folds('C:\Users\thais\Documents\Doutorado\UniboServer-Documenti\MATLAB\DatasetsFNN\Imagens Originais\NHL', '.png');
    [UCSBOrigTrainFolds, UCSBOrigTestFolds] = Create10Folds('C:\Users\thais\Documents\Doutorado\UniboServer-Documenti\MATLAB\DatasetsFNN\Imagens Originais\UCSB', '.png');
    save(strcat('FoldsSet', num2str(k), 'Orig.mat'));
   
    %Criar Folds para imagens Reshape Comum
    CRFracTestFolds = CROrigTestFolds;
    LAFracTestFolds = LAOrigTestFolds;
    LGFracTestFolds = LGOrigTestFolds;
    UCSBFracTestFolds = UCSBOrigTestFolds;
    NHLFracTestFolds = NHLOrigTestFolds;
    for i = 1:10
        for j = 1:size(CROrigTestFolds{1,i}.Files, 1)
            CRFracTestFolds{1,i}.Files{j,1} = strrep(CROrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Fractais Reshape Comum');
        end
        for j = 1:size(LAOrigTestFolds{1,i}.Files, 1)
            LAFracTestFolds{1,i}.Files{j,1} = strrep(LAOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Fractais Reshape Comum');
        end
        for j = 1:size(LGOrigTestFolds{1,i}.Files, 1)
            LGFracTestFolds{1,i}.Files{j,1} = strrep(LGOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Fractais Reshape Comum');
        end
        for j = 1:size(UCSBOrigTestFolds{1,i}.Files, 1)
            UCSBFracTestFolds{1,i}.Files{j,1} = strrep(UCSBOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Fractais Reshape Comum');
        end
        for j = 1:size(NHLOrigTestFolds{1,i}.Files, 1)
            NHLFracTestFolds{1,i}.Files{j,1} = strrep(NHLOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Fractais Reshape Comum');
        end
    end
    CRFracTrainFolds = CROrigTrainFolds;
    LAFracTrainFolds = LAOrigTrainFolds;
    LGFracTrainFolds = LGOrigTrainFolds;
    UCSBFracTrainFolds = UCSBOrigTrainFolds;
    NHLFracTrainFolds = NHLOrigTrainFolds;
    for i = 1:10
        for j = 1:size(CROrigTrainFolds{1,i}.Files, 1)
            CRFracTrainFolds{1,i}.Files{j,1} = strrep(CROrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Fractais Reshape Comum');
        end
        for j = 1:size(LAOrigTrainFolds{1,i}.Files, 1)
            LAFracTrainFolds{1,i}.Files{j,1} = strrep(LAOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Fractais Reshape Comum');
        end
        for j = 1:size(LGOrigTrainFolds{1,i}.Files, 1)
            LGFracTrainFolds{1,i}.Files{j,1} = strrep(LGOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Fractais Reshape Comum');
        end
        for j = 1:size(UCSBOrigTrainFolds{1,i}.Files, 1)
            UCSBFracTrainFolds{1,i}.Files{j,1} = strrep(UCSBOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Fractais Reshape Comum');
        end
        for j = 1:size(NHLOrigTrainFolds{1,i}.Files, 1)
            NHLFracTrainFolds{1,i}.Files{j,1} = strrep(NHLOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Fractais Reshape Comum');
        end
    end
    clear i j
    load(strcat('FoldsSet', num2str(k), 'Orig.mat'))
    save(strcat('FoldsSet', num2str(k), 'Orig+Frac.mat'));
    
    %Criar Folds imagens Recurrence Plot
    CRRecTestFolds = CROrigTestFolds;
    LARecTestFolds = LAOrigTestFolds;
    LGRecTestFolds = LGOrigTestFolds;
    UCSBRecTestFolds = UCSBOrigTestFolds;
    NHLRecTestFolds = NHLOrigTestFolds;
    for i = 1:10
        for j = 1:size(CROrigTestFolds{1,i}.Files, 1)
            CRRecTestFolds{1,i}.Files{j,1} = strrep(CROrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Recurrence Plot');
        end
        for j = 1:size(LAOrigTestFolds{1,i}.Files, 1)
            LARecTestFolds{1,i}.Files{j,1} = strrep(LAOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Recurrence Plot');
        end
        for j = 1:size(LGOrigTestFolds{1,i}.Files, 1)
            LGRecTestFolds{1,i}.Files{j,1} = strrep(LGOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Recurrence Plot');
        end
        for j = 1:size(UCSBOrigTestFolds{1,i}.Files, 1)
            UCSBRecTestFolds{1,i}.Files{j,1} = strrep(UCSBOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Recurrence Plot');
        end
        for j = 1:size(NHLOrigTestFolds{1,i}.Files, 1)
            NHLRecTestFolds{1,i}.Files{j,1} = strrep(NHLOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Recurrence Plot');
        end
    end
    CRRecTrainFolds = CROrigTrainFolds;
    LARecTrainFolds = LAOrigTrainFolds;
    LGRecTrainFolds = LGOrigTrainFolds;
    UCSBRecTrainFolds = UCSBOrigTrainFolds;
    NHLRecTrainFolds = NHLOrigTrainFolds;
    for i = 1:10
        for j = 1:size(CROrigTrainFolds{1,i}.Files, 1)
            CRRecTrainFolds{1,i}.Files{j,1} = strrep(CROrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Recurrence Plot');
        end
        for j = 1:size(LAOrigTrainFolds{1,i}.Files, 1)
            LARecTrainFolds{1,i}.Files{j,1} = strrep(LAOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Recurrence Plot');
        end
        for j = 1:size(LGOrigTrainFolds{1,i}.Files, 1)
            LGRecTrainFolds{1,i}.Files{j,1} = strrep(LGOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Recurrence Plot');
        end
        for j = 1:size(UCSBOrigTrainFolds{1,i}.Files, 1)
            UCSBRecTrainFolds{1,i}.Files{j,1} = strrep(UCSBOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Recurrence Plot');
        end
        for j = 1:size(NHLOrigTrainFolds{1,i}.Files, 1)
            NHLRecTrainFolds{1,i}.Files{j,1} = strrep(NHLOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens Recurrence Plot');
        end
    end
    clear i j
    load(strcat('FoldsSet', num2str(k), 'Orig+Frac.mat'))
    save(strcat('FoldsSet', num2str(k), 'Orig+Frac+Rec.mat'))
    
    %Criar Folds imagens GADF
    CRGADFTestFolds = CROrigTestFolds;
    LAGADFTestFolds = LAOrigTestFolds;
    LGGADFTestFolds = LGOrigTestFolds;
    UCSBGADFTestFolds = UCSBOrigTestFolds;
    NHLGADFTestFolds = NHLOrigTestFolds;
    for i = 1:10
        for j = 1:size(CROrigTestFolds{1,i}.Files, 1)
            CRGADFTestFolds{1,i}.Files{j,1} = strrep(CROrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GADF');
        end
        for j = 1:size(LAOrigTestFolds{1,i}.Files, 1)
            LAGADFTestFolds{1,i}.Files{j,1} = strrep(LAOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GADF');
        end
        for j = 1:size(LGOrigTestFolds{1,i}.Files, 1)
            LGGADFTestFolds{1,i}.Files{j,1} = strrep(LGOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GADF');
        end
        for j = 1:size(UCSBOrigTestFolds{1,i}.Files, 1)
            UCSBGADFTestFolds{1,i}.Files{j,1} = strrep(UCSBOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GADF');
        end
        for j = 1:size(NHLOrigTestFolds{1,i}.Files, 1)
            NHLGADFTestFolds{1,i}.Files{j,1} = strrep(NHLOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GADF');
        end
    end
    CRGADFTrainFolds = CROrigTrainFolds;
    LAGADFTrainFolds = LAOrigTrainFolds;
    LGGADFTrainFolds = LGOrigTrainFolds;
    UCSBGADFTrainFolds = UCSBOrigTrainFolds;
    NHLGADFTrainFolds = NHLOrigTrainFolds;
    for i = 1:10
        for j = 1:size(CROrigTrainFolds{1,i}.Files, 1)
            CRGADFTrainFolds{1,i}.Files{j,1} = strrep(CROrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GADF');
        end
        for j = 1:size(LAOrigTrainFolds{1,i}.Files, 1)
            LAGADFTrainFolds{1,i}.Files{j,1} = strrep(LAOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GADF');
        end
        for j = 1:size(LGOrigTrainFolds{1,i}.Files, 1)
            LGGADFTrainFolds{1,i}.Files{j,1} = strrep(LGOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GADF');
        end
        for j = 1:size(UCSBOrigTrainFolds{1,i}.Files, 1)
            UCSBGADFTrainFolds{1,i}.Files{j,1} = strrep(UCSBOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GADF');
        end
        for j = 1:size(NHLOrigTrainFolds{1,i}.Files, 1)
            NHLGADFTrainFolds{1,i}.Files{j,1} = strrep(NHLOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GADF');
        end
    end
    clear i j
    load(strcat('FoldsSet', num2str(k), 'Orig+Frac+Rec.mat'))
    save(strcat('FoldsSet', num2str(k), 'Orig+Frac+Rec+GADF.mat'))
    
    
    %Criar Folds imagens GASF
    CRGASFTestFolds = CROrigTestFolds;
    LAGASFTestFolds = LAOrigTestFolds;
    LGGASFTestFolds = LGOrigTestFolds;
    UCSBGASFTestFolds = UCSBOrigTestFolds;
    NHLGASFTestFolds = NHLOrigTestFolds;
    for i = 1:10
        for j = 1:size(CROrigTestFolds{1,i}.Files, 1)
            CRGASFTestFolds{1,i}.Files{j,1} = strrep(CROrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GASF');
        end
        for j = 1:size(LAOrigTestFolds{1,i}.Files, 1)
            LAGASFTestFolds{1,i}.Files{j,1} = strrep(LAOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GASF');
        end
        for j = 1:size(LGOrigTestFolds{1,i}.Files, 1)
            LGGASFTestFolds{1,i}.Files{j,1} = strrep(LGOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GASF');
        end
        for j = 1:size(UCSBOrigTestFolds{1,i}.Files, 1)
            UCSBGASFTestFolds{1,i}.Files{j,1} = strrep(UCSBOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GASF');
        end
        for j = 1:size(NHLOrigTestFolds{1,i}.Files, 1)
            NHLGASFTestFolds{1,i}.Files{j,1} = strrep(NHLOrigTestFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GASF');
        end
    end
    CRGASFTrainFolds = CROrigTrainFolds;
    LAGASFTrainFolds = LAOrigTrainFolds;
    LGGASFTrainFolds = LGOrigTrainFolds;
    UCSBGASFTrainFolds = UCSBOrigTrainFolds;
    NHLGASFTrainFolds = NHLOrigTrainFolds;
    for i = 1:10
        for j = 1:size(CROrigTrainFolds{1,i}.Files, 1)
            CRGASFTrainFolds{1,i}.Files{j,1} = strrep(CROrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GASF');
        end
        for j = 1:size(LAOrigTrainFolds{1,i}.Files, 1)
            LAGASFTrainFolds{1,i}.Files{j,1} = strrep(LAOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GASF');
        end
        for j = 1:size(LGOrigTrainFolds{1,i}.Files, 1)
            LGGASFTrainFolds{1,i}.Files{j,1} = strrep(LGOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GASF');
        end
        for j = 1:size(UCSBOrigTrainFolds{1,i}.Files, 1)
            UCSBGASFTrainFolds{1,i}.Files{j,1} = strrep(UCSBOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GASF');
        end
        for j = 1:size(NHLOrigTrainFolds{1,i}.Files, 1)
            NHLGASFTrainFolds{1,i}.Files{j,1} = strrep(NHLOrigTrainFolds{1,i}.Files{j,1}, 'Imagens Originais', 'Imagens GASF');
        end
    end
    clear i j
   
	
	
    save(strcat('FoldsSet', num2str(k), 'All.mat'))
end