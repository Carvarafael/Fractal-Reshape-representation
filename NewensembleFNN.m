function [dados, fit] = NewensembleFNN(ORIG, FRAC, REC, GASF, GADF, DTL, LOGL, RFL, SVML, MLPL, DTG, LOGG, RFG, SVMG, MLPG)

dados = zeros(5,4);
%ORIG = 0;

accCR = zeros(1,10);
accLA = zeros(1,10);
accLG = zeros(1,10);
accNHL = zeros(1,10);
accUCSB = zeros(1,10);
%accDIS = zeros(1,10);

f1CR = zeros(1,10);
f1LA = zeros(1,10);
f1LG = zeros(1,10);
f1NHL = zeros(1,10);
f1UCSB = zeros(1,10);
%f1DIS = zeros(1,10);

for i = 1:10
    load(strcat('ResultsMultiCNNSet', num2str(i), '.mat'));
    load(strcat('ResultsResNet-50NoWeights50EpochsOrigSet', num2str(i), 'All.mat'));
    %load(strcat('ResNet-50DAAllMUltiCNNSet', num2str(i), '.mat'));
    %load(strcat('DAGeoResultsMUltiCNNSet', num2str(i), '.mat'));
    load(strcat('ResultsGADFGASFResNet-50Set', num2str(i), '.mat'));
    %load(strcat('ResultsWeightsNoneFoldsResNet-50Set', num2str(i), 'All.mat'))
    load(strcat('ProbDistr', num2str(i), 'GlobalsESWA.mat'));
    load(strcat('ProbDistr', num2str(i), 'LocalsESWA.mat'));
    load(strcat('ProbDistr', num2str(i), 'GlobalsESWAMLP.mat'));
    load(strcat('ProbDistr', num2str(i), 'LocalsESWAMLP.mat'));
    load(strcat('TrueFoldsSet', num2str(i), 'All.mat'));
    ConfusionCR = cell(1,10);
    ConfusionLA = cell(1,10);
    ConfusionLG = cell(1,10);
    ConfusionNHL = cell(1,10);
    ConfusionUCSB = cell(1,10);
    %ConfusionDIS = cell(1,10);
    
    confFinalCR = zeros(2);
    confFinalLA = zeros(4);
    confFinalLG = zeros(2);
    confFinalNHL = zeros(3);
    confFinalUCSB = zeros(2);
    %confFinalDIS = zeros(4);
    
%     % VOTE
%     for vi = 1:10
%         for vj = 1:size(CROrigFeatures, 1)%CR
%             [M, I] = max(CROrigFeatures{1,vi}(vj, :));
%             CROrigFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(CRfracFeatures{1,vi}(vj, :));
%             CRfracFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(CRrecFeatures{1,vi}(vj, :));
%             CRrecFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(CRGASFManhFeatures{1,vi}(vj, :));
%             CRGASFManhFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(CRGASFEuclFeatures{1,vi}(vj, :));
%             CRGASFEuclFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(CRGASFMinkFeatures{1,vi}(vj, :));
%             CRGASFMinkFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(CRGADFManhFeatures{1,vi}(vj, :));
%             CRGADFManhFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(CRGADFEuclFeatures{1,vi}(vj, :));
%             CRGADFEuclFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(CRGADFMinkFeatures{1,vi}(vj, :));
%             CRGADFMinkFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(CRProbDT{1,vi}(vj, :));
%             CRProbDT{1,vi}(vj, I) = 1;
%             [M, I] = max(CRProbLOG{1,vi}(vj, :));
%             CRProbLOG{1,vi}(vj, I) = 1;
%             [M, I] = max(CRProbRF{1,vi}(vj, :));
%             CRProbRF{1,vi}(vj, I) = 1;
%             [M, I] = max(CRProbSVM{1,vi}(vj, :));
%             CRProbSVM{1,vi}(vj, I) = 1;
%             [M, I] = max(CRProbMLPLocals{1,vi}(vj, :));
%             CRProbMLPLocals{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWACRDT{1,vi}(vj, :));
%             ESWACRDT{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWACRLOG{1,vi}(vj, :));
%             ESWACRLOG{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWACRRF{1,vi}(vj, :));
%             ESWACRRF{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWACRSVM{1,vi}(vj, :));
%             ESWACRSVM{1,vi}(vj, I) = 1;
%             [M, I] = max(CRProbMLPGlobals{1,vi}(vj, :));
%             CRProbMLPGlobals{1,vi}(vj, I) = 1;
%         end
%         for vj = 1:size(LAOrigFeatures, 1)%LA
%             [M, I] = max(LAOrigFeatures{1,vi}(vj, :));
%             LAOrigFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LAfracFeatures{1,vi}(vj, :));
%             LAfracFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LArecFeatures{1,vi}(vj, :));
%             LArecFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LAGASFManhFeatures{1,vi}(vj, :));
%             LAGASFManhFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LAGASFEuclFeatures{1,vi}(vj, :));
%             LAGASFEuclFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LAGASFMinkFeatures{1,vi}(vj, :));
%             LAGASFMinkFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LAGADFManhFeatures{1,vi}(vj, :));
%             LAGADFManhFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LAGADFEuclFeatures{1,vi}(vj, :));
%             LAGADFEuclFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LAGADFMinkFeatures{1,vi}(vj, :));
%             LAGADFMinkFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LAProbDT{1,vi}(vj, :));
%             LAProbDT{1,vi}(vj, I) = 1;
%             [M, I] = max(LAProbLOG{1,vi}(vj, :));
%             LAProbLOG{1,vi}(vj, I) = 1;
%             [M, I] = max(LAProbRF{1,vi}(vj, :));
%             LAProbRF{1,vi}(vj, I) = 1;
%             [M, I] = max(LAProbSVM{1,vi}(vj, :));
%             LAProbSVM{1,vi}(vj, I) = 1;
%             [M, I] = max(LAProbMLPLocals{1,vi}(vj, :));
%             LAProbMLPLocals{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWALADT{1,vi}(vj, :));
%             ESWALADT{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWALALOG{1,vi}(vj, :));
%             ESWALALOG{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWALARF{1,vi}(vj, :));
%             ESWALARF{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWALASVM{1,vi}(vj, :));
%             ESWALASVM{1,vi}(vj, I) = 1;
%             [M, I] = max(LAProbMLPGlobals{1,vi}(vj, :));
%             LAProbMLPGlobals{1,vi}(vj, I) = 1;
%         end
%         for vj = 1:size(LGOrigFeatures, 1)%LG
%             [M, I] = max(LGOrigFeatures{1,vi}(vj, :));
%             LGOrigFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LGfracFeatures{1,vi}(vj, :));
%             LGfracFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LGrecFeatures{1,vi}(vj, :));
%             LGrecFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LGGASFManhFeatures{1,vi}(vj, :));
%             LGGASFManhFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LGGASFEuclFeatures{1,vi}(vj, :));
%             LGGASFEuclFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LGGASFMinkFeatures{1,vi}(vj, :));
%             LGGASFMinkFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LGGADFManhFeatures{1,vi}(vj, :));
%             LGGADFManhFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LGGADFEuclFeatures{1,vi}(vj, :));
%             LGGADFEuclFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LGGADFMinkFeatures{1,vi}(vj, :));
%             LGGADFMinkFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(LGProbDT{1,vi}(vj, :));
%             LGProbDT{1,vi}(vj, I) = 1;
%             [M, I] = max(LGProbLOG{1,vi}(vj, :));
%             LGProbLOG{1,vi}(vj, I) = 1;
%             [M, I] = max(LGProbRF{1,vi}(vj, :));
%             LGProbRF{1,vi}(vj, I) = 1;
%             [M, I] = max(LGProbSVM{1,vi}(vj, :));
%             LGProbSVM{1,vi}(vj, I) = 1;
%             [M, I] = max(LGProbMLPLocals{1,vi}(vj, :));
%             LGProbMLPLocals{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWALGDT{1,vi}(vj, :));
%             ESWALGDT{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWALGLOG{1,vi}(vj, :));
%             ESWALGLOG{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWALGRF{1,vi}(vj, :));
%             ESWALGRF{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWALGSVM{1,vi}(vj, :));
%             ESWALGSVM{1,vi}(vj, I) = 1;
%             [M, I] = max(LGProbMLPGlobals{1,vi}(vj, :));
%             LGProbMLPGlobals{1,vi}(vj, I) = 1;
%         end
%         for vj = 1:size(NHLOrigFeatures, 1)%NHL
%             [M, I] = max(NHLOrigFeatures{1,vi}(vj, :));
%             NHLOrigFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(NHLfracFeatures{1,vi}(vj, :));
%             NHLfracFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(NHLrecFeatures{1,vi}(vj, :));
%             NHLrecFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(NHLGASFManhFeatures{1,vi}(vj, :));
%             NHLGASFManhFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(NHLGASFEuclFeatures{1,vi}(vj, :));
%             NHLGASFEuclFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(NHLGASFMinkFeatures{1,vi}(vj, :));
%             NHLGASFMinkFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(NHLGADFManhFeatures{1,vi}(vj, :));
%             NHLGADFManhFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(NHLGADFEuclFeatures{1,vi}(vj, :));
%             NHLGADFEuclFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(NHLGADFMinkFeatures{1,vi}(vj, :));
%             NHLGADFMinkFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(NHLProbDT{1,vi}(vj, :));
%             NHLProbDT{1,vi}(vj, I) = 1;
%             [M, I] = max(NHLProbLOG{1,vi}(vj, :));
%             NHLProbLOG{1,vi}(vj, I) = 1;
%             [M, I] = max(NHLProbRF{1,vi}(vj, :));
%             NHLProbRF{1,vi}(vj, I) = 1;
%             [M, I] = max(NHLProbSVM{1,vi}(vj, :));
%             NHLProbSVM{1,vi}(vj, I) = 1;
%             [M, I] = max(NHLProbMLPLocals{1,vi}(vj, :));
%             NHLProbMLPLocals{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWANHLDT{1,vi}(vj, :));
%             ESWANHLDT{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWANHLLOG{1,vi}(vj, :));
%             ESWANHLLOG{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWANHLRF{1,vi}(vj, :));
%             ESWANHLRF{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWANHLSVM{1,vi}(vj, :));
%             ESWANHLSVM{1,vi}(vj, I) = 1;
%             [M, I] = max(NHLProbMLPGlobals{1,vi}(vj, :));
%             NHLProbMLPGlobals{1,vi}(vj, I) = 1;
%         end
%         for vj = 1:size(UCSBOrigFeatures, 1)%UCSB
%             [M, I] = max(UCSBOrigFeatures{1,vi}(vj, :));
%             UCSBOrigFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(UCSBfracFeatures{1,vi}(vj, :));
%             UCSBfracFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(UCSBrecFeatures{1,vi}(vj, :));
%             UCSBrecFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(UCSBGASFManhFeatures{1,vi}(vj, :));
%             UCSBGASFManhFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(UCSBGASFEuclFeatures{1,vi}(vj, :));
%             UCSBGASFEuclFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(UCSBGASFMinkFeatures{1,vi}(vj, :));
%             UCSBGASFMinkFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(UCSBGADFManhFeatures{1,vi}(vj, :));
%             UCSBGADFManhFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(UCSBGADFEuclFeatures{1,vi}(vj, :));
%             UCSBGADFEuclFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(UCSBGADFMinkFeatures{1,vi}(vj, :));
%             UCSBGADFMinkFeatures{1,vi}(vj, I) = 1;
%             [M, I] = max(UCSBProbDT{1,vi}(vj, :));
%             UCSBProbDT{1,vi}(vj, I) = 1;
%             [M, I] = max(UCSBProbLOG{1,vi}(vj, :));
%             UCSBProbLOG{1,vi}(vj, I) = 1;
%             [M, I] = max(UCSBProbRF{1,vi}(vj, :));
%             UCSBProbRF{1,vi}(vj, I) = 1;
%             [M, I] = max(UCSBProbSVM{1,vi}(vj, :));
%             UCSBProbSVM{1,vi}(vj, I) = 1;
%             [M, I] = max(UCSBProbMLPLocals{1,vi}(vj, :));
%             UCSBProbMLPLocals{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWAUCSBDT{1,vi}(vj, :));
%             ESWAUCSBDT{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWAUCSBLOG{1,vi}(vj, :));
%             ESWAUCSBLOG{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWAUCSBRF{1,vi}(vj, :));
%             ESWAUCSBRF{1,vi}(vj, I) = 1;
%             [M, I] = max(ESWAUCSBSVM{1,vi}(vj, :));
%             ESWAUCSBSVM{1,vi}(vj, I) = 1;
%             [M, I] = max(UCSBProbMLPGlobals{1,vi}(vj, :));
%             UCSBProbMLPGlobals{1,vi}(vj, I) = 1;
%         end
%     end
    %%
    for j = 1:10
        ensembleCR = (ORIG*CROrigFeatures{1,j})+(FRAC*CRfracFeatures{1,j})+(REC*CRrecFeatures{1,j})+(GASF*CRGASFFeatures{1,j})+(GADF*CRGADFFeatures{1,j})+(DTL*CRProbDT{1,j})+(LOGL*CRProbLOG{1,j})+(RFL*CRProbRF{1,j})+(SVML*CRProbSVM{1,j})+(MLPL*CRProbMLPLocals{1,j})+(DTG*ESWACRDT{1,j})+(LOGG*ESWACRLOG{1,j})+(RFG*ESWACRRF{1,j})+(SVMG*ESWACRSVM{1,j})+(MLPG*CRProbMLPGlobals{1,j});
        ensembleLA = (ORIG*LAOrigFeatures{1,j})+(FRAC*LAfracFeatures{1,j})+(REC*LArecFeatures{1,j})+(GASF*LAGASFFeatures{1,j})+(GADF*LAGADFFeatures{1,j})+(DTL*LAProbDT{1,j})+(LOGL*LAProbLOG{1,j})+(RFL*LAProbRF{1,j})+(SVML*LAProbSVM{1,j})+(MLPL*LAProbMLPLocals{1,j})+(DTG*ESWALADT{1,j})+(LOGG*ESWALALOG{1,j})+(RFG*ESWALARF{1,j})+(SVMG*ESWALASVM{1,j})+(MLPG*LAProbMLPGlobals{1,j});
        ensembleLG = (ORIG*LGOrigFeatures{1,j})+(FRAC*LGfracFeatures{1,j})+(REC*LGrecFeatures{1,j})+(GASF*LGGASFFeatures{1,j})+(GADF*LGGADFFeatures{1,j})+(DTL*LGProbDT{1,j})+(LOGL*LGProbLOG{1,j})+(RFL*LGProbRF{1,j})+(SVML*LGProbSVM{1,j})+(MLPL*LGProbMLPLocals{1,j})+(DTG*ESWALGDT{1,j})+(LOGG*ESWALGLOG{1,j})+(RFG*ESWALGRF{1,j})+(SVMG*ESWALGSVM{1,j})+(MLPG*LGProbMLPGlobals{1,j});
        ensembleNHL = (ORIG*NHLOrigFeatures{1,j})+(FRAC*NHLfracFeatures{1,j})+(REC*NHLrecFeatures{1,j})+(GASF*NHLGASFFeatures{1,j})+(GADF*NHLGADFFeatures{1,j})+(DTL*NHLProbDT{1,j})+(LOGL*NHLProbLOG{1,j})+(RFL*NHLProbRF{1,j})+(SVML*NHLProbSVM{1,j})+(MLPL*NHLProbMLPLocals{1,j})+(DTG*ESWANHLDT{1,j})+(LOGG*ESWANHLLOG{1,j})+(RFG*ESWANHLRF{1,j})+(SVMG*ESWANHLSVM{1,j})+(MLPG*NHLProbMLPGlobals{1,j});
        ensembleUCSB = (ORIG*UCSBOrigFeatures{1,j})+(FRAC*UCSBfracFeatures{1,j})+(REC*UCSBrecFeatures{1,j})+(GASF*UCSBGASFFeatures{1,j})+(GADF*UCSBGADFFeatures{1,j})+(DTL*UCSBProbDT{1,j})+(LOGL*UCSBProbLOG{1,j})+(RFL*UCSBProbRF{1,j})+(SVML*UCSBProbSVM{1,j})+(MLPL*UCSBProbMLPLocals{1,j})+(DTG*ESWAUCSBDT{1,j})+(LOGG*ESWAUCSBLOG{1,j})+(RFG*ESWAUCSBRF{1,j})+(SVMG*ESWAUCSBSVM{1,j})+(MLPG*UCSBProbMLPGlobals{1,j});
        %ensembleDIS = (ORIG*DISOrigFeatures{1,j})+(FRAC*DISfracFeatures{1,j})+(REC*DISrecFeatures{1,j})+(SMA*DISGASFManhFeatures{1,j})+(SE*DISGASFEuclFeatures{1,j})+(SMI*DISGASFMinkFeatures{1,j})+(DMA*DISGADFManhFeatures{1,j})+(DE*DISGADFEuclFeatures{1,j})+(DMI*DISGADFMinkFeatures{1,j})+(DT*DISProbDT{1,j})+(LOG*DISProbLOG{1,j})+(RF*DISProbRF{1,j})+(SVM*DISProbSVM{1,j});
        
        predictedCR = zeros(1, size(ensembleCR, 1));
        predictedLA = zeros(1, size(ensembleLA, 1));
        predictedLG = zeros(1, size(ensembleLG, 1));
        predictedNHL = zeros(1, size(ensembleNHL, 1));
        predictedUCSB = zeros(1, size(ensembleUCSB, 1));
        %predictedDIS = zeros(1, size(ensembleDIS, 1));
        
        
        %Define Predicted labels
        for k = 1:size(ensembleCR, 1)
            [M, I] = max(ensembleCR(k, :));
            predictedCR(1, k) = I-1;
        end
        
        for k = 1:size(ensembleLA, 1)
            [M, I] = max(ensembleLA(k, :));
            predictedLA(1, k) = I-1;
        end
        
        for k = 1:size(ensembleLG, 1)
            [M, I] = max(ensembleLG(k, :));
            predictedLG(1, k) = I-1;
        end
        
        for k = 1:size(ensembleNHL, 1)
            [M, I] = max(ensembleNHL(k, :));
            predictedNHL(1, k) = I-1;
        end
        
        for k = 1:size(ensembleUCSB, 1)
            [M, I] = max(ensembleUCSB(k, :));
            predictedUCSB(1, k) = I-1;
        end
        
%         for k = 1:size(ensembleDIS, 1)
%             [M, I] = max(ensembleDIS(k, :));
%             predictedDIS(1, k) = I-1;
%         end
        
        ConfusionCR{1,j} = confusionmat(CRTrue{1,j}, predictedCR);
        ConfusionLA{1,j} = confusionmat(LATrue{1,j}, predictedLA);
        ConfusionLG{1,j} = confusionmat(LGTrue{1,j}, predictedLG);
        ConfusionNHL{1,j} = confusionmat(NHLTrue{1,j}, predictedNHL);
        ConfusionUCSB{1,j} = confusionmat(UCSBTrue{1,j}, predictedUCSB);
       % ConfusionDIS{1,j} = confusionmat(DISTrue{1,j}, predictedDIS);
    end
    for j = 1:10
        confFinalCR = confFinalCR + ConfusionCR{1,j};
        confFinalLA = confFinalLA + ConfusionLA{1,j};
        confFinalLG = confFinalLG + ConfusionLG{1,j};
        confFinalNHL = confFinalNHL + ConfusionNHL{1,j};
        confFinalUCSB = confFinalUCSB + ConfusionUCSB{1,j};
       % confFinalDIS = confFinalDIS + ConfusionDIS{1,j};
    end
    
    statsCR = confusionmatStats(confFinalCR);
    statsLA = confusionmatStats(confFinalLA);
    statsLG = confusionmatStats(confFinalLG);
    statsNHL = confusionmatStats(confFinalNHL);
    statsUCSB = confusionmatStats(confFinalUCSB);
    %statsDIS = confusionmatStats(confFinalDIS);
    
    accCR(1,i) = (statsCR.confusionMat(1,1)+statsCR.confusionMat(2,2))/165;
    accLA(1,i) = (statsLA.confusionMat(1,1)+statsLA.confusionMat(2,2)+statsLA.confusionMat(3,3)+statsLA.confusionMat(4,4))/528;
    accLG(1,i) = (statsLG.confusionMat(1,1)+statsLG.confusionMat(2,2))/265;
    accNHL(1,i) = (statsNHL.confusionMat(1,1)+statsNHL.confusionMat(2,2)+statsNHL.confusionMat(3,3))/374;
    accUCSB(1,i) = (statsUCSB.confusionMat(1,1)+statsUCSB.confusionMat(2,2))/58;
    %accDIS(1,i) = (statsDIS.confusionMat(1,1)+statsDIS.confusionMat(2,2)+statsDIS.confusionMat(3,3)+statsDIS.confusionMat(4,4))/296;
    
    f1CR(1,i) = mean(statsCR.Fscore);
    f1LA(1,i) = mean(statsLA.Fscore);
    f1LG(1,i) = mean(statsLG.Fscore);
    f1NHL(1,i) = mean(statsNHL.Fscore);
    f1UCSB(1,i) = mean(statsUCSB.Fscore);
    %f1DIS(1,i) = mean(statsDIS.Fscore);
end

dados(1,1) = mean(accCR);
dados(2,1) = mean(accLA);
dados(3,1) = mean(accLG);
dados(4,1) = mean(accNHL);
dados(5,1) = mean(accUCSB);
%aaaDISacc = mean(accDIS)

dados(1,3) = mean(f1CR);
dados(2,3) = mean(f1LA);
dados(3,3) = mean(f1LG);
dados(4,3) = mean(f1NHL);
dados(5,3) = mean(f1UCSB);
%aaaDISf1 = mean(f1DIS)

dados(1,2) = std(accCR);
dados(2,2) = std(accLA);
dados(3,2) = std(accLG);
dados(4,2) = std(accNHL);
dados(5,2) = std(accUCSB);
%aaaDISaccstd = std(accDIS)

dados(1,4) = std(f1CR);
dados(2,4) = std(f1LA);
dados(3,4) = std(f1LG);
dados(4,4) = std(f1NHL);
dados(5,4) = std(f1UCSB);
%aaaDISf1std = std(f1DIS)

fit = (dados(1,3)+dados(2,3)+dados(3,3)+dados(4,3)+dados(5,3))/5;% +aaaDISf1;



end
