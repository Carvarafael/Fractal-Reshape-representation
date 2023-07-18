function [Resultado] = NewensembleFNN(OrigTest,o_Features,ORIG,rec_Features,REC,class_Features,CLASS,folds)
%,GASFEucl_Features,GASFEucl,GADF_Features,GADF,MTF_Features,MTF,SSM_Features,SSM,folds)
%,rec_Features, REC,class_Features,CLASS,GASFEucl_Features, GASFEucl,GASFManh_Features, GASFManh,GASFMink_Features,GASFMink,GADFEucl_Features, GADFEucl,GADFManh_Features, GADFManh,GADFMink_Features,GADFMink)
accDIS = zeros(1,folds);
f1DIS = zeros(1,folds);
Orginal = [];

for j = 1:folds
    ensembleDIS = (ORIG*o_Features{1,j})+(REC*rec_Features{1,j})+(CLASS*class_Features{1,j});
    ...
        %+(GASFEucl*GASFEucl_Features{1,j}) +(GADF*GADF_Features{1,j})...
        %+(MTF*MTF_Features{1,j})+(SSM*SSM_Features{1,j});
 
    
    %Define Predicted labels
    for k = 1:size(ensembleDIS, 1)
        [~, I] = max(ensembleDIS(k, :));
        predicted(1, k) = I;
        
        Categoria = OrigTest{1, j}.Labels(k);
%         switch Categoria
%             case 'healthy'
%                 Orginal(1, k) = 1;
%             case 'mild'
%                 Orginal(1, k) = 2;
%             case 'moderate'
%                 Orginal(1, k) = 3;
%             case 'severe'
%                 Orginal(1, k) = 4;
%         end
        
%         switch Categoria
%             case 'grave'
%                  Orginal(1, k) = 1;
%             case 'leve'
%                  Orginal(1, k) = 2;
%             case 'moderado'
%                  Orginal(1, k) = 3;
%             case 'normal'
%                  Orginal(1, k) = 4;
%         end
        
%         switch Categoria
%             case 'Doente'
%                  Orginal(1, k) = 1;
%             case 'Normal'
%                  Orginal(1, k) = 2;
%         end

        switch Categoria
            case 'leve'
                 Orginal(1, k) = 1;
            case 'moderado'
                 Orginal(1, k) = 2;
        end



    end
    
    Dados.Orginal{1, j} = Orginal;
    Dados.predicted{1, j} = predicted;
    Dados.stats{1, j} = confusionmatStats(confusionmat(Orginal,predicted));
    Dados.acc{1, j} = mean(Dados.stats{1, j}.accuracy);
    acurr = (Dados.stats{1, j}.accuracy);
    Dados.acc1{1, j} = acurr(1);
    Dados.acc2{1, j} = acurr(2);
    Dados.Fscore{1, j} = mean(Dados.stats{1, j}.Fscore);
    Resultado = Dados;
    Orginal = [];
    predicted = [];
    
end
end


