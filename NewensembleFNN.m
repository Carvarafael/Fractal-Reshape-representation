function [Resultado] = NewensembleFNN(OrigTest,o_Features,ORIG,rec_Features, REC,class_Features,CLASS,GASFEucl_Features, GASFEucl,GASFManh_Features, GASFManh,GASFMink_Features,GASFMink,GADFEucl_Features, GADFEucl,GADFManh_Features, GADFManh,GADFMink_Features,GADFMink)
accDIS = zeros(1,10);
f1DIS = zeros(1,10);


for j = 1:10
    ensembleDIS = (ORIG*o_Features{1,j})+(REC*rec_Features{1,j})+(CLASS*class_Features{1,j})+(GASFEucl*GASFEucl_Features{1,j})+(GASFManh*GASFManh_Features{1,j})+(GASFMink*GASFMink_Features{1,j})+(GADFEucl*GADFEucl_Features{1,j})+(GADFManh*GADFManh_Features{1,j})+(GADFMink*GADFMink_Features{1,j});
    
    %Define Predicted labels
    for k = 1:size(ensembleDIS, 1)
        [~, I] = max(ensembleDIS(k, :));
        predicted(1, k) = I;
        
        Categoria = OrigTest{1, j}.Labels(k);
        switch Categoria
            case 'normal'
                Orginal(1, k) = 4;
            case 'leve'
                Orginal(1, k) = 2;
            case 'moderado'
                Orginal(1, k) = 3;
            case 'grave'
                Orginal(1, k) = 1;
        end
    end
    
    Dados.Orginal{1, j} = Orginal;
    Dados.predicted{1, j} = predicted;
    Dados.stats = confusionmatStats(confusionmat(Orginal,predicted));
    Dados.acc = mean(Dados.stats.accuracy);
    Dados.Fscore = mean(Dados.stats.Fscore);
    Resultado{1,j} = Dados;
    Orginal = [];
    predicted = []; 
end
end


