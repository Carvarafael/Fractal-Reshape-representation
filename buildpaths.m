function [Paths] = buildpaths(Folds,recdir,classdir,gasfdir,gadfdir,MTFdir,SSMfdir)

for j=1:length(Folds)
%     for i=1:length(Folds{1, 1})
%         Paths{1,j}{1,i} = split(Folds{1, j}{1, i}.Files,'\');
%         Paths{1,j}{1,i} = split(Paths{1,j}{1,i}(:,5),".");
%         Paths{1,j}{1,i} = Paths{1,j}{1,i}(:,1);
%         Paths{1,j}{1,i}= regexp(Paths{1,j}{1,i}(:,1), '[0-9]*', 'match');
for j=1:length(Folds)
    for i=1:length(Folds{1, 1})
        Paths{1,j}{1,i} = split(Folds{1, j}{1, i}.Files,'\');
        Paths{1,j}{1,i} = split(Paths{1,j}{1,i}(:,6),".");
        Paths{1,j}{1,i} = Paths{1,j}{1,i}(:,1);
        Paths{1,j}{1,i}= regexp(Paths{1,j}{1,i}(:,1), '[0-9]*', 'match');
        
        for k=1:length(Folds{1, j}{1, i}.Files)
            Paths{1,j}{1,i}{k,2} = char(Folds{1,j}{1,i}.Labels(k));
            Paths{1,j}{1,i}{k,3} = Folds{1,j}{1,i}.Files{k,1};
            Paths{1,j}{1,i}{k,4} = char(strcat(recdir{j},"\","F-RecPlot",Paths{1,j}{1,i}{k,1},".png"));
            Paths{1,j}{1,i}{k,5} = char(strcat(classdir{j},"\","F-Classical",Paths{1,j}{1,i}{k,1},".png"));
            Paths{1,j}{1,i}{k,6} = char(strcat(gasfdir{j},"\","GRAYGASFEucl",Paths{1,j}{1,i}{k,1},".png"));
            Paths{1,j}{1,i}{k,7} = char(strcat(gasfdir{j},"\","GRAYGASFManh",Paths{1,j}{1,i}{k,1},".png"));
            Paths{1,j}{1,i}{k,8} = char(strcat(gasfdir{j},"\","GRAYGASFMink",Paths{1,j}{1,i}{k,1},".png"));
            Paths{1,j}{1,i}{k,9} = char(strcat(gadfdir{j},"\","GRAYGADFEucl",Paths{1,j}{1,i}{k,1},".png"));
            Paths{1,j}{1,i}{k,10} = char(strcat(gadfdir{j},"\","GRAYGADFManh",Paths{1,j}{1,i}{k,1},".png"));
            Paths{1,j}{1,i}{k,11} = char(strcat(gadfdir{j},"\","GRAYGADFMink",Paths{1,j}{1,i}{k,1},".png"));
            Paths{1,j}{1,i}{k,12} = char(strcat(gasfdir{j},"\","GASF",Paths{1,j}{1,i}{k,1},".png"));
            Paths{1,j}{1,i}{k,13} = char(strcat(gadfdir{j},"\","GADF",Paths{1,j}{1,i}{k,1},".png"));
            Paths{1,j}{1,i}{k,14} = char(strcat(MTFdir{j},"\","GRAYMTFMink",Paths{1,j}{1,i}{k,1},".png"));
            Paths{1,j}{1,i}{k,15} = char(strcat(MTFdir{j},"\","GRAYMTFEucl",Paths{1,j}{1,i}{k,1},".png"));
            Paths{1,j}{1,i}{k,16} = char(strcat(MTFdir{j},"\","GRAYMTFManh",Paths{1,j}{1,i}{k,1},".png"));
            Paths{1,j}{1,i}{k,17} = char(strcat(MTFdir{j},"\","MTF",Paths{1,j}{1,i}{k,1},".png"));
            Paths{1,j}{1,i}{k,18} = char(strcat(SSMfdir{j},"\","SSM",Paths{1,j}{1,i}{k,1},".png"));
        end
        
    end
end

end