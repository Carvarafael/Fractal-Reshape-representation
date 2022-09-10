%%Extração de atributos DF LAC 
%%com uso das métricas de distância Chessboard, Euclidiana e Manhatan a partir de imagens RGB.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Guilherme Freire Roberto

%Base LA: uma imagem da class4 foi removida. Corrigir os folds

%%Valor máximo de L
maxr = 41;

%%
%Classe 1
diretorio_org='C:\Users\thais\Documents\Doutorado\Bases de imagens\LiverGender\1'; %imagens
destino='C:\Users\thais\Documents\Doutorado\Bases de imagens\LiverGender\1'; %local onde será salvo o arquivo .mat
for n=1:150 %número de imagens da classe
    tic
originais=strcat('Class 1 - (',num2str(n),').png');
fullname=fullfile(diretorio_org,originais);
PIC=imread(fullname);



%Minkowski LAC DF
MatrizProb = pmr(PIC, maxr);
MinkLAC = lacunaridade(MatrizProb);
r = 3:2:maxr;
MinkAreaLAC = trapz(MinkLAC);
MinkSkewnessLAC = skewness(MinkLAC);
half = ceil(length(MinkLAC)/2);
MinkAreaRatioLAC = trapz(MinkLAC(half+1:end))/trapz(MinkLAC(1:half));
[MinkMaxLAC, MinkMaxLACIndex] = max(MinkLAC);
Minknn = N(MatrizProb);
x = log(r);
y = -log(Minknn);
X = x.';
Y = y.';
mdl = fitlm(X, Y, 'RobustOpts', 'on');
MinkDF = mdl.Coefficients.Estimate(2);

%Euclidian LAC DF
MatrizProb = pmrEucl(PIC, maxr);
EuclLAC = lacunaridade(MatrizProb);
r = 3:2:maxr;
EuclAreaLAC = trapz(EuclLAC);
EuclSkewnessLAC = skewness(EuclLAC);
half = ceil(length(EuclLAC)/2);
EuclAreaRatioLAC = trapz(EuclLAC(half+1:end))/trapz(EuclLAC(1:half));
[EuclMaxLAC, EuclMaxLACIndex] = max(EuclLAC);
Euclnn = N(MatrizProb);
x = log(r);
y = -log(Euclnn);
X = x.';
Y = y.';
mdl = fitlm(X, Y, 'RobustOpts', 'on');
EuclDF = mdl.Coefficients.Estimate(2);

%Manhattan LAC DF
MatrizProb = pmrManh(PIC, maxr);
ManhLAC = lacunaridade(MatrizProb);
r = 3:2:maxr;
ManhAreaLAC = trapz(ManhLAC);
ManhSkewnessLAC = skewness(ManhLAC);
half = ceil(length(ManhLAC)/2);
ManhAreaRatioLAC = trapz(ManhLAC(half+1:end))/trapz(ManhLAC(1:half));
[ManhMaxLAC, ManhMaxLACIndex] = max(ManhLAC);
Manhnn = N(MatrizProb);
x = log(r);
y = -log(Manhnn);
X = x.';
Y = y.';
mdl = fitlm(X, Y, 'RobustOpts', 'on');
ManhDF = mdl.Coefficients.Estimate(2);

clear PIC;
%clear AC
final=strcat(num2str(n),'.mat');
filename=fullfile(destino,final);
save(filename);
toc
end
