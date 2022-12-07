function CriaARFF_4Multi(camin, nArq, dados, classe1, classe2,classe3,classe4)
%{
%% function SEPARA_CC_ML_TWav2(camin,nomeArq,tWav);
%%
%% Author:
%%   PhD. Marcelo Zanchetta do Nascimento
%%   (marcelo.nascimento@ufabc.edu.br)
%%
%% Author:
%%   Ms. Rogério Daniel Dantas
%%
%% Date:    November - 2010
%% Course:  Information Engineering (Federal University of ABC)
%%
%% ______________________________ Function ________________________________
%%
%% Separa as imagens CC's das MLO's gerando arquivo .ARFF para WEKA
%%
%% ___________________________ Input Parameters ___________________________
%%
%%
%%  CAMIN - Choose the size of the ROI to be processed
%%
%% ___________________________ Output Parameters __________________________
%%
%% ________________________________ Sample ________________________________
%%  tWav = 'db2';
%%  camin = '/home/DDSM/SimulMestado/ScriptsOut2010/Dados_Wavelets/';
%%  nomeArq = strcat('Dados_Wav_SVD_480_CC_ML_',tWav);
%% ________________________________________________________________________
%}

a = 0;

% Pega as dimensões da matriz de dados
[L C] = size(dados);

nome = strcat(camin,nArq,'.arff');

% Cria o arquivo
save(nome, 'a', '-ascii');

%Abre o arquivo
fid = fopen(nome,'w');

%Grava a primeira linha do arquivo .arff
fprintf(fid,'%s\n',strcat('@RELATION',' ''', nArq, '.arff'''));
fprintf(fid,'%s\n\','');

for ind_C = 1:(C-1)
    a = ['@ATTRIBUTE','     ','x', int2str(ind_C), '    REAL'];
    fprintf(fid,'%s\n',a);
end
fprintf(fid,'%s\n\n','');
class = strcat('@ATTRIBUTE	class   {',classe1,',',classe2,',',classe3,',',classe4,'}');
fprintf(fid, '%s\n\n', class);
fprintf(fid,'%s\n\n','@DATA');
fprintf(fid,'%s\n','');

for iL = 1:L;
    for iC = 1:C;
        k = dados(iL,iC);
        fprintf(fid,'%e%s',k,' ');
        end
    fprintf(fid,'%f\n','');
    
end

fclose(fid);
end