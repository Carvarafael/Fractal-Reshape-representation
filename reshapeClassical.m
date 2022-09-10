%%Converte os atributos locais salvos no arquivo CSV com uso do script
%%SaveCSVPercLACDF3Distances em uma imagem de atributos fractais
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%input: variável finalcsv, número de amostras
%%Guilherme Freire Roberto


function reshapeClassical(file, samples)

readFile = file;
features = readFile(369:368+samples,1:300);
features = cell2mat(features);
featuresSplit = zeros(samples,100,3);
featuresSplit(:,:,1) = features(:, 1:100);
featuresSplit(:,:,2) = features(:, 101:200);
featuresSplit(:,:,3) = features(:, 201:300);
newFeatures = zeros(samples,100,3);
newFeatures(:, 1:20, :) = uint8(255 * mat2gray(featuresSplit(:, 1:20, :)));
newFeatures(:, 21:40, :) = uint8(255 * mat2gray(featuresSplit(:, 21:40, :)));
newFeatures(:, 41:60, :) = uint8(255 * mat2gray(featuresSplit(:, 41:60, :)));
newFeatures(:, 61:80, :) = uint8(255 * mat2gray(featuresSplit(:, 61:80, :)));
newFeatures(:, 81:100, :) = uint8(255 * mat2gray(featuresSplit(:, 81:100, :)));

for n = 1:samples
    

IMG = reshape(newFeatures(n, :, :),[10, 10, 3]);

IMG = uint8(IMG);
imwrite(IMG, strcat('F-Classical ', num2str(n), '.png'));
end

end