function [eval] = rocWeka(filename1,filename2,type)
% ROC Curve
 javaaddpath('C:\Program Files\Weka-3-8\weka.jar');

if(~wekaPathCheck); return,end

import weka.classifiers.*;
import weka.core.Instances;

%train = loadARFF(filename1);

v1 = java.lang.String('-t');
v2 = java.lang.String(filename1);

v3 = java.lang.String('-T');
v4 = java.lang.String(filename2);
% v5 = java.lang.String('-i');

prm = cat(1,v1,v2,v3,v4);

classifier = javaObject(['weka.classifiers.',type]);

eval = Evaluation.evaluateModel(classifier,prm);

end
