%%
close all;
clear;
clc;
%% Load the data
load('trafficsign.mat');
[trainInd, testInd] = crossvalind('HoldOut',phoglabels,0.2);
trainSample = phogsamples(trainInd,:);
testSample = phogsamples(testInd,:);
trainLabel = ind2vec(phoglabels(trainInd,:)')';
testLabel = ind2vec(phoglabels(testInd,:)')';
%% Network training
net = newff(trainSample',trainLabel',[5,5],{},'traingd');
net = init(net);

for i = 1:2
    net.layers{i}.transferFcn = 'logsig'
end
net.trainParam.epochs = 10000;
net = train(net,trainSample',trainLabel')
view(net)
%% Testing
testProb = sim(net,testSample')
testClass = vec2ind(testProb)
C=confusionmat(vec2ind(testLabel'),testClass');
confusionchart(C)