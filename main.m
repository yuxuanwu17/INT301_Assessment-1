%% getting data
clc;clear;close all;
load('trafficsign.mat');
y = ind2vec(phoglabels')';
X = phogsamples;

%% split data into 8:2
[ndata, D] = size(X);        %ndata样本数，D维数
R = randperm(ndata);         %1到n这些数随机打乱得到的一个随机数字序列作为索引
num_test = floor(ndata*0.2);
X_test = X(R(1:num_test),:);  %以索引的前1000个数据点作为测试样本Xtest
R(1:num_test) = [];
X_train = X(R,:);          %剩下的数据作为训练样本Xtraining
num_training = size(X_train,1);%num_training；训练样本数

y_test = y(R(1:num_test),:);
y_train = y(R,:);

%% Normalization of training and testing data

[X_train,X_output]=mapminmax(X_train',0,1);
X_test = mapminmax('apply',X_test',X_output);
[y_train,y_output]=mapminmax(y_train',0,1);
y_test = mapminmax('apply',y_test',y_output);

%% Assign the new value

data_tr = X_train;
target_tr = y_train;
data_ts = X_test;
target_ts = y_test;
%% train NN

% build the NN
nhidden = 25; %number of hidden layers
net=newff(data_tr,target_tr,[nhidden,nhidden],{'logsig','logsig','logsig'},'traingd');
% set the hyperparameter (epochs, learning rate)
net.trainParam.epochs = 1000; %number of training epochs
net.trainParam.lr = 0.1;
net.trainParam.mc = 0.9;

% train a neural network
[net,tr] = train(net,data_tr,target_tr);
% show network
view(net);
% predict the value
y_predict = sim(net,data_ts)
Y_predict = mapminmax('reverse',y_predict,y_output);
%%
