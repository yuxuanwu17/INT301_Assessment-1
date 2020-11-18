%% getting data
clc;clear;close all;
load('trafficsign.mat');
y = ind2vec(phoglabels')';
X = phogsamples;

%% split data into 8:2
rng(1)
[ndata, D] = size(X);        %ndata，D: feature dimension
R = randperm(ndata);         %1到n randomize the number as index
num_test = floor(ndata*0.2);
X_test = X(R(1:num_test),:);  
R(1:num_test) = [];
X_train = X(R,:);          
num_training = size(X_train,1);

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

% seperate the value 
neurons = [32,64,128];
lr_rate = 0.1;
mc_rate = 0.9;
% lr_rate = [0.1,0.4,0.7];
% mc_rate = [0.6,0.75,0.9];
perf_store = [];
count = 0;
minim = [];
for i = neurons 
    for j = lr_rate
        for k = mc_rate
            % build the NN
            nhidden = i; %number of hidden layers
            net=newff(data_tr,target_tr,[nhidden,nhidden,nhidden],{'logsig','logsig','logsig','logsig'},'traingd');
            % set the hyperparameter (epochs, learning rate)
            net.trainParam.epochs = 500; %number of training epochs
            net.trainParam.lr = j;
            net.trainParam.mc = k;
            % train a neural network
            net = train(net,data_tr,target_tr);
            % predict the value
            y_predict = sim(net,data_ts);
            Y_predict = mapminmax('reverse',y_predict,y_output);
            perf = perform(net, y_test, Y_predict);
            if count == 0
                perf_store = perf;
                minim = perf;
                count = count + 1;
                fprintf('value of mse in neuron %d, learning rate %2.1f, momentum is %3.1f,is %6.4f\n ',i,j,k,perf);
            else 
                perf_store(end+1) = perf;
                if perf < minim;
                    minim = perf;
                end
                count = count + 1;
                fprintf('value of mse in neuron %d, learning rate %2.1f, momentum is %3.1f,is %6.4f\n ',i,j,k,perf);
            end 
        
            
        end 
    end 
end 
%%
% return the min value
perf_store
min(perf_store)

%%
minim=[];
for i = perf_store
    minim = perf_store(1);
    if i < minim
        minim = i;
    end 
    
end
minim
min(minim)
%%

% build the NN
nhidden = 32; %number of hidden layers
net=newff(data_tr,target_tr,[nhidden,nhidden,nhidden],{'logsig','logsig','logsig','logsig'},'traingd');
% set the hyperparameter (epochs, learning rate)
net.trainParam.epochs = 500; %number of training epochs
net.trainParam.lr = 0.1;
net.trainParam.mc = 0.7;

% train a neural network
[net,tr] = train(net,data_tr,target_tr);
% show network

% view(net);

% predict the value
y_predict = sim(net,data_ts);
Y_predict = mapminmax('reverse',y_predict,y_output);

%% Calculate the performance
% calculate the MSE
perf = perform(net, y_test, Y_predict)


%% watch the distribution of predict density and y_test density
num1 = vec2ind(Y_predict); % Predicted groups
num2 = vec2ind(y_test); % Known groups
% figure(1)
% [f,xi]=ksdensity(num1);
% plot(xi,f)
% 
% figure(2)
% [f,xi]=ksdensity(num2);
% plot(xi,f);

%% Create the confusion matrix
% size(num1,2)
figure(4)
C = confusionmat(num2,num1);
confusionchart(C)



%% evaluation of the model 
figure(3)
N = size(y_test,2);
plot(1:N,vec2ind(Y_predict),':b*',1:N,vec2ind(y_test))
legend('预测输出','期望输出')
title('BP网络预测输出','fontsize',12)
ylabel('函数输出','fontsize',12)
xlabel('样本','fontsize',12)
