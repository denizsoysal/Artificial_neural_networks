clear
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%generation of examples and targets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%sampling interval
dx=0.05;% Decrease this value to increase the number of data points

%input vector : sine function
x=0:dx:3*pi;y=sin(x.^2);

%input vector : noisy sine function
sigma=0.2;% Standard deviation of added noise
yn=y+sigma*randn(size(y));% Add gaussian noise

%target : s
t=yn;% Targets. Change to yn to train on noisy data
    





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot MSE vs #epochs experiment - learning sine function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

H = 180;

algs{1} =  'trainbr';%  Levenberg-Marquardt algorithm
algs{2} =  'trainlm'; %  Levenberg-Marquardt algorithm with bayesian reularization

for i=1:2
    nets{i}=feedforwardnet(H,algs{i});
    nets{i}=configure(nets{i},x,t);% Set the input and output sizes of the net
    nets{i}.divideFcn = 'dividetrain';
    nets{i}.trainParam.epochs=5;  % set the number of epochs for the training 
    [nets{i},tr{i}]=train(nets{i},x,t);
end

figure
subplot(1,2,1);
semilogy(tr{1}.epoch, tr{1}.perf, tr{2}.epoch, tr{2}.perf,'LineWidth',2);
xlabel('epoch') 
ylabel('MSE') 
legend(algs{1},algs{2},'Location','north');

subplot(1,2,2);
semilogy(tr{1}.time, tr{1}.perf, tr{2}.time, tr{2}.perf,'LineWidth',2);
legend(algs{1},algs{2},'Location','north');
xlabel('time [s]') 
ylabel('MSE') 

sgtitle('Performance comparison of different algorithm - one layer MLP learning a sine function')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plots best MSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
y = [tr{1}.best_perf,tr{2}.best_perf];
barh(y)
set(gca,'XScale','log')
title('MSE of the different algorithms after 2000 epochs : Learning of the sine function (logarithmic scale)')
yticklabels({algs{1},algs{2}})




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Bayes Experiment Personnal regression %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d1=8;
d2=7;
d3=7;
d4=5;
d5=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%DATASET CREATION%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%load data and create my own dataset
load("Data_Problem1_regression.mat")
T_new = (d1*T1+d2*T2+d3*T3+d4*T4+d5*T5)/(d1+d2+d3+d4+d5);

%draw 3 indepedents sets of 1000 points (training - validation - set)
%input = (X1,X2), in domain ([0,1]x[0,1])
%output = T_new

%we take 1000 samples
k = 1000;

dataset = [X1 X2 T_new];

%let's samples 3000 samples without replacement (to have different
%train-val-test set)
[sampled_data,index] = datasample(dataset,k*3,'Replace',false);
%let's take the first 1000 samples as training set
training_set = sampled_data(1:1000,:);
%let's take the following 1000 samples as validation set
validation_set = sampled_data(1001:2000,:);
%let's take the last 1000 samples as test set
test_set = sampled_data(2001:3000,:);


inputX1 = [training_set(:,1);validation_set(:,1)];
inputX2 = [training_set(:,2);validation_set(:,2)];
input = [inputX1,inputX2];

target = [training_set(:,3);validation_set(:,3)];

nets{1} = feedforwardnet([50,50],'trainbr');
nets{2} = feedforwardnet([50,50], 'trainlm');





tr=cell(1,7);
for i=1:7
    nets{i}.trainParam.epochs=300;  % set the number of epochs for the training 
    nets{i}.divideFcn = 'divideind';
    nets{i}.divideParam.trainInd = 1:1000;
    nets{i}.divideParam.valInd = 1001:2000;
    [nets{i},tr{i}]=train(nets{i},input.',target.');
end

figure
y = [tr{1}.best_perf,tr{2}.best_perf,tr{3}.best_perf,tr{4}.best_perf,tr{5}.best_perf,tr{6}.best_perf,tr{7}.best_perf];
bar(y)
set(gca,'YScale','log')
title('MSE of a 2 hidden layers MLP after 300 epoch')
xticklabels(['trainbr', 'trainlm']);

