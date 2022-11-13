clear
clc
close all

%my student number = r0875700
%5 largest number in descending order = 8 7 7 5 0
% --> d1=8, d2=7, d3=7, d4=5, d5=0
%T_new = (8T1+7T2+3T3+5T4+0*T5)/(8+7+3+5+0)

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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%Training of the Neural Network%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

algo = 'trainbr';
net= feedforwardnet([20,20],algo);% Define the feedfoward net (3 hidden layers)
net.trainParam.epochs=20;
%train the network with training and validation set

%we will concatenate train, validation set 
%then, in the training function, we select net.divideFcn = 'divideind' to select
%training, validation  based on index 
%important to do that to avoid mixing the different datasets ! 

inputX1 = [training_set(:,1);validation_set(:,1)];
inputX2 = [training_set(:,2);validation_set(:,2)];
target = [training_set(:,3);validation_set(:,3)];

net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:1000;
net.divideParam.valInd = 1001:2000;

input = [inputX1,inputX2];
[net,tr]=train(net,input.',target.');



%let's plot the surface of the training set 
F=scatteredInterpolant(training_set(:,1), training_set(:,2), training_set(:,3));
[x,y] = meshgrid(0:0.01:1);
vq1 = F(x,y);
plot3(training_set(:,1),training_set(:,2),training_set(:,3),'.')
hold on
mesh(x,y,vq1)
title('Training set')
xlabel('X1'), ylabel('X2'), zlabel('Tnew')
legend('Sample Points','Interpolated Surface','Location','NorthWest')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Comparison test surface and approximation of model %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure
%actual test surface
subplot(2,1,1);
F=scatteredInterpolant(test_set(:,1), test_set(:,2), test_set(:,3));
[x,y] = meshgrid(0:0.01:1);
vq1 = F(x,y);
plot3(test_set(:,1),test_set(:,2),test_set(:,3),'.')
hold on
mesh(x,y,vq1)
title('Test set')
xlabel('X1'), ylabel('X2'), zlabel('Tnew (actual label)')
legend('Sample Points','Interpolated Surface','Location','NorthWest')
%approximation of neural network 
test_res=sim(net,test_set(:,1:2)');
subplot(2,1,2);
F=scatteredInterpolant(test_set(:,1), test_set(:,2), test_res');
[x2,y2] = meshgrid(0:0.01:1);
% F.Method = 'nearest';
vq2 = F(x2,y2);
plot3(test_set(:,1),test_set(:,2),test_res','.')
hold on
mesh(x2,y2,vq2)
title('Approximation by the network')
xlabel('X1'), ylabel('X2'), zlabel('Tnew (prediction of model)')
legend('Sample Points','Interpolated Surface','Location','NorthWest')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Comparison test surface and approximation of model %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%here, we will see the MSE for different neural network 

nets{1} = feedforwardnet([3,3],algo);
nets{2} = feedforwardnet([5,5],algo);
nets{3} = feedforwardnet([10,10],algo);
nets{4} = feedforwardnet([15,15],algo);
nets{5} = feedforwardnet([20,20],algo);
nets{6} = feedforwardnet([25,25],algo);
nets{7} = feedforwardnet([30,30],algo);




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
title('MSE of a 2 hidden layers MLP after 300 epoch, for different number of neurons per hidden layer (logarithmic scale)')
xticklabels(["3","5","10","15","20","25","30"]);



