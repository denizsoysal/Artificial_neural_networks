clear
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of different algorithms
% traingd gradient descent
% traingda gradient descent with adaptive learning rate
% traingdm gradient descent with momentum
% traingdx gradient descent with momentum and adaptive learning rate
% traincgf Fletcher-Reeves conjugate gradient algorithm
% traincgp Polak-Ribiere conjugate gradient algorithm
% trainbfg BFGS quasi Newton algorithm (quasi Newton)
% trainlm Levenberg-Marquardt algorithm (adaptive mixture of Newton and steepest descent algorithms)
%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Configuration:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%create a cell array of the different algo we will use
algs{1} =  'traingd';% gradient descent
algs{2} = 'trainlm';%  Levenberg-Marquardt algorithm
algs{3} = 'trainbfg';% BFGS quasi Newton algorithm (quasi Newton)
algs{4} = 'traingda';% gradient descent with adaptive learning rate'
algs{5} = 'traincgf';% Fletcher-Reeves conjugate gradient algorithm'


H = 50;% Number of neurons in the hidden layer
delta_epochs = [1,14,2000];% Number of epochs to train in each step
epochs = cumsum(delta_epochs);

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
t=y;% Targets. Change to yn to train on noisy data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%creation of networks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%for all networks :
for i=1:5
    nets{i} = feedforwardnet(H,algs{i});% Define the feedfoward net (hidden layers
    nets{i} = configure(nets{i},x,t);% Set the input and output sizes of the net
    nets{i}.divideFcn = 'dividetrain';% Use training set only (no validation and test split)
end



% Initialize the weights (randomly)
nets{1}=init(nets{1});

% Set the same weights for all networks 
for i=2:5
    nets{i}.iw{1,1} =nets{1}.iw{1,1};
    nets{i}.iw{2,1} =nets{1}.lw{2,1};
    nets{i}.b{1}=nets{1}.b{1};
    nets{i}.b{2}=nets{1}.b{2};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training and simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% set the number of epochs for the training 
for i=1:5
    nets{i}.trainParam.epochs=delta_epochs(1); 
end

% train the networks
for i=1:5
    nets{i}=train(nets{i},x,t);
end

% simulate the networks with the input vector x
for i=1:5
    a1{i}=sim(nets{i},x)
end

% set the number of epochs for the training 
for i=1:5
    net{i}.trainParam.epochs=delta_epochs(2); 
end

% train the networks
for i=1:5
    nets{i}=train(nets{i},x,t);
end

% simulate the networks with the input vector x
for i=1:5
    a2{i}=sim(nets{i},x)
end

% set the number of epochs for the training 
for i=1:5
    net{i}.trainParam.epochs=delta_epochs(3); 
end

% train the networks
for i=1:5
    nets{i}=train(nets{i},x,t);
end

% simulate the networks with the input vector x
for i=1:5
    a3{i}=sim(nets{i},x)
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plots of estimation and regression between targets and outputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
subplot(3,6,1);
% plot the sine function and the output of the networks
plot(x,t,'bx',x,a1{1},'r',x,a1{2},'g',x,a1{3},'m',x,a1{4},'m',x,a1{5},'y');
title([num2str(epochs(1)),' epochs']);
legend('target',algs{1},algs{2},algs{3},algs{4},algs{5},'Location','north');
% perform a linear regression analysis and plot the result
for i=2:6
    subplot(3,6,i);
    postregm(a1{i-1},y);
end
subplot(3,6,7);
plot(x,t,'bx',x,a2{1},'r',x,a2{2},'g',x,a2{3},'m',x,a2{4},'m',x,a2{5},'y');
title([num2str(epochs(2)),' epoch']);
legend('target',algs{1},algs{2},algs{3},algs{4},algs{5},'Location','north');

for i=8:12
    subplot(3,6,i);
    postregm(a2{i-7},y);
end 
subplot(3,6,13);
plot(x,t,'bx',x,a3{1},'r',x,a3{2},'g',x,a3{3},'m',x,a3{4},'m',x,a3{5},'y');
title([num2str(epochs(3)),' epoch','wsh']);
legend('target',algs{1},algs{2},algs{3},algs{4},algs{5},'Location','north');
for i=14:18
    subplot(3,6,i);
    postregm(a3{i-13},y);
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %plot MSE vs #epochs
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% 
% 
% 
% figure
