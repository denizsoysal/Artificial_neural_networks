
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

alg0 =  'traingd';% gradient descent
alg1 = 'trainlm';%  Levenberg-Marquardt algorithm
alg2 = 'trainbfg';% BFGS quasi Newton algorithm (quasi Newton)
alg3 = 'traingda';% gradient descent with adaptive learning rate'
alg4 = 'traincgf';% Fletcher-Reeves conjugate gradient algorithm'
alg5 = 'traingdm';%gradient descent with momentum
alg6 = 'traingdx';%gradient descent with momentum and adaptive learning rate
alg7 = "traincgp";%Polak-Ribiere conjugate gradient algorithm

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

net0= feedforwardnet(H,alg0);% Define the feedfoward net (hidden layers)
net1=feedforwardnet(H,alg1);
net2=feedforwardnet(H,alg2);
net3=feedforwardnet(H,alg3);
net4=feedforwardnet(H,alg4);
net5=feedforwardnet(H,alg5);
net6=feedforwardnet(H,alg6);
net7=feedforwardnet(H,alg7);

net0=configure(net0,x,t);% Set the input and output sizes of the net
net1=configure(net1,x,t);
net2=configure(net2,x,t);
net3=configure(net3,x,t);
net4=configure(net4,x,t);
net5=configure(net5,x,t);
net6=configure(net6,x,t);
net7=configure(net7,x,t);

net0.divideFcn = 'dividetrain';% Use training set only (no validation and test split)
net1.divideFcn = 'dividetrain';
net2.divideFcn = 'dividetrain';
net3.divideFcn = 'dividetrain';
net4.divideFcn = 'dividetrain';
net5.divideFcn = 'dividetrain';
net6.divideFcn = 'dividetrain';
net7.divideFcn = 'dividetrain';


% Initialize the weights (randomly)
net0=init(net0);
% Set the same weights for all networks 
net1.iw{1,1}=net0.iw{1,1};
net2.iw{1,1}=net0.iw{1,1};
net3.iw{1,1}=net0.iw{1,1};
net4.iw{1,1}=net0.iw{1,1};
net1.lw{2,1}=net0.lw{2,1};
net2.lw{2,1}=net0.lw{2,1};
net3.lw{2,1}=net0.lw{2,1};
net4.lw{2,1}=net0.lw{2,1};
net5.lw{2,1}=net0.lw{2,1};
net6.lw{2,1}=net0.lw{2,1};
net7.lw{2,1}=net0.lw{2,1};
% Set same biases for all networks 
net1.b{1}=net0.b{1};
net1.b{2}=net0.b{2};
net2.b{1}=net0.b{1};
net2.b{2}=net0.b{2};
net3.b{1}=net0.b{1};
net3.b{2}=net0.b{2};
net4.b{1}=net0.b{1};
net4.b{2}=net0.b{2};
net5.b{1}=net0.b{1};
net5.b{2}=net0.b{2};
net6.b{1}=net0.b{1};
net6.b{2}=net0.b{2};
net7.b{1}=net0.b{1};
net7.b{2}=net0.b{2};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training and simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
net0.trainParam.epochs=delta_epochs(1);  % set the number of epochs for the training 
net1.trainParam.epochs=delta_epochs(1);
net2.trainParam.epochs=delta_epochs(1);
net3.trainParam.epochs=delta_epochs(1);
net4.trainParam.epochs=delta_epochs(1);
net5.trainParam.epochs=delta_epochs(1);
net6.trainParam.epochs=delta_epochs(1);
net7.trainParam.epochs=delta_epochs(1);

net0=train(net0,x,t);   % train the networks
net1=train(net1,x,t);
net2=train(net2,x,t);
net3=train(net3,x,t);
net4=train(net4,x,t);
net5=train(net5,x,t);
net6=train(net6,x,t);
net7=train(net7,x,t);

a01=sim(net0,x); a11=sim(net1,x);  % simulate the networks with the input vector x
a21=sim(net2,x); a31=sim(net3,x);
a41=sim(net4,x);a51=sim(net5,x);
a61=sim(net6,x);a71=sim(net7,x);

net0.trainParam.epochs=delta_epochs(2);
net1.trainParam.epochs=delta_epochs(2);
net2.trainParam.epochs=delta_epochs(2);
net3.trainParam.epochs=delta_epochs(2);
net4.trainParam.epochs=delta_epochs(2);
net5.trainParam.epochs=delta_epochs(2);
net6.trainParam.epochs=delta_epochs(2);
net7.trainParam.epochs=delta_epochs(2);



net0=train(net0,x,t);
net1=train(net1,x,t);
net2=train(net2,x,t);
net3=train(net3,x,t);
net4=train(net4,x,t);
net5=train(net5,x,t);
net6=train(net6,x,t);
net7=train(net7,x,t);


a02=sim(net0,x); a12=sim(net1,x);
a22=sim(net2,x); a32=sim(net3,x);
a42=sim(net4,x); a52=sim(net5,x);
a62=sim(net6,x); a72=sim(net7,x);

net0.trainParam.epochs=delta_epochs(3);
net1.trainParam.epochs=delta_epochs(3);
net2.trainParam.epochs=delta_epochs(3);
net3.trainParam.epochs=delta_epochs(3);
net4.trainParam.epochs=delta_epochs(3);
net5.trainParam.epochs=delta_epochs(3);
net6.trainParam.epochs=delta_epochs(3);
net7.trainParam.epochs=delta_epochs(3);

net0=train(net0,x,t);
net1=train(net1,x,t);
net2=train(net2,x,t);
net3=train(net3,x,t);
net4=train(net4,x,t);
net5=train(net5,x,t);
net6=train(net6,x,t);
net7=train(net7,x,t);


a03=sim(net0,x); a13=sim(net1,x);
a23=sim(net2,x); a33=sim(net3,x);
a43=sim(net4,x); a53=sim(net5,x);
a63=sim(net6,x); a73=sim(net7,x);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plots of estimation of functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
subplot(2,4,1);
plot(x,t,'bx',x,a01);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg0, ' after ', num2str(epochs(1)),' epochs']);
subplot(2,4,2);
plot(x,t,'bx',x,a11);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg1, ' after ', num2str(epochs(1)),' epochs']);
subplot(2,4,3);
plot(x,t,'bx',x,a21);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg2, ' after ', num2str(epochs(1)),' epochs']);
subplot(2,4,4);
plot(x,t,'bx',x,a31);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg3, ' after ', num2str(epochs(1)),' epochs']);
subplot(2,4,5);
plot(x,t,'bx',x,a41);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg4, ' after ', num2str(epochs(1)),' epochs']);
subplot(2,4,6);
plot(x,t,'bx',x,a51);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg5, ' after ', num2str(epochs(1)),' epochs']);
subplot(2,4,7);
plot(x,t,'bx',x,a61);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg6, ' after ', num2str(epochs(1)),' epochs']);
subplot(2,4,8);
plot(x,t,'bx',x,a71);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg7, 'after ', num2str(epochs(1)),' epochs']);




figure
subplot(2,4,1);
plot(x,t,'bx',x,a02);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg0, ' after ', num2str(epochs(2)),' epochs']);
subplot(2,4,2);
plot(x,t,'bx',x,a12);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg1, ' after ', num2str(epochs(2)),' epochs']);
subplot(2,4,3);
plot(x,t,'bx',x,a22);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg2, ' after ', num2str(epochs(2)),' epochs']);
subplot(2,4,4);
plot(x,t,'bx',x,a32);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg3, ' after ', num2str(epochs(2)),' epochs']);
subplot(2,4,5);
plot(x,t,'bx',x,a42);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg4, ' after ', num2str(epochs(2)),' epochs']);
subplot(2,4,6);
plot(x,t,'bx',x,a52);
title(['Approximation of ', alg5, ' after ', num2str(epochs(2)),' epochs']);
subplot(2,4,7);
plot(x,t,'bx',x,a62);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg6, ' after ', num2str(epochs(2)),' epochs']);
subplot(2,4,8);
plot(x,t,'bx',x,a72);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg7, 'after ', num2str(epochs(2)),' epochs']);


figure
subplot(2,4,1);
plot(x,t,'bx',x,a03);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg0, ' after ', num2str(epochs(3)),' epochs']);
subplot(2,4,2);
plot(x,t,'bx',x,a13);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg1, ' after ', num2str(epochs(3)),' epochs']);
subplot(2,4,3);
plot(x,t,'bx',x,a23);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg2, ' after ', num2str(epochs(3)),' epochs']);
subplot(2,4,4);
plot(x,t,'bx',x,a33);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg3, ' after ', num2str(epochs(3)),' epochs']);
subplot(2,4,5);
plot(x,t,'bx',x,a43);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg4, ' after ', num2str(epochs(3)),' epochs']);
subplot(2,4,6);
plot(x,t,'bx',x,a53);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg5, ' after ', num2str(epochs(3)),' epochs']);
subplot(2,4,7);
plot(x,t,'bx',x,a63);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg6, ' after ', num2str(epochs(3)),' epochs']);
subplot(2,4,8);
plot(x,t,'bx',x,a73);
xlim([0 10]);
ylim([-1 1]);
title(['Approximation of ', alg7, 'after ', num2str(epochs(3)),' epochs']);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plots of estimation and regression between targets and outputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
subplot(3,6,1);
% plot the sine function and the output of the networks
plot(x,t,'bx',x,a01,'r',x,a11,'g',x,a21,'m',x,a31,'m',x,a41,'y');
title([num2str(epochs(1)),' epochs']);
legend('target',alg0,alg1,alg2,alg3,alg4,'Location','north');
subplot(3,6,2);
postregm(a01,y); % perform a linear regression analysis and plot the result
subplot(3,6,3);
postregm(a11,y);
subplot(3,6,4);
postregm(a21,y);
subplot(3,6,5);
postregm(a31,y);
subplot(3,6,6);
postregm(a41,y);
%
subplot(3,6,7);
plot(x,t,'bx',x,a02,'r',x,a12,'g',x,a22,'m',x,a32,'m',x,a42,'y');
title([num2str(epochs(2)),' epoch']);
legend('target',alg0,alg1,alg2,alg3,alg4,'Location','north');
subplot(3,6,8);
postregm(a02,y);
subplot(3,6,9);
postregm(a12,y);
subplot(3,6,10);
postregm(a22,y);
subplot(3,6,11);
postregm(a32,y);
subplot(3,6,12);
postregm(a42,y);
%
subplot(3,6,13);
plot(x,t,'bx',x,a03,'r',x,a13,'g',x,a23,'m',x,a33,'m',x,a43,'y');
title([num2str(epochs(3)),' epoch','wsh']);
legend('target',alg0,alg1,alg2,alg3,alg4,'Location','north');
subplot(3,6,14);
postregm(a03,y);
subplot(3,6,15);
postregm(a13,y);
subplot(3,6,16);
postregm(a23,y);
subplot(3,6,17);
postregm(a33,y);
subplot(3,6,18);
postregm(a43,y);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot MSE vs #epochs experiment - learning sine function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


algs{1} =  'trainlm';%  Levenberg-Marquardt algorithm
algs{2} = 'trainbfg';% BFGS quasi Newton algorithm (quasi Newton)
algs{3} = "traincgp";%Polak-Ribiere conjugate gradient algorithm
algs{4} = 'traincgf';% Fletcher-Reeves conjugate gradient algorithm'
algs{5} = 'traingdx';%gradient descent with momentum and adaptive learning rate
algs{6} = 'traingda';% gradient descent with adaptive learning rate'
algs{7} = 'traingdm';%gradient descent with momentum
algs{8} = "traingd";%Polak-Ribiere conjugate gradient algorithm
algs{9} = "trainbr";%Bayesian Learning



for i=1:9
    nets{i}=feedforwardnet(H,algs{i})
    nets{i}=configure(nets{i},x,t);% Set the input and output sizes of the net
    nets{i}.divideFcn = 'dividetrain';
    nets{i}.trainParam.epochs=2000;  % set the number of epochs for the training 
    [nets{i},tr{i}]=train(nets{i},x,t);
end

figure
subplot(1,2,1);
semilogy(tr{1}.epoch, tr{1}.perf, tr{2}.epoch, tr{2}.perf,tr{3}.epoch,... 
    tr{3}.perf,tr{4}.epoch, tr{4}.perf,tr{5}.epoch, tr{5}.perf,tr{6}.epoch, tr{6}.perf,tr{7}.epoch, tr{7}.perf,'m',...
    tr{8}.epoch, tr{8}.perf,'r',tr{9}.epoch,tr{9}.perf,'g','LineWidth',2);
xlabel('epoch') 
ylabel('MSE') 
legend(algs{1},algs{2},algs{3},algs{4},algs{5},algs{6},algs{7},algs{8},algs{9},'Location','north');

%I don't know why but gradient descent time does not start at 0
%this is a work around I have found
tr{1}.time = tr{1}.time - tr{1}.time(1);
subplot(1,2,2);
semilogy(tr{1}.time, tr{1}.perf, tr{2}.time, tr{2}.perf,tr{3}.time,... 
    tr{3}.perf,tr{4}.time, tr{4}.perf,tr{5}.time, tr{5}.perf,tr{6}.time, tr{6}.perf,tr{7}.time, tr{7}.perf,'m',...
    tr{8}.time, tr{8}.perf,'r',tr{9}.time,tr{9}.perf,'g','LineWidth',2);
legend(algs{1},algs{2},algs{3},algs{4},algs{5},algs{6},algs{7},algs{8},algs{9},'Location','north');
xlabel('time [s]') 
ylabel('MSE') 


sgtitle('Performance comparison of different algorithm - one layer MLP learning a sine function')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plots best MSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% figure
% y = [tr{1}.best_perf,trCopy{1}.best_perf;tr{2}.best_perf,trCopy{2}.best_perf;tr{3}.best_perf,trCopy{3}.best_perf;tr{4}.best_perf,trCopy{4}.best_perf;tr{5}.best_perf,trCopy{5}.best_perf;tr{6}.best_perf,trCopy{6}.best_perf;tr{7}.best_perf,trCopy{7}.best_perf;tr{8}.best_perf,trCopy{8}.best_perf;tr{9}.best_perf,trCopy{9}.best_perf];
% 
% barh(y)
% set(gca,'XScale','log')
% title('MSE of the different algorithms after 2000 epochs : Learning of the sine function (logarithmic scale)')
% yticklabels({algs{1},algs{2},algs{3},algs{4},algs{5},algs{6},algs{7},algs{8},algs{9}})
% 



figure
y = [tr{1}.best_perf,tr{2}.best_perf,tr{3}.best_perf,tr{4}.best_perf,tr{5}.best_perf,tr{6}.best_perf,tr{7}.best_perf,tr{8}.best_perf,tr{9}.best_perf];
barh(y)
set(gca,'XScale','log')
title('MSE of the different algorithms after 2000 epochs : Learning of the sine function (logarithmic scale)')
yticklabels({algs{1},algs{2},algs{3},algs{4},algs{5},algs{6},algs{7},algs{8},algs{9}})
