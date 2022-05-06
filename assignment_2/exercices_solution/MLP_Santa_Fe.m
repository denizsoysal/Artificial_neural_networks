clear all;
close all;
clc;
clf;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%preparation of data%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%
%%training set

%load dataset : first 1000 training points
training_set = load('..\Files\lasertrain.dat');

%normalize dataset
mu = mean(training_set);
sig = std(training_set);
dataTrainStandardized = (training_set - mu) / sig;

%build data and target for training set
%we will use a window of size p. This will allows us to train the network
%to predict a point based on "p" points. We will use the network after to
%predict the unknown 100 points based on all the training set (1000 points)
p = 40;
training = getTimeSeriesTrainData(dataTrainStandardized,p);

X_train = training(1:p-1,:);
y_train = training(p,:);

% %%%%%%%%%%%%%%
%%test set

%load dataset : next 100 test points to predict (after the first 1000
%training points of the training dataset
test_set = load('..\Files\laserpred.dat');

%normalize dataset
mu = mean(test_set);
sig = std(test_set);
dataTestStandardized = (test_set - mu) / sig;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%train on training set %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%let's use trainlm
algo = 'trainlm';
%number of hidden layer :
H = 20;
%create net with one hidden laeyr
net = feedforwardnet(H, algo);


% Initialize the weights (randomly)
net=init(net);
net.trainParam.epochs = 1000;
net.divideFcn = 'divideblock';
net.trainParam.max_fail = 6;
net.divideParam.trainRatio = 0.9;
net.divideParam.valRatio   = 0.1;
net.divideParam.testRatio  = 0;


net = train(net, X_train , y_train);

%predict on training set
y_train_predict = sim(net, X_train);
%compare actual value with predicted
plot(y_train,'r')
hold on
plot(y_train_predict,'b')
hold off
title('Prediction of the training set (first 1000 points)');
legend('Target','Predicted')


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%predict the test set  %%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %based on windows of size "p" 
% %predict the next point based on window

window = training(p, end-p+1:end);

%To predict the test set we write a for loop that
%includes the predicted value from the previous timestep in the input vector 
% to predict the next timestep
%at the end, the last 100 elem of window will be our prediction ! 

for i=1:100
    window(p+i) = sim(net, window(i+1:p+i-1)');
end
%calculate MSE between actual (dataTestStandardized) and predicted
%(window(end-99:end))

err = immse(dataTestStandardized, window(end-99:end)')


%actual value
fig = figure;
plot(dataTestStandardized)
hold on
%prediction
plot(window(end-98:end),'r')
legend('Target','Predicted')
title('Prediction of the test set (next 100 points)');



