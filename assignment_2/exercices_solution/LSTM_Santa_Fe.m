clear all;
close all;
clc;
clf;


%inspired by matlab demo that you can run with the command : 
% openExample('nnet/TimeSeriesForecastingUsingDeepLearningExample')


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
p = 50;
training = getTimeSeriesTrainData(dataTrainStandardized,p);

%based on first "p-1" samples
%predict sample number "p"
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%Define LSTM Network Architecture%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create an LSTM regression network. Specify the LSTM layer to have 200 hidden units.

numFeatures = p-1;
numResponses = 1;
numHiddenUnits = 50;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%Specify the training options. Set the solver to 'adam' and train for 
% 250 epochs. To prevent the gradients from exploding, set the gradient 
% threshold to 1. Specify the initial learn rate 0.005, and drop the 
% learn rate after 125 epochs by multiplying by a factor of 0.2.

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%Train LSTM Network %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


net = trainNetwork(X_train,y_train,layers,options);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%Forecast Future Time Steps %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


plot(dataTestStandardized)
hold on
net = predictAndUpdateState(net,X_train);
predictions = training(p, end-p+1:end);

for i=1:100
    [net, predictions(p+i)] = predictAndUpdateState(net, predictions(i+1:p+i-1)','ExecutionEnvironment','cpu');
end
% 
err = immse(dataTestStandardized(1:end-1), predictions(end-98:end)')

%actual value
fig = figure;
plot(dataTestStandardized)
hold on
%prediction
plot(predictions(end-98:end),'r')
legend('Target','Predicted')
title('Prediction of the test set (next 100 points)');



