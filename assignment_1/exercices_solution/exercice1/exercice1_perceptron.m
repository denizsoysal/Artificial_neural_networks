%% Classification with a 2-Input Perceptron
% A 2-input hard limit neuron is trained to classify 5 input vectors into two
% categories.
%
% Copyright 1992-2014 The MathWorks, Inc.

%%
% Each of the four column vectors in X defines a 2-element input vectors and a
% row vector T defines the vector's target categories.  We can plot these
% vectors with PLOTPV.

X = [ -0.5 -0.5 +0.3 -0.1;  ... %each row of X correspond to a feature
      -0.5 +0.5 -0.5 +1.0];     %each column of X correspond to a sample
                                %we have sample 1 = -0.5 -0.5  (feature_1 = -0.5, feature_2 = -0.5)   ; sample 2  = -0.5  +0.5 (feature_1 = -0.5, feature_2 = +0.5) 
T = [1 1 0 0];                  %target values (for sample 1, target is 1 ; for sample ...)
plotpv(X,T);                    %plot the points and the target
                                %--> the input X will be plot in a 2D space, and the target will be represent by 'circle' and '+' 
                      

%%
% The perceptron must properly classify the 4 input vectors in X into the two
% categories defined by T.  Perceptrons have HARDLIM neurons.  These neurons are
% capable of separating an input space with a straight line into two categories
% (0 and 1).
%
% Here PERCEPTRON creates a new neural network with a single neuron. The
% network is then configured to the data, so we can examine its
% initial weight and bias values. (Normally the configuration step can be
% skipped as it is automatically done by ADAPT or TRAIN.)

net = perceptron;               %we create the percerpetron
net = configure(net,X,T);       %we configure it to the date (to have the right number of input to the perceptron, here 2 inputs)

%%
% The input vectors are replotted with the neuron's initial attempt at
% classification.
%
% The initial weights are set to zero, so any input gives the same output and
% the classification line does not even appear on the plot.  Fear not... we are
% going to train it!

plotpv(X,T);    
plotpc(net.IW{1},net.b{1});     %we can try to plot the decision line of the perceptron without training it, but because all the weights are intialized to 0, it will not draw anything

%%
% Here the input and target data are converted to sequential data (cell
% array where each column indicates a timestep) and copied three times
% to form the series XX and TT.
%
% ADAPT updates the network for each timestep in the series and returns
% a new network object that performs as a better classifier.

XX = repmat(con2seq(X),1,3);    %here, we convert the data X and the target T to sequential data that can be inputted to the perceptron and copy these 
TT = repmat(con2seq(T),1,3);    %values 3 times
net = adapt(net,XX,TT);         %"adapt" is the "fit" function. It will have 3 iteration because we have copied the samples 3 time
plotpc(net.IW{1},net.b{1});     %then we plot the decision line

%%
% Now SIM is used to classify any other input vector, like [0.7; 1.2]. A plot of
% this new point with the original training set shows how the network performs.
% To distinguish it from the training set, color it red.

x = [0.7; 1.2];                          %we define a new sample (a test sample)
y = net(x);                              %we compute the output of our model with the sample "x" as an input
plotpv(x,y);                             %we plot the sample and the prediction "y" : this prediction is represented by a 'circle' and '+' : in the case [0.7; 1.2], it is a circle
point = findobj(gca,'type','line');
point.Color = 'red';

%%
% Turn on "hold" so the previous plot is not erased and plot the training set
% and the classification line.
%
% The perceptron correctly classified our new point (in red) as category "zero"
% (represented by a circle) and not a "one" (represented by a plus).

hold on;
plotpv(X,T);
plotpc(net.IW{1},net.b{1});
hold off;


displayEndOfDemoMessage(mfilename)
