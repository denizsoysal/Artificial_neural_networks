%number of neurons
numN=10;
%training algorithm (ex : traingd)
% traingd gradient descent
% traingda gradient descent with adaptive learning rate
% traingdm gradient descent with momentum
% traingdx gradient descent with momentum and adaptive learning rate
% traincgf Fletcher-Reeves conjugate gradient algorithm
% traincgp Polak-Ribiere conjugate gradient algorithm
% trainbfg BFGS quasi Newton algorithm (quasi Newton)
% trainlm Levenberg-Marquardt algorithm (adaptive mixture of Newton and steepest descent algorithms)
trainAlg=traingd;
%instanciate an MLP network
net = feedforwardnet(numN,trainAlg);


%input
P=0;
%targets
t=0;
%training the network :
net=train(net,P,T);
%simulate network
a =sim(net,P);
%postreg which calculates and visualizes regression between targets and outputs
[m,b,r]=postreg(a,T); %m and b are the slope and the y-intercept of the best linear regression respectively. r is a correlation between targets T and
                      %outputs a.