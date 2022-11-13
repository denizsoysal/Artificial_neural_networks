%%%%%%%%%%%
% rep2.m
% A script which generates n random initial points 
%and visualises results of simulation of a 2d Hopfield network 'net'
%%%%%%%%%%

clear
close all

% We would like to obtain a Hopfield network that has the 3 stable points
% defined by the 3 target (column) vectors in T.

T = [1 1; -1 -1; 1 -1]';

% The function NEWHOP creates Hopfield networks given the stable points T.

net = newhop(T);

%We will test and see the Hopfield networks activity for 100 random inputs
% Note that if the Hopfield network starts out closer to the upper-left, it will
% go to the upper-left, and vise versa.  This ability to find the closest memory
% to an initial input is what makes the Hopfield network useful.

n=200;
epochs=30;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%Generatate random initial points%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for i=1:n
    a={rands(2,1)};                     % generate an initial point 
    [y,Pf,Af] = sim(net,{1 epochs},{},a);   % simulation of the network for 50 timesteps              
    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    start=cell2mat(a);                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r'); % plot evolution
    hold on;
    plot(record(1,epochs),record(2,epochs),'gO',LineWidth=5);  % plot the final point with a green circle
end
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');

%Let's also see the activity of the Hopfield networks when the inputs
%points are on the origin

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%Generatate origin initial points%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a={[0;0]};                     % generate an initial point 
[y,Pf,Af] = sim(net,{1 epochs},{},a);   % simulation of the network for 50 timesteps              
record=[cell2mat(a) cell2mat(y)];   % formatting results  
start=cell2mat(a);                  % formatting results 
plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r'); % plot evolution
hold on;
plot(record(1,epochs),record(2,epochs),'gO',LineWidth=5);  % plot the final point with a green circle
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');


%Let's also see the activity of the Hopfield networks when the inputs
%points are on the x-axis


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%Generatate initial points on  x axis%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:n/10
    x_value = rands(1,1);
    a={[x_value;0]};                     % generate an initial point 
    [y,Pf,Af] = sim(net,{1 epochs},{},a);   % simulation of the network for 50 timesteps              
    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    start=cell2mat(a);                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r'); % plot evolution
    hold on;
    plot(record(1,epochs),record(2,epochs),'gO',LineWidth=5);  % plot the final point with a green circle
    legend('initial state','time evolution','attractor','Location', 'northeast');
    title('Time evolution in the phase space of 2d Hopfield model');
end

%Let's also see the activity of the Hopfield networks when the inputs
%points are on the y-axis

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%Generatate initial points on  y axis%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:n/10
    y_value = rands(1,1);
    a={[0;y_value]};                     % generate an initial point 
    [y,Pf,Af] = sim(net,{1 epochs},{},a);   % simulation of the network for 50 timesteps              
    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    start=cell2mat(a);                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r'); % plot evolution
    hold on;
    plot(record(1,epochs),record(2,epochs),'gO',LineWidth=5);  % plot the final point with a green circle
    legend('initial state','time evolution','attractor','Location', 'northeast');
    title('Time evolution in the phase space of 2d Hopfield model');
end
