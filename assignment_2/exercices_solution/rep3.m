clear
close all

%%%%%%%%%%%
% rep3.m
% A script which generates n random initial points for
% and visualise results of simulation of a 3d Hopfield network net
%%%%%%%%%%
T = [1 1 1; -1 -1 1; 1 -1 -1]';
net = newhop(T);
epochs=100
n=8;
x = linspace(-1,1,n);
y = linspace(-1,1,n);
z = linspace(-1,1,n);
[X, Y, Z] = ndgrid(x, y, z);
Xi = [X(:), Y(:), Z(:)]';
converge_time = 0;
for i=1:n*n*n
    a={rands(3,1)};                         % generate an initial point                   
    [y,Pf,Af] = sim(net,{1 epochs},{},Xi(:,i));       % simulation of the network  for 50 timesteps
    record=[Xi(:,i) cell2mat(y)];       % formatting results
        for j=1:size(record, 2)
        if (ismember(record(:,j)', T', 'rows'))
            converge_time = converge_time + j;
            break
        end
    end
    start=Xi(:,i);                      % formatting results 
    plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r');  % plot evolution
    hold on;
    plot3(record(1,epochs),record(2,epochs),record(3,epochs),'gO',LineWidth=5);  % plot the final point with a green circle
end
grid on;
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 3d Hopfield model');
