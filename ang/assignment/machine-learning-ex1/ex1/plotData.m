function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

figure; % open a new figure window
scatter(x, y, 10);
xlabel('profit in 10000$');
ylabel('population in 10000');

end
