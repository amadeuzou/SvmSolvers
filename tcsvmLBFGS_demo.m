function tcsvmLBFGS_demo
clc
clear all
close all


%% generate data
nsamples = 200;
% training data
[x, y] = tcdataGenerator(nsamples);
% testing data
[xt, yt] = tcdataGenerator(nsamples);
[m n] = size(x);

%% Solver: LBFGS Pegasos SGD
option.C = 0.01;
option.debug = 1;
w = tcsvmSGD(x, y, option)


%% Visualize Results
figure(1)
xmin = min(x(:))-1;
xmax = max(x(:))+1;
data_pos = x(find(y==1),:);
data_neg = x(find(y==-1),:);
subplot(121)
hold on
scatter(data_pos(:, 1), data_pos(:, 2), 'b+', 'SizeData', 200, 'LineWidth', 2);
scatter(data_neg(:, 1), data_neg(:, 2), 'gx', 'SizeData', 200, 'LineWidth', 2);
axis tight

margin = xmin:0.1:xmax;
plot(margin, (-w(3)-margin*w(1))/w(2), 'r', 'LineWidth', 2);
plot(margin, (1-w(3)-margin*w(1))/w(2), 'r:', 'LineWidth', 1.5);
plot(margin, (-1-w(3)-margin*w(1))/w(2), 'r:', 'LineWidth', 1.5);
hold off

%% predict
%training data
X = [x ones(m, 1)];
acc = length(find(y.*(X*w)>0))/length(y);
disp(['training acc: ', num2str(acc)])
title(['C = ', num2str(option.C), ', acc = ', num2str(acc)])
T = [xt ones(m, 1)];
acc = length(find(yt.*(T*w)>0))/length(yt);
disp(['testing acc: ', num2str(acc)])


%% LBFGS Solver
option.C = 0.01;
w = tcsvmLBFGS(x, y, option)

%% Visualize Results
figure(1)
subplot(122)
hold on
title(['C = ', num2str(option.C)])
scatter(data_pos(:, 1), data_pos(:, 2), 'b+', 'SizeData', 200, 'LineWidth', 2);
scatter(data_neg(:, 1), data_neg(:, 2), 'gx', 'SizeData', 200, 'LineWidth', 2);
axis tight

margin = xmin:0.1:xmax;
plot(margin, (-w(3)-margin*w(1))/w(2), 'r', 'LineWidth', 2);
plot(margin, (1-w(3)-margin*w(1))/w(2), 'r:', 'LineWidth', 1.5);
plot(margin, (-1-w(3)-margin*w(1))/w(2), 'r:', 'LineWidth', 1.5);
hold off

%% predict
X = [x ones(m, 1)];
acc = length(find(y.*(X*w)>0))/length(y);
disp(['training acc: ', num2str(acc)]);
title(['C = ', num2str(option.C), ', acc = ', num2str(acc)])
T = [xt ones(m, 1)];
acc = length(find(yt.*(T*w)>0))/length(yt);
disp(['testing acc: ', num2str(acc)])

