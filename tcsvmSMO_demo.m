function tcsvmSMO_demo
clc
clear all
close all

%% generate data
nsamples = 100;
% training data
[x, y] = tcdataGenerator(nsamples, 0.5, 'normal');
% testing data
[xt, yt] = tcdataGenerator(nsamples, 0.5, 'normal');
[m n] = size(x);

%% SMO Solver
option.C = 1;
option.kernel = 'linear';
[alphay, b, sv, w, index] = tcsvmSMO(x, y, option);
w = [w; b];

%% Visualize Results
figure(1)
subplot(121)
xmin = min(x(:))-1;
xmax = max(x(:))+1;
data_pos = x(find(y==1),:);
data_neg = x(find(y==-1),:);
hold on
scatter(data_pos(:, 1), data_pos(:, 2), 'b+', 'SizeData', 200, 'LineWidth', 2);
scatter(data_neg(:, 1), data_neg(:, 2), 'gx', 'SizeData', 200, 'LineWidth', 2);
axis tight

scatter(x(index, 1), x(index, 2), 'yo', 'SizeData', 200, 'LineWidth', 2);
legend('positive', 'negative', 'support vector')

margin = xmin:0.1:xmax;
plot(margin, (-w(3)-margin*w(1))/w(2), 'r', 'LineWidth', 2);
plot(margin, (1-w(3)-margin*w(1))/w(2), 'r:', 'LineWidth', 1.5);
plot(margin, (-1-w(3)-margin*w(1))/w(2), 'r:', 'LineWidth', 1.5);

%% predict
%training data
svn = length(alphay);
f = x*(repmat(alphay, 1, n).*sv)';
f = sum(f, 2) + b;
accuracy = length(find(y.*f>0)) / length(y);
disp(['training accuracy: ', num2str(accuracy)])
title(['C = ', num2str(option.C), ', accuracy = ', num2str(accuracy)])

%testing data
f = xt*(repmat(alphay, 1, n).*sv)';
f = sum(f, 2) + b;
accuracy = length(find(yt.*f>0)) / length(y);
disp(['testing accuracy: ', num2str(accuracy)])

%%
%% SMO Solver
option.C = 0.1;
[alphay, b, sv, w, index] = tcsvmSMO(x, y, option);
w = [w; b];

%% Visualize Results
figure(1)
subplot(122)
xmin = min(x(:))-1;
xmax = max(x(:))+1;

hold on
scatter(data_pos(:, 1), data_pos(:, 2), 'b+', 'SizeData', 200, 'LineWidth', 2);
scatter(data_neg(:, 1), data_neg(:, 2), 'gx', 'SizeData', 200, 'LineWidth', 2);
axis tight

scatter(x(index, 1), x(index, 2), 'yo', 'SizeData', 200, 'LineWidth', 2);
legend('positive', 'negative', 'support vector')

margin = xmin:0.1:xmax;
plot(margin, (-w(3)-margin*w(1))/w(2), 'r', 'LineWidth', 2);
plot(margin, (1-w(3)-margin*w(1))/w(2), 'r:', 'LineWidth', 1.5);
plot(margin, (-1-w(3)-margin*w(1))/w(2), 'r:', 'LineWidth', 1.5);
%% predict
%training data
svn = length(alphay);
f = x*(repmat(alphay, 1, n).*sv)';
f = sum(f, 2) + b;
accuracy = length(find(y.*f>0)) / m;
disp(['training accuracy: ', num2str(accuracy)])
title(['C = ', num2str(option.C), ', accuracy = ', num2str(accuracy)])

%testing data
f = xt*(repmat(alphay, 1, n).*sv)';
f = sum(f, 2) + b;
accuracy = length(find(yt.*f>0)) / m;
disp(['testing accuracy: ', num2str(accuracy)])


