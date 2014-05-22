function tcsvmLBFGS_Demo_mc

clear all
close all
clc

%% generate data
nsamples = 100;
ds_c1 = mvnrnd ( [1,1]*2, eye(2), nsamples );
lab_c1 = ones(nsamples, 1);
ds_c2 = mvnrnd ( [-1,-1]*2, eye(2), nsamples );
lab_c2 = 2*ones(nsamples, 1);
ds_c3 = mvnrnd ( [-1.5,1.5]*3, 1.5*eye(2), nsamples );
lab_c3 = 3*ones(nsamples, 1);
ds = [ds_c1; ds_c2; ds_c3];
lab = [lab_c1; lab_c2; lab_c3];
%scatter(ds(:, 1), ds(:, 2), 'b+', 'SizeData', 200, 'LineWidth', 2);


%% train

x = ds;
y = lab;
nclass = length(unique(y));


[m n] = size(x);
model = {};
option.C = 1;
option.debug = 0;
option.max_itr = 100;
disp('training...');
for c = 1:nclass
    disp([num2str(c), '-th loop:']);
    idc = find(y==c);
    yc = -ones(size(y));
    yc(idc) = 1;
    % tcsvmLBFGS
    wc = tcsvmLBFGS(x, yc, option);
    model{c} = wc;
end


%% Visualize Results
figure(1)

xx = [x ones(size(x, 1), 1)];
xmin = min(x(:, 1))-1;
xmax = max(x(:, 1))+1;
margin = xmin:0.1:xmax;
colors = ['r' 'g' 'b' 'y' 'k'];
stlyes = ['r' 'g' 'b' 'y' 'k'];
accuracy = [];
hold on
for c = 1:nclass
    idc = find(y==c);
    data_c = x(idc,:);
    
    scatter(data_c(:, 1), data_c(:, 2), colors(c),'LineWidth', 2);

    w = model{c};
    plot(margin, (-w(3)-margin*w(1))/w(2), [colors(c) ''], 'LineWidth', 2);
    plot(margin, (1-w(3)-margin*w(1))/w(2), [colors(c) ':'], 'LineWidth', 1.5);
    plot(margin, (-1-w(3)-margin*w(1))/w(2), [colors(c) ':'], 'LineWidth', 1.5);


    yc = -ones(size(y));
    yc(idc) = 1;
    wc = model{c};
    % predict
    acc = length(find(yc.*(xx*wc)>0))/length(yc);
    accuracy = [accuracy acc];
    disp(['accuracy: ', num2str(acc)])
end
disp(['avg-accuracy: ', num2str(mean(accuracy))])
axis tight
hold off