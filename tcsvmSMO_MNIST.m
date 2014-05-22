function tcsvmSMO_MNIST

clear all
close all
clc

%images = loadMNISTImages('train-images.idx3-ubyte');size(images)
%labels = loadMNISTLabels('train-labels.idx1-ubyte');size(labels)

%% load data
[I,labels,I_test,labels_test] = readMNIST(1000);

%% train
nclass = 10;
y_train = double(labels) + 1.0;
x_train = [];
for i = 1:length(I)
    x_train = [x_train; I{i}(:)'];
end
x_train = im2double(x_train);
%clear I
%clear lables



[m n] = size(x_train);
model = {};
option.C = 1;
disp('training...');
for c = 1:nclass
    disp([num2str(c), '-th loop:']);
    idc = find(y_train==c);
    yc_train = -ones(size(y_train));
    yc_train(idc) = 1;
    % tcsvmSMO
    [alphay, b, sv, w] = tcsvmSMO(x_train, yc_train, option);
    mc.alphay = alphay;
    mc.b = b;
    mc.sv = sv;
    model{c} = mc;
end

clear x_train
clear y_train
%% test
y_test = double(labels_test) + 1.0;
x_test = [];
for i = 1:length(I)
    x_test = [x_test; I_test{i}(:)'];
end
x_test = im2double(x_test);
clear I_test
clear lables_test

accuracy = [];
disp('testing...');
for c = 1:nclass
    disp([num2str(c), '-th loop:']);
    idc = find(y_test==c);
    yc_test = -ones(size(y_test));
    yc_test(idc) = 1;
    wc = model{c};
    % predict
    f = x_test*(repmat(wc.alphay, 1, n).*wc.sv)';
    f = sum(f, 2) + wc.b;
    acc = length(find(yc_test.*f>0)) / length(yc_test);
    accuracy = [accuracy acc];
    disp(['accuracy: ', num2str(acc)])
end
disp(['avg-accuracy: ', num2str(mean(accuracy))])