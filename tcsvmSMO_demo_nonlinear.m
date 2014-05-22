function tcsvmSMO_demo_nonlinear
clc
clear all
close all

%% generate data
nsamples = 100;
% training data
[x, y] = nonlinearData(nsamples);
% testing data
[xt, yt] = nonlinearData(nsamples);
[m n] = size(x);

%% SMO Solver with linear kernel
disp('linear kernel:');
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

margin = xmin:0.1:xmax;
plot(margin, (-w(3)-margin*w(1))/w(2), 'r', 'LineWidth', 2);
plot(margin, (1-w(3)-margin*w(1))/w(2), 'r:', 'LineWidth', 1.5);
plot(margin, (-1-w(3)-margin*w(1))/w(2), 'r:', 'LineWidth', 1.5);

ezpolar(@(x)1);
ezpolar(@(x)2);
%% predict
%training data
svn = length(alphay);
f = x*(repmat(alphay, 1, n).*sv)';
f = sum(f, 2) + b;
accuracy = length(find(y.*f>0)) / length(y);
disp(['training accuracy: ', num2str(accuracy)])
title(['C = ', num2str(option.C), ', accuracy = ', num2str(accuracy), ', linear kernel'])

%testing data
f = xt*(repmat(alphay, 1, n).*sv)';
f = sum(f, 2) + b;
accuracy = length(find(yt.*f>0)) / length(y);
disp(['testing accuracy: ', num2str(accuracy)])




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SMO Solver with Gaussian kernel
disp('gaussian kernel:')
option.C = 1;
option.kernel = 'gaussian';
option.gaussian_sigma = 4;
[alphay, b, sv, w, index] = tcsvmSMO(x, y, option);
if length(alphay) < 1
    disp('failed');
    return;
end
svmModel.alphay = alphay;
svmModel.b = b;
svmModel.sv = sv;
svmModel.option = option;
w = [w; b];
%% Visualize Results
figure(1)
subplot(122)
hold on
scatter(data_pos(:, 1), data_pos(:, 2), 'b+', 'SizeData', 200, 'LineWidth', 2);
scatter(data_neg(:, 1), data_neg(:, 2), 'gx', 'SizeData', 200, 'LineWidth', 2);
axis tight

idx = find(alphay>0);
scatter(x(index, 1), x(index, 2), 'yo', 'SizeData', 200, 'LineWidth', 2);
scatter(x(index(idx), 1), x(index(idx), 2), 'ro', 'SizeData', 200, 'LineWidth', 2);

legend('+positive', '-negative', '-support vector', '+support vector')
% Circle Fit for support vector
c = CircleFitByTaubin(x(index,:));
plot(c(1)+c(3)*cos(0:pi/50:2*pi), c(1)+c(3)*sin(0:pi/50:2*pi), 'r-')

ezpolar(@(x)1);
ezpolar(@(x)2);
%% predict
%training data
accuracy =  predict(x, y, svmModel);
disp(['training accuracy: ', num2str(accuracy)])
title(['C = ', num2str(option.C), ', accuracy = ', num2str(accuracy), ', gaussian kernel'])

%testing data
accuracy =  predict(xt, yt, svmModel);
disp(['testing accuracy: ', num2str(accuracy)])
%% 
function p = predict(xt, yt, svmModel)
[m n] = size(xt);
alphay = svmModel.alphay;
b = svmModel.b;
sv = svmModel.sv;
option = svmModel.option;

switch lower(option.kernel)
    case 'gaussian'
        zi = repmat(alphay, 1, n).*sv;
        sn = length(alphay);
        sigma = option.gaussian_sigma;
        accurary = 0;
        for i = 1:m
            xi = repmat(xt(i, :), sn, 1);
            k = exp(-sum((xi - zi)*(xi - zi)', 2)/sigma);
            f = sum(k) + b;
            accurary = accurary + (f*yt(i)>0);
        end
        p = accurary / m;
end


%%
function [x y] = nonlinearData(nsamples, pos_per)
if nargin == 1
    pos_per = 0.5;
end
npos = round(nsamples*pos_per);
nneg = nsamples - npos;
%generate  random radius
r = sqrt(rand(npos,1));
%generate  random angles, in range [0,2*pi]
t = 2*pi*rand(npos,1);
data_pos = [r.*cos(t),r.*sin(t)];
labels_pos = ones(npos, 1);

%generate data
r2 = sqrt(3*rand(nneg,1)+1);
t2 = 2*pi*rand(nneg,1);
data_neg = [r2.*cos(t2),r2.*sin(t2)];
labels_neg = -ones(nneg, 1);

x = [data_pos; data_neg];
y = [labels_pos; labels_neg];

%%
function [x, y] = linearData(nsamples)
% Linear Seperable Data in 2-dimension
data_pos= mvnrnd ( [1,1]*2, eye(2), nsamples );
labels_pos = ones(nsamples, 1);
data_neg= mvnrnd ( [-1,-1]*2, eye(2), nsamples);
labels_neg = -ones(nsamples, 1);
x = [data_pos; data_neg];
y = [labels_pos; labels_neg];

function Par = CircleFitByTaubin(XY)

%--------------------------------------------------------------------------
%  
%     Circle fit by Taubin
%      G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
%                  Space Curves Defined By Implicit Equations, With 
%                  Applications To Edge And Range Image Segmentation",
%      IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)
%
%     Input:  XY(n,2) is the array of coordinates of n points x(i)=XY(i,1), y(i)=XY(i,2)
%
%     Output: Par = [a b R] is the fitting circle:
%                           center (a,b) and radius R
%
%     Note: this fit does not use built-in matrix functions (except "mean"),
%           so it can be easily programmed in any programming language
%
%--------------------------------------------------------------------------

n = size(XY,1);      % number of data points

centroid = mean(XY);   % the centroid of the data set

%     computing moments (note: all moments will be normed, i.e. divided by n)

Mxx = 0; Myy = 0; Mxy = 0; Mxz = 0; Myz = 0; Mzz = 0;

for i=1:n
    Xi = XY(i,1) - centroid(1);  %  centering data
    Yi = XY(i,2) - centroid(2);  %  centering data
    Zi = Xi*Xi + Yi*Yi;
    Mxy = Mxy + Xi*Yi;
    Mxx = Mxx + Xi*Xi;
    Myy = Myy + Yi*Yi;
    Mxz = Mxz + Xi*Zi;
    Myz = Myz + Yi*Zi;
    Mzz = Mzz + Zi*Zi;
end

Mxx = Mxx/n;
Myy = Myy/n;
Mxy = Mxy/n;
Mxz = Mxz/n;
Myz = Myz/n;
Mzz = Mzz/n;

%    computing the coefficients of the characteristic polynomial

Mz = Mxx + Myy;
Cov_xy = Mxx*Myy - Mxy*Mxy;
A3 = 4*Mz;
A2 = -3*Mz*Mz - Mzz;
A1 = Mzz*Mz + 4*Cov_xy*Mz - Mxz*Mxz - Myz*Myz - Mz*Mz*Mz;
A0 = Mxz*Mxz*Myy + Myz*Myz*Mxx - Mzz*Cov_xy - 2*Mxz*Myz*Mxy + Mz*Mz*Cov_xy;
A22 = A2 + A2;
A33 = A3 + A3 + A3;

xnew = 0;
ynew = 1e+20;
epsilon = 1e-12;
IterMax = 20;

% Newton's method starting at x=0

for iter=1:IterMax
    yold = ynew;
    ynew = A0 + xnew*(A1 + xnew*(A2 + xnew*A3));
    if abs(ynew) > abs(yold)
       disp('Newton-Taubin goes wrong direction: |ynew| > |yold|');
       xnew = 0;
       break;
    end
    Dy = A1 + xnew*(A22 + xnew*A33);
    xold = xnew;
    xnew = xold - ynew/Dy;
    if (abs((xnew-xold)/xnew) < epsilon), break, end
    if (iter >= IterMax)
        disp('Newton-Taubin will not converge');
        xnew = 0;
    end
    if (xnew<0.)
        fprintf(1,'Newton-Taubin negative root:  x=%f\n',xnew);
        xnew = 0;
    end
end

%  computing the circle parameters

DET = xnew*xnew - xnew*Mz + Cov_xy;
Center = [Mxz*(Myy-xnew)-Myz*Mxy , Myz*(Mxx-xnew)-Mxz*Mxy]/DET/2;

Par = [Center+centroid , sqrt(Center*Center'+Mz)];
