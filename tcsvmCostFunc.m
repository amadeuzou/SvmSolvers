function [cost grad] = tcsvmCostFunc(x, y, w, C)
% Two Classes SVM Cost Function
% min sum( max{0, 1-y(wx+b)} ) + alpha / 2 * W'W
% x      -- input data, size = [m, n], m:samples number, n:feature dimension;
% y      -- labels data, size = [m, 1], values=[-1 1], m:samples number;
% w      -- parameters[W, b], size = [n, 1], n:elements nubmer;
% C      -- penalty factor
% cost   -- cost
% grad   -- gradient, size = [n, 1];

% author -- amadeuzou AT gmail
% date   -- 11/19/2013, Beijing, China

%% Hinge l1-Loss
m = size(x, 1);
yp = x*w;
yy = 1 - yp.*y;
idx = find(yy>0);
cost = sum(yy(idx)) + C*w'*w;
cost = cost / m;
if nargout == 2
    grad = -x(idx,:)'*y(idx) + 2*C*w;
    grad = grad / m;
end
%% Hinge l2-Loss
%{
m = size(x, 1);
yp = x*w;
idx = find(yp.*y<1);
err = yp(idx)-y(idx);
cost = err'*err/m + C*w'*w;
if nargout == 2
    grad = 2*x(idx,:)'*err + 2*C*w;
end
%}
%% Exponential Loss
%{
m = length(y);
yp=x*w;
cost = sum(exp(1-y.*yp))/m + C*w'*w;
if nargout == 2
    grad = -x'*(y.*exp(1-y.*yp)) + 2*C*w;
end
%}
%% Log loss
%{
m = length(y);
yp=x*w;
cost = sum(log(1+exp(-y.*yp))) / m  + C*w'*w;
if nargout == 2
    grad = -x'*(y./ (1+exp(-y.*yp)))  + 2*C*w;
end
%}

