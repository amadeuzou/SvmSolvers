function w = tcsvmQP(x, y)
% Two Classes SVM Solver by Quadratic Programming
% min   1/2 W'W
% s.t.  y(wx+b)>=1
% x      -- input data, size = [m, n], m:samples number, n:feature dimension;
% y      -- labels data, size = [m, 1], values=[-1 1], m:samples number;
% w      -- parameters[W, b], size = [n+1, 1], n:elements nubmer;
% author -- amadeuzou AT gmail
% date   -- 11/19/2013, Beijing, China

[m, n] = size(x);
x = [x, ones(m, 1)];

H = zeros(n+1, n+1);
H(1:n, 1:n) = eye(n);
f = zeros(1, n+1);
A = -repmat(y, 1, n+1).*x;
b = -ones(m, 1);
opt = optimset('Algorithm','active-set');
[w,fval,exitflag,output,lambda] = quadprog(H,f,A,b,[],[],[],[],[],opt);

