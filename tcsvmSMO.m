function [alphay, b, sv, w, index] = tcsvmSMO(x, y, option)
% x      -- input data, size = [m, n], m:samples number, n:feature dimension;
% y      -- labels data, size = [m, 1], values=[-1 1], m:samples number;
% option -- option struct
%        C:       penalty factor
%        debug:   show debug message
%        kernel:  linear, gaussian

% w      -- output parameters when linear kernel, size = [n, 1], n:elements nubmer;
% b      -- bias
% alphay -- alpha*y for support vector, size = [sn, 1], 
% sv     -- support vector, size = [sn, n], sn: support vectors number

% author -- amadeuzou AT gmail
% date   -- 11/19/2013, Beijing, China

%%
% SVM Sequential Minimal Optimization
% http://www.cnblogs.com/vivounicorn/archive/2011/06/01/2067496.html
% John Platt, Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines, no. MSR-TR-98-14, April 1998
% http://research.microsoft.com/apps/pubs/default.aspx?id=69644
% John C. Platt, Fast Training of Support Vector Machines Using Sequential Minimal Optimization, in Advances in Kernel Methods - Support Vector Learning, MIT Press, January 1998
% http://research.microsoft.com/apps/pubs/?id=68391
% Sequential Minimal Optimization for svm 
% http://www.cs.iastate.edu/~honavar/smo-svm.pdf

if nargin == 2
    option.C = 1;
    option.accuracy = 1e-3;
    option.tolerance = 1e-3;
    option.iterators = 10;
    option.debug = 1;
    option.kernel = 'linear';
end
if ~isfield(option, 'C')
    option.C = 1;
end
if ~isfield(option, 'kernel')
    option.kernel = 'linear';
end
if ~isfield(option, 'iterators')
    option.iterator = 100;
end
if ~isfield(option, 'accuracy')
    option.accuracy = 1e-3;
end
if ~isfield(option, 'tolerance')
    option.tolerance = 1e-3;
end
if ~isfield(option, 'debug')
    option.debug = 1;
end
if strcmp(lower(option.kernel), 'gaussian') && ~isfield(option, 'gaussian_sigma')
    option.gaussian_sigma = 4;%2*sigma^2
end

[m, n] = size(x);
w = zeros(n, 1);
alpha = zeros(m, 1);
alphay = [];
sv = [];
index = [];
b = 0;

numChanged = 0;
examineAll = 1;
while numChanged > 0 || examineAll
    numChanged = 0;
    if examineAll
        for i = 1:m
            [flag, w, alpha, b] = examineExample(x, y, w, alpha, b, i, option);
            numChanged = numChanged + flag;
        end
    else
        idx = find(0 ~= alpha & option.C ~= alpha);
        for i = 1:length(idx)
            [flag, w, alpha, b] = examineExample(x, y, w, alpha, b, idx(i), option);
            numChanged = numChanged + flag;
        end
    end
    if examineAll == 1
        examineAll = 0;
    elseif numChanged == 0
        examineAll = 1;
    else
        index = find(alpha > 0);
        alphay = alpha(index).*y(index);
        sv = x(index, :);
    end
    
    
    
end

function [flag, w, alpha, b] = examineExample(x, y, w, alpha, b, i1, option)

y1 = y(i1);
alpha1 = alpha(i1);
E1 = gFunc(x, y, alpha, w, b, i1, option) - y1;%g(i1) - y1;
kkt = y1*E1;
flag = 0;
if(kkt > option.tolerance && alpha1 > 0) || (kkt < -option.tolerance && alpha1 < option.C)
    [mflag, w, alpha, b] = findMaxNonbound(x, y, w, alpha, b, i1, option);
    if(mflag)
        flag = 1;
        return;
    end
    [mflag, w, alpha, b] = findRandomNonbound(x, y, w, alpha, b, i1, option);
    if(mflag)
        flag = 1;
        return;
    end
    
end


function [flag, w, alpha, b] = updateStep(x, y, w, alpha, b, i1, i2, option)

flag = 0;
if i1 == i2
    return ;
end

C = option.C;
accuracy = option.accuracy;
tolerance = option.tolerance;

alpha1 = alpha(i1);
alpha2 = alpha(i2);
alpha1new = -1.0;
alpha2new = -1.0;
alpha2newclipped = -1.0;
y1 = y(i1);
y2 = y(i2);
s = y1 * y2;

k11 = kernelFunc(x(i1, :), x(i1, :), option);%K(i1, i1);
k22 = kernelFunc(x(i2, :), x(i2, :), option);%K(i2, i2);
k12 = kernelFunc(x(i1, :), x(i2, :), option);%K(i1, i2);
eta = k11 + k22 - 2*k12;
E1 = gFunc(x, y, alpha, w, b, i1, option) - y1;%g(i1) - y1;
E2 = gFunc(x, y, alpha, w, b, i2, option) - y2;%g(i2) - y2;

L = 0.0;
H = 0.0;
if s == -1
    gamma = alpha2 - alpha1;
    if gamma > 0
        L = gamma;
        H = C;
    else
        L = 0;
        H = C + gamma;
    end
elseif s == 1
    gamma = alpha2 + alpha1;
    if gamma - C > 0
        L = gamma - C;
        H = C;
    else
        L = 0;
        H = gamma;
    end
end
if H == L
    flag = 0;
    return ;
end

if -eta < 0
    alpha2new = alpha2 + y2*(E1 - E2)/eta;

    if alpha2new < L
        alpha2newclipped = L;
    elseif alpha2new > H
        alpha2newclipped = H;
    else
        alpha2newclipped = alpha2new;
    end
else
    %
    w1 = updateW(y,alpha, b, alpha1 + s * (alpha2 - L),L,i1,i2,E1,E2, k11, k22, k12);
    w2 = updateW(y,alpha, b, alpha1 + s * (alpha2 - H),H,i1,i2,E1,E2, k11, k22, k12);
    if w1 - w2 > accuracy
        alpha2newclipped = L;
    elseif w2 - w1 > accuracy
        alpha2newclipped = H;
    else
        alpha2newclipped = alpha2;
    end
end

if abs(alpha2newclipped - alpha2) < accuracy * (alpha2newclipped + alpha2 + accuracy)
    flag = 0;
    return ;
end

alpha1new = alpha1 + s*(alpha2 - alpha2newclipped);
if alpha1new < 0
    alpha2newclipped = alpha2newclipped + s*alpha1new;
    alpha1new = 0;
elseif alpha1new > C
    alpha2newclipped = alpha2newclipped + s*(alpha1new - C);
    alpha1new = C;
end

if alpha1new > 0 && alpha1new < C
    b = b + (alpha1-alpha1new) * y1 * k11 + (alpha2 - alpha2newclipped) * y2 *k12 - E1;
elseif alpha2newclipped > 0 && alpha2newclipped < C
    b = b + (alpha1-alpha1new) * y1 * k12 + (alpha2 - alpha2newclipped) * y2 *k22 - E2;
else
    b1 = (alpha1-alpha1new) * y1 * k11 + (alpha2 - alpha2newclipped) * y2 *k12 - E1 + b;
    b2 = (alpha1-alpha1new) * y1 * k12 + (alpha2 - alpha2newclipped) * y2 *k22 - E2 + b;
    b = (b1 + b2)/2;
end

w = w + (alpha1new - alpha1) * y1 * x(i1, :)' + (alpha2newclipped - alpha2) * y2 * x(i2, :)';

alpha(i1) = alpha1new;
alpha(i2) = alpha2newclipped;
        
flag = 1;

function w1 = updateW(y, alpha, b, alpha1new,alpha2newclipped,i1,i2,E1,E2, k11, k22, k12)

alpha1 = alpha(i1);
alpha2 = alpha(i2);
y1 = y(i1);
y2 = y(i2);
s = y1 * y2;
        
w1 = alpha1new * (y1 * (b - E1) + alpha1 * k11 + s * alpha2 * k12);
w1 = w1 + alpha2newclipped * (y2 * (b - E2) + alpha2 * k22 + s * alpha1 * k12);
w1 = w1 - k11 * alpha1new * alpha1new/2 - k22 * alpha2newclipped * alpha2newclipped/2 - s * k12 * alpha1new * alpha2newclipped;

function g = gFunc(x, y, alpha, w, b, idx, option)
[m, n] = size(x);
mi = length(idx);
switch lower(option.kernel)
    case 'linear'
        % Linear kernel
        if mi > 1
            g = sum(repmat(w', mi, 1).*x(idx, :), 2) + b;
        else
            g = x(idx, :)*w + b;
        end
        %{
        ai = find(alpha>0);
        if length(ai) < 0
            g = 0;
            return;
        end
        g = sum(x(idx, :)*(repmat(alpha(ai).*y(ai), 1, n).*x(ai, :))', 2) +b;
        %}
        %g = sum(x(idx, :)*(repmat(alpha.*y, 1, n).*x)', 2) +b;
    case 'gaussian'
        % Gaussian kernel
        if mi > 1
            g = [];
            for i = 1:mi
                gi = sum(kernelFunc(repmat(x(idx(i), :), m, 1), x, option).*alpha.*y) + b;
                g = [g; gi];
            end
        else
            g = sum(kernelFunc(repmat(x(idx, :), m, 1), x, option).*alpha.*y) + b;
        end
        
    otherwise
        % Linear kernel
        if mi > 1
            g = sum(repmat(w', mi, 1).*x(idx, :), 2) + b;
        else
            g = x(idx, :)*w + b;
        end
end


function k = kernelFunc(xi, zi, option)

switch lower(option.kernel)
    case 'linear'
        % Linear kernel
        k = xi*zi';
    case 'gaussian'
        % Gaussian kernel
        k = exp(-diag((xi - zi)*(xi - zi)')/option.gaussian_sigma);
    otherwise
        % Linear kernel
        k = xi*zi';
end


function r = randomI(r_min, r_max, i)

while(true)
    r = randi([r_min, r_max], 1);
    if r ~= i
        break;
    end
end

function [flag, w, alpha, b] = findMaxNonbound(x, y, w, alpha, b, i1, option)

i2 = -1;
E2 = -inf;
flag = 0;

idx = find(alpha>0 & alpha<option.C);
if length(idx) < 1
    return;
end

E1 = gFunc(x, y, alpha, w, b, i1, option) - y(i1);%g(i1) - y(i1);
E2 = gFunc(x, y, alpha, w, b, idx, option) - y(idx);%g(idx) - y(idx);
[ev ei] = max(abs(E1 - E2));
flag = 1;
i2 = ei(end);
E2 = ev(end);
[mflag, w, alpha, b] = updateStep(x, y, w, alpha, b, i1, i2, option);
flag = mflag;

function [flag, w, alpha, b] = findRandomNonbound(x, y, w, alpha, b, i1, option)
m = length(y);
i2 = randomI(1, m, i1);
[mflag, w, alpha, b] = updateStep(x, y, w, alpha, b, i1, i2, option);
flag = mflag;