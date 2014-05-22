function [w cost] = tcsvmPegasos(x, y, options)
% Two Classes SVM Solver -- Pegasos: Primal Estimated sub-GrAdient SOlver for SVM
% http://eprints.pascal-network.org/archive/00004062/01/ShalevSiSr07.pdf

% x      -- input data, size = [m, n], m:samples number, n:feature dimension;
% y      -- labels data, size = [m, 1], values=[-1 1], m:samples number;
% options -- options struct
%        max_itr: max iterators
%        min_eps: min eps
%        C:       penalty factor
%        debug:   show debug message

% theta  -- parameters, size = [n+1, 1], n:elements nubmer;
% cost   -- cost

% author -- amadeuzou AT gmail
% date   -- 11/19/2013, Beijing, China

if nargin == 2
    options.C = 1;
    options.max_itr = 100;
    options.min_eps = 1e-3;
    options.debug = 1;
end
if ~isfield(options, 'C')
    options.C = 1;
end
if ~isfield(options, 'max_itr')
    options.max_itr = 100;
end
if ~isfield(options, 'min_eps')
    options.min_eps = 1e-3;
end
if ~isfield(options, 'debug')
    options.debug = 1;
end

[m, n] = size(x);
k = ceil(0.5*m);
lambda = options.C + eps;
wt = rand(n, 1);
wt = wt/(sqrt(lambda)*norm(wt));
bt = 0;
xb = [x, ones(m, 1)];
w = [wt; bt];
cost = 0;

J = [];
t = 1;
err = 0;
warning('off','comm:obsolete:randint');
while(1)
    
    it = randint(k, 1, [1, m]);
    xt = x(it, :);
    yt = y(it);
    eta = 1 / lambda / t;
    idx = (xt*wt + bt).*yt < 1;
    
    % update
    wtu = (1-eta*lambda)*wt + (eta/k)*sum(xt(idx,:).*repmat(yt(idx,:), 1, n), 1)';
    wt = min(1, 1/(sqrt(lambda)*norm(wtu)))*wtu;
    %bt = bt + eta/k * sum(yt(idx,:));
    bt = mean(y - x*wt);
    
    err = norm(w(1:n) - wt);
    w = [wt; bt];
    t = t + 1;
    cost = tcsvmCostFunc(xb, y, w, options.C);
    J = [J; cost];
    
    if(options.debug)
        disp(['itr = ', num2str(t), ', cost = ', num2str(cost), ', err = ', num2str(err)]);
    end
    if t >= options.max_itr || err <= options.min_eps || norm(wt)<=eps
        break;
    end
end






