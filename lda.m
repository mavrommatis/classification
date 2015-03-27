
function [Xb,x_all,y_hat] = lda(x)
% [Xb,x_all,y_hat] = lda(x)
% 
% Perform Linear Discriminant Analysis for classifying a single predictor
% x, based on observations from a number of known classes, y(x).
%  y values are assumed to be coded as 1, 2, ..., K, where K is the number
%  of classes.
%
% Input:
%  x = K-long cell array containing values of predictor variable from each
%      class, where K is the number of classes
%
% Output:
%  Xb    = decision boundary(ies)
%  x_all = vector of values of independent variable, ordered by observed
%          class and order in input x
%  y_hat = predicted class for each observation

K = length(x);  % number of classes

[mu_hat,var_hat,prior,N] = deal(nan(K,1)); % initialization

for k = 1:K
    N(k) = length(x{k});  % number of observations per class
end
N_all = sum(N);         % number of all observations

% Compute sample means, variance, and priors
for k = 1:K
    mu_hat(k) = mean(x{k});
    var_hat(k) = sum(x{k} - mu_hat(k))^2;
    prior(k) = N(k)/N_all;
end
var_hat = sum(var_hat)/(N_all-K);

% Compute predicted classes
x_all = (cat(2,x{:}));
delta = nan(K,length(x_all));
for i = 1:length(x_all)
    delta(:,i) = x_all(i)*mu_hat/var_hat - (mu_hat.^2)/(2*var_hat)  + log(prior);
end
[~,y_hat] = max(delta);

% Compute decision boundaries
[Y_hat,IS] = sort(y_hat);
X_all = x_all(IS);
Ib = find(diff(Y_hat)~=0);
Xb = nan(K-1,1);
for k = 1:K-1
    Xb1 = X_all(Ib(k));
    Xb2 = X_all(Ib(k)+1);
    Xb(k) = mean([Xb1 Xb2]);
end

