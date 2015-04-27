function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

h = sigmoid(X * theta);
J = sum(-y .* log(h) - (1 - y) .* log(1 - h)) ./ m ...
    + lambda * (theta(2:end)'*theta(2:end)) ./ (2*m);

grad = zeros(size(theta));
grad(1) = X(:,1)' * (h-y) ./ m;
grad(2:end) =  (X(:,2:end)' * (h-y) ./ m) + lambda .* theta(2:end) ./ m;
end
