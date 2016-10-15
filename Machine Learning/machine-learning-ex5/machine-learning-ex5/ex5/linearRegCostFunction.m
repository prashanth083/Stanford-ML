function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
% You should set J to the cost and grad to the gradient.
%
%calculate hypothesis
h = X*theta;
%compute the error difference between hypothesis and output y
error =  h-y;
%compute the error square
error_sqr = error.^2;
%compute the cost function
J = sum(error_sqr) / (2*m);
%compute reguralization parameter
theta(1) =0;
reg = (lambda / (2*m)) * sum(theta.^2);
%regularized cost function
J = J + reg;


% =========================================================================
%compute the unreguralized gradient
grad = (X' * error) / m;

%reg derivative
rd = (lambda/m) * theta;
%regularized gradient descent
grad = grad  + rd;

grad = grad(:);

end
