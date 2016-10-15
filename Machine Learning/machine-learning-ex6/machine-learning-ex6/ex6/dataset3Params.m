function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
c_vector = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vector = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
m = size(c_vector,1);
n = size(sigma_vector,1);
error_vector = zeros(m,n);
for i = 1:m
    C = c_vector(i);
    for j = 1:n
        sigma = sigma_vector(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
       error_vector(i,j)= mean(double(predictions ~= yval));
    end
end
fprintf('The error matrix is:\n');
fprintf([repmat('%f\t', 1, size(error_vector, 2)) '\n'], error_vector');
[x,y]=find(error_vector==min(min(error_vector))) %index minimum of the matrix error_vector 
C = c_vector(x);
sigma = sigma_vector(y);




% =========================================================================

end
