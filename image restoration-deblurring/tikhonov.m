function [xa] = tikhonov(lambda, A, Da)

%input 
% lambda = regularization parameter
% A = blurring matrix created
% Da = noisy data matrix

%output
% xa = the approximate solution of x, true value of the unknown vector



end

% Viewiwing a matrix as an image
D = load('dollarblur.m');
D = mat2gray(D)
imshow(D)
