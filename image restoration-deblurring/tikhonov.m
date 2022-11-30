function [xa] = tikhonov(lambda, A, Dn)

% SEE CODE BELOW THAT MUST BE RAN BEFORE THIS FUNCTION TO GET A & Dn

% Input :
%   lambda = regularization parameter
%   A = blurring matrix created
%   Dn = noisy data matrix

% Output:
%   xa = the approximate solution of x, true value of the unknown vector

% Storage for vectors:
    xa = zeros(220,1);

% Other variables: 
%   dn = individual column of noisy data matrix
%   x0 = initial guess
%   I = identity matrix (m*m)
%   AA = A matrix in CG method given context of problem
%   b = b vector in CG method given context of problem
%   x = current vector approximation in CG iteration
%   normrk = norm of residual vector 
%   beta = used in CG iteration as "steplength" on descent direction pk

% Initilization for CG method
I = eye(220,220);
AA = A.'*A + (lambda^2)*I;

% Use Tikhonov regularization with CG method to deblur noisy 
%   data/matrix one column at a time
for i= 1:520

    dn = Dn(:,i);
    b = A.'*dn;
 
    x = zeros(220,1);
    rk = AA*x - b;
    normrk = norm(rk);
    pk = (-1)*rk;
   

    while (normrk >= 10^(-3))
        alphak = (rk.'*rk)/(pk.'*AA*pk);
        x = x + alphak*pk;

        rkp1 = rk + alphak*AA*pk;
        beta = (rkp1.'*rkp1)/(rk.'*rk);
        pk = (-1)*rkp1 + beta*pk;
        
        rk = rkp1;
        normrk = norm(rk);

    end
    
    xa(:,i) = x;

end

imagesc(xa);
colormap(gray)
hold on

end


%%%%%%%% RUN THE FOLLOWING BEFORE RUNNING THE TIKHONOV FUNCTION  %%%%%%%%%%
% % Create/Define blurring matrix A
% L = 0.45;
% N = 220;
% a = 1-2*L;
% b = L;
% c = L;
% B = diag(a*ones(1,N)) + diag(b*ones(1,N-1),1) + diag(c*ones(1,N-1),-1);
% A = B^(25);
% 
% % Load noisy data matrix Dn
% load dollarblur.m
% Dn = dollarblur;
