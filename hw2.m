function [sdsol, sk, sfuneval, nsol, nk, nfuneval] = hw2(tol, x0, a0, row, c, A)

%input 
% tol = tolerance level for solution
% x0 = initial guess point
% a0 = initial alpha step used in backtracing line search algorithm
% row = constant used to decrease alpha in backtracing line search
% c = constant used in backtracing line search
% A = parameter for Rosenbrock function

%output
% sdsol = the approximate solution of the steepest descent method
% sk = number of iterations ran in steepest descent method
% sfuneval = number of cost function iterations when using steepest descent
% nsol, nk, nfuneval - same but for Newton's method
% SEE ALSO: Plotting (below) for information on plot outputs

%Other Variables not involved in input or output (but could be if desired)
% xs = a matrix of dimension (2, sk) that stores the optimization
%   path produced by the steepest descent method with iterations stored on columns,
%   including first guess point x0
% fs = value of cost function along optimization path produced by steepest
%   descent method, stores as an array (column) of values with iter rows
% normgrads = value of the norm of the gradient produced by the streepest
%   descent method, stores an array (column) of values with sk rows
% xn, fn, normgradn - same but for Newton's method

%storage for vectors and matrices
xs = zeros(2,1);
fs = zeros(1,1);
normgrads = zeros(1,1);

xn = xs;
fn = fs;
normgradn = normgrads;

%initialization of vectors and matrices
xs(:,1) = x0;
fs(1) = fcost(xs(:,1), A);
normgrads(1) = norm(fgrad(xs(:,1), A));

xn(:,1) = x0;
fn(1) = fs(1);
normgradn(1) = norm(fgrad(xn(:,1), A));


%initialization for steepest descent iteration 
%   sk = iteration counter for steepest descent
%   sfuneval = counter for number of cost function evaluations (steepest
%              descent)
sk = 1;
sfuneval = 1;

%generate iterations with varying alpha length decided by backtracing line
%   search algorithm until norm of the gradient of the function at current iteration is
%   less than tol
    
    %steepest descent iteration
    while (normgrads(sk) >= tol)
        gs = fgrad(xs(:,sk), A);

        %initialize backtracing line search algorithm for alpha steplength
        a = a0;

        while fcost((xs(:,sk) - a*gs), A) > (fs(sk) - c*a*dot(gs.',gs))
            a = (row)*a;
            sfuneval = sfuneval + 1;
        end

        xs(:,sk+1) = xs(:,sk) - a*gs;
        fs(sk+1) = fcost(xs(:,sk+1),A);
        sfuneval = sfuneval + 1;
        normgrads(sk+1) = norm(fgrad(xs(:,sk+1), A));

        sk = sk + 1;
    end 

%initialization k for Newton iteration
%   nk = iteration counter for newton iteration
%   nfuneval = counter for number of cost function evaluations (newton 
%              iteration)
 nk = 1;
 nfuneval = 1;

    %Newton iteration 
    while (normgradn(nk) >= tol)
        gn = fgrad(xn(:,nk), A);
        H = fhess(xn(:,nk), A);

        %initialize backtracing line search algorithm for alpha steplength
        a = a0;
        hinvgn = H\gn;

        while fcost((xn(:,nk) - a*hinvgn), A) > (fn(nk) - c*a*dot((hinvgn.'),gn))
            a = (row)*a;
            nfuneval = nfuneval + 1;
        end
         

        xn(:,nk+1) = xn(:,nk) - a*(inv(H))*gn;
        fn(nk+1) = fcost(xn(:,nk+1), A);
        nfuneval = nfuneval + 1;
        normgradn(nk+1) = norm(fgrad(xn(:,nk+1), A));

        nk = nk + 1;
    end 

 sdsol = xs(:,sk);
 nsol = xn(:,nk);

%Plotting
% xiters = iteration count of steepest descent listed as a vector (i.e 1,2,3,...ns) used as x-axis
%      values when plotting 
% snormplot = plot showing the evolution of the norm of the gradient
%      during steepest descent optimization process
% scosfunplot - plot showing the evolution of the cost function during the
%      steepest descent optimization process
% soptpath - plot showing the optimization path produced by steepest
%      descent algorithm
% xitern, nnormplot, ncosfunplot, noptpath - same but for newton iteration

xiters = 1:1:sk;
xitern = 1:1:nk;

%snormplot
figure;
plot(xiters,normgrads, '-*')
title('Evolution of Norm of Gradient using Steepest Descent')
xlabel ('Steepest Descent Iteration Number')
ylabel('Norm of Gradient')

%nnormplot
figure;
plot(xitern,normgradn, '-*')
title('Evolution of Norm of Gradient using Newton')
xlabel ('Newton Iteration Number')
ylabel('Norm of Gradient')

%scosfunplot
figure;
plot(xiters,fs, '-*')
title('Evolution of Cost Function using Steepest Descent')
xlabel ('Steepest Descent Iteration Number')
ylabel('Cost Function Value')

%ncosfunplot
figure;
plot(xitern,fn, '-*')
title('Evolution of Cost Function using Newton')
xlabel ('Newton Iteration Number')
ylabel('Cost Function Value')

% Create base contour plot of Rosenbrock function
[X,Y]=meshgrid(-2:0.1:2, -2:0.1:4);
Z=(1-X).^2 + A.*(Y-X.^2).^2;

%soptpath
figure;
contour(X,Y,Z,25)
hold on
plot(xs(1,:), xs(2,:), '-*')
title('Steepest Descent Optimization Path')
hold off

%noptpath
figure;
contour(X,Y,Z,25)
hold on
plot(xn(1,:), xn(2,:), '-*')
title('Newton Iteration Optimization Path')
hold off


end


%OUTSIDE hw2() function

% Difine necesarry cost function (user defined)
function f = fcost(x,A)
    f = A*((x(2) - x(1)^2)^2) + (1 - x(1))^2;
end

%define gradient function (user defined)
function g = fgrad(x,A)

   g = zeros(2,1);

   g(1,1) = -4*A*(x(2) - x(1)^2)*x(1) - 2*(1-x(1));
   g(2,1) = 2*A*(x(2) - x(1)^2);
end

%define hessian function (user defined)
function H = fhess(x,A)

    H = zeros(2,2);

    H(1,1)= 12*A*x(1)^2 - 4*A*x(2) + 2;
    H(1,2)= -4*A*x(1);
    H(2,1)= -4*A*x(1);
    H(2,2)= 2*A;
   
end
