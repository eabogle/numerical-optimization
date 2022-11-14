function [sol, k, funeval, restart] = hw3(tol, x0, a0, row, c, A)

%input 
% tol = tolerance level for solution
% x0 = initial guess point
% a0 = initial alpha step used in backtracing line search algorithm
% row = constant used to decrease alpha in backtracing line search
% c = constant used in backtracing line search
% A = parameter for Rosenbrock function

%output
% sol = the approximate solution of the Fletcher-Reeves Nonlinear CG method
% k = number of iterations ran in Fletcher-Reeves Nonlinear CG method
% funeval = number of cost function iterations when using Fletcher-Reeves Nonlinear CG method
% restart = number of restarts required when using Fletcher-Reeves Nonlinear CG method
% SEE ALSO: Plotting (below) for information on plot outputs

%Other Variables not involved in input or output (but could be if desired)
% xvect = a matrix of dimension (2, k) that stores the optimization
%   path produced by the Fletcher-Reeves Nonlinear CG method with iterations stored on columns,
%   including first guess point x0
% funvect = value of cost function along optimization path produced by Fletcher-Reeves Nonlinear CG method, 
%   stores as an array (column) of values with iter rows
% normgradvect = value of the norm of the gradient produced by the Fletcher-Reeves Nonlinear CG method,
%   stores an array (column) of values with k rows


%storage for vectors and matrices
xvect = zeros(2,1);
funvect = zeros(1,1);
normgradvect = zeros(1,1);


%initialization for Fletcher-Reeves Nonlinear CG iteration
%   k = iteration counter for Fletcher-Reeves Nonlinear CG method
%   funeval = counter for number of cost function evaluations (Fletcher-Reeves Nonlinear CG method)
%   restart = counter for number of restarts required (Fletcher-Reeves Nonlinear CG method)
%   x = current iteration point
%   grad = gradient at current iteration point
%   normgrad = norm of the gradient at current iteration point
%   pk = descent direction at current iteration point
%   funxk = function value at current iteration point
k = 0;
funeval = 1; 
restart = 0;

x = [x0(1);x0(2)];
grad = fgrad(x,A);
normgrad = norm(grad);
pk = (-1)*grad;
funxk = fcost(x, A);


%generate iterations with varying alpha length decided by backtracing line
%   search algorithm until norm of the gradient of the function at current iteration is
%   less than tol
    
    %nonnlinear CG iteration (Flectcher-Reeves)
    while (normgrad >= tol)

        %initialize backtracing line search algorithm for alpha steplength
        a = a0;
            
        while ((fcost((x + a*pk), A)) > (funxk + c*a*dot(pk,grad)))
            a = (row)*a;
            funeval = funeval + 1;
        end
   
        x = x + a*pk; 
       
        %gradkp1 = gradient at new iteration point
        gradkp1 = fgrad(x, A);

        beta = dot(gradkp1, gradkp1)/dot(grad, grad);
        
        pk = (-1)*gradkp1 + beta*pk;

        grad = gradkp1;
        normgrad = norm(grad);

        if (dot(pk, grad) >= 0)
            pk = (-1)*grad;
            fprintf('RESTART AT ITERATION:\n')
            disp(k)
            restart = restart + 1;
        end

        funxk = fcost(x,A);
        funeval = funeval + 1;

        k = k + 1;
        
        xvect(:,k) = x;
        normgradvect(k) = normgrad;
        funvect(k) = funxk;
       
    end 
    
 sol = x;

%Plotting
% xiters = iteration count of steepest descent listed as a vector (i.e 1,2,3,...ns) used as x-axis
%      values when plotting 
% normplot = plot showing the evolution of the norm of the gradient
%      during Fletcher-Reeves Nonlinear CG iteration
% cosfunplot - plot showing the evolution of the cost function during the
%      Fletcher-Reeves Nonlinear CG iteration
% optpath - plot showing the optimization path using Fletcher-Reeves
%      Nonlinear CG iteration

 xiters = 1:1:k;

%normplot
figure;
loglog(xiters,normgradvect, '-*')
title('Evolution of Norm of Gradient using Fletcher-Reeves Nonlinear CG')
xlabel ('Fletcher-Reeves Nonlinear CG Iterations')
ylabel('Norm of Gradient')

%cosfunplot
figure;
loglog(xiters,funvect, '-*')
title('Evolution of Cost Function using Fletcher-Reeves Nonlinear CG')
xlabel ('Fletcher-Reeves Nonlinear CG Iterations')
ylabel('Cost Function Value')


% Create base contour plot of Rosenbrock function
[X,Y]=meshgrid(-2:0.1:2, -2:0.1:4);
Z=(1-X).^2 + A.*(Y-X.^2).^2;

%optpath
figure;
contour(X,Y,Z,25)
hold on
plot(1,1,'*g')
hold on
plot(-1.2,1, "*r")
hold on
plot(xvect(1,:), xvect(2,:), '-*b')
title('Fletcher-Reeves Nonlinear CG Optimization Path')
hold off

end


%OUTSIDE hw3() function

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
