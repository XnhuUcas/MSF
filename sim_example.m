clear all;
close all;

% seed = 1;
% rand('state', seed);
% rand('state', seed);

%  ////////////  Simulation settings //////////////////////

p = 150;
q = 150;
n = 240;
k = 6; % number of nonzero features % 

% Correlation
rho = 0.3;
sigma_matrix = zeros(p, p);
for i = 1:p
    for j = 1:p
        sigma_matrix(i, j) = rho^abs(i-j);
    end    
end

% Error variance
beta_type = 'diag_non_bal';
sigma = [];
err_rho = 0.3; 
sigma_err_matrix = zeros(q, q);
for i = 1:q
    for j = 1:q
        sigma_err_matrix(i, j) = err_rho^abs(i-j);
    end    
end
sigma_err_matrix = 0.5^2 * sigma_err_matrix;


%Network regularization
lam1 = 140;

%Exclusive regularization (Sample-wise sparsity)
lam2 = 0.1;

%0: no bias, 1: add bias (i.e., y_i = w_i^t x_i + b_i)
%Bias term is regularized only in network regularization
biasflag = 0;

%Graph information
L = 1;
E = ones(n, 2 * L + 1);
R = spdiags(E, -L:L, n, n);
R = R - diag(diag(R));

# Breakpoints threshold
thr1 = 0.05;
thr2 = 0.16; 

%  ////////////////// Generating Dataset ///////////////////////////
X = zscore(mvnrnd(zeros(1, p), sigma_matrix, n));
X_sparse = zeros(n, n*p);
for i = 1:n
    X_sparse(i,((i-1)*p+1):(i*p)) = X(i,:);
end
X_sparse = sparse(X_sparse);

W0 = DataGeneration(p, q, k, n, beta_type);
W0_reshape = cellfun(@(x) x(:)', W0, 'UniformOutput', false);
Y_total = X_sparse * cat(1, W0_reshape{:})' + mvnrnd(zeros(1, q), sigma_err_matrix, n);
X_reshape = X'; %  X:p*n % 


# /////////////////// Running MSF /////////////////////
[breakpoint_total, sparse_estimate, group_estimate] = msf(X_reshape, Y_total, R, lam1, lam2, biasflag, thr1, thr2); 

