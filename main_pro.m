clear all;
close all;

% seed = 1;
% rand('state', seed);
% rand('state', seed);

%//////////////////////////////////////////////
%////////// Simulation settings ///////////////
%//////////////////////////////////////////////

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
beta_type = 'non_unbal';
if strcmpi(beta_type, 'non_bal')||strcmpi(beta_type, 'non_unbal')||strcmpi(beta_type, 'same_bal')||strcmpi(beta_type, 'same_unbal')
%     sigma = sqrt(1); 
    sigma = sqrt(3);
    sigma_err_matrix = [];
else
    sigma = [];
    err_rho = 0.3; 
%     err_rho = 0.8;
    sigma_err_matrix = zeros(q, q);
    for i = 1:q
        for j = 1:q
            sigma_err_matrix(i, j) = err_rho^abs(i-j);
        end    
    end
    sigma_err_matrix = 0.5^2 * sigma_err_matrix;
end


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


input_para = struct('pinp', p, 'ninp', n, 'qinp', q, 'kinp', k, 'sigma_matrix_inp', sigma_matrix, 'sigma_err_matrix_inp', sigma_err_matrix,...
                    'sigma_inp', sigma, 'R_inp', R, 'lam1_inp', lam1, 'lam2_inp', lam2, 'bias_inp', biasflag);


%//////////////////////////////////////////////
%//////////// Start All /////////// 
%//////////////////////////////////////////////
rep = 100;
parpool('local',3)

tic

results_sparse_rep = cell(rep, 1);
results_group_rep = cell(rep, 1);
beta_mse_rep = zeros(4, rep);
prediction_mse_rep = zeros(4, rep);
groupnum_est_rep = zeros(4, rep);

breakpoint_total_rep = cell(rep, 1);

parfor rr = 1:rep
    
    fprintf('Now the %d th replicate.\n', rr)
   
    [W_true, breakpoint_total, sparse_estimate, group_estimate, results_sparse, results_group, beta_mse, prediction_mse, groupnum_est] = clust_comparsion(input_para, beta_type); 
    results_sparse_rep{rr} = results_sparse;
    results_group_rep{rr} = results_group;
    beta_mse_rep(:,rr) = beta_mse;
    prediction_mse_rep(:,rr) = prediction_mse;
    groupnum_est_rep(:,rr) = groupnum_est;
    breakpoint_total_rep{rr} = breakpoint_total;
end
toc

delete(gcp('nocreate'))

save non_unbal_rho03_d1.mat 

