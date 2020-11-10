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

% Breakpoints threshold
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

% /////////////////// Running MSF /////////////////////
[breakpoint_total, sparse_estimate, group_estimate] = MSF(X, Y_total, R, lam1, lam2, biasflag, thr1, thr2); 

% ////////////////// Get results //////////////////////
% Sparse Estimate
W_true = cat(1,W0{:});
TP = length(find(((abs(sparse_estimate) > 0)==1) & ((W_true~=0)==1)));
FP = length(find(((abs(sparse_estimate) > 0)==1) & ((W_true~=0)==0)));  
TN = length(find(((abs(sparse_estimate) > 0)==0) & ((W_true~=0)==0))); 
FN = length(find(((abs(sparse_estimate) > 0)==0) & ((W_true~=0)==1)));  
Accuracy = (TP+TN)/(TP+FP+TN+FN);
Precision = TP/(TP+FP);
Recall = TP/(TP+FN);
FPR = FP/(FP+TN);
Fscore = 2*Precision*Recall/(Precision+Recall);
results_sparse = [Accuracy, Precision, Recall, FPR, Fscore];

% Group Estimate
group_nums = breakpoint_total(2:length(breakpoint_total)) - breakpoint_total(1:length(breakpoint_total)-1);
group_nums_mat = cell(length(group_nums), 1);
for j18 = 1:length(group_nums)
    group_nums_mat{j18} = ones(group_nums(j18), group_nums(j18));
end
predict_matrix_ad = blkdiag(group_nums_mat{:});
if strcmpi(beta_type, 'diag_non_bal')||strcmpi(beta_type, 'diag_same_bal')
    true_matrix = blkdiag(ones(n/3,n/3),ones(n/3,n/3),ones(n/3,n/3));
else
    true_matrix = blkdiag(ones(60,60),ones(80,80),ones(100,100));
end
TP = length(find((predict_matrix_ad==1)&(true_matrix==1)));
FP = length(find((predict_matrix_ad==1)&(true_matrix==0)));  
TN = length(find((predict_matrix_ad==0)&(true_matrix==0))); 
FN = length(find((predict_matrix_ad==0)&(true_matrix==1)));
Accuracy = (TP+TN)/(TP+FP+TN+FN);
Precision = TP/(TP+FP);
Recall = TP/(TP+FN);
FPR = FP/(FP+TN);
Fscore = 2*Precision*Recall/(Precision+Recall);
results_group = [Accuracy, Precision, Recall, FPR, Fscore];


% Group Num
groupnum_est = length(breakpoint_total) - 1;
