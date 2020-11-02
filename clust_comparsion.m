function[W_true, breakpoint_total, sparse_estimate, group_estimate, results_sparse, results_group, beta_mse, prediction_mse, groupnum_est] = clust_comparsion(input_para, beta_type)

p = input_para.pinp;
n = input_para.ninp;
q = input_para.qinp;
k = input_para.kinp;
sigma_matrix = input_para.sigma_matrix_inp;
sigma_err_matrix = input_para.sigma_err_matrix_inp;
sigma = input_para.sigma_inp;
R = input_para.R_inp;
lam1 = input_para.lam1_inp;
lam2 = input_para.lam2_inp;
biasflag = input_para.bias_inp;

%//////////////////////////////////////////////
%////////////////// Input ///////////////////// 
%//////////////////////////////////////////////

X = zscore(mvnrnd(zeros(1, p), sigma_matrix, n));
X_sparse = zeros(n, n*p);
for i = 1:n
    X_sparse(i,((i-1)*p+1):(i*p)) = X(i,:);
end
X_sparse = sparse(X_sparse);

if strcmpi(beta_type, 'non_bal')||strcmpi(beta_type, 'non_unbal')||strcmpi(beta_type, 'same_bal')||strcmpi(beta_type, 'same_unbal')
    % 不分块情况
    W0 = generate_solution(p, q, k, n, beta_type);
else
    % 分块
    W0 = generate_solution(p, q, k, n, beta_type);
    W0_reshape = cellfun(@(x) x(:)', W0, 'UniformOutput', false);
    Y_total = X_sparse * cat(1, W0_reshape{:})' + mvnrnd(zeros(1, q), sigma_err_matrix, n);
end

%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

W_est_multilasso = cell(q, 1);
X1_cell = cell(q, 1);
row_ind_cell = cell(q, 1);

% tic
for res_dim = 1:q

    %% Localized Lasso
    %Training
    [W, ~] = LocalizedLasso(X_reshape, Y, R, lam1, lam2, biasflag);

    W = W.*(abs(W) > 0.01);
    [row, ~] = find(W);
    row = unique(row);
    X1 = X_reshape(row, :);
    [W1, ~] = LocalizedLasso(X1, Y, R, lam1, 2 * lam2, biasflag);
%     W1 = W1.*(abs(W1) > 0.01);
    
%     row_ind_cell{res_dim} = row;
%     X1_cell{res_dim} = X1;
    W_est_multilasso{res_dim} = W1;

    %% Stop
    if(res_dim == q)
%        fprintf ('The p*q beta estimate has done.\n')
    end
    
end
% toc

group_estimate = W_est_multilasso;

%///////////////////////////////////////////////////////////////////////////////////

% tic
%% 稀疏估计
% thr1 = 0.02;
% thr2 = 0.18;
thr1 = 0.05;
thr2 = 0.16;
%% Locallasso
W_loclasso_est_mat = cat(1,W_est_multilasso{:});
dist_loclasso = diag(squareform(pdist(W_loclasso_est_mat')), 1);
[dist_sort_loclasso, I_loclasso] = sort(dist_loclasso, 'descend');
breakpoints_loclasso = I_loclasso(dist_sort_loclasso > thr1); 

breakpoints_loclasso_in = breakpoints_loclasso(1);
i1=2;
while i1 <= length(breakpoints_loclasso)
    breakpoints_loclasso_in = [breakpoints_loclasso_in;breakpoints_loclasso(i1)];
    DB_loclasso = squareform(pdist(breakpoints_loclasso_in));
    in_tril_loclasso = tril(true(size(DB_loclasso,1),size(DB_loclasso,1)),-1);
    DB_loclasso_vec = DB_loclasso(in_tril_loclasso);
    if any(DB_loclasso_vec < thr2 * n)
        breakpoints_loclasso_in = breakpoints_loclasso_in(1:end-1);
        i1=i1+1;
    else
        i1=i1+1;
    end
end
breakpoints_loclasso_tmp = sort(breakpoints_loclasso_in);
if(breakpoints_loclasso_tmp(1)-1 > thr2 * n && n-breakpoints_loclasso_tmp(end) > thr2 * n)
    breakpoint_total = [1;breakpoints_loclasso_tmp;n+1];
elseif(breakpoints_loclasso_tmp(1)-1 > thr2 * n && n-breakpoints_loclasso_tmp(end) < thr2 * n)
    breakpoint_total = [1;breakpoints_loclasso_tmp(1:end-1);n+1];
elseif(breakpoints_loclasso_tmp(1)-1 < thr2 * n && n-breakpoints_loclasso_tmp(end) > thr2 * n)
    breakpoint_total = [1;breakpoints_loclasso_tmp(2:end);n+1];
else
    breakpoint_total = [1;breakpoints_loclasso_tmp(2:end-1);n+1];
end

W_loclasso_est_sparse = zeros(p*q, n);
W_loclasso_est_sparse_mse = zeros(p*n, q);
for i2 = 1:q
%     fprintf('Now the %d th response.\n', i2)
    Y_loclasso_est = Respon_mat(:,i2);
    Wf_loclasso = zeros(p, size(breakpoint_total{1}, 1) - 1);
    for i3 = 1:size(breakpoint_total{1}, 1) - 1
        [B_loclasso, info_loclasso] = lasso(X(breakpoint_total{1}(i3):breakpoint_total{1}(i3 + 1) - 1, :), Y_loclasso_est(breakpoint_total{1}(i3):breakpoint_total{1}(i3 + 1) - 1), 'Alpha', 1, 'CV', 5);
        Wf_loclasso(:, i3) = B_loclasso(:, info_loclasso.Index1SE);
    end
    Wf_loclasso_mat = zeros(p, n);
    for i4 = 1:size(breakpoint_total{1}, 1) - 1
        Wf_loclasso_mat(:,breakpoint_total{1}(i4):breakpoint_total{1}(i4 + 1) - 1) = repmat(Wf_loclasso(:, i4), 1, breakpoint_total{1}(i4 + 1) - breakpoint_total{1}(i4));
    end
    W_loclasso_est_sparse(((i2-1)*p+1):(i2*p), :) = Wf_loclasso_mat;
    W_loclasso_est_sparse_mse(:, i2) = Wf_loclasso_mat(:);
end
sparse_estimate = W_loclasso_est_sparse;
sparse_estimate_mse = W_loclasso_est_sparse_mse;


%% get results
% 稀疏结果
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


% 分组结果
group_nums = breakpoint_total(2:length(breakpoint_total)) - breakpoint_total(1:length(breakpoint_total)-1);
group_nums_mat = cell(length(group_nums), 1);
for j18 = 1:length(group_nums)
    group_nums_mat{j18} = ones(group_nums(j18), group_nums(j18));
end
predict_matrix_ad = blkdiag(group_nums_mat{:});
if strcmpi(beta_type, 'non_bal')||strcmpi(beta_type, 'same_bal')||strcmpi(beta_type, 'diag_non_bal')||strcmpi(beta_type, 'diag_same_bal')
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


% 估计和预测MSE
est_square_err = (sparse_estimate - W_true).^2;
beta_mse = sum(est_square_err(:))/(n*p*q);

est_square_err_y = (X_sparse * sparse_estimate_mse - Respon_mat).^2;
prediction_mse = sum(est_square_err_y(:))/(n*q);


% 组数
groupnum_est = length(breakpoint_total) - 1;


