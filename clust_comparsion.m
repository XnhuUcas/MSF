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
Resi_mat = zeros(n, q);
Respon_mat = zeros(n, q);
W_est_ridge_fusion_clust = cell(q, 1);
group_estimate = cell(4,1);

% tic
for res_dim = 1:q
    
%     fprintf('Now the %d th response.\n', res_dim)
        
    % Response
    if strcmpi(beta_type, 'non_bal')||strcmpi(beta_type, 'non_unbal')||strcmpi(beta_type, 'same_bal')||strcmpi(beta_type, 'same_unbal')
%         Y1 = X(1:n/3,:)*W0{res_dim}(:,1) + normrnd(0,sigma);
%         Y2 = X(n/3+1:2*n/3,:)*W0{res_dim}(:,81) + normrnd(0,sigma);
%         Y3 = X(n/3*2+1:n,:)*W0{res_dim}(:,161) + normrnd(0,sigma);
%         Y = [Y1;Y2;Y3];
        Y =  X_sparse * W0{res_dim}(:) + normrnd(0, sigma, n, 1);
        X_reshape = X'; %  X:p*n % 
    else
        Y = Y_total(:, res_dim);
        X_reshape = X'; %  X:p*n % 
    end

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

    %% Residual clustering
    [B_las, info_las] = lasso(X, Y, 'Alpha', 1, 'CV', 5);
    Lasso_est = B_las(:, info_las.Index1SE);
    Resi_reg = Y - X * Lasso_est;
    Resi_mat(:,res_dim) = Resi_reg;

    %% Response clustering
    Respon_mat(:,res_dim) = Y;
    
    %% Ridge fusion 
    lam_rid = 0.001;
    D_tmp = zeros(n-1, n);
    for i0 = 1:n-1
        D_tmp(i0, i0) = 1;
        D_tmp(i0, i0+1) = -1;
    end
    A_tmp = kron(sparse(D_tmp), speye(p));
    Bet_ridge_fuse = ((X_sparse') * X_sparse + lam_rid * (A_tmp') * A_tmp) \ (X_sparse') * Y;
    Bet_ridge_fuse_mat = reshape(Bet_ridge_fuse, p, n);
    W_est_ridge_fusion_clust{res_dim} = Bet_ridge_fuse_mat;

    %% Stop
    if(res_dim == q)
%        fprintf ('The p*q beta estimate has done.\n')
    end
    
end
% toc

group_estimate{1} = W_est_multilasso;
group_estimate{2} = Resi_mat;
group_estimate{3} = Respon_mat;
group_estimate{4} = W_est_ridge_fusion_clust;

%///////////////////////////////////////////////////////////////////////////////////

breakpoint_total = cell(4,1);
sparse_estimate = cell(4,1);
sparse_estimate_mse = cell(4,1);

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
    breakpoint_total{1} = [1;breakpoints_loclasso_tmp;n+1];
elseif(breakpoints_loclasso_tmp(1)-1 > thr2 * n && n-breakpoints_loclasso_tmp(end) < thr2 * n)
    breakpoint_total{1} = [1;breakpoints_loclasso_tmp(1:end-1);n+1];
elseif(breakpoints_loclasso_tmp(1)-1 < thr2 * n && n-breakpoints_loclasso_tmp(end) > thr2 * n)
    breakpoint_total{1} = [1;breakpoints_loclasso_tmp(2:end);n+1];
else
    breakpoint_total{1} = [1;breakpoints_loclasso_tmp(2:end-1);n+1];
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
sparse_estimate{1} = W_loclasso_est_sparse;
sparse_estimate_mse{1} = W_loclasso_est_sparse_mse;

%% residual
dist_resi = diag(squareform(pdist(Resi_mat)), 1);
[dist_sort_resi, I_resi] = sort(dist_resi, 'descend');
breakpoints_resi = I_resi(dist_sort_resi > thr1); 

breakpoints_resi_in = breakpoints_resi(1);
i5=2;
while i5 <= length(breakpoints_resi)
    breakpoints_resi_in = [breakpoints_resi_in;breakpoints_resi(i5)];
    DB_resi = squareform(pdist(breakpoints_resi_in));
    in_tril_resi = tril(true(size(DB_resi,1),size(DB_resi,1)),-1);
    DB_resi_vec = DB_resi(in_tril_resi);
    if any(DB_resi_vec < thr2 * n)
        breakpoints_resi_in = breakpoints_resi_in(1:end-1);
        i5=i5+1;
    else
        i5=i5+1;
    end
end
breakpoints_resi_tmp = sort(breakpoints_resi_in);
if(breakpoints_resi_tmp(1)-1 > thr2 * n && n-breakpoints_resi_tmp(end) > thr2 * n)
    breakpoint_total{2} = [1;breakpoints_resi_tmp;n+1];
elseif(breakpoints_resi_tmp(1)-1 > thr2 * n && n-breakpoints_resi_tmp(end) < thr2 * n)
    breakpoint_total{2} = [1;breakpoints_resi_tmp(1:end-1);n+1];
elseif(breakpoints_resi_tmp(1)-1 < thr2 * n && n-breakpoints_resi_tmp(end) > thr2 * n)
    breakpoint_total{2} = [1;breakpoints_resi_tmp(2:end);n+1];
else
    breakpoint_total{2} = [1;breakpoints_resi_tmp(2:end-1);n+1];
end

W_resi_est_sparse = zeros(p*q, n);
W_resi_est_sparse_mse = zeros(p*n, q);
for i6 = 1:q
%     fprintf('Now the %d th response.\n', i6)
    Y_resi_est = Respon_mat(:,i6);
    Wf_resi = zeros(p, size(breakpoint_total{2}, 1) - 1);
    for i7 = 1:size(breakpoint_total{2}, 1) - 1
        [B_resi, info_resi] = lasso(X(breakpoint_total{2}(i7):breakpoint_total{2}(i7 + 1) - 1, :), Y_resi_est(breakpoint_total{2}(i7):breakpoint_total{2}(i7 + 1) - 1), 'Alpha', 1, 'CV', 5);
        Wf_resi(:, i7) = B_resi(:, info_resi.Index1SE);
    end
    Wf_resi_mat = zeros(p, n);
    for i8 = 1:size(breakpoint_total{2}, 1) - 1
        Wf_resi_mat(:,breakpoint_total{2}(i8):breakpoint_total{2}(i8 + 1) - 1) = repmat(Wf_resi(:, i8), 1, breakpoint_total{2}(i8 + 1) - breakpoint_total{2}(i8));
    end
    W_resi_est_sparse(((i6-1)*p+1):(i6*p), :) = Wf_resi_mat;
    W_resi_est_sparse_mse(:, i6) = Wf_resi_mat(:);
end
sparse_estimate{2} = W_resi_est_sparse;
sparse_estimate_mse{2} = W_resi_est_sparse_mse;

%% response
dist_respon = diag(squareform(pdist(Respon_mat)), 1);
[dist_sort_respon, I_respon] = sort(dist_respon, 'descend');
breakpoints_respon = I_respon(dist_sort_respon > thr1); 

breakpoints_respon_in = breakpoints_respon(1);
i9=2;
while i9 <= length(breakpoints_respon)
    breakpoints_respon_in = [breakpoints_respon_in;breakpoints_respon(i9)];
    DB_respon = squareform(pdist(breakpoints_respon_in));
    in_tril_respon = tril(true(size(DB_respon,1),size(DB_respon,1)),-1);
    DB_respon_vec = DB_respon(in_tril_respon);
    if any(DB_respon_vec < thr2 * n)
        breakpoints_respon_in = breakpoints_respon_in(1:end-1);
        i9=i9+1;
    else
        i9=i9+1;
    end
end
breakpoints_respon_tmp = sort(breakpoints_respon_in);
if(breakpoints_respon_tmp(1)-1 > thr2 * n && n-breakpoints_respon_tmp(end) > thr2 * n)
    breakpoint_total{3} = [1;breakpoints_respon_tmp;n+1];
elseif(breakpoints_respon_tmp(1)-1 > thr2 * n && n-breakpoints_respon_tmp(end) < thr2 * n)
    breakpoint_total{3} = [1;breakpoints_respon_tmp(1:end-1);n+1];
elseif(breakpoints_respon_tmp(1)-1 < thr2 * n && n-breakpoints_respon_tmp(end) > thr2 * n)
    breakpoint_total{3} = [1;breakpoints_respon_tmp(2:end);n+1];
else
    breakpoint_total{3} = [1;breakpoints_respon_tmp(2:end-1);n+1];
end

W_respon_est_sparse = zeros(p*q, n);
W_respon_est_sparse_mse = zeros(p*n, q);
for i10 = 1:q
%     fprintf('Now the %d th response.\n', i10)
    Y_respon_est = Respon_mat(:,i10);
    Wf_respon = zeros(p, size(breakpoint_total{3}, 1) - 1);
    for i11 = 1:size(breakpoint_total{3}, 1) - 1
        [B_respon, info_respon] = lasso(X(breakpoint_total{3}(i11):breakpoint_total{3}(i11 + 1) - 1, :), Y_respon_est(breakpoint_total{3}(i11):breakpoint_total{3}(i11 + 1) - 1), 'Alpha', 1, 'CV', 5);
        Wf_respon(:, i11) = B_respon(:, info_respon.Index1SE);
    end
    Wf_respon_mat = zeros(p, n);
    for i12 = 1:size(breakpoint_total{3}, 1) - 1
        Wf_respon_mat(:,breakpoint_total{3}(i12):breakpoint_total{3}(i12 + 1) - 1) = repmat(Wf_respon(:, i12), 1, breakpoint_total{3}(i12 + 1) - breakpoint_total{3}(i12));
    end
    W_respon_est_sparse(((i10-1)*p+1):(i10*p), :) = Wf_respon_mat;
    W_respon_est_sparse_mse(:, i10) = Wf_respon_mat(:);
end
sparse_estimate{3} = W_respon_est_sparse;
sparse_estimate_mse{3} = W_respon_est_sparse_mse;

%% ridge fusion
W_ridge_est_mat = cat(1,W_est_ridge_fusion_clust{:});
dist_ridge = diag(squareform(pdist(W_ridge_est_mat')), 1);
[dist_sort_ridge, I_ridge] = sort(dist_ridge, 'descend');
breakpoints_ridge = I_ridge(dist_sort_ridge > thr1); 

breakpoints_ridge_in = breakpoints_ridge(1);
i13=2;
while i13 <= length(breakpoints_ridge)
    breakpoints_ridge_in = [breakpoints_ridge_in;breakpoints_ridge(i13)];
    DB_ridge = squareform(pdist(breakpoints_ridge_in));
    in_tril_ridge = tril(true(size(DB_ridge,1),size(DB_ridge,1)),-1);
    DB_ridge_vec = DB_ridge(in_tril_ridge);
    if any(DB_ridge_vec < thr2 * n)
        breakpoints_ridge_in = breakpoints_ridge_in(1:end-1);
        i13=i13+1;
    else
        i13=i13+1;
    end
end
breakpoints_ridge_tmp = sort(breakpoints_ridge_in);
if(breakpoints_ridge_tmp(1)-1 > thr2 * n && n-breakpoints_ridge_tmp(end) > thr2 * n)
    breakpoint_total{4} = [1;breakpoints_ridge_tmp;n+1];
elseif(breakpoints_ridge_tmp(1)-1 > thr2 * n && n-breakpoints_ridge_tmp(end) < thr2 * n)
    breakpoint_total{4} = [1;breakpoints_ridge_tmp(1:end-1);n+1];
elseif(breakpoints_ridge_tmp(1)-1 < thr2 * n && n-breakpoints_ridge_tmp(end) > thr2 * n)
    breakpoint_total{4} = [1;breakpoints_ridge_tmp(2:end);n+1];
else
    breakpoint_total{4} = [1;breakpoints_ridge_tmp(2:end-1);n+1];
end

W_ridge_est_sparse = zeros(p*q, n);
W_ridge_est_sparse_mse = zeros(p*n, q);
for i14 = 1:q
%     fprintf('Now the %d th response.\n', i14)
    Y_ridge_est = Respon_mat(:,i14);
    Wf_ridge = zeros(p, size(breakpoint_total{4}, 1) - 1);
    for i15 = 1:size(breakpoint_total{4}, 1) - 1
        [B_ridge, info_ridge] = lasso(X(breakpoint_total{4}(i15):breakpoint_total{4}(i15 + 1) - 1, :), Y_ridge_est(breakpoint_total{4}(i15):breakpoint_total{4}(i15 + 1) - 1), 'Alpha', 1, 'CV', 5);
        Wf_ridge(:, i15) = B_ridge(:, info_ridge.Index1SE);
    end
    Wf_ridge_mat = zeros(p, n);
    for i16 = 1:size(breakpoint_total{4}, 1) - 1
        Wf_ridge_mat(:,breakpoint_total{4}(i16):breakpoint_total{4}(i16 + 1) - 1) = repmat(Wf_ridge(:, i16), 1, breakpoint_total{4}(i16 + 1) - breakpoint_total{4}(i16));
    end
    W_ridge_est_sparse(((i14-1)*p+1):(i14*p), :) = Wf_ridge_mat;
    W_ridge_est_sparse_mse(:, i14) = Wf_ridge_mat(:);
end
sparse_estimate{4} = W_ridge_est_sparse;
sparse_estimate_mse{4} = W_ridge_est_sparse_mse;

% toc

%% get results
% 稀疏结果
W_true = cat(1,W0{:});
results_sparse = zeros(4, 5);
for i17 = 1:4
    TP = length(find(((abs(sparse_estimate{i17}) > 0)==1) & ((W_true~=0)==1)));
    FP = length(find(((abs(sparse_estimate{i17}) > 0)==1) & ((W_true~=0)==0)));  
    TN = length(find(((abs(sparse_estimate{i17}) > 0)==0) & ((W_true~=0)==0))); 
    FN = length(find(((abs(sparse_estimate{i17}) > 0)==0) & ((W_true~=0)==1)));  
%     TP = length(find(((sparse_estimate{i17} ~= 0)==1) & ((W_true~=0)==1)));
%     FP = length(find(((sparse_estimate{i17} ~= 0)==1) & ((W_true~=0)==0)));  
%     TN = length(find(((sparse_estimate{i17} ~= 0)==0) & ((W_true~=0)==0))); 
%     FN = length(find(((sparse_estimate{i17} ~= 0)==0) & ((W_true~=0)==1)));
    Accuracy = (TP+TN)/(TP+FP+TN+FN);
    Precision = TP/(TP+FP);
    Recall = TP/(TP+FN);
    FPR = FP/(FP+TN);
    Fscore = 2*Precision*Recall/(Precision+Recall);
    results_sparse(i17,:) = [Accuracy, Precision, Recall, FPR, Fscore];
end

% 分组结果
results_group = zeros(4, 5);
for i18 = 1:4
    group_nums = breakpoint_total{i18}(2:length(breakpoint_total{i18})) - breakpoint_total{i18}(1:length(breakpoint_total{i18})-1);
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
    results_group(i18,:) = [Accuracy, Precision, Recall, FPR, Fscore];
end

% 估计和预测MSE
beta_mse = zeros(4,1);
for i19 = 1:4
    est_square_err = (sparse_estimate{i19} - W_true).^2;
    beta_mse(i19) = sum(est_square_err(:))/(n*p*q);
end

prediction_mse = zeros(4,1);
for i20 = 1:4
    est_square_err_y = (X_sparse * sparse_estimate_mse{i20} - Respon_mat).^2;
    prediction_mse(i20) = sum(est_square_err_y(:))/(n*q);
end

% 组数
groupnum_est = zeros(4, 1);
for i21 = 1:4
    groupnum_est(i21) = length(breakpoint_total{i21}) - 1;
end

