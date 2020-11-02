function[breakpoint_total, sparse_estimate, group_estimate] = msf(X_reshape, Y, R, lam1, lam2, biasflag)

#

W_est_multilasso = cell(q, 1);
X1_cell = cell(q, 1);
row_ind_cell = cell(q, 1);

% tic
for res_dim = 1:q

    %% Localized Lasso
    %Training
    [W, ~] = LocLasso(X_reshape, Y, R, lam1, lam2, biasflag);

    W = W.*(abs(W) > 0.01);
    [row, ~] = find(W);
    row = unique(row);
    X1 = X_reshape(row, :);
    [W1, ~] = LocLasso(X1, Y, R, lam1, 2 * lam2, biasflag);
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
