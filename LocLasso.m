function [W,fval,Yest] = LocLasso(Xtr,Ytr,R,lam1,lam2,biasflag)

num_iter = 50;

if biasflag == 1
    Xtr = [Xtr;ones(1,size(Xtr,2))];
end

[d,ntr] = size(Xtr);
%computing kernel matrices

vecW = ones(d*ntr,1);

% seed = 1;
% rand('state',seed);
% randn('state',seed);

dinv = rand(d*ntr,1);
index = 1:(d*ntr);
D = sparse(index,index,dinv);

A = zeros(d*ntr,ntr);

for ii = 1:ntr
    A(ii:ntr:end,ii) = Xtr(:,ii);
end

A = sparse(A');
fval = zeros(num_iter,1);
epsi = 0.001;

for iter = 1:num_iter
    
%     fprintf('The iteration now in %d step.\n', iter)
    
    index = 1:(d*ntr);
    DinvA = D\A';
    B = eye(size(A,1)) + A*DinvA;
    vecW = B\Ytr;
    vecW = DinvA*vecW;
    
    W = reshape(vecW,ntr,d);
    
    %Network regularization
    tmp = Distl2(W',W');
    tmp1 = tmp.*R;
    U_net = sum(tmp1(:));
    
    tmp = (0.5./(tmp + 10e-10)).*R;
    
    td1 = diag(sum(tmp,1));
    td2 = diag(sum(tmp,2));
    
    AA = td1 + td2 - 2*tmp;
    AA = (AA + AA')/2 + eye(ntr)*0.001;

    D_net = kron(speye(d),sparse(AA));
           
    %Exclusive constraint
    if biasflag == 0
        tmp = (sum(abs(W),2));
        tmp2 = repmat(tmp,1,d);
    
        D_exc = sparse(index,index,tmp2(:)./(abs(vecW)+10e-10));
    else
        tmp = (sum(abs(W(:,1:(end-1))),2));
        tmp2 = repmat(tmp,1,d);
        
        tmp2(:,end) = 0;
        D_exc = sparse(index,index,tmp2(:)./(abs(vecW)+10e-10));
    end
    
    
    U_exc = sum(tmp.^2);
    D = lam1*D_net + lam2*D_exc;
    
    fval(iter) = sum((Ytr - A*vecW).^2) + lam1*U_net + lam2*U_exc;
    
    if iter >= 2 
        stop_rule = abs(fval(iter)-fval(iter-1))/abs(fval(iter-1));
        if stop_rule < epsi
%            fprintf('The iteration stop at %d step.\n', iter)
           break 
        end
    end
    
end

% logL = log(sum((Ytr - A*vecW).^2) / ntr);
% [~,BIC_val] = aicbic(logL, ntr * d, ntr);
% BIC_val = log(sum((Ytr - A*vecW).^2) / ntr) + log(ntr*d) * (log(ntr)/ntr) * (3*d);

W = reshape(vecW,ntr,d)';
Yest = A*vecW;

