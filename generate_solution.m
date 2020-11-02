function [W0] = generate_solution(p, q, k, n, option)

% generate FOUR types of pq*n regression coefficients % 
% there are k nonzero coefficients in each set% 
% option, 6 values:
% 1. 'non_bal': all-different structure between balanced groups
% 2. 'non_unbal': all-different structure between unbalanced groups 
% 3. 'same_bal': all-same structure but different values between balanced groups
% 4. 'same_unbal': all-same structure but different values between unbalanced groups
% 5. 'diag_non_bal': diagnal-beta with different structures and balanced groups
% 6. 'diag_non_unbal': diagnal-beta with different structures and unbalanced groups
% 7. 'diag_same_bal': diagnal-beta with same structures but different values and balanced groups
% 8. 'diag_same_unbal': diagnal-beta with same structures but different values and unbalanced groups

if(strcmp(option,'non_bal'))
    B1 = zeros(p,1);
    B2 = zeros(p,1);
    B3 = zeros(p,1);
    B1(1:k) = 1;
    B2(k+1:2*k) = 1;
    B3(2*k+1:3*k) = 1;
    W0 = cell(q, 1);
    for i = 1:q
        W0{i} = [repmat(B1, 1, n/3), repmat(B2, 1, n/3), repmat(B3, 1, n/3)];
    end
    return;
end

if(strcmp(option,'non_unbal'))
    B1 = zeros(p,1);
    B2 = zeros(p,1);
    B3 = zeros(p,1);
    B1(1:k) = 1;
    B2(k+1:2*k) = 1;
    B3(2*k+1:3*k) = 1;
    W0 = cell(q, 1);
    for i = 1:q
        W0{i} = [repmat(B1, 1, 60), repmat(B2, 1, 80), repmat(B3, 1, 100)];
    end
    return;
end

if(strcmp(option,'same_bal'))
    bet_single_1 = zeros(p, q);
    bet_single_1(1:k,:) = unifrnd(-2.2,-2,k,q);
    bet_single_2 = zeros(p, q);
    bet_single_2(3:k+2,:) = unifrnd(1,1.2,k,q);
    bet_single_3 = zeros(p, q);
    bet_single_3(5:k+4,:) = unifrnd(2.5,2.8,k,q);
   
    W0 = cell(q, 1);
    for i = 1:q
        W0{i} = [repmat(bet_single_1(:,i), 1, n/3), repmat(bet_single_2(:,i), 1, n/3), repmat(bet_single_3(:,i), 1, n/3)];
    end
    return;
end

if(strcmp(option,'same_unbal'))
    bet_single_1 = zeros(p, q);
    bet_single_1(1:k,:) = unifrnd(-2.2,-2,k,q);
    bet_single_2 = zeros(p, q);
    bet_single_2(3:k+2,:) = unifrnd(1,1.2,k,q);
    bet_single_3 = zeros(p, q);
    bet_single_3(5:k+4,:) = unifrnd(2.5,2.8,k,q);
   
    W0 = cell(q, 1);
    for i = 1:q
        W0{i} = [repmat(bet_single_1(:,i), 1, 60), repmat(bet_single_2(:,i), 1, 80), repmat(bet_single_3(:,i), 1, 100)];
    end
    return;
end

if(strcmp(option,'diag_non_bal'))
    if(p > q)
        blksize = 5;
        blk0 = cell(10, 1);
        for i = 1:10
            blk0{i} = ones(blksize).* binornd(1, 0.8, blksize, blksize);
        end
        bet_temp_1 = zeros(q, q);
        bet_temp_1(1:10*blksize, 1:10*blksize) = blkdiag(blk0{:});
        bet_temp_2 = zeros(q, q);
        bet_temp_2((10*blksize+1):20*blksize, (10*blksize+1):20*blksize) = blkdiag(blk0{:});
        bet_temp_3 = zeros(q, q);
        bet_temp_3((20*blksize+1):30*blksize, (20*blksize+1):30*blksize) = blkdiag(blk0{:});
        bet_single_1 = [bet_temp_1; zeros(p-q, q)];
        bet_single_2 = [bet_temp_2; zeros(p-q, q)];
        bet_single_3 = [bet_temp_3; zeros(p-q, q)];
        
        W0 = cell(q, 1);
        for i = 1:q
            W0{i} = [repmat(bet_single_1(:,i), 1, n/3), repmat(bet_single_2(:,i), 1, n/3), repmat(bet_single_3(:,i), 1, n/3)];
        end
    else
        blksize = 5;
        blk0 = cell(10, 1);
        for i = 1:10
            blk0{i} = ones(blksize).* binornd(1, 0.8, blksize, blksize);
        end
        bet_single_1 = zeros(p, q);
        bet_single_1(1:10*blksize, 1:10*blksize) = blkdiag(blk0{:});
        bet_single_2 = zeros(p, q);
        bet_single_2((10*blksize+1):20*blksize, (10*blksize+1):20*blksize) = blkdiag(blk0{:});
        bet_single_3 = zeros(p, q);
        bet_single_3((20*blksize+1):30*blksize, (20*blksize+1):30*blksize) = blkdiag(blk0{:});
        
        W0 = cell(q, 1);
        for i = 1:q
            W0{i} = [repmat(bet_single_1(:,i), 1, n/3), repmat(bet_single_2(:,i), 1, n/3), repmat(bet_single_3(:,i), 1, n/3)];
        end
    end
    return;
end

if(strcmp(option,'diag_non_unbal'))
    if(p > q)
        blksize = 5;
        blk0 = cell(10, 1);
        for i = 1:10
            blk0{i} = ones(blksize).* binornd(1, 0.8, blksize, blksize);
        end
        bet_temp_1 = zeros(q, q);
        bet_temp_1(1:10*blksize, 1:10*blksize) = blkdiag(blk0{:});
        bet_temp_2 = zeros(q, q);
        bet_temp_2((10*blksize+1):20*blksize, (10*blksize+1):20*blksize) = blkdiag(blk0{:});
        bet_temp_3 = zeros(q, q);
        bet_temp_3((20*blksize+1):30*blksize, (20*blksize+1):30*blksize) = blkdiag(blk0{:});
        bet_single_1 = [bet_temp_1; zeros(p-q, q)];
        bet_single_2 = [bet_temp_2; zeros(p-q, q)];
        bet_single_3 = [bet_temp_3; zeros(p-q, q)];
        
        W0 = cell(q, 1);
        for i = 1:q
            W0{i} = [repmat(bet_single_1(:,i), 1, 60), repmat(bet_single_2(:,i), 1, 80), repmat(bet_single_3(:,i), 1, 100)];
        end
    else
        blksize = 5;
        blk0 = cell(10, 1);
        for i = 1:10
            blk0{i} = ones(blksize).* binornd(1, 0.8, blksize, blksize);
        end
        bet_single_1 = zeros(p, q);
        bet_single_1(1:10*blksize, 1:10*blksize) = blkdiag(blk0{:});
        bet_single_2 = zeros(p, q);
        bet_single_2((10*blksize+1):20*blksize, (10*blksize+1):20*blksize) = blkdiag(blk0{:});
        bet_single_3 = zeros(p, q);
        bet_single_3((20*blksize+1):30*blksize, (20*blksize+1):30*blksize) = blkdiag(blk0{:});
        
        W0 = cell(q, 1);
        for i = 1:q
            W0{i} = [repmat(bet_single_1(:,i), 1, 60), repmat(bet_single_2(:,i), 1, 80), repmat(bet_single_3(:,i), 1, 100)];
        end
    end
    return;
end


if(strcmp(option,'diag_same_bal'))
    blksize = 5;
    blk1 = cell(30, 1);
    for i = 1:30
        blk1{i} = unifrnd(-2.2,-2, blksize, blksize).* binornd(1, 0.8, blksize, blksize);
    end
    bet_single_1 = blkdiag(blk1{:});

   blk2 = cell(30, 1);
    for i = 1:30
        blk2{i} = unifrnd(1,1.2, blksize, blksize).* binornd(1, 0.8, blksize, blksize);
    end
    bet_single_2 = blkdiag(blk2{:});

    blk3 = cell(30, 1);
    for i = 1:30
        blk3{i} = unifrnd(2.5,2.8, blksize, blksize).* binornd(1, 0.8, blksize, blksize);
    end
    bet_single_3 = blkdiag(blk3{:});

    W0 = cell(q, 1);
    for i = 1:q
        W0{i} = [repmat(bet_single_1(:,i), 1, n/3), repmat(bet_single_2(:,i), 1, n/3), repmat(bet_single_3(:,i), 1, n/3)];
    end
    return;
end


if(strcmp(option,'diag_same_unbal'))
    blksize = 5;
    blk1 = cell(30, 1);
    for i = 1:30
        blk1{i} = unifrnd(-2.2,-2, blksize, blksize).* binornd(1, 0.8, blksize, blksize);
    end
    bet_single_1 = blkdiag(blk1{:});

   blk2 = cell(30, 1);
    for i = 1:30
        blk2{i} = unifrnd(1,1.2, blksize, blksize).* binornd(1, 0.8, blksize, blksize);
    end
    bet_single_2 = blkdiag(blk2{:});

    blk3 = cell(30, 1);
    for i = 1:30
        blk3{i} = unifrnd(2.5,2.8, blksize, blksize).* binornd(1, 0.8, blksize, blksize);
    end
    bet_single_3 = blkdiag(blk3{:});

    W0 = cell(q, 1);
    for i = 1:q
        W0{i} = [repmat(bet_single_1(:,i), 1, 60), repmat(bet_single_2(:,i), 1, 80), repmat(bet_single_3(:,i), 1, 100)];
    end
    return;
end

disp('option is wrong!');
end    