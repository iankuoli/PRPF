function [X] = LorSLIM(tol, maxIter, z, ro, lambda, beta, topK)

    %% Parameters declaration
    
    global K                 % number of topics
    global M                 % number of users
    global N                 % number of items
    
    global matX              % dim(M, N): consuming records for training
    global matX_test         % dim(M, N): consuming records for testing
    global matX_valid        % dim(M, N): consuming records for validation
    
    global best_TestPrecRecall_precision
    
    best_TestPrecRecall_precision = zeros(1, length(topK)*2);
    
    %% Problem
    %
    %  min  1/2|| X-XW||_F^2 + 1/2 beta * ||W||_F^2 + lambda * ||W||_1 +
    %  z*||W||_{*}
    %  s.t.  W >=0
    %        diag(W)=0
    %
    %% Reformulation
    %
    %  min 1/2||X-XW||_F^2 + 1/2 beta * ||W||_F^2 +
    %  lambda * ||W||_1+z*||W||_{*}+<V,W-S>+1/2 ro||W-S||_F^2
    %  s.t. W >= 0
    %       diag(W)=0
    %
    %% Input parameters:
    %
    % matX -        Matrix of the training data
    % test -         Cell of the testing data excludes rated/purchased one
    % test_zhong -       The rated/purchased one in the testing data
    % z -        Nuclear norm regularization parameter
    % ro -       Parameter associated with the consensus L_2 constraint
    % lambda -       L_1 norm regularization parameter
    % beta -         L_2 norm regularization parameter
    % maxIter -      Maximum number of iterations
    % tol -      Tolerance parameter
    %
    %% Output parameters
    % hr -      Matrix of the HR corresponds to N = 5,10,15,20,25
    % arhr -        ARHR corresponds to N = 10
    %
    
    
    %% Initialization
    %Initialize W, V and S
    [M, N] = size(matX);
    
    matW = rand(N);
    for i = 1:size(matW, 1)
        matW(i, i) = 0;
    end
    matV = ones(size(matX, 2));
    matS = matW;
    
    
    %% Training
    now_goal = 0;
    IterStep = 0;
    while(1)
        if IterStep > maxIter
            break
        end
        % Update
        parfor j = 1:size(matW,1)
            matW(:,j) = nnLeastR1(matX, matX(:,j), lambda, matS(:,j), ro, matV(:,j), beta);
        end
        for j = 1:size(matW,1)
            matW(j,j) = 0;
        end
        matS = cal_nuclear(matW, matV, z, ro);
        matV = matV + ro * (matW - matS);
        IterStep = IterStep + 1;
        last_goal = now_goal;
        now_goal = cal_goal(matW, matX, beta, ro, lambda, matV, matS, z);
        if((abs(last_goal - now_goal) < last_goal * tol) && (IterStep>1))
            break
        end
    end
    
    
    %% Testing
    %[hr,arhr] = cal_res(W,matX,test,test_zhong);
    
    % Normalization
    As = matX;
    mu = mean(matX, 1);
    nu=(sum(matX.^2, 1) / size(matX, 1)) .^ (0.5); 
    nu=nu';
    As= (As - repmat(mu, size(As, 1), 1)) * diag(nu)^(-1);
    
    %Calculation
    X = As * W;
    
    [test_usr_idx, test_itm_idx, test_val] = find(matX_test);
    test_usr_idx = unique(test_usr_idx);
    list_vecPrecision = zeros(1, length(topK));
    list_vecRecall = zeros(1, length(topK));
    Tlog_likelihood = 0;
    step_size = 10000;

    for j = 1:ceil(length(test_usr_idx)/step_size)
        range_step = (1 + (j-1) * step_size):min(j*step_size, length(test_usr_idx));

        % Compute the Precision and Recall
        test_matPredict = X(test_usr_idx(range_step), :);
        test_matPredict = test_matPredict - test_matPredict .* (matX(test_usr_idx(range_step), :) > 0);
        [vec_precision, vec_recall] = MeasurePrecisionRecall(matX_test(test_usr_idx(range_step), :), test_matPredict, topK);
        list_vecPrecision = list_vecPrecision + sum(vec_precision, 1);
        list_vecRecall = list_vecRecall + sum(vec_recall, 1);

        % Compute the log likelihood
        Tlog_likelihood = Tlog_likelihood + DistributionPoissonLogNZ(matX_test(test_usr_idx(range_step), :), test_matPredict);
    end

    test_precision = list_vecPrecision / length(test_usr_idx);
    test_recall = list_vecRecall / length(test_usr_idx);
    
    best_TestPrecRecall_precision = [test_precision test_recall];
end
    