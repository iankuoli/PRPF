function [matClusteringRes] = NMF(k, topK)
    %
    % Coordinate Ascent Algorithm
    %
    
    %% Parameters declaration
    
    global K                 % number of topics
    global M                 % number of users
    global N                 % number of items
    
    global matX              % dim(M, N): consuming records for training
    global matX_test         % dim(M, N): consuming records for testing
    
    global matTheta          % dim(M, K): latent document-topic intensities
    global matBeta           % dim(N, K): latent word-topic intensities
    
    global best_TestPrecRecall_precision
    
    
    best_TestPrecRecall_precision = zeros(1, length(topK)*2);
    
    [M, N] = size(matX); 
    K = k;
    
    [matTheta, matBeta] = nnmf(matX, K);
    matBeta = matBeta';
    
    [test_usr_idx, test_itm_idx, test_val] = find(matX_test);
    test_usr_idx = unique(test_usr_idx);
    list_vecPrecision = zeros(1, length(topK));
    list_vecRecall = zeros(1, length(topK));
    Tlog_likelihood = 0;
    step_size = 10000;

    for j = 1:ceil(length(test_usr_idx)/step_size)
        range_step = (1 + (j-1) * step_size):min(j*step_size, length(test_usr_idx));

        % Compute the Precision and Recall
        test_matPredict = matTheta(test_usr_idx(range_step),:) * matBeta';
        test_matPredict = test_matPredict - test_matPredict .* (matX(test_usr_idx(range_step), :) > 0);
        [vec_precision, vec_recall] = MeasurePrecisionRecall(matX_test(test_usr_idx(range_step), :), test_matPredict, topK);
        list_vecPrecision = list_vecPrecision + sum(vec_precision, 1);
        list_vecRecall = list_vecRecall + sum(vec_recall, 1);

        % Compute the log likelihood
        Tlog_likelihood = Tlog_likelihood + DistributionPoissonLogNZ(matX_test(test_usr_idx(range_step), :), test_matPredict);
    end

    test_precision = list_vecPrecision / length(test_usr_idx);
    test_recall = list_vecRecall / length(test_usr_idx);
    best_TestPrecRecall_precision = [test_precision, test_recall];
end