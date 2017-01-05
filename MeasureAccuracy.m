function [ accurate ] = MeasureAccuracy( vecLabel, vecPred, K )

    N = size(vecLabel, 1);

    matLabel = sparse(1:N, vecLabel, ones(N,1), N, K);
    matPred = sparse(1:N, vecPred, ones(N,1), N, K);

    accurate_instance = 0;
    
    for k = 1:K
        
        match = matLabel(:,k)' * matPred;        
        [max_match_val, max_match_idx] = max(match);
        
        accurate_instance = accurate_instance + max_match_val;
        matPred(:, max_match_idx) = [];
    end
    
    accurate = accurate_instance / N;
end

