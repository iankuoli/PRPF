function [ cellRes ] = TopNItem( cellItemNames, topN, matClusteringRes)
    % 
    %
    %
    
    K = size(matClusteringRes, 2);
    cellRes = cell(topN, K);
    matRes = zeros(topN, K);
    
    for k = 1:K
        topN_Res = MaxK(matClusteringRes(:, k), topN);
        
        for n = 1:topN  
            idx = find(matClusteringRes(:, k) == topN_Res(n), 1);
            matRes(n, k) = idx;
            
            if matClusteringRes(idx, k) ~= 0
                id = find([cellItemNames{:,1}] == idx, 1);
                if ~isempty(id)
                    cellRes{n, k} = cellItemNames{id,2};
                else
                    cellRes{n, k} = 'n / a';
                end
            end
                
            matClusteringRes(idx, k) = 0;
        end
    end

    
end

