function [ normX ] = Normalize( X, type )
    %
    % Normalize matrix X
    % type: 1. column normalization
    %       2. row normalization
    %       3. symmetric normalization
    %

    if type == 1
        vecSum = sum(X, 1);
        [x_idx, y_idx, val] = find(vecSum);
        invSum = sparse(x_idx, y_idx, 1./val, 1, size(X, 2));
        normX = bsxfun(@times, X, invSum);
    elseif type == 2
        vecSum = sum(X, 2);
        [x_idx, y_idx, val] = find(vecSum);
        invSum = sparse(x_idx, y_idx, 1./val, size(X, 1), 1);
        normX = bsxfun(@times, X, invSum);
    elseif type == 3
        vecSum = sum(X, 1);
        [x_idx, y_idx, val] = find(vecSum);
        invSum = sparse(x_idx, y_idx, 1./sqrt(val), 1, size(X, 2));
        normX = bsxfun(@times, X, invSum);
        
        vecSum = sum(X, 2);
        [x_idx, y_idx, val] = find(vecSum);
        invSum = sparse(x_idx, y_idx, 1/sqrt(val), size(X, 1), 1);
        normX = bsxfun(@times, normX, invSum);
    end

end

