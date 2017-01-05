function [ matRet ] = Similarity( X, Y, type, r)
    % Compute the similarity between two data lists (i.e., X and Y)
    
    if strcmp(type, 'rbf')
        matRet = -2 * full(X * Y');
        matRet = bsxfun(@plus, matRet, diag(X * X'));
        matRet = bsxfun(@plus, matRet, diag(Y * Y')');
        %h = mean(mean(matRet));  
        h = std(reshape(matRet, size(matRet,1) * size(matRet,2), 1))/20;
        matRet = exp(- matRet / h);
    elseif strcmp(type, 'cos')
        matRet = full(X * Y');
        a = sqrt(diag(X * X'));
        b = sqrt(diag(Y * Y'));
        %matRet = bsxfun(@plus, matRet, a);
        %matRet = bsxfun(@plus, matRet, b');
        matRet = bsxfun(@rdivide, matRet, a);
        matRet = bsxfun(@rdivide, matRet, b');
        
    elseif strcmp(type, 'gamma')
        
        if nargin < 4
            r = 1;
        end
        
        X = X + r;
        Y = Y + r;
        X_denominator = gammaln(X);
        Y_denominator = gammaln(Y);
        
        matRet = zeros(size(X,1), size(Y,1));
        
        for i = 1:size(X,1)
            ret = gammaln(0.5 * bsxfun(@plus, Y, X(i,:)));
            ret = bsxfun(@plus, ret, -0.5 * X_denominator(i,:));
            ret = ret -0.5 * Y_denominator;
            matRet(i, :) = sum(ret');
        end
        
        matRet = matRet ./ size(X,2);
        matRet = exp(matRet);
    
    elseif strcmp(type, 'gamma2')
        
        if nargin < 4
            r = 1;
        end
        
        [x_i, x_j, x_v] = find(X);
        [y_i, y_j, y_v] = find(Y);
        
        x_nz_denominator = sparse(x_i, x_j, gammaln(x_v + r), size(X,1), size(X,2));
        y_nz_denominator = sparse(y_i, y_j, gammaln(y_v + r), size(Y,1), size(Y,2));
        
        matRet = zeros(size(X,1), size(X,1));
        mat_norm = zeros(size(X,1), size(X,1));
        
        for i = 1:size(X,1)
            nz_mean = 0.5 * (Y + ones(size(Y, 1), 1) * X(i, :));
            [nz_i, nz_j, nz_v] = find(nz_mean);
            ret = sparse(nz_i, nz_j, gammaln(nz_v + r), size(nz_mean,1), size(nz_mean,2));
            ret = ret - 0.5 * ones(size(Y,1), 1) * x_nz_denominator(i, :) - 0.5 * y_nz_denominator;
            
            mat_norm(i,:) = sum(nz_mean > 0, 2)';
            matRet(i,:) = sum(ret, 2);
        end
        
        matRet = exp(matRet ./ mat_norm);
        
    elseif strcmp(type, 'gamma3')
        
        if nargin < 4
            r = 1;
        end
        
        scaleX = sum(X, 2);
        scaleY = sum(Y, 2);
        
        [x_i, x_j, x_v] = find(scaleX);
        [y_i, y_j, y_v] = find(scaleY);
        
        x_nz_denominator = sparse(x_i, x_j, gammaln(x_v + r), size(scaleX,1), size(scaleX,2));
        y_nz_denominator = sparse(y_i, y_j, gammaln(y_v + r), size(scaleY,1), size(scaleY,2));
        
        matRetScale = zeros(size(scaleX,1), size(scaleX,1));
        
        for i = 1:size(scaleX,1)
            nz_mean = 0.5 * (scaleY + ones(size(scaleY, 1), 1) * scaleX(i, :));
            [nz_i, nz_j, nz_v] = find(nz_mean);
            ret = sparse(nz_i, nz_j, gammaln(nz_v + r), size(nz_mean,1), size(nz_mean,2));
            ret = ret - 0.5 * ones(size(scaleY,1), 1) * x_nz_denominator(i, :) - 0.5 * y_nz_denominator;
            
            matRetScale(i,:) = sum(ret, 2);
        end
        
        multX = bsxfun(@rdivide, X, scaleX);
        multY = bsxfun(@rdivide, Y, scaleY);
        
        matRetMult = sqrt(multX) * sqrt(multY)' - eye(size(X,1));
        a = exp(matRetScale);
        
        matRet = matRetMult .* exp(matRetScale);
    
    elseif strcmp(type, 'poisson')
        
        if nargin < 4
            r = 1;
        end
        
        scaleX = sum(X, 2);
        scaleY = sum(Y, 2);
        
        mean_scaleX = scaleX ./ sum(X>0, 2);
        mean_scaleY = scaleY ./ sum(Y>0, 2);
        
        matRetScale = sqrt(mean_scaleX) * sqrt(mean_scaleY)';
        matRetScale = bsxfun(@plus, matRetScale, -0.5 * mean_scaleX);
        matRetScale = exp(bsxfun(@plus, matRetScale, -0.5 * mean_scaleY'));
        
        multX = bsxfun(@rdivide, X, scaleX);
        multY = bsxfun(@rdivide, Y, scaleY);
        
        matRetMult = sqrt(multX) * sqrt(multY)';
        %matRetMult = matRetMult - diag(diag(matRetMult));
        %matRetMult = 1 ./ (1-matRetMult) - 1;
        
        matRet = (matRetMult.^8) .* matRetScale;
    
        
    elseif strcmp(type, 'pfs')
        
        if nargin < 4
            r = 1;
        end
        
        scaleX = sum(X, 2);
        scaleY = sum(Y, 2);
        
        nnzX = sum(X>0, 2);
        nnzY = sum(Y>0, 2);
        
        mean_scaleX = scaleX ./ nnzX;
        mean_scaleY = scaleY ./ nnzY;
        
        matRetScale = sqrt(mean_scaleX) * sqrt(mean_scaleY)';
        matRetScale = bsxfun(@plus, matRetScale, -0.5 * mean_scaleX);
        matRetScale = exp(bsxfun(@plus, matRetScale, -0.5 * mean_scaleY'));
        
        multX = bsxfun(@rdivide, X, scaleX);
        multY = bsxfun(@rdivide, Y, scaleY);
        
        matRetMult = sqrt(multX) * sqrt(multY)';
        
        % arithmetic mean
        %matPower = zeros(length(nnzX), length(nnzY));
        %matPower = bsxfun(@plus, matPower, nnzX);
        %matPower = 0.5 * bsxfun(@plus, matPower, nnzY');
        
        % geometric mean
        matPower = sqrt(nnzX * nnzY');
        
        matRet = (matRetMult.^ matPower) .* matRetScale;
    
    elseif strcmp(type, 'pfs2')
        
        scaleX = sum(X, 2) + 10e-10;
        scaleY = sum(Y, 2) + 10e-10;
        
        nnzX = sum(X>0, 2) + 10e-10;
        nnzY = sum(Y>0, 2) + 10e-10;
        
        mean_scaleX = scaleX ./ nnzX;
        mean_scaleY = scaleY ./ nnzY;      
        matRetScale = sqrt(mean_scaleX) * sqrt(mean_scaleY)';
        matRetScale = bsxfun(@plus, matRetScale, -0.5 * mean_scaleX);
        matRetScale = exp(bsxfun(@plus, matRetScale, -0.5 * mean_scaleY'));
        
        multX = bsxfun(@rdivide, X, scaleX);
        multY = bsxfun(@rdivide, Y, scaleY);        
        matRetMult = sqrt(multX) * sqrt(multY)';
        
        matRetNNZ = sqrt(nnzX) * sqrt(nnzY)';
        matRetNNZ = bsxfun(@plus, matRetNNZ, -0.5 * nnzX);
        matRetNNZ = exp(bsxfun(@plus, matRetNNZ, -0.5 * nnzY'));
        
        matRet = (matRetMult.^r) .* matRetScale .* matRetNNZ;
    end
    
end

