function [matClusteringRes] = LogisticPF(type_model, k, prior, ini_scale, usr_batch_size, delta, kappa, topK, test_size, test_step, ini, MaxItr, check_step)
    %
    % Coordinate Ascent Algorithm
    %
    
    %% Parameters declaration
    
    global K                 % number of topics
    global M                 % number of users
    global N                 % number of items
    
    global matX              % dim(M, N): consuming records for training
    global matX_test         % dim(M, N): consuming records for testing
    global matX_valid        % dim(M, N): consuming records for validation
    global matX_predict

    global matEpsilon        % dim(M, 1): latent document-topic offsets
    global matTheta          % dim(M, K): latent document-topic intensities

    global matEta            % dim(N, 1): latent word-topic offsets
    global matBeta           % dim(N, K): latent word-topic intensities

    global matEpsilon_Shp    % dim(M, 1): varational param of matEpsilon (shape)
    global matEpsilon_Rte    % dim(M, 1): varational param of matEpsilon (rate)
    global matTheta_Shp      % dim(M, K): varational param of matTheta (shape)
    global matTheta_Rte      % dim(M, K): varational param of matTheta (rate)

    global matEta_Shp        % dim(N, 1): varational param of matEta (shape)
    global matEta_Rte        % dim(N, 1): varational param of matEta (rate)
    global matBeta_Shp       % dim(N, K): varational param of matBeta (shape)
    global matBeta_Rte       % dim(N, K): varational param of matBeta (rate)

    global tensorPhi         % dim(K, M, N) = cell{dim(M,N)}: varational param of matX
    global tensorRho         % dim(K, M, M) = cell{dim(M,M)}: varational param of matW
    global tensorSigma       % dim(K, N, N) = cell{dim(N,N)}: varational param of matS
    
    global list_ValidPrecRecall
    global list_TestPrecRecall
    global list_ValidLogLikelihhod
    global list_TrainLogLikelihhod
    global best_TestPrecRecall_precision
    global best_TestPrecRecall_likelihood
    global best_TestLogLikelihhod
    global bestVlog_likelihood
    
    list_ValidPrecRecall = zeros(MaxItr, length(topK)*2);
    list_TestPrecRecall = zeros(MaxItr/20, length(topK)*2);
    list_ValidLogLikelihhod = zeros(MaxItr/check_step, 1);
    list_TrainLogLikelihhod = zeros(MaxItr, 1);
    best_TestPrecRecall_precision = zeros(1, length(topK)*2);
    best_TestPrecRecall_likelihood = zeros(1, length(topK)*2);
    best_TestLogLikelihhod = -Inf;
    bestVlog_likelihood = -Inf;
    best_ValidPrecision = 0;
    
    %% Initialization
    
    a = prior(1);
    b = prior(2);
    c = prior(3);
    d = prior(4);
    e = prior(5);
    f = prior(6);
    m = 100;
    
    if ini == 0
        
        [M, N] = size(matX); 

        K = k;    
        
        %offset = 0.1;

        matEpsilon_Shp = ini_scale * rand(M, 1) + b;
        matEpsilon_Rte = ini_scale * rand(M, 1) + c;
        matEpsilon = matEpsilon_Shp ./ matEpsilon_Rte;

        matEta_Shp = ini_scale * rand(N, 1) + e;
        matEta_Rte = ini_scale * rand(N, 1) + f;
        matEta = matEta_Shp ./ matEta_Rte;

        matBeta_Shp = ini_scale * rand(N, K) + d;
        matBeta_Rte = bsxfun(@plus, ini_scale * rand(N, K), matEta);
        matBeta = matBeta_Shp ./ matBeta_Rte;

        matTheta_Shp = ini_scale * rand(M, K) + a;
        matTheta_Rte = bsxfun(@plus, ini_scale * rand(M, K), matEpsilon);
        matTheta = matTheta_Shp ./ matTheta_Rte;
        
        matX_predict = (matTheta(1,:) * matBeta(1,:)') * (matX > 0);
        
        tensorPhi = cell(K, 1);
        for i = 1:K
            tensorPhi{i} = sparse(M, N);
        end

        tensorRho = cell(K, 1);
        for i = 1:K
            tensorRho{i} = sparse(M, M);
        end

        tensorSigma = cell(K, 1);
        for i = 1:K
            tensorSigma{i} = sparse(N, N);
        end
        
        zero_idx_usr = sum(matX,2)==0;
        matTheta_Shp(zero_idx_usr(:,1),:) = 0;
        matTheta(zero_idx_usr(:,1),:) = 0;
        zero_idx_itm = sum(matX,1)==0;
        matTheta_Shp(zero_idx_itm(:,1),:) = 0;
        matBeta(zero_idx_itm(:,1),:) = 0;
    end

    
    %% CTPF Coordinate ascent
    
    IsConverge = false;
    i = ini;
    l = 0;
    
    
    %[x_i, x_j, x_v] = find(matX);
    
    while IsConverge == false && i < MaxItr
                
        i = i + 1;
        
        %
        % Set the learning rate 
        % ref: Content-based recommendations with Poisson factorization. NIPS, 2014
        %
        if usr_batch_size == M
            lr = 1;
        else
            offset = 1024;
            lr = (offset + i) ^ -kappa;
        end
        
        %
        % Sample data
        %
        if usr_batch_size == size(matX,1)
            usr_idx = 1:size(matX,1);
        else
            usr_idx = randsample(size(matX,1), usr_batch_size);
            usr_idx(sum(matX(usr_idx,:),2)==0) = [];
        end
        
        itm_idx = find(sum(matX(usr_idx, :))>0);
        usr_idx_len = length(usr_idx);
        itm_idx_len = length(itm_idx);
        
        fprintf('\nIndex: %d  ---------------------------------- %d , %d , lr: %f \n', i, usr_idx_len, itm_idx_len, lr);
        
        matTheta_Shp_psi = psi(matTheta_Shp(usr_idx,:));
        matTheta_Rte_log = log(matTheta_Rte(usr_idx,:));
        matBeta_Shp_psi = psi(matBeta_Shp(itm_idx,:));
        matBeta_Rte_log = log(matBeta_Rte(itm_idx,:));
        
        
        %% Update Utility Predicitons
        approx_dist = type_model;
        if approx_dist == 3
            %
            % Objective function is based on logistic function.
            % To approximate \hat{x}_{ui} by a Poisson distribution.
            % Use 2-th Taylor series for approximation.
            % Do not use conjugate prior, minimize the difference between the two expectation directly!
            %
            prior_X = (matTheta(usr_idx,:) * matBeta(itm_idx, :)') .* (matX(usr_idx,itm_idx)>0);
            predict_X = matX_predict(usr_idx, itm_idx) .* (matX(usr_idx,itm_idx)>0);
            
            %predict_X = bsxfun(@times, predict_X, (sum(matX(usr_idx, itm_idx), 2) ./ sum(predict_X,2)));
            vec_s = prior_X;
            
            inv_exp_predict_X = spfun(@(x) exp(-x), predict_X .* (matX(usr_idx,itm_idx))>0);
            %log_sigma_predict_X = spfun(@(x) -log(1+x), inv_exp_predict_X);
            diff1_log_sigma_predict_X = spfun(@(x) x./(1+x), inv_exp_predict_X);
            diff2_log_sigma_predict_X = spfun(@(x) -x./(1+x).^2, inv_exp_predict_X);
            
            l_function_s = spfun(@(x) x+1, 10*matX(usr_idx,itm_idx)) .* diff2_log_sigma_predict_X;
            h_function_s = spfun(@(x) x+1, 10*matX(usr_idx,itm_idx)) .* (diff1_log_sigma_predict_X + (0.5 - vec_s) .* diff2_log_sigma_predict_X) + spfun(@(x) log(x), prior_X) - (prior_X>0);
            
            W_tmp = -l_function_s .* spfun(@(x) exp(x), h_function_s);
            W_toosmall_mask = W_tmp <= -1/exp(1);
            W_toolarge_mask = W_tmp > 10e+30;
            W_mask = (ones(size(W_tmp)) - W_toosmall_mask - W_toolarge_mask - (matX(usr_idx,itm_idx)==0)) > 0;
            tmp1 = sparse(size(prior_X,1), size(prior_X,2));
            tmp2 = sparse(size(prior_X,1), size(prior_X,2));
            tmp1(W_mask) = Lambert_W(W_tmp(W_mask), 0);
            tmp2(W_mask) = Lambert_W(W_tmp(W_mask), -1);
            
            mask_better = abs(tmp1 - vec_s) - abs(tmp2 - vec_s);
            solution_xui_xuj = tmp1;
            solution_xui_xuj(mask_better>0) = tmp2((mask_better>0));
            solution_xui_xuj(W_toolarge_mask) = (-h_function_s(W_toolarge_mask) + 1) ./ l_function_s(W_toolarge_mask);
            solution_xui_xuj(isnan(solution_xui_xuj)) = predict_X(isnan(solution_xui_xuj));
            solution_xui_xuj(solution_xui_xuj==inf) = 1;
            
            if nnz(solution_xui_xuj<0)>0
                fprintf('ZZ');
            end
            
            matX_predict(usr_idx, itm_idx) = (1-lr) * matX_predict(usr_idx, itm_idx) + lr * solution_xui_xuj;
            %matX_predict(usr_idx, itm_idx) = (1-0.01) * matX_predict(usr_idx, itm_idx) + 0.01 * predict_X;
            predict_X = matX_predict(usr_idx, itm_idx);
        elseif approx_dist == -1
            %
            % To approximate \hat{x}_{ui} by a Gamma distribution.
            % Poor mathematical support, but works!
            %
            predict_X = (matTheta(usr_idx,:) * matBeta(itm_idx, :)') .* (matX(usr_idx,itm_idx)>0);
            for u = 1:usr_idx_len
                u_idx = usr_idx(u);
                if nnz(matX(u_idx, itm_idx)) < 2
                    continue;
                end
                [is, js, vs] = find(matX(u_idx, itm_idx));
                [val, idx] = sort(vs);

                inverse_pred = spfun(@(x) 1./x, predict_X(u,:));
                sum_inverse_pred = sum(inverse_pred);
                pred_beta = spfun(@(x) -x+sum_inverse_pred, inverse_pred);

                tmpV = sparse(ones(1,length(idx)), js(idx), (1:nnz(val)), 1, length(itm_idx));
                predict_X(u, :) = 100 * nnz(tmpV) * tmpV / sum(tmpV);
            end
            matX_predict(usr_idx, itm_idx) = predict_X;
        elseif approx_dist == -2
            %
            % Point-wise PRPF
            %
            predict_X = (matTheta(usr_idx,:) * matBeta(itm_idx, :)') .* (matX(usr_idx,itm_idx)>0);
            for u = 1:usr_idx_len
                u_idx = usr_idx(u);
                if nnz(matX(u_idx, itm_idx)) < 2
                    continue;
                end
                [is, js, vs] = find(matX(u_idx, itm_idx));
                [val, idx] = sort(vs);

                inverse_pred = spfun(@(x) 1./x, predict_X(u,:));
                sum_inverse_pred = sum(inverse_pred);
                pred_beta = spfun(@(x) -x+sum_inverse_pred, inverse_pred);

                tmpV = sparse(ones(1,length(idx)), js(idx), (1:nnz(val)), 1, length(itm_idx));
                predict_X(u, :) = tmpV;
            end
            matX_predict(usr_idx, itm_idx) = predict_X;
        else
            %
            % Poisson matrix Factorization
            %
            matX_predict(usr_idx, itm_idx) = matX(usr_idx, itm_idx);
            predict_X = matX(usr_idx, itm_idx);
        end
        
        
        %% Update Latent Tensor Variables
        
        %
        % Update tensorPhi
        %
        fprintf('Update tensorPhi ...  k = ');
        matX_One = predict_X > 0;

        tensorPhi = cellfun(@(x,y) spfun(@exp, x+y), cellfun(@(x) bsxfun(@times, matX_One, x), num2cell(matTheta_Shp_psi - matTheta_Rte_log, 1), 'UniformOutput', false), ...
                                                     cellfun(@(x) bsxfun(@times, matX_One, x'), num2cell(matBeta_Shp_psi - matBeta_Rte_log, 1), 'UniformOutput', false), 'UniformOutput', false);

        tmp = cellfun(@(x) reshape(x, [], 1), tensorPhi, 'UniformOutput', false);
        tensorPhi_inv = reshape(spfun(@(x) 1 ./ x, sum([tmp{:}],2)), usr_idx_len, itm_idx_len);
        tensorPhi = cellfun(@(x) x .* tensorPhi_inv, tensorPhi,  'UniformOutput', false);
        
        
        %% Update Latent Matrix Variables
        fprintf('\nUpdate Latent Matrix Variables ...');
        
        %
        % Update matTheta_Shp , matTheta_Rte, matThetaD
        %              
        matTheta_Shp(usr_idx, :) = (1 - lr) * matTheta_Shp(usr_idx, :) +...
                                    lr * (a + cell2mat(cellfun(@(x) sum(predict_X .* x, 2), tensorPhi, 'UniformOutput', false)));
                                
        matTheta_Rte(usr_idx, :) = (1 - lr) * matTheta_Rte(usr_idx, :) + lr * (repmat(sum(matBeta, 1), [usr_idx_len,1]));         
        matTheta_Rte(usr_idx, :) = bsxfun(@plus, matTheta_Rte(usr_idx, :), lr * matEpsilon(usr_idx));
        
        matTheta(usr_idx, :) = matTheta_Shp(usr_idx, :) ./ matTheta_Rte(usr_idx, :);
        
        if any(isnan(matTheta))>0
            fprintf('NaN');
        end
        
        
        %
        % Update matBeta_Shp , matBeta_Rte, matBetaD
        %
        scale1 = sum(matX(:, itm_idx) > 0, 1)' ./ sum(matX(usr_idx, itm_idx) > 0, 1)';
          
        matBeta_Shp(itm_idx, :) = (1 - lr) * matBeta_Shp(itm_idx, :) + ...
                                  lr * (d + bsxfun(@times, cell2mat(cellfun(@(x) sum(predict_X .* x, 1)', tensorPhi, 'UniformOutput', false)), scale1));
        
        matBeta_Rte(itm_idx, :) = (1 - lr) * matBeta_Rte(itm_idx, :) + lr * (repmat(sum(matTheta, 1), [itm_idx_len,1]));
        matBeta_Rte(itm_idx, :) = bsxfun(@plus, matBeta_Rte(itm_idx,:), lr * matEta(itm_idx));
        
        matBeta(itm_idx, :) = matBeta_Shp(itm_idx, :) ./ matBeta_Rte(itm_idx, :);
        
        if any(isnan(matBeta))>0
            fprintf('NaN');
        end
        
        %
        % Update matEpsilon_Shp , matEpsilon_Rte
        %
        %fprintf('\nUpdate matEpsilon_Shp , matEpsilon_Rte ...');
        matEpsilon_Shp(usr_idx) = (1-lr) * matEpsilon_Shp(usr_idx) + lr * (b + K * a);
        matEpsilon_Rte(usr_idx) = (1-lr) * matEpsilon_Rte(usr_idx) + lr * (c + sum(matTheta(usr_idx), 2));
        matEpsilon(usr_idx) = matEpsilon_Shp(usr_idx) ./ matEpsilon_Rte(usr_idx);
        
        %
        % Update matEta_Shp , matEta_Rte
        %
        %fprintf('\nUpdate matEta_Shp , matEta_Rte ...');
        matEta_Shp(itm_idx) = (1-lr) * matEta_Shp(itm_idx) + lr * (e + K * d);
        matEta_Rte(itm_idx) = (1-lr) * matEta_Rte(itm_idx) + lr * (f + sum(matBeta(itm_idx), 2) .* scale1);
        matEta(itm_idx) = matEta_Shp(itm_idx) ./ matEta_Rte(itm_idx);
        
        
        %% Terminiation Checkout
        fprintf('\nTerminiation Checkout ...');
        
        %
        % Compute the precision & recall of the testing set.
        %
        if mod(i, test_step) == 0 && test_size > 0
            [test_usr_idx, test_itm_idx, test_val] = find(matX_test);
            test_usr_idx = unique(test_usr_idx);
            list_vecPrecision = zeros(1, length(topK));
            list_vecRecall = zeros(1, length(topK));
            Tlog_likelihood = 0;
            step_size = 10000;
            
            for j = 1:1%ceil(length(test_usr_idx)/step_size)
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
            list_TestPrecRecall(i/test_step, :) = [test_precision test_recall];
            Tlog_likelihood = Tlog_likelihood / nnz(matX_test);
        else
            test_precision = 0;
            test_recall = 0;
        end
        
        %
        % Compute the precision & recall of the validation set.
        %
        if mod(i, check_step) == 0
            [valid_usr_idx, valid_itm_idx, valid_val] = find(matX_valid);
            valid_usr_idx = unique(valid_usr_idx);
            list_vecPrecision = zeros(1, length(topK));
            list_vecRecall = zeros(1, length(topK));
            Vlog_likelihood = 0;
            step_size = 10000;
            
            for j = 1:1%ceil(length(valid_usr_idx)/step_size)
                range_step = (1 + (j-1) * step_size):min(j*step_size, length(valid_usr_idx));
                
                % Compute the Precision and Recall
                valid_matPredict = matTheta(valid_usr_idx(range_step),:) * matBeta';
                valid_matPredict = valid_matPredict - valid_matPredict .* (matX(valid_usr_idx(range_step), :) > 0);
                [vec_precision, vec_recall] = MeasurePrecisionRecall(matX_valid(valid_usr_idx(range_step), :), valid_matPredict, topK);
                list_vecPrecision = list_vecPrecision + sum(vec_precision, 1);
                list_vecRecall = list_vecRecall + sum(vec_recall, 1);
                
                % Compute the log likelihood
                Vlog_likelihood = Vlog_likelihood + DistributionPoissonLogNZ(matX_valid(valid_usr_idx(range_step), :), valid_matPredict);
            end
            valid_precision = list_vecPrecision / length(valid_usr_idx);
            valid_recall = list_vecRecall / length(valid_usr_idx);
            Vlog_likelihood = Vlog_likelihood / nnz(matX_valid);
            list_ValidPrecRecall(i/check_step, :) = [valid_precision valid_recall];
            list_ValidLogLikelihhod(i/check_step) = Vlog_likelihood;
            new_l = Vlog_likelihood;
            
            if abs(new_l - l) < 0.000001
                IsConverge = true;
            end
            
            l = new_l;
            
            if bestVlog_likelihood < Vlog_likelihood || best_ValidPrecision <= valid_precision(1)
                if mod(i, test_step) == 0 && test_size > 0
                    if bestVlog_likelihood < Vlog_likelihood
                        best_TestPrecRecall_likelihood = [test_precision, test_recall];
                    end
                    if best_ValidPrecision <= valid_precision(1) && best_TestPrecRecall_precision(1) < test_precision(1)
                        best_TestPrecRecall_precision = [test_precision, test_recall];
                    end
                else
                    [test_usr_idx, test_itm_idx, test_val] = find(matX_test);
                    test_usr_idx = unique(test_usr_idx);
                    list_vecPrecision = zeros(1, length(topK));
                    list_vecRecall = zeros(1, length(topK));
                    Tlog_likelihood = 0;
                    step_size = 10000;

                    for j = 1:1%ceil(length(test_usr_idx)/step_size)
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
                    if bestVlog_likelihood < Vlog_likelihood
                        best_TestPrecRecall_likelihood = [test_precision, test_recall];
                    end
                    if best_ValidPrecision <= valid_precision(1) && best_TestPrecRecall_precision(1) < test_precision(1)
                        best_TestPrecRecall_precision = [test_precision, test_recall];
                    end
                    best_TestLogLikelihhod = Tlog_likelihood / nnz(matX_test);
                end
                if bestVlog_likelihood < Vlog_likelihood
                    bestVlog_likelihood = Vlog_likelihood;
                end
                if best_ValidPrecision < valid_precision(1)
                    best_ValidPrecision = valid_precision(1);
                end
            end
        end
        
        %
        % Calculate the likelihood to determine when to terminate.
        %      
        if type_model == 3
%             prior_X = (matTheta(usr_idx,:) * matBeta(itm_idx, :)') .* (matX(usr_idx,itm_idx)>0);
%             predict_X = matX_predict(usr_idx, itm_idx) .* (matX(usr_idx,itm_idx)>0);
%             
%             tmp = 0;
%             %predict_X = bsxfun(@times, predict_X, (sum(matX(usr_idx, itm_idx), 2) ./ sum(predict_X,2)));
%             for u = 1:usr_idx_len
%                 u_idx = usr_idx(u);
%                 if nnz(matX(u_idx, itm_idx)) < 2
%                     continue;
%                 end
%                 [is, js, vs] = find(matX(u_idx, itm_idx));
%                 
%                 vec_prior_X_u = prior_X(u, js);
%                 vec_predict_X_u = predict_X(u, js);
%                 vec_matX_u = matX(u_idx, itm_idx(js));
%                 
%                 [v_X, i_X] = sort(vec_matX_u);
%                 
%                 mask = bsxfun(@plus, vec_matX_u', -vec_matX_u);
%                 
%                 % calculate (i, j) = -\hat{s}_{ui} + \hat{x}_{uj}  for f(x): x_{ui} > x_{uj}
%                 predict_X_diff_xui_xuj = bsxfun(@plus, -vec_prior_X_u', vec_prior_X_u);
%                 logsig_diff_predict_xij_f = -log(1 + exp(predict_X_diff_xui_xuj));
%                 logsig_diff_predict_xij_g = -log(1 + exp(-predict_X_diff_xui_xuj));
%                 
%                 tmp = tmp + sum(sum((mask > 0) .* logsig_diff_predict_xij_f)) + sum(sum((mask < 0) .* logsig_diff_predict_xij_g));
%             end
%             obj_func = tmp;
            obj_func = 1 / nnz(matX(usr_idx, :)) * DistributionPoissonLog(matX(usr_idx, :), matTheta(usr_idx,:), matBeta');
        else
            obj_func = 1 / nnz(matX(usr_idx, :)) * DistributionPoissonLog(matX(usr_idx, :), matTheta(usr_idx,:), matBeta');
        end
        list_TrainLogLikelihhod(i) = obj_func;
                
        if mod(i, check_step) == 0
            if mod(i, test_step) == 0
                fprintf('\nVlikelihood: %f Tlikelihood: %f ObjFunc: %f ( Vprecision: %f , Vrecall: %f Tprecision: %f , Trecall: %f )\n',...
                        Vlog_likelihood, Tlog_likelihood, obj_func, valid_precision(1), valid_recall(1), test_precision(1), test_recall(1));
            else
                fprintf('\nVlikelihood: %f ObjFunc: %f ( Vprecision: %f , Vrecall: %f )\n',...
                        Vlog_likelihood, obj_func, valid_precision(1), valid_recall(1));
            end
        else
            fprintf('\nObjFunc: %f \n', obj_func);
        end
        
        if mod(i,1) == 0
            %plot(matTheta(1:50,:));figure(gcf);
            
            %plot(matTheta);figure(gcf);
            %idx_show = 32;
            idx_show = 20;
            tmppp = matTheta(idx_show,:) * matBeta';
            %plot([tmppp(1,matX_predict(idx_show,:)>0); matX_predict(idx_show,matX_predict(idx_show,:)>0); matX(idx_show, matX_predict(idx_show,:)>0)]');figure(gcf);
            plot([tmppp(1,matX_predict(idx_show,:)>0); matX_predict(idx_show,matX_predict(idx_show,:)>0)]');figure(gcf);
            %plot([tmppp(1,matX_predict(idx_show,:)>0); matX_predict(idx_show,matX_predict(idx_show,:)>0)]');figure(gcf);
            
            %bbb = [sort(full(matX(u_idx, matX(u_idx, :)>0))); sort(full(vec_prior_X_u)); sort(full(vec_predict_X_u)); sort(full(solution_xui_xuj))]';
            %bbb = bsxfun(@times, bbb, 1./sum(bbb));
            %bbb = [sort(full(vec_prior_X_u)); sort(full(vec_predict_X_u)); sort(full(solution_xui_xuj))]';
            %plot(bbb)
        end
    end
end