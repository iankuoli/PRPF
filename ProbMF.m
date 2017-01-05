function [matClusteringRes] = ProbMF(k)
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
    
    
    best_TestPrecRecall_precision = zeros(1, length(topK)*2);
    
    if ini == 1
        [M, N] = size(matX); 
        K = k;
        
        matTheta = ini_scale * rand(M, K) + prior;
        matBeta = ini_scale * rand(M, K) + prior;
    end
    
    IsConverge = false;
    i = 0;
    l = 0;
    
    while IsConverge == false && i < MaxItr
        i = i + 1;
        
        %% Update the latent parameters
        %
        % Set the learning rate 
        % ref: Content-based recommendations with Poisson factorization. NIPS, 2014
        %
        offset = 1024;
        lr = (offset + i) ^ -kappa;
        
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
        
        %
        % Update matTheta, matBeta
        %
        matLr_theta = 2 * lambda_theta * matTheta(usr_idx,:);
        matLr_beta = 2 * lambda_beta * matBeta(itm_idx,:);
        
        predict_X = (matTheta(usr_idx,:) * matBeta(itm_idx,:)') .* (matX(usr_idx, itm_idx)>0);
        sigma_predX = spfun(@(x) 1/(1+exp(-x)), predict_X);
        diff_sigma_predX = sigma_predX .* (1 - sigma_predX);
        for u = 1:usr_idx_len
            u_idx = usr_idx(u);
            if nnz(matX(u_idx, itm_idx)) < 2
                continue;
            end
            [is, js, vs] = find(matX(u_idx, itm_idx));
            
            vec_predict_X_u = predict_X(u, js);
            vec_matX_u = matX(u_idx, itm_idx(js));
            
            [val_sort, idx_sort] = sort(vec_matX_u, 'descend');
            
            %
            % Update matTheta
            %
            tmp_theta = 0;
            
            for j = 1:length(vec_matX_u)
                tmp_theta = tmp_theta + ...
                            (diff_sigma_predX(u, idx_sort(j:end)) * matBeta(itm_idx(idx_sort(j:end)),:)') ./ sum(sigma_predX(u, idx_sort(j:end)), 2) - ...
                            diff_sigma_predX(u, idx_sort(j)) ./ sigma_predX(u, idx_sort(j)) * matBeta(itm_idx(idx_sort(j)),:)';
            end
            matLr_theta(u,:) = matLr_theta(u,:) + tmp_theta;

            %
            % Update matBeta
            %
            tmp_beta = 0;
            for l = 1:length(vec_matX_u)
                tmp = 1 ./ sum(sigma_predX(u, idx_sort(l:end)), 2);
                tmp_beta(l:end) = tmp_beta(l:end) + tmp;              
            end
            tmp_beta = tmp_beta .* diff_sigma_predX(u, idx_sort) - ...
                       (diff_sigma_predX(u, idx_sort) ./ sigma_predX(u, idx_sort))' * matTheta(u,:);
            matLr_beta(js(idx_sort),:) = matLr_beta(itm_idxjs(idx_sort),:) + tmp_beta;
        end
        matTheta(usr_idx,:) = matTheta(usr_idx,:) - lr * matLr_theta;
        matBeta(itm_idx,:) = matBeta(itm_idx,:) - N / usr_idx_len * lr * matLr_beta;      
        
        
        
        %% Terminiation Checkout
        fprintf('\nTerminiation Checkout ...');
        %
        % Compute the precision & recall of the validation set.
        %
        if mod(i, test_step) == 0 && test_size > 0
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
            list_TestPrecRecall(i/test_step, :) = [test_precision test_recall];
            Tlog_likelihood = Tlog_likelihood / nnz(matX_test);
        else
            test_precision = 0;
            test_recall = 0;
        end
        
        if mod(i, check_step) == 0
            [valid_usr_idx, valid_itm_idx, valid_val] = find(matX_valid);
            valid_usr_idx = unique(valid_usr_idx);
            list_vecPrecision = zeros(1, length(topK));
            list_vecRecall = zeros(1, length(topK));
            Vlog_likelihood = 0;
            step_size = 10000;
            
            for j = 1:ceil(length(valid_usr_idx)/step_size)
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
            
            if bestVlog_likelihood < Vlog_likelihood || best_ValidPrecision < valid_precision(1)
                if mod(i, test_step) == 0 && test_size > 0
                    if bestVlog_likelihood < Vlog_likelihood
                        best_TestPrecRecall_likelihood = [test_precision, test_recall];
                    end
                    if best_ValidPrecision < valid_precision(1)
                        best_TestPrecRecall_precision = [test_precision, test_recall];
                    end
                else
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
                    if bestVlog_likelihood < Vlog_likelihood
                        best_TestPrecRecall_likelihood = [test_precision, test_recall];
                    end
                    if best_ValidPrecision < valid_precision(1)
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
        obj_func = 1 / nnz(matX(usr_idx, :)) * DistributionPoissonLog(matX(usr_idx, :), matTheta(usr_idx,:), matBeta');
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
        
        if mod(i,80) == 0
            plot(matTheta(1:50,:)');figure(gcf);
            %bbb = [sort(full(matX(u_idx, matX(u_idx, :)>0))); sort(full(vec_prior_X_u)); sort(full(vec_predict_X_u)); sort(full(solution_xui_xuj))]';
            %bbb = bsxfun(@times, bbb, 1./sum(bbb));
            %bbb = [sort(full(vec_prior_X_u)); sort(full(vec_predict_X_u)); sort(full(solution_xui_xuj))]';
            %plot(bbb)
        end
    end
    
end