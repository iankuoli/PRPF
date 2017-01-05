function [matClusteringRes] = LogMF(k, prior, ini_scale, usr_batch_size, gamma, lambda, alpha, topK, test_size, test_step, ini, MaxItr, check_step)
    %
    % Logistic Matrix Factorization for Implicit Feedback Data
    % In NIPS, 2014.
    %
    
    %% Parameters declaration
    
    global K                 % number of topics
    global M                 % number of users
    global N                 % number of items
    
    global matX              % dim(M, N): consuming records for training
    global matX_test         % dim(M, N): consuming records for testing
    global matX_valid        % dim(M, N): consuming records for validation
    
    global matTheta          % dim(M, K): latent document-topic intensities
    global matBeta           % dim(N, K): latent word-topic intensities
    global vecBiasU
    global vecBiasI

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
        
        matTheta = ini_scale * rand(M, K) + prior(1);
        vecBiasU = ini_scale * rand(M, 1) + prior(1);
        matBeta = ini_scale * rand(N, K) + prior(2);
        vecBiasI = ini_scale * rand(N, 1) + prior(1);
    end
    
    IsConverge = false;
    i = 0;
    l = 0;
    
    grad_sqr_sum_theta = ones(M, K);
    grad_sqr_sum_biasU = ones(M, 1);
    grad_sqr_sum_beta = ones(N, K);
    grad_sqr_sum_biasI = ones(N, 1);
    
    while IsConverge == false && i < MaxItr
        i = i + 1;
        
        %% Update the latent parameters
        
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
        
        fprintf('\nIndex: %d  ---------------------------------- %d , %d , lr: %f \n', i, usr_idx_len, itm_idx_len, gamma);
        
        %
        % Update matBeta & vecBiasU
        %
        predict_X = bsxfun(@plus, bsxfun(@plus, matTheta(usr_idx,:) * matBeta(itm_idx,:)', vecBiasU(usr_idx)), vecBiasI(itm_idx)');
        %predict_X = matTheta(usr_idx,:) * matBeta(itm_idx,:)';
        matTmp = exp(predict_X);
        matTmp = matTmp ./ (1 + matTmp);
        matTmp(isnan(matTmp)) = 1;
        matTmp = (1 + alpha * matX(usr_idx, itm_idx)) .* matTmp;
        matTmp = alpha * matX(usr_idx, itm_idx) - matTmp;
        
        % Update matBeta
        partial_beta = matTmp' * matTheta(usr_idx, :) - 0.5 * lambda * matBeta(itm_idx,:);
        matBeta(itm_idx,:) = matBeta(itm_idx,:) + gamma * (partial_beta);% ./ sqrt(grad_sqr_sum_beta(itm_idx,:)));
        grad_sqr_sum_beta(itm_idx,:) = grad_sqr_sum_beta(itm_idx,:) + partial_beta .^ 2;

        % Update vecBiasU
        partial_biasI = sum(matTmp, 1)';
        vecBiasI(itm_idx) = vecBiasI(itm_idx) + gamma * (partial_biasI);% ./ sqrt(grad_sqr_sum_biasI(itm_idx)));
        grad_sqr_sum_biasI(itm_idx) = grad_sqr_sum_biasI(itm_idx) + partial_biasI .^ 2;
        
        %
        % Update matTheta & vecBiasU
        %
        predict_X = bsxfun(@plus, bsxfun(@plus, matTheta(usr_idx,:) * matBeta(itm_idx,:)', vecBiasU(usr_idx)), vecBiasI(itm_idx)');
        matTmp = exp(predict_X);
        matTmp = matTmp ./ (1 + matTmp);
        matTmp(isnan(matTmp)) = 1;
        matTmp = (1 + alpha * matX(usr_idx, itm_idx)) .* matTmp;
        matTmp = alpha * matX(usr_idx, itm_idx) - matTmp;
        
        % Update matTheta    
        partial_theta = matTmp * matBeta(itm_idx, :) - 0.5 * lambda * matTheta(usr_idx,:);
        matTheta(usr_idx,:) = matTheta(usr_idx,:) + gamma * (partial_theta);% ./ sqrt(grad_sqr_sum_theta(usr_idx,:)));
        grad_sqr_sum_theta(usr_idx,:) = grad_sqr_sum_theta(usr_idx,:) + partial_theta .^ 2;
        
        % Update vecBiasU
        partial_biasU = sum(matTmp, 2);
        vecBiasU(usr_idx) = vecBiasU(usr_idx) + gamma * (partial_biasU);% ./ sqrt(grad_sqr_sum_biasU(usr_idx)));
        grad_sqr_sum_biasU(usr_idx) = grad_sqr_sum_biasU(usr_idx) + partial_biasU .^ 2;
        
        
        
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
                test_matPredict = bsxfun(@plus, bsxfun(@plus, matTheta(test_usr_idx(range_step),:) * matBeta', vecBiasU(test_usr_idx(range_step))), vecBiasI');
                %test_matPredict = matTheta(test_usr_idx(range_step),:) * matBeta';
                test_matPredict = test_matPredict - test_matPredict .* (matX(test_usr_idx(range_step), :) > 0);
                [vec_precision, vec_recall] = MeasurePrecisionRecall(matX_test(test_usr_idx(range_step), :), test_matPredict, topK);
                list_vecPrecision = list_vecPrecision + sum(vec_precision, 1);
                list_vecRecall = list_vecRecall + sum(vec_recall, 1);
                
                % Compute the log likelihood
                tmp = alpha * sum(sum(matX_test(test_usr_idx(range_step), :) .* test_matPredict)) - ...
                      sum(sum((1 + alpha * matX_test(test_usr_idx(range_step), :)) .* log(1 + exp(test_matPredict)))) - ...
                      0.5 * lambda * (vecBiasU((test_usr_idx(range_step)))'*vecBiasU((test_usr_idx(range_step))) + vecBiasI'*vecBiasI);
                Tlog_likelihood = Tlog_likelihood + tmp;
            end
            
            test_precision = list_vecPrecision / length(test_usr_idx);
            test_recall = list_vecRecall / length(test_usr_idx);
            list_TestPrecRecall(i/test_step, :) = [test_precision test_recall];
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
            
            for j = 1:1%ceil(length(valid_usr_idx)/step_size)
                range_step = (1 + (j-1) * step_size):min(j*step_size, length(valid_usr_idx));
                
                % Compute the Precision and Recall
                valid_matPredict = bsxfun(@plus, bsxfun(@plus, matTheta(valid_usr_idx(range_step),:) * matBeta', vecBiasU(valid_usr_idx(range_step))), vecBiasI');
                %valid_matPredict = matTheta(valid_usr_idx(range_step),:) * matBeta';
                valid_matPredict = valid_matPredict - valid_matPredict .* (matX(valid_usr_idx(range_step), :) > 0);
                [vec_precision, vec_recall] = MeasurePrecisionRecall(matX_valid(valid_usr_idx(range_step), :), valid_matPredict, topK);
                list_vecPrecision = list_vecPrecision + sum(vec_precision, 1);
                list_vecRecall = list_vecRecall + sum(vec_recall, 1);
                
                % Compute the log likelihood
                tmp = alpha * sum(sum(matX_test(valid_usr_idx(range_step), :) .* valid_matPredict)) - ...
                      sum(sum((1 + alpha * matX_test(valid_usr_idx(range_step), :)) .* log(1 + exp(valid_matPredict)))) - ...
                      0.5 * lambda * (vecBiasU((valid_usr_idx(range_step)))'*vecBiasU((valid_usr_idx(range_step))) + vecBiasI'*vecBiasI);
                Vlog_likelihood = Vlog_likelihood + tmp;
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
                        test_matPredict = bsxfun(@plus, bsxfun(@plus, matTheta(test_usr_idx(range_step),:) * matBeta', vecBiasU(test_usr_idx(range_step))), vecBiasI');                        
                        %test_matPredict = matTheta(test_usr_idx(range_step),:) * matBeta';
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
        
        %predict_X = bsxfun(@plus, bsxfun(@plus, matTheta(usr_idx,:) * matBeta(itm_idx,:)', vecBiasU(usr_idx)), vecBiasI(itm_idx)');
        predict_X = matTheta(usr_idx,:) * matBeta(itm_idx,:)';
        obj_func = alpha * sum(sum(matX(usr_idx, itm_idx) .* predict_X)) - ...
                   sum(sum((1 + alpha * matX(usr_idx, itm_idx)) .* log(1 + exp(predict_X)))) - ...
                   0.5 * lambda * (vecBiasU(usr_idx)'*vecBiasU(usr_idx) + vecBiasI(itm_idx)'*vecBiasI(itm_idx));     
        
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
            fprintf('\nObjFunc: %f   lr: %f \n', obj_func, sum(sum(partial_theta ./ sqrt(grad_sqr_sum_theta(usr_idx,:)))));
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