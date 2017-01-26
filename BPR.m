function [matClusteringRes] = BPR(k, lr, lambda, prior, ini_scale, topK, test_size, test_step, Itr_step, MaxItr, check_step)
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
    
    list_ValidPrecRecall = zeros(MaxItr/Itr_step, length(topK)*2);
    list_TestPrecRecall = zeros(MaxItr/Itr_step, length(topK)*2);
    list_ValidLogLikelihhod = zeros(MaxItr/(check_step*Itr_step), 1);
    list_TrainLogLikelihhod = zeros(MaxItr/Itr_step, 1);
    best_TestPrecRecall_precision = zeros(1, length(topK)*2);
    best_TestPrecRecall_likelihood = zeros(1, length(topK)*2);
    best_TestLogLikelihhod = -Inf;
    bestVlog_likelihood = -Inf;
    best_ValidPrecision = 0;

    prior_theta = prior(1);
    prior_beta = prior(2);
    
    matTheta = ini_scale * rand(M, K) - 0.5 * ini_scale;% + prior_theta;
    matBeta = ini_scale * rand(N, K) - 0.5 * ini_scale;% + prior_beta;

    zero_idx_usr = sum(matX,2)==0;
    matTheta(zero_idx_usr(:,1),:) = 0;
    zero_idx_itm = sum(matX,1)==0;
    matBeta(zero_idx_itm(:,1),:) = 0;

    IsConverge = false;
    ii = 0;
    K = k;

    while IsConverge == false && ii < MaxItr
        ii = ii + 1;
        
        %
        % Sample data
        %
        u_idx = randsample(size(matX,1), 1);
        if nnz(matX(u_idx,:)) == 0
            continue;
        end
        i_idx = randsample(find(matX(u_idx, :)>0),1);
        j_idx = randsample(find(matX(u_idx, :)==0),1);
        
        x_cap_uij = matTheta(u_idx,:) * (matBeta(i_idx,:) - matBeta(j_idx, :))';
        tmp_uij = 1 / (1 + exp(x_cap_uij));
        tmp_uij(isnan(tmp_uij))=1;
        
        %
        % Update matTheta
        %
        matTheta(u_idx,:) = matTheta(u_idx,:) + lr * (tmp_uij * (matBeta(i_idx,:) - matBeta(j_idx,:)) - lambda * matTheta(u_idx,:));
        if sum(sum(isnan(matTheta)))>0
            fprintf('\nFUCKKKKKKKKKKK ...');
        end

        %
        % Update matBeta
        %
        matBeta(i_idx,:) = matBeta(i_idx,:) + lr * (tmp_uij * matTheta(u_idx,:) - lambda * matBeta(i_idx,:));
        matBeta(j_idx,:) = matBeta(j_idx,:) + lr * (tmp_uij * (-matTheta(u_idx,:)) - lambda * matBeta(j_idx,:));
        if sum(sum(isnan(matBeta)))>0
            fprintf('\nFUCKKKKKKKKKKK ...');
        end
        
        
        if mod(ii, Itr_step) ~=0
            continue;
        end
        
        %% Terminiation Checkout
        fprintf('\n%d Terminiation Checkout ...', ii/Itr_step);
        l = inf;
        %
        % Compute the precision & recall of the testing set.
        %
        i = ii/Itr_step;
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
        obj_func = 1 / nnz(matX(u_idx, :)) * sum(sum(((matX(u_idx, :)>0) - ((matTheta(u_idx,:) * matBeta') > 0)).^2));
        DistributionGaussianLog(matX(u_idx, :)>0, matTheta(u_idx,:), matBeta');
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
        
        if mod(i,50) == 0
            plot(matTheta(1:50,:)');figure(gcf);
            %bbb = [sort(full(matX(u_idx, matX(u_idx, :)>0))); sort(full(vec_prior_X_u)); sort(full(vec_predict_X_u)); sort(full(solution_xui_xuj))]';
            %bbb = bsxfun(@times, bbb, 1./sum(bbb));
            %bbb = [sort(full(vec_prior_X_u)); sort(full(vec_predict_X_u)); sort(full(solution_xui_xuj))]';
            %plot(bbb)
        end
    end
%         %
%         % Sample data
%         %
%         if usr_batch_size == size(matX,1)
%             usr_idx = 1:size(matX,1);
%         else
%             usr_idx = randsample(size(matX,1), usr_batch_size);
%             usr_idx(sum(matX(usr_idx,:),2)==0) = [];
%         end
%         
%         itm_idx = find(sum(matX(usr_idx, :))>0);
%         usr_idx_len = length(usr_idx);
%         itm_idx_len = length(itm_idx);
%         
%         fprintf('\nIndex: %d  ---------------------------------- %d , %d , lr: %f \n', i, usr_idx_len, itm_idx_len, lr);
%         
% 
%         for u = 1:usr_idx_len
%             u_idx = usr_idx(u);
%             if nnz(matX(u_idx, itm_idx)) < 2
%                 continue;
%             end
%             
%             vec_predict_X_u = matTheta(u_idx,:) * matBeta';
%             vec_matX_u = matX(u_idx, :);
%             mask_x_uij = bsxfun(@plus, vec_matX_u', -vec_matX_u)>0;
%             x_cap_uij = bsxfun(@plus, vec_predict_X_u', -vec_predict_X_u);
%             exp_x_uij = exp(-x_cap_uij);
%             tmp_x_uij = exp_x_uij ./ (1 + exp_x_uij);
%             
%             update_theta = ones(1,K);
%             for k = 1:K
%                 partial__x_uij__theta_u = bsxfun(@plus, matBeta(:,k), -matBeta(:,k)') .* mask_x_uij;
%                 update_theta(1,k) = sum(sum(partial__x_uij__theta_u .* tmp_x_uij));
%                 partial__x_uij__beta_i = 
%             end
%             
%             %
%             % Update matTheta
%             %
%             matTheta(u_idx,:) = matTheta(u_idx,:) + lr * update_theta;
% 
%             %
%             % Update matBeta
%             %
%         end
        
end