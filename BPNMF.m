function [matClusteringRes] = BPNMF(k, prior, ini_scale, usr_batch_size, kappa, topK, test_size, test_step, ini, MaxItr, check_step)
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

    global matTheta_Shp      % dim(M, K): varational param of matTheta 

    global matBeta_Shp       % dim(N, K): varational param of matBeta (shape)
    global matBeta_Rte       % dim(N, K): varational param of matBeta (rate)

    global tensorPhi         % dim(K, M, N) = cell{dim(M,N)}: varational param of matX
    
    global bestTheta
    global bestBeta_Shp
    global bestBeta_Rte
    
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
    
    if ini == 0
        
        [M, N] = size(matX); 
        
        usr_zeros = sum(matX, 2)==0;
        itm_zeros = sum(matX, 1)==0;

        K = k;    
        
        %offset = 0.1;

        matBeta_Shp = ini_scale * rand(N, K) + b;
        matBeta_Rte = ini_scale * rand(N, K) + b;
        matBeta_Shp(itm_zeros, :) = 0;
        matBeta_Rte(itm_zeros, :) = 0;

        matTheta_Shp = ini_scale * rand(M, K) + a;
        matTheta_Shp(usr_zeros,:) = 0;
        
        tensorSigma = cell(K, 1);
        for i = 1:K
            tensorSigma{i} = sparse(N, N);
        end
        
        zero_idx_usr = sum(matX,2)==0;
        matTheta_Shp(zero_idx_usr(:,1),:) = 0;
        zero_idx_itm = sum(matX,1)==0;
        matBeta_Shp(zero_idx_itm(:,1),:) = 0;
        matBeta(zero_idx_itm(:,1),:) = 0;
    end

    
    %% CTPF Coordinate ascent
    
    IsConverge = false;
    i = ini;
    l = 0;
    
    
    %[x_i, x_j, x_v] = find(matX);
    R = 4;
    [xxx, yyy, val] = find(matX);
    max_val = max(max(val));
    min_val = min(min(val));
    matR_star = spfun(@(x) (x-min_val) / (max_val - min_val), matX);
    clear xxx;
    clear yyy;
        
    
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
        if usr_batch_size == M
            usr_idx = 1:M;
            itm_idx = 1:N;
            usr_idx(usr_zeros) = [];
            itm_idx(itm_zeros) = [];
            usr_idx_len = length(usr_idx);
            itm_idx_len = length(itm_idx);
        else
            if usr_batch_size == size(matX,1)
                usr_idx = 1:size(matX,1);
            else
                usr_idx = randsample(size(matX,1), usr_batch_size);
                usr_idx(sum(matX(usr_idx,:),2)==0) = [];
            end

            itm_idx = find(sum(matX(usr_idx, :))>0);
            usr_idx_len = length(usr_idx);
            itm_idx_len = length(itm_idx);
        end
        
        fprintf('\nIndex: %d  ---------------------------------- %d , %d , lr: %f \n', i, usr_idx_len, itm_idx_len, lr);
        
        matTheta_Shp_psi = psi(matTheta_Shp(usr_idx,:));
        matBeta_Shp_psi = psi(matBeta_Shp(itm_idx,:));
        matBeta_Rte_psi = psi(matBeta_Rte(itm_idx,:));
        
        
        %% Update Latent Tensor Variables
        
        %
        % Update tensorPhi
        %
        fprintf('Update tensorPhi ...  k = ');
        matX_One = matX(usr_idx, itm_idx) > 0;

%         tensorPhi = cellfun(@(x,y) spfun(@exp, x+y), cellfun(@(x) bsxfun(@times, matX_One, x), num2cell(matTheta_Shp_psi, 1), 'UniformOutput', false), ...
%                                                      cellfun(@(x) bsxfun(@times, matX_One, x'), num2cell(R * matR_star * matBeta_Shp_psi + ...
%                                                                                                          R * (1-matR_star) * matBeta_Rte_psi - ...
%                                                                                                          R * psi(matBeta_Shp(itm_idx,:) + matBeta_Rte(itm_idx,:)), 1), 'UniformOutput', false), 'UniformOutput', false);
                                                                                                   
        tensorPhi = cellfun(@(w,x,y,z) spfun(@exp, w+R*(x+y-z)), cellfun(@(x) bsxfun(@times, ones(usr_idx_len,itm_idx_len), x), num2cell(matTheta_Shp_psi, 1), 'UniformOutput', false), ...
                                                                 cellfun(@(x) bsxfun(@times, matR_star(usr_idx,itm_idx) .* ones(usr_idx_len,itm_idx_len), x'), num2cell(matBeta_Shp_psi, 1), 'UniformOutput', false), ...
                                                                 cellfun(@(x) bsxfun(@times, (1-matR_star(usr_idx,itm_idx)) .* ones(usr_idx_len,itm_idx_len), x'), num2cell(matBeta_Rte_psi, 1), 'UniformOutput', false), ...
                                                                 cellfun(@(x) bsxfun(@times, ones(usr_idx_len,itm_idx_len), x'), num2cell(psi(matBeta_Shp(itm_idx,:) + matBeta_Rte(itm_idx,:)), 1), 'UniformOutput', false), 'UniformOutput', false);

        tmp = cellfun(@(x) reshape(x, [], 1), tensorPhi, 'UniformOutput', false);
        tensorPhi_inv = reshape(spfun(@(x) 1 ./ x, sum([tmp{:}],2)), usr_idx_len, itm_idx_len);
        tensorPhi = cellfun(@(x) x .* tensorPhi_inv, tensorPhi,  'UniformOutput', false);
        
        
        %% Update Latent Matrix Variables
        fprintf('\nUpdate Latent Matrix Variables ...');
        
        %
        % Update matTheta_Shp
        %              
        matTheta_Shp(usr_idx, :) = (1 - lr) * matTheta_Shp(usr_idx, :) +...
                                    lr * (a + cell2mat(cellfun(@(x) sum(matX_One .* x, 2), tensorPhi, 'UniformOutput', false)));
                                
        if any(isnan(matTheta_Shp))>0
            fprintf('NaN');
        end
        
        
        %
        % Update matBeta_Shp , matBeta_Rte
        %
        if usr_batch_size == M
            scale1 = ones(length(itm_idx), 1);
        else
            scale1 = sum(matX(:, itm_idx) > 0, 1)' ./ sum(matX(usr_idx, itm_idx) > 0, 1)';
        end
          
        matBeta_Shp(itm_idx, :) = (1 - lr) * matBeta_Shp(itm_idx, :) + ...
                                  lr * (b + R * bsxfun(@times, cell2mat(cellfun(@(x) sum(matR_star(usr_idx, itm_idx) .* matX_One .* x, 1)', tensorPhi, 'UniformOutput', false)), scale1));
        matBeta_Rte(itm_idx, :) = (1 - lr) * matBeta_Shp(itm_idx, :) + ...
                                  lr * (b + R * bsxfun(@times, cell2mat(cellfun(@(x) sum((1 - matR_star(usr_idx, itm_idx)) .* matX_One .* x, 1)', tensorPhi, 'UniformOutput', false)), scale1));        
        
        if any(isnan(matBeta))>0
            fprintf('NaN');
        end
        
        
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
            step_size = 3000;
            
            for j = 1:3%ceil(length(test_usr_idx)/step_size)
                range_step = (1 + (j-1) * step_size):min(j*step_size, length(test_usr_idx));
                
                % Compute the Precision and Recall
                matA = bsxfun(@times, matTheta_Shp(test_usr_idx(range_step),:), 1./sum(matTheta_Shp(test_usr_idx(range_step),:),2));
                matB = matBeta_Shp ./ (matBeta_Shp + matBeta_Rte);
                test_matPredict = matA * matB';
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
            step_size = 3000;
                   
            for j = 1:3%ceil(length(valid_usr_idx)/step_size)
                range_step = (1 + (j-1) * step_size):min(j*step_size, length(valid_usr_idx));
                
                % Compute the Precision and Recall
                matA = bsxfun(@times, matTheta_Shp(valid_usr_idx(range_step),:), 1./sum(matTheta_Shp(valid_usr_idx(range_step),:),2));
                matB = matBeta_Shp ./ (matBeta_Shp + matBeta_Rte);
                valid_matPredict = matA * matB';
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
                        bestTheta = matTheta_Shp;
                        bestBeta_Shp = matBeta_Shp;
                        bestBeta_Rte = matBeta_Rte;
                    end
                else
                    [test_usr_idx, test_itm_idx, test_val] = find(matX_test);
                    test_usr_idx = unique(test_usr_idx);
                    list_vecPrecision = zeros(1, length(topK));
                    list_vecRecall = zeros(1, length(topK));
                    Tlog_likelihood = 0;
                    step_size = 3000;

                    for j = 1:3%ceil(length(test_usr_idx)/step_size)
                        range_step = (1 + (j-1) * step_size):min(j*step_size, length(test_usr_idx));

                        % Compute the Precision and Recall
                        matA = bsxfun(@times, matTheta_Shp(test_usr_idx(range_step),:), 1./sum(matTheta_Shp(test_usr_idx(range_step),:),2));
                        matB = matBeta_Shp ./ (matBeta_Shp + matBeta_Rte);
                        test_matPredict = matA * matB';
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
                        bestTheta = matTheta_Shp;
                        bestBeta_Shp = matBeta_Shp;
                        bestBeta_Rte = matBeta_Rte;
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
        matA = bsxfun(@times, matTheta_Shp(usr_idx,:), 1./sum(matTheta_Shp(usr_idx,:),2));
        matB = matBeta_Shp ./ (matBeta_Shp + matBeta_Rte);
        obj_func = 1 / nnz(matX(usr_idx, :)) * DistributionPoissonLog(matX(usr_idx, :), matA, matB');
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
        
        if mod(i,10) == 0
            %plot(matTheta(1:50,:));figure(gcf);
            
            %plot(matTheta);figure(gcf);
            idx_show = 32;
            %idx_show = 220;
            tmppp = matTheta_Shp(idx_show,:) / sum(matTheta_Shp(idx_show,:)) * matB';
            plot([tmppp(1,matX_predict(idx_show,:)>0)]');figure(gcf);
            %plot([tmppp(1,matX_predict(idx_show,:)>0); matX_predict(idx_show,matX_predict(idx_show,:)>0)]');figure(gcf);
            
            %bbb = [sort(full(matX(u_idx, matX(u_idx, :)>0))); sort(full(vec_prior_X_u)); sort(full(vec_matX(usr_idx, itm_idx)_u)); sort(full(solution_xui_xuj))]';
            %bbb = bsxfun(@times, bbb, 1./sum(bbb));
            %bbb = [sort(full(vec_prior_X_u)); sort(full(vec_matX(usr_idx, itm_idx)_u)); sort(full(solution_xui_xuj))]';
            %plot(bbb)
        end
    end
    
    [test_usr_idx, test_itm_idx, test_val] = find(matX_test);
    test_usr_idx = unique(test_usr_idx);
    list_vecPrecision = zeros(1, length(topK));
    list_vecRecall = zeros(1, length(topK));
    vecTlog_likelihood = ones(1, ceil(length(test_usr_idx)/step_size));
    step_size = 3000;

    for j = 1:ceil(length(test_usr_idx)/step_size)
        range_step = (1 + (j-1) * step_size):min(j*step_size, length(test_usr_idx));

        % Compute the Precision and Recall
        matA = bsxfun(@times, matTheta_Shp(test_usr_idx(range_step),:), 1./sum(matTheta_Shp(test_usr_idx(range_step),:),2));
        matB = matBeta_Shp ./ (matBeta_Shp + matBeta_Rte);
        test_matPredict = matA * matB';
        test_matPredict = test_matPredict - test_matPredict .* (matX(test_usr_idx(range_step), :) > 0);
        [vec_precision, vec_recall] = MeasurePrecisionRecall(matX_test(test_usr_idx(range_step), :), test_matPredict, topK);
        list_vecPrecision = list_vecPrecision + sum(vec_precision, 1);
        list_vecRecall = list_vecRecall + sum(vec_recall, 1);

        % Compute the log likelihood
        vecTlog_likelihood(1,j) = DistributionPoissonLogNZ(matX_test(test_usr_idx(range_step), :), test_matPredict);
    end
    Tlog_likelihood = sum(vecTlog_likelihood);

    test_precision = list_vecPrecision / length(test_usr_idx);
    test_recall = list_vecRecall / length(test_usr_idx);
    if bestVlog_likelihood < Vlog_likelihood
        best_TestPrecRecall_likelihood = [test_precision, test_recall];
    end
    if best_ValidPrecision <= valid_precision(1) && best_TestPrecRecall_precision(1) < test_precision(1)
        best_TestPrecRecall_precision = [test_precision, test_recall];
    end
end