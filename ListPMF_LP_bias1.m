function ListPMF_LP_bias1(k, lambda, lambda_Theta, lambda_Beta, lambda_B, topK, test_step, ini, MaxItr, check_step)
    %% Parameters declaration
    
    global K                 % number of topics
    global M                 % number of users
    global N                 % number of items
    
    global matX              % dim(M, N): consuming records for training
    global matX_test         % dim(M, N): consuming records for testing
    global matX_valid        % dim(M, N): consuming records for validation
    
    global matTheta          % dim(M, K): latent document-topic intensities
    global matBeta           % dim(N, K): latent word-topic intensities
    global vecBias
      
    global list_ValidPrecRecall
    global list_TestPrecRecall
    global list_TrainLogLikelihhod
    global best_TestPrecRecall_precision
    global best_TestPrecRecall_likelihood
    
    list_ValidPrecRecall = zeros(MaxItr/check_step, length(topK)*2);
    list_TestPrecRecall = zeros(MaxItr/test_step, length(topK)*2);
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
        
%         lambda   = 0.01; % learning rate
%         lambda_Theta = 0.001;
%         lambda_Beta = 0.001;
%         lambda_B = 0.001;

        epoch = 1; 
        maxepoch = MaxItr; 

        %fprintf(1, 'Loading data ......');
        %load([dataset,'.mat']);
        %fprintf(1, 'Finish!\n');

        %pairs_tr = length(train_vec); % training data 
        %pairs_pr = length(probe_vec); % validation data 

        matBeta  = 0.01 * randn(N, K); % Movie feature vectors
        matTheta = 0.01 * randn(M, K); % User feature vecators
        vecBias  = 0.01 * randn(M, 1); % bias vector

        best_train_err = inf;

        train_err = zeros(1, maxepoch);
        %test_NDCG = [];
        %test_ERRv = [];
    end

    train_PI = cell(M);
    for i = 1:M
    	[idx_x, idx_y, val] = find(matX(i,:));
        [sort_val, sort_idx] = sort(val, 'descend');
        train_PI{i} = idx_y(sort_idx);
    end
    
    for epoch = epoch:maxepoch

        t1 = cputime;    
	
        %        
        % gradient    
        %
        d_P = zeros(M,K);
        d_B = zeros(M,1);
        d_M = zeros(N,K);
        for i = 1:M
            p_s = 0;
            d_p_w = zeros(1,K);
            d_m_w = zeros(1,K);
            d_p_b = 0;

            matTheta_m = repmat(matTheta(i,:), [numel(train_PI{i}),1]);
            matBeta_m = matBeta(train_PI{i},:);
            e_p_M = exp(-sum(matTheta_m .* matBeta_m, 2) - vecBias(i));
            %e_p_M = exp(-sum(matTheta_m .* matBeta_m,2));
            f_m = zeros(numel(train_PI{i}), 1);
            
            for j = numel(train_PI{i}):-1:1
                item_id = train_PI{i}(j);
                p_s = p_s + 1 / (1+e_p_M(j));
                
                % for user matrix U
                d_p_w = d_p_w + e_p_M(j) * matBeta(item_id,:) / ((1 + e_p_M(j)) ^ 2);
                d_P(i,:) = d_P(i,:) + e_p_M(j) * matBeta(item_id,:) / (1+e_p_M(j)) - d_p_w / p_s;

                %for bias vector B
                d_p_b = d_p_b + e_p_M(j) / ((1 + e_p_M(j)) ^ 2);
                d_B(i) = d_B(i) + e_p_M(j) / (1 + e_p_M(j)) - d_p_b / p_s;

                % for item matrix V
                for k = j:numel(train_PI{i})
                   f_m(k) = f_m(k) - e_p_M(k) / ((1 + e_p_M(k)) ^ 2) / p_s;
                end
                f_m(j) = f_m(j) + e_p_M(j) / (1 + e_p_M(j));

                % for target function
                train_err(epoch) = train_err(epoch) - log(1 + e_p_M(j)) - log(p_s);

            end
            
            d_M(train_PI{i},:) = d_M(train_PI{i},:) + matTheta_m .* repmat(f_m,[1,K]);
        end

        train_err(epoch) = -train_err(epoch) + lambda_Theta * sum(sum(matTheta.^2)) + lambda_Beta * sum(sum(matBeta.^2)) + lambda_B * sum(vecBias.^2);
        list_TrainLogLikelihhod(epoch) = train_err(epoch);
        obj_func = train_err(epoch);

        %pred_out = sum(matTheta(probe_vec(:,1),:) .* matBeta(probe_vec(:,2),:),2) + vecBias(probe_vec(:,1));
        %test_NDCG(epoch)=NDCGAtN(M,pred_out,probe_vec,10);
        %test_ERR(epoch)=ERRAtN(M,pred_out,probe_vec,10);
        %fprintf(1, 'epoch %4i:\ttrain error %6.4f;\tNDCG@10 %6.4f\tERR@10 %6.4f\n', epoch-1,  train_err(epoch),test_NDCG(epoch),test_ERR(epoch));
        fprintf(1, 'epoch %4i:\ttrain error %6.4f\n', epoch-1,  train_err(epoch));

        if(best_train_err - train_err(epoch) > 1e-5)
            best_train_err = train_err(epoch);
        else
            fprintf(1, 'termilate at epoch %4i.\n', epoch-1);
            break
        end

        %% update
        matTheta = (1 - 2 * lambda * lambda_Theta) * matTheta + lambda * d_P;
        matBeta  = (1 - 2 * lambda * lambda_Beta) * matBeta + lambda * d_M;
        vecBias  = (1 - 2 * lambda * lambda_B) * vecBias + lambda * d_B;


        %% Terminiation Checkout
        fprintf('\nTerminiation Checkout ...');
        
        %
        % Compute the precision & recall of the testing set.
        %
        if mod(epoch, test_step) == 0 && test_step > 0
            [test_usr_idx, test_itm_idx, test_val] = find(matX_test);
            test_usr_idx = unique(test_usr_idx);
            list_vecPrecision = zeros(1, length(topK));
            list_vecRecall = zeros(1, length(topK));
            Tlog_likelihood = 0;
            step_size = 3000;
            %veclist_vecPrecision = zeros(1, min(4, ceil(length(test_usr_idx)/step_size)));
            %veclist_vecRecall = zeros(1, min(4, ceil(length(test_usr_idx)/step_size)));
            
            parfor j = 1:min(4, ceil(length(test_usr_idx)/step_size))
                range_step = (1 + (j-1) * step_size):min(j*step_size, length(test_usr_idx));
                
                % Compute the Precision and Recall
                test_matPredict = bsxfun(@plus, matTheta(test_usr_idx(range_step),:) * matBeta', vecBias(test_usr_idx(range_step),:));
                test_matPredict = test_matPredict - test_matPredict .* (matX(test_usr_idx(range_step), :) > 0);
                [vec_precision, vec_recall] = MeasurePrecisionRecall(matX_test(test_usr_idx(range_step), :), test_matPredict, topK);
                
                %list_vecPrecision = list_vecPrecision + sum(vec_precision, 1);
                %list_vecRecall = list_vecRecall + sum(vec_recall, 1);
                %Tlog_likelihood = Tlog_likelihood + DistributionPoissonLogNZ(matX_test(test_usr_idx(range_step), :), test_matPredict);
                
                veclist_vecPrecision(j) = sum(vec_precision, 1);
                veclist_vecRecall(j) = sum(vec_recall, 1);
            end
            
            %test_precision = list_vecPrecision / length(test_usr_idx);
            %test_recall = list_vecRecall / length(test_usr_idx);
            
            test_precision = sum(veclist_vecPrecision) / length(test_usr_idx);
            test_recall = sum(veclist_vecRecall) / length(test_usr_idx);
                    
            list_TestPrecRecall(epoch/test_step, :) = [test_precision test_recall];
            
        else
            test_precision = 0;
            test_recall = 0;
        end
        
        %
        % Compute the precision & recall of the validation set.
        %
        if mod(epoch, check_step) == 0
            [valid_usr_idx, valid_itm_idx, valid_val] = find(matX_valid);
            valid_usr_idx = unique(valid_usr_idx);
            list_vecPrecision = zeros(1, length(topK));
            list_vecRecall = zeros(1, length(topK));
            Vlog_likelihood = 0;
            step_size = 3000;
            %veclist_vecPrecision = zeros(1, min(4, ceil(length(valid_usr_idx)/step_size)));
            %veclist_vecRecall = zeros(1, min(4, ceil(length(valid_usr_idx)/step_size)));
            
            parfor j = 1:min(4, ceil(length(valid_usr_idx)/step_size))
                range_step = (1 + (j-1) * step_size):min(j*step_size, length(valid_usr_idx));
                
                % Compute the Precision and Recall
                valid_matPredict = bsxfun(@plus, matTheta(valid_usr_idx(range_step),:) * matBeta', vecBias(valid_usr_idx(range_step),:));
                valid_matPredict = valid_matPredict - valid_matPredict .* (matX(valid_usr_idx(range_step), :) > 0);
                [vec_precision, vec_recall] = MeasurePrecisionRecall(matX_valid(valid_usr_idx(range_step), :), valid_matPredict, topK);
                
                %list_vecPrecision = list_vecPrecision + sum(vec_precision, 1);
                %list_vecRecall = list_vecRecall + sum(vec_recall, 1);
                
                veclist_vecPrecision(j) = sum(vec_precision, 1);
                veclist_vecRecall(j) = sum(vec_recall, 1);
            end
            %valid_precision = list_vecPrecision / length(valid_usr_idx);
            %valid_recall = list_vecRecall / length(valid_usr_idx);
            
            valid_precision = sum(veclist_vecPrecision) / length(valid_usr_idx);
            valid_recall = sum(veclist_vecRecall) / length(valid_usr_idx);
            
            list_ValidPrecRecall(epoch, :) = [valid_precision valid_recall];
            list_ValidLogLikelihhod(epoch) = Vlog_likelihood;
            new_l = Vlog_likelihood;
            
            if bestVlog_likelihood < Vlog_likelihood || best_ValidPrecision <= valid_precision(1)
                if mod(epoch, test_step) == 0 && test_step > 0
                    if bestVlog_likelihood < Vlog_likelihood
                        best_TestPrecRecall_likelihood = [test_precision, test_recall];
                    end
                    if best_ValidPrecision <= valid_precision(1) && best_TestPrecRecall_precision(1) < test_precision(1)
                        best_TestPrecRecall_precision = [test_precision, test_recall];
                        bestTheta = matTheta;
                        bestBeta = matBeta;
                    end
                else
                    [test_usr_idx, test_itm_idx, test_val] = find(matX_test);
                    test_usr_idx = unique(test_usr_idx);
                    list_vecPrecision = zeros(1, length(topK));
                    list_vecRecall = zeros(1, length(topK));
                    Tlog_likelihood = 0;
                    step_size = 3000;
                    %veclist_vecPrecision = zeros(1, min(4, ceil(length(test_usr_idx)/step_size)));
                    %veclist_vecRecall = zeros(1, min(4, ceil(length(test_usr_idx)/step_size)));

                    parfor j = 1:min(4, ceil(length(test_usr_idx)/step_size))
                        range_step = (1 + (j-1) * step_size):min(j*step_size, length(test_usr_idx));

                        % Compute the Precision and Recall
                        test_matPredict = bsxfun(@plus, matTheta(test_usr_idx(range_step),:) * matBeta', vecBias(test_usr_idx(range_step),:));
                        test_matPredict = test_matPredict - test_matPredict .* (matX(test_usr_idx(range_step), :) > 0);
                        [vec_precision, vec_recall] = MeasurePrecisionRecall(matX_test(test_usr_idx(range_step), :), test_matPredict, topK);
                        
                        %list_vecPrecision = list_vecPrecision + sum(vec_precision, 1);
                        %list_vecRecall = list_vecRecall + sum(vec_recall, 1);
                        
                        veclist_vecPrecision(j) = sum(vec_precision, 1);
                        veclist_vecRecall(j) = sum(vec_recall, 1);
                    end

                    test_precision = sum(veclist_vecPrecision) / length(test_usr_idx);
                    test_recall = sum(veclist_vecRecall) / length(test_usr_idx);
                    
                    list_TestPrecRecall(epoch/test_step, :) = [test_precision test_recall];
                    
                    if best_ValidPrecision <= valid_precision(1) && best_TestPrecRecall_precision(1) < test_precision(1)
                        best_TestPrecRecall_precision = [test_precision, test_recall];
                        bestTheta = matTheta;
                        bestBeta = matBeta;
                    end
                end
                if best_ValidPrecision < valid_precision(1)
                    best_ValidPrecision = valid_precision(1);
                end
            end
        end
        
        %
        % Calculate the likelihood to determine when to terminate.
        %      
                
        if mod(epoch, check_step) == 0
            if mod(epoch, test_step) == 0
                fprintf('\nObjFunc: %f ( Vprecision: %f , Vrecall: %f Tprecision: %f , Trecall: %f )\n',...
                        obj_func, valid_precision(1), valid_recall(1), test_precision(1), test_recall(1));
            else
                fprintf('\nObjFunc: %f ( Vprecision: %f , Vrecall: %f )\n',...
                        Vlog_likelihood, obj_func, valid_precision(1), valid_recall(1));
            end
        else
            fprintf('\nObjFunc: %f \n', obj_func);
        end
        
        if mod(epoch,10) == 0
            %plot(matTheta(1:50,:));figure(gcf);
            
            %plot(matTheta);figure(gcf);
            idx_show = 32;
            %idx_show = 220;
            tmppp = bsxfun(@plus, matTheta(idx_show,:), vecBias(idx_show,:)) * matBeta';
            plot([tmppp(1,matX(idx_show,:)>0); matX(idx_show, matX(idx_show,:)>0)]');figure(gcf);
            %plot([tmppp(1,matX_predict(idx_show,:)>0); matX_predict(idx_show,matX_predict(idx_show,:)>0)]');figure(gcf);
            
            %bbb = [sort(full(matX(u_idx, matX(u_idx, :)>0))); sort(full(vec_prior_X_u)); sort(full(vec_predict_X_u)); sort(full(solution_xui_xuj))]';
            %bbb = bsxfun(@times, bbb, 1./sum(bbb));
            %bbb = [sort(full(vec_prior_X_u)); sort(full(vec_predict_X_u)); sort(full(solution_xui_xuj))]';
            %plot(bbb)
        end

    end

end
