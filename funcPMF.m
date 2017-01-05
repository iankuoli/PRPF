function [mean_rating] = funcPMF(restart, batch_size, maxepoch, lr, regular_param, check_step, test_step, test_size, topK)
   % Version 1.000
    %
    % Code provided by Ruslan Salakhutdinov
    %
    % Permission is granted for anyone to copy, use, modify, or distribute this
    % program and accompanying programs and documents for any purpose, provided
    % this copyright notice is retained and prominently displayed, along with
    % a note saying that the original programs are available from our
    % web page.
    % The programs and documents are distributed without any warranty, express or
    % implied.  As the programs were written for research purposes only, they have
    % not been tested to the degree that would be advisable in any important
    % application.  All use of these programs is entirely at the user's own risk.

    global K                 % number of topics
    
    global matX              % dim(M, N): consuming records for training
    global matX_test         % dim(M, N): consuming records for testing
    global matX_valid        % dim(M, N): consuming records for validation
    
    global matTheta
    global matBeta
    
    global best_TestPrecRecall_precision
    
    
    best_TestPrecRecall_precision = zeros(1, length(topK)*2);
    best_ValidPrecision = 0;
    
    rand('state',0); 
    randn('state',0); 
    [numM, numN] = size(matX);

    if restart==1 
        restart = 0;
        epsilon = lr; %50; % Learning rate 
        lambda = regular_param; %0.01; % Regularization parameter 
        momentum = 0.8; 

        epoch=1;

        % Triplets: {user_id, movie_id, rating}
        [ii, jj, vv] = find(matX);
        train_vec = [ii jj vv];
        
        [ii, jj, vv] = find(matX_test);
        probe_vec = [ii jj vv];
        
        mean_rating = mean(train_vec(:,3)); 

        pairs_tr = length(train_vec); % training data 
        pairs_pr = length(probe_vec); % validation data 

        N = batch_size;   % size of per batch  
        numbatches = ceil(size(train_vec,1) / N);    % Number of batches  
        num_m = numN;     % Number of movies 
        num_p = numM;     % Number of users 
        num_feat = K;     % Rank K decomposition 

        matBeta     = 0.1 * randn(num_m, num_feat); % Movie feature vectors
        matTheta     = 0.1 * randn(num_p, num_feat); % User feature vecators
        matBeta_inc = zeros(num_m, num_feat);
        matTheta_inc = zeros(num_p, num_feat);

    end


    for epoch = epoch:maxepoch
        rr = randperm(pairs_tr);
        train_vec = train_vec(rr,:);
        clear rr 

        for batch = 1:numbatches
            fprintf(1,'epoch %d batch %d \r',epoch,batch);

            start_pos = (batch-1) * N + 1;
            end_pos = min(batch*N, size(train_vec,1));
            NNN = end_pos - start_pos + 1;

            aa_p   = double(train_vec(start_pos:end_pos, 1));
            aa_m   = double(train_vec(start_pos:end_pos, 2));
            rating = double(train_vec(start_pos:end_pos, 3));

            rating = rating-mean_rating; % Default prediction is the mean rating. 

            %%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
            pred_out = sum(matBeta(aa_m,:).*matTheta(aa_p,:),2);
            f = sum( (pred_out - rating).^2 + ...
                0.5*lambda*( sum( (matBeta(aa_m,:).^2 + matTheta(aa_p,:).^2),2)));

            %%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
            IO = repmat(2*(pred_out - rating),1,num_feat);
            Ix_m=IO.*matTheta(aa_p,:) + lambda*matBeta(aa_m,:);
            Ix_p=IO.*matBeta(aa_m,:) + lambda*matTheta(aa_p,:);

            dmatBeta = zeros(num_m,num_feat);
            dmatTheta = zeros(num_p,num_feat);

            for ii=1:NNN
              dmatBeta(aa_m(ii),:) =  dmatBeta(aa_m(ii),:) +  Ix_m(ii,:);
              dmatTheta(aa_p(ii),:) =  dmatTheta(aa_p(ii),:) +  Ix_p(ii,:);
            end

            %%%% Update movie and user features %%%%%%%%%%%

            matBeta_inc = momentum*matBeta_inc + epsilon*dmatBeta/NNN;
            matBeta =  matBeta - matBeta_inc;

            matTheta_inc = momentum*matTheta_inc + epsilon*dmatTheta/NNN;
            matTheta =  matTheta - matTheta_inc;
        end 

        %%%%%%%%%%%%%% Compute Predictions after Paramete Updates %%%%%%%%%%%%%%%%%
        pred_out = sum(matBeta(aa_m,:).*matTheta(aa_p,:),2);
        f_s = sum( (pred_out - rating).^2 + ...
            0.5*lambda*( sum( (matBeta(aa_m,:).^2 + matTheta(aa_p,:).^2),2)));
        err_train(epoch) = sqrt(f_s/NNN);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 
        NN=pairs_pr;

        aa_p = double(probe_vec(:,1));
        aa_m = double(probe_vec(:,2));
        rating = double(probe_vec(:,3));

        pred_out = sum(matBeta(aa_m,:) .* matTheta(aa_p,:),2) + mean_rating;
        %ff = find(pred_out>5); pred_out(ff)=5; % Clip predictions 
        %ff = find(pred_out<1); pred_out(ff)=1;

        err_valid(epoch) = sqrt(sum((pred_out- rating).^2)/NN);
        fprintf(1, 'epoch %4i batch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n', ...
                  epoch, batch, err_train(epoch), err_valid(epoch));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if (rem(epoch, check_step))==0
            save pmf_weight matBeta matTheta
        end
      
        %
        % Compute the precision & recall of the testing set.
        %
        if mod(epoch, test_step) == 0 && test_size > 0
            [test_usr_idx, test_itm_idx, test_val] = find(matX_test);
            test_usr_idx = unique(test_usr_idx);
            list_vecPrecision = zeros(1, length(topK));
            list_vecRecall = zeros(1, length(topK));
            Tlog_likelihood = 0;
            step_size = 10000;
            
            for j = 1:ceil(length(test_usr_idx)/step_size)
                range_step = (1 + (j-1) * step_size):min(j*step_size, length(test_usr_idx));
                
                % Compute the Precision and Recall
                test_matPredict = matTheta(test_usr_idx(range_step),:) * matBeta' + mean_rating;
                test_matPredict = test_matPredict - test_matPredict .* (matX(test_usr_idx(range_step), :) > 0);
                [vec_precision, vec_recall] = MeasurePrecisionRecall(matX_test(test_usr_idx(range_step), :), test_matPredict, topK);
                list_vecPrecision = list_vecPrecision + sum(vec_precision, 1);
                list_vecRecall = list_vecRecall + sum(vec_recall, 1);
                
                % Compute the log likelihood
                Tlog_likelihood = Tlog_likelihood + DistributionPoissonLogNZ(matX_test(test_usr_idx(range_step), :), test_matPredict);
            end
            
            test_precision = list_vecPrecision / length(test_usr_idx);
            test_recall = list_vecRecall / length(test_usr_idx);
            list_TestPrecRecall(epoch/test_step, :) = [test_precision test_recall];
            Tlog_likelihood = Tlog_likelihood / nnz(matX_test);
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
            step_size = 10000;
            
            for j = 1:ceil(length(valid_usr_idx)/step_size)
                range_step = (1 + (j-1) * step_size):min(j*step_size, length(valid_usr_idx));
                
                % Compute the Precision and Recall
                valid_matPredict = matTheta(valid_usr_idx(range_step),:) * matBeta' + mean_rating;
                valid_matPredict = valid_matPredict - valid_matPredict .* (matX(valid_usr_idx(range_step), :) > 0);
                [vec_precision, vec_recall] = MeasurePrecisionRecall(matX_valid(valid_usr_idx(range_step), :), valid_matPredict, topK);
                list_vecPrecision = list_vecPrecision + sum(vec_precision, 1);
                list_vecRecall = list_vecRecall + sum(vec_recall, 1);
            end
            valid_precision = list_vecPrecision / length(valid_usr_idx);
            valid_recall = list_vecRecall / length(valid_usr_idx);
            list_ValidPrecRecall(epoch/check_step, :) = [valid_precision valid_recall];
            
            
            if best_ValidPrecision <= valid_precision(1)
                if mod(epoch, test_step) == 0 && test_size > 0
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
                    end

                    test_precision = list_vecPrecision / length(test_usr_idx);
                    test_recall = list_vecRecall / length(test_usr_idx);
                    if best_ValidPrecision <= valid_precision(1) && best_TestPrecRecall_precision(1) < test_precision(1)
                        best_TestPrecRecall_precision = [test_precision, test_recall];
                    end
                    best_TestLogLikelihhod = Tlog_likelihood / nnz(matX_test);
                end
                if best_ValidPrecision < valid_precision(1)
                    best_ValidPrecision = valid_precision(1);
                end
            end
        end
        
        if mod(epoch, check_step) == 0
            if mod(epoch, test_step) == 0
                fprintf('Vprecision: %f , Vrecall: %f Tprecision: %f , Trecall: %f \n',...
                        valid_precision(1), valid_recall(1), test_precision(1), test_recall(1));
            else
                fprintf('Vprecision: %f , Vrecall: %f \n', valid_precision(1), valid_recall(1));
            end
        end

    end  
    
    
    
end