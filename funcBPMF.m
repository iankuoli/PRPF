function funcBPMF(restart, batch_size, maxepoch, lr, regular_param, check_step, test_step, test_size, topK)
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
        restart=0; 
        epoch=1; 
        maxepoch=50; 

        iter=0; 
        num_m = numN;     % Number of movies 
        num_p = numM;     % Number of users 
        num_feat = K;     % Rank K decomposition 

        % Initialize hierarchical priors 
        beta = 2; % observation noise (precision) 
        mu_u = zeros(num_feat,1);
        mu_m = zeros(num_feat,1);
        alpha_u = eye(num_feat);
        alpha_m = eye(num_feat);  

        % parameters of Inv-Whishart distribution (see paper for details) 
        WI_u = eye(num_feat);
        b0_u = 2;
        df_u = num_feat;
        mu0_u = zeros(num_feat,1);

        WI_m = eye(num_feat);
        b0_m = 2;
        df_m = num_feat;
        mu0_m = zeros(num_feat,1);

        [ii, jj, vv] = find(matX);
        train_vec = [ii jj vv];
        
        [ii, jj, vv] = find(matX_test);
        probe_vec = [ii jj vv];
        
        mean_rating = mean(train_vec(:,3));
        ratings_test = double(probe_vec(:,3));

        pairs_tr = length(train_vec);
        pairs_pr = length(probe_vec);

        fprintf(1,'Initializing Bayesian PMF using MAP solution found by PMF \n'); 
        %makematrix

        load pmf_weight
        err_test = cell(maxepoch,1);

        % Initialization using MAP solution found by PMF. 
        %% Do simple fit
        mu_u = mean(matTheta)';
        d=num_feat;
        alpha_u = inv(cov(matTheta));

        mu_m = mean(matBeta)';
        alpha_m = inv(cov(matTheta));

        matX=matX';
        probe_rat_all = pred(matBeta, matTheta, probe_vec, mean_rating);
        counter_prob=1; 

    end


    for epoch = epoch:maxepoch

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Sample from movie hyperparams (see paper for details)  
        N = size(matBeta,1);
        x_bar = mean(matBeta)'; 
        S_bar = cov(matBeta); 

        WI_post = inv(inv(WI_m) + N/1*S_bar + ...
                N*b0_m*(mu0_m - x_bar)*(mu0_m - x_bar)'/(1*(b0_m+N)));
        WI_post = (WI_post + WI_post')/2;

        df_mpost = df_m+N;
        alpha_m = wishrnd(WI_post,df_mpost);   
        mu_temp = (b0_m*mu0_m + N*x_bar)/(b0_m+N);  
        lam = chol( inv((b0_m+N)*alpha_m) ); lam=lam';
        mu_m = lam*randn(num_feat,1)+mu_temp;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Sample from user hyperparams
        N = size(matTheta,1);
        x_bar = mean(matTheta)';
        S_bar = cov(matTheta);

        WI_post = inv(inv(WI_u) + N/1*S_bar + ...
                N*b0_u*(mu0_u - x_bar)*(mu0_u - x_bar)'/(1*(b0_u+N)));
        WI_post = (WI_post + WI_post')/2;
        df_mpost = df_u+N;
        alpha_u = wishrnd(WI_post,df_mpost);
        mu_temp = (b0_u*mu0_u + N*x_bar)/(b0_u+N);
        lam = chol( inv((b0_u+N)*alpha_u) ); lam=lam';
        mu_u = lam*randn(num_feat,1)+mu_temp;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Start doing Gibbs updates over user and 
        % movie feature vectors given hyperparams.  

        for gibbs=1:2 
            fprintf(1,'\t\t Gibbs sampling %d \r', gibbs);

            %%% Infer posterior distribution over all movie feature vectors 
            matX=matX';
            for mm=1:num_m
                fprintf(1,'movie =%d\r',mm);
                ff = find(matX(:,mm)>0);
                MM = matTheta(ff,:);
                rr = matX(ff,mm)-mean_rating;
                covar = inv((alpha_m+beta*MM'*MM));
                mean_m = covar * (beta*MM'*rr+alpha_m*mu_m);
                lam = chol(covar); lam=lam'; 
                matBeta(mm,:) = lam*randn(num_feat,1)+mean_m;
            end

            %%% Infer posterior distribution over all user feature vectors 
            matX=matX';
            for uu=1:num_p
                fprintf(1,'user  =%d\r',uu);
                ff = find(matX(:,uu)>0);
                MM = matBeta(ff,:);
                rr = matX(ff,uu)-mean_rating;
                covar = inv((alpha_u+beta*MM'*MM));
                mean_u = covar * (beta*MM'*rr+alpha_u*mu_u);
                lam = chol(covar); lam=lam'; 
                matTheta(uu,:) = lam*randn(num_feat,1)+mean_u;
            end
        end 

        probe_rat = pred(matBeta,matTheta,probe_vec,mean_rating);
        probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1);
        counter_prob=counter_prob+1;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%% Make predictions on the validation data %%%%%%%
        temp = (ratings_test - probe_rat_all).^2;
        err = sqrt( sum(temp)/pairs_pr);

        iter=iter+1;
        overall_err(iter)=err;

        fprintf(1, '\nEpoch %d \t Average Test RMSE %6.4f \n', epoch, err);
      
        matX=matX';
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
        matX=matX';
    end  
end