global K                 % number of topics
global M                 % number of users
global N                 % number of items

global matX              % dim(M, N): consuming records for training
global matX_test         % dim(M, N): consuming records for testing
global matX_valid        % dim(M, N): consuming records for validation
global matX_predict

global matW              % dim(M, M): document manifold
global matS              % dim(N, N): word manifold

global matEpsilon        % dim(M, 1): latent document-topic offsets
global matPi             % dim(M, 1): latent user normalizer
global matTheta          % dim(M, K): latent document-topic intensities

global matEta            % dim(N, 1): latent word-topic offsets
global matGamma          % dim(N, 1): latent item normalizer
global matBeta           % dim(N, K): latent word-topic intensities

global matEpsilon_Shp    % dim(M, 1): varational param of matEpsilon (shape)
global matEpsilon_Rte    % dim(M, 1): varational param of matEpsilon (rate)
global matPi_Shp         % dim(M, 1): varational param of matPi (shape)
global matPi_Rte         % dim(M, 1): varational param of matPi (rate)
global matTheta_Shp      % dim(M, K): varational param of matTheta (shape)
global matTheta_Rte      % dim(M, K): varational param of matTheta (rate)

global matEta_Shp        % dim(N, 1): varational param of matEta (shape)
global matEta_Rte        % dim(N, 1): varational param of matEta (rate)
global matGamma_Shp      % dim(N, 1): varational param of matGamma (shape)
global matGamma_Rte      % dim(N, 1): varational param of matGamma (rate)
global matBeta_Shp       % dim(N, K): varational param of matBeta (shape)
global matBeta_Rte       % dim(N, K): varational param of matBeta (rate)

global tensorPhi         % dim(K, M, N) = cell{dim(M,N)}: varational param of matX
global tensorRho         % dim(K, M, M) = cell{dim(M,M)}: varational param of matW
global tensorSigma       % dim(K, N, N) = cell{dim(N,N)}: varational param of matS

global vecBiasU
global vecBiasI

global valZeta
global valDelta
global valMu

global list_ll
global list_ValidPrecRecall
global list_TestPrecRecall
global list_ValidLogLikelihhod
global list_TrainLogLikelihhod
global best_TestPrecRecall_precision
global best_TestPrecRecall_likelihood
global best_TestLogLikelihhod
global bestVlog_likelihood

TEST_TYPE = 5;
ENV = 3;
models = {'pointPRPF', 'pairPRPF', 'HPF', 'BPR', 'ListPMF', 'LorSLIM', 'RecomMC', 'PMF', 'BPMF', 'NMF', 'LogMF'};
test_size = 1;

run_model = models{2};
Ks = [5 20 50 100 150 200 250 300];
%Ks = [100];
topK = [5 10 15 20 50 100];
%topK = [10];
initialize = 0;

%% Load Data
if TEST_TYPE == 1
    % Read Last.fm data (User-Item-Word)
    meta_info = 'LastFm';
    
    if strcmp(run_model, 'pointPRPF') || strcmp(run_model, 'pairPRPF') || strcmp(run_model, 'HPF')
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        ini_scale = prior(1)/100;
        batch_size = 100;        
        MaxItr = 3600;
        test_step = 20;
        check_step = 10;
        kappa = 0.5;
        delta = 1;
        alpha = 0.05;
        if strcmp(run_model, 'pairPRPF')
            type_model = 3;
        elseif strcmp(run_model, 'pointPRPF')
            type_model = -2;
        elseif strcmp(run_model, 'HPF')
            type_model = 0;
        end

    elseif strcmp(run_model, 'BPR')
        lr = 0.2;       % learning rate
        lambda = 0;     % regularization weight
        prior = [0.3 0.3];
        ini_scale = prior(1)/100;
        Itr_step = 2000;
        MaxItr = 2500 * Itr_step;
        
    elseif strcmp(run_model, 'ListPMF')
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, ...
                0.3, 0.3, 0.3, 0.3, 0.3];
        ini_scale = prior(1)/100;
        batch_size = 100;        
        MaxItr = 2600;
        test_step = 20;
        check_step = 10;
        kappa = 0.5;
        
    elseif strcmp(run_model, 'LorSLIM') 
        tol = 10e-3;
        maxIter = 1000;
        z = 5;
        ro = 3;
        lambda = 8;
        beta = 10;
        
    elseif strcmp(run_model, 'RecomMC')
        mu = 10;
        rho = 1;
        
    elseif strcmp(run_model, 'PMF') || strcmp(run_model, 'BPMF')
        batch_size = 2000;
        maxepoch = 1000;
        lr = 10;
        regular_param = 0.01;
        
    elseif strcmp(run_model, 'NMF')
    
    elseif strcmp(run_model, 'LogMF')
        prior = [0.3, 0.3];
        ini_scale = prior(1)/100;
        usr_batch_size = 100;
        %lr = 0.0001;
        lr = 0.000001;
        lambda = 0;
        alpha = 1;
        test_step = 800;
        ini = 1;
        MaxItr = 5000;
        check_step = 400;
        
    end
    
    if ENV == 1           
        [ M, N ] = LoadUtilities('/Users/iankuoli/Dataset/LastFm_train.csv', '/Users/iankuoli/Dataset/LastFm_test.csv', '/Users/iankuoli/Dataset/LastFm_valid.csv');
    elseif ENV == 2
        %
        % 1585, 1879 are zero entries in test set.
        %
        [ M, N ] = LoadUtilities('/home/iankuoli/dataset/LastFm_train.csv', '/home/iankuoli/dataset/LastFm_test.csv', '/home/iankuoli/dataset/LastFm_valid.csv');
    else
    end
elseif TEST_TYPE == 2

    % ----- The Echo Nest Taste Profile Subset -----
    % 1,019,318 unique users
    % 384,546 unique MSD songs
    % 48,373,586 user - song - play count triplets
    meta_info = 'EchoNest';
    
    if strcmp(run_model, 'pointPRPF') || strcmp(run_model, 'pairPRPF') || strcmp(run_model, 'HPF')
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, ...
                0.3, 0.3, 0.3, 0.3, 0.3];
        ini_scale = prior(1)/100;
        batch_size = 100;        
        MaxItr = 2600;
        test_step = 20;
        check_step = 10;
        kappa = 0.5;
        if strcmp(run_model, 'pairPRPF')
            type_model = 3;
        elseif strcmp(run_model, 'pointPRPF')
            type_model = -2;
        elseif strcmp(run_model, 'HPF')
            type_model = 0;
        end

    elseif strcmp(run_model, 'BPR')
        lr = 0.2;       % learning rate
        lambda = 0;     % regularization weight
        prior = [0.3 0.3];
        ini_scale = prior(1)/100;
        Itr_step = 2000;
        MaxItr = 2500 * Itr_step;
        
    elseif strcmp(run_model, 'ListPMF')
        
    elseif strcmp(run_model, 'LorSLIM') 
        tol = 10e-3;
        maxIter = 1000;
        z = 5;
        ro = 3;
        lambda = 8;
        beta = 10;
        
    elseif strcmp(run_model, 'RecomMC')
        mu = 10;
        rho = 1;
        
    elseif strcmp(run_model, 'PMF') || strcmp(run_model, 'BPMF')
        batch_size = 2000;
        maxepoch = 1000;
        lr = 10;
        regular_param = 0.01;
        
    elseif strcmp(run_model, 'NMF')
        
    end

    if ENV == 1
        [ M, N ] = LoadUtilities('/Users/iankuoli/Dataset/EchoNest_train.csv', '/Users/iankuoli/Dataset/EchoNest_test.csv', '/Users/iankuoli/Dataset/EchoNest_valid.csv');
    elseif ENV == 2
        [ M, N ] = LoadUtilities('/home/iankuoli/dataset/EchoNest_train.csv', '/home/iankuoli/dataset/EchoNest_test.csv', '/home/iankuoli/dataset/EchoNest_valid.csv');
    end
elseif TEST_TYPE == 3
    % ----- MovieLens 20M Dataset -----
    % 138,000 users
    % 27, 000movies
    % 20 million ratings
    % 465, 000 tag applications
    meta_info = 'MovieLens20M';

    if ENV == 1
        [ M, N ] = LoadUtilities('/Users/iankuoli/Dataset/MovieLens_train.csv', '/Users/iankuoli/Dataset/MovieLens_test.csv', '/Users/iankuoli/Dataset/MovieLens_valid.csv');
    elseif ENV == 2
        [ M, N ] = LoadUtilities('/home/iankuoli/dataset/MovieLens_train.csv', '/home/iankuoli/dataset/MovieLens_test.csv', '/home/iankuoli/dataset/MovieLens_valid.csv');
    else
    end
elseif TEST_TYPE == 4
    % ----- MovieLens 10M Dataset -----
    % 71,567 users
    % 10,681 movies
    % 10 million ratings
    % 95,580 tag applications
    meta_info = 'MovieLens10M';
    
    K = 100;
    prior = [0.3, 0.3, 0.3, 0.3, 0.3,...
            0.3, 0.3, 0.3, 0.3, 0.3];
    ini_scale = prior(1)/100;
    batch_size = 1024;
    kappa = 0.5;
    alpha = 0.1;
    topK = [10];
    test_step = 100;
    check_step = 10;
    if ENV == 1
        [ M, N ] = LoadUtilities('/Users/iankuoli/Dataset/MovieLens10M_train.csv', '/Users/iankuoli/Dataset/MovieLens10M_test.csv', '/Users/iankuoli/Dataset/MovieLens10M_valid.csv');
    elseif ENV == 2
        [ M, N ] = LoadUtilities('/home/iankuoli/dataset/MovieLens10M_train.csv', '/home/iankuoli/dataset/MovieLens10M_test.csv', '/home/iankuoli/dataset/MovieLens10M_valid.csv');
    else
    end
elseif TEST_TYPE == 5
    % ----- MovieLens 1M Dataset -----
    % 6,040 users
    % 3,980 movies
    % 1 million ratings
    meta_info = 'MovieLens1M';
    
    if strcmp(run_model, 'pointPRPF') || strcmp(run_model, 'pairPRPF') || strcmp(run_model, 'HPF')
        % from BNPF, theta=Gamma(1, 0.3/0.3); beta=Gamma(0.3, 0.3/1)
        %prior = [1, 0.3, 0.3, 1, 0.3, 0.3];
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        ini_scale = prior(1)/100;
        %batch_size = 100; 
        batch_size=6040;
        MaxItr = 100;
        test_step = 5;
        check_step = 5;
        kappa = 0.5;
        delta = 0.5;
        alpha = 3;
        if strcmp(run_model, 'pairPRPF')
            type_model = 3;
        elseif strcmp(run_model, 'pointPRPF')
            type_model = -2;
        elseif strcmp(run_model, 'HPF')
            type_model = 0;
        end

    elseif strcmp(run_model, 'BPR')
        lr = 0.2;       % learning rate
        lambda = 0;     % regularization weight
        prior = [0.3 0.3];
        ini_scale = prior(1)/100;
        Itr_step = 2000;
        MaxItr = 2500 * Itr_step;
        
    elseif strcmp(run_model, 'ListPMF')
        
    elseif strcmp(run_model, 'LorSLIM') 
        tol = 10e-3;
        maxIter = 1000;
        z = 5;
        ro = 3;
        lambda = 8;
        beta = 10;
        
    elseif strcmp(run_model, 'RecomMC')
        mu = 10;
        rho = 1;
        
    elseif strcmp(run_model, 'PMF') || strcmp(run_model, 'BPMF')
        batch_size = 2000;
        maxepoch = 1000;
        lr = 10;
        regular_param = 0.01;
        
    elseif strcmp(run_model, 'NMF')
        
    elseif strcmp(run_model, 'LogMF')
        prior = [0.3, 0.3];
        ini_scale = prior(1)/100;
        usr_batch_size = 100;
        lr = 0.001;
        lambda = 0;
        alpha = 1;
        test_step = 1000;
        ini = 1;
        MaxItr = 50000;
        check_step = 500;
        
    end
    
    if ENV == 1
        [ M, N ] = LoadUtilities('/Users/iankuoli/Dataset/MovieLens1M_train.csv', '/Users/iankuoli/Dataset/MovieLens1M_test.csv', '/Users/iankuoli/Dataset/MovieLens1M_valid.csv');
    elseif ENV == 2
        [ M, N ] = LoadUtilities('/home/iankuoli/dataset/MovieLens1M_train.csv', '/home/iankuoli/dataset/MovieLens1M_test.csv', '/home/iankuoli/dataset/MovieLens1M_valid.csv');
    elseif ENV == 3
        [ M, N ] = LoadUtilities('/home/csist/Dataset/MovieLens1M_train.csv', '/home/csist/Dataset/MovieLens1M_test.csv', '/home/csist/Dataset/MovieLens1M_valid.csv');
    else
    end    
elseif TEST_TYPE == 0
    % Read Toy-graph
    meta_info = 'toy';
    Ks = [6];
    topK = [5];
    
    if strcmp(run_model, 'pointPRPF') || strcmp(run_model, 'pairPRPF') || strcmp(run_model, 'HPF')
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        %prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];    
        ini_scale = prior(1)/100;
        batch_size = 50;      
        kappa = 0.5;
        MaxItr = 100;
        test_step = 20;
        check_step = 10;
        delta = 0.5;
        alpha = 1;
        if strcmp(run_model, 'pairPRPF')
            type_model = 3;
        elseif strcmp(run_model, 'pointPRPF')
            type_model = -2;
        elseif strcmp(run_model, 'HPF')
            type_model = 0;
        end

    elseif strcmp(run_model, 'BPR')
        lr = 0.2;       % learning rate
        lambda = 0;     % regularization weight
        prior = [0.3 0.3];
        ini_scale = prior(1)/100;
        Itr_step = 2000;
        MaxItr = 2500 * Itr_step;
        
    elseif strcmp(run_model, 'ListPMF')
        prior = [0.3, 0.3, 0.3, 0.3, 0.3, ...
                0.3, 0.3, 0.3, 0.3, 0.3];
        ini_scale = prior(1)/100;
        usr_batch_size = 50;
        lr = 1;
        lambda = 0.1;
        test_step = 10;
        ini = 1;
        MaxItr = 1000;
        check_step = 10;
    
    elseif strcmp(run_model, 'LorSLIM') 
        tol = 10e-3;
        maxIter = 1000;
        z = 5;
        ro = 3;
        lambda = 8;
        beta = 10;
        
    elseif strcmp(run_model, 'RecomMC')
        mu = 1;
        rho = 1;
        
    elseif strcmp(run_model, 'PMF') || strcmp(run_model, 'BPMF')
        batch_size = 671;
        maxepoch = 1000;
        lr = 10;
        regular_param = 0.1;
        
    elseif strcmp(run_model, 'NMF')
        
    elseif strcmp(run_model, 'LogMF')
        prior = [0.3, 0.3];
        ini_scale = prior(1)/100;
        usr_batch_size = 50;
        lr = 0.0005;
        %lr = 0.0001;
        lambda = 0.01;
        alpha = 1;
        test_step = 10;
        ini = 1;
        %MaxItr = 800;
        MaxItr = 4000;
        check_step = 10;
        topK = [5];
        
    end
    
    if ENV == 1
        [ M, N ] = LoadUtilities('/Users/iankuoli/Dataset/SmallToy_train.csv', '/Users/iankuoli/Dataset/SmallToy_test.csv', '/Users/iankuoli/Dataset/SmallToy_valid.csv');
    elseif ENV == 2
        [ M, N ] = LoadUtilities('/home/iankuoli/dataset/SmallToy_train.csv', '/home/iankuoli/dataset/SmallToy_test.csv', '/home/iankuoli/dataset/SmallToy_valid.csv');
    else
    end
    
%     matX(find((matX<50) .* (matX>0))) = 1;
%     matX(find((matX<100) .* (matX>=50))) = 2;
%     matX(find((matX<150) .* (matX>=100))) = 3;
%     matX(find((matX<200) .* (matX>=150))) = 4;
%     matX(find(matX>=200)) = 5;
end

%
% Point-wise PRPF
%
if strcmp(run_model, 'pointPRPF')
    best_TestPrecRecall_pointPRPF = zeros(length(Ks), 3*length(topK)*2);
    for ik = 1:length(Ks)
        K = Ks(ik);
        StochasticCoordinateAscent_PMFR2(type_model, K, prior, ini_scale, batch_size, alpha, delta, kappa, topK, test_size, test_step, initialize, MaxItr, check_step);
        
        save(strcat(meta_info, '_PMF_batch600_K', K, '_[10]_1000_100_0.1.mat'));
        best_TestPrecRecall_pointPRPF(ik, 1:length(topK)*2) = best_TestPrecRecall_precision;
        best_TestPrecRecall_pointPRPF(ik, (length(topK)*2+1):length(topK)*4) = best_TestPrecRecall_likelihood;
        [a, b] = max(list_TestPrecRecall(:,1));
        best_TestPrecRecall_pointPRPF(ik, (length(topK)*4+1):length(topK)*6) = list_TestPrecRecall(b,:);
    end
end

%
% Pair-wise PRPF
%
if strcmp(run_model, 'pairPRPF')
    best_TestPrecRecall_pairPRPF = zeros(length(Ks), 3*length(topK)*2);
    for ik = 1:length(Ks)
        K = Ks(ik);
        StochasticCoordinateAscent_PMFR2(type_model, K, prior, ini_scale, batch_size, alpha, delta, kappa, topK, test_size, test_step, initialize, MaxItr, check_step);
        %LogisticPF(type_model, K, prior, ini_scale, batch_size, delta, kappa, topK, test_size, test_step, initialize, MaxItr, check_step);
        
        save(strcat(meta_info, '_PMF_batch600_K', K, '_[10]_1000_100_0.1.mat'));
        best_TestPrecRecall_pairPRPF(ik, 1:length(topK)*2) = best_TestPrecRecall_precision;
        best_TestPrecRecall_pairPRPF(ik, (length(topK)*2+1):length(topK)*4) = best_TestPrecRecall_likelihood;
        [a, b] = max(list_TestPrecRecall(:,1));
        best_TestPrecRecall_pairPRPF(ik, (length(topK)*4+1):length(topK)*6) = list_TestPrecRecall(b,:);
    end
end

%
% HPF
%
if strcmp(run_model, 'HPF')
    best_TestPrecRecall_pairPRPF = zeros(length(Ks), 3*length(topK)*2);
    for ik = 1:length(Ks)
        K = Ks(ik);
        StochasticCoordinateAscent_PMFR2(type_model, K, prior, ini_scale, batch_size, alpha, delta, kappa, topK, test_size, test_step, initialize, MaxItr, check_step);
        
        save(strcat(meta_info, '_PMF_batch600_K', K, '_[10]_1000_100_0.1.mat'));
        best_TestPrecRecall_pairPRPF(ik, 1:length(topK)*2) = best_TestPrecRecall_precision;
        best_TestPrecRecall_pairPRPF(ik, (length(topK)*2+1):length(topK)*4) = best_TestPrecRecall_likelihood;
        [a, b] = max(list_TestPrecRecall(:,1));
        best_TestPrecRecall_pairPRPF(ik, (length(topK)*4+1):length(topK)*6) = list_TestPrecRecall(b,:);
    end
end

%
% BPR -- Beyasian Personalized Ranking, UAI, 2009
%
if strcmp(run_model, 'BPR')
    best_TestPrecRecall_BPR = zeros(length(Ks), length(topK)*2);
    for ik = 1:length(Ks)
        K = Ks(ik);
        BPR(K, lr, lambda, prior, ini_scale, topK, test_size, test_step, Itr_step, MaxItr, check_step);
        best_TestPrecRecall_BPR(ik, :) = best_TestPrecRecall_precision;
    end
end

%
% ListPMF
%
if strcmp(run_model, 'ListPMF')
    best_TestPrecRecall_ListPMF = zeros(length(Ks), length(topK)*2);
    for ik = 1:length(Ks)
        K = Ks(ik);
        ListProbMF(K, prior, ini_scale, usr_batch_size, lr, lambda, topK, test_size, test_step, ini, MaxItr, check_step);
        %ListProbMF1(K, prior, ini_scale, usr_batch_size, lr, lambda, topK, test_size, test_step, ini, MaxItr, check_step);
        %ListProbMF2(k, prior, ini_scale, usr_batch_size, lr, lambda, topK, test_size, test_step, ini, MaxItr, check_step);
        
        best_TestPrecRecall_ListPMF(ik, :) = best_TestPrecRecall_precision;
    end
end

%
% LorSLIM -- Low Rank Sparse Linear Methods for Top-N Recommendations, ICDM, 2014.  
%
if strcmp(run_model, 'LorSLIM')
    X = LorSLIM(tol, maxIter, z, ro, lambda, beta, topK);
end

%
% RecomMC -- Top-N Recommender System via Matrix Completion, AAAI, 2016.
% 
if strcmp(run_model, 'RecomMC')
    best_TestPrecRecall_ListPMF = zeros(1, length(topK)*2);
    X = recom_mc(mu, rho, topK);
    best_TestPrecRecall_RecomMC(1, :) = best_TestPrecRecall_precision;
end

%
% BPMF -- Bayesian Probabilistic Matrix Factorization, ICML, 2008
%
if strcmp(run_model, 'PMF')
    best_TestPrecRecall_PMF = zeros(length(Ks), length(topK)*2);
    for ik = 1:length(Ks)
        K = Ks(ik);
        mean_rating = funcPMF(1, batch_size, maxepoch, lr, regular_param, check_step, test_step, test_size, topK);
        best_TestPrecRecall_PMF(ik, :) = best_TestPrecRecall_precision;
    end
end
if strcmp(run_model, 'BPMF')
    best_TestPrecRecall_BPMF = zeros(length(Ks), length(topK)*2);
    for ik = 1:length(Ks)
        K = Ks(ik);
        mean_rating1 = funcPMF(1, batch_size, maxepoch, lr, regular_param, check_step, test_step, test_size, topK);
        mean_rating2 = funcBPMF(1, batch_size, maxepoch, lr, regular_param, check_step, test_step, test_size, topK);
        best_TestPrecRecall_BPMF(ik, :) = best_TestPrecRecall_precision;
    end
end

%
% NMF -- Nonnegative Matrix Factorization, NIPS, 2001
%
if strcmp(run_model, 'NMF')
    best_TestPrecRecall_NMF = zeros(length(Ks), length(topK)*2);
    for ik = 1:length(Ks)
        K = Ks(ik);
        NMF(K, topK);
        best_TestPrecRecall_NMF(ik, :) = best_TestPrecRecall_precision;
    end
end

%
% LogMF -- Logistic Matrix Factorization, NIPS, 2014
%
if strcmp(run_model, 'LogMF')
    best_TestPrecRecall_LogMF = zeros(length(Ks), length(topK)*2);
    for ik = 1:length(Ks)
        K = Ks(ik);
        LogMF(K, prior, ini_scale, usr_batch_size, lr, lambda, alpha, topK, test_size, test_step, ini, MaxItr, check_step);
        best_TestPrecRecall_LogMF(ik, :) = best_TestPrecRecall_precision;
    end
end