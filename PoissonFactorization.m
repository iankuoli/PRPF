%  The implemntation is according to "Content-based recommendation with Poisson factorization", in NIPS, 2014.
% 
%

%% Global Setting.

% Re-read => 0: not re-read; 1: re-read
REREAD = 0;

% Test Type => 1: toy graph;  2: JAIN;  3: IRIS;  4: YEAST  5: Last.fm;  
TEST_TYPE = 1;

% Enviornment => 1: OSX;  2: CentOS
ENV = 2;

%% Parameters declaration

global K                 % number of topics
global M                 % number of users
global N                 % number of items

global matX              % dim(M, N): consuming records for training
global matX_test         % dim(M, N): consuming records for testing
global matX_valid        % dim(M, N): consuming records for validation

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

global valZeta
global valDelta
global valMu

global list_ll
global list_ValidPrecRecall
global list_TestPrecRecall


%% Load Data
if REREAD == 1
    if TEST_TYPE == 1
        if ENV == 1
            [ M, N ] = LoadUtilities('/Users/iankuoli/Dataset/SmallToy_train.csv', '/Users/iankuoli/Dataset/SmallToy_test.csv', '/Users/iankuoli/Dataset/SmallToy_valid.csv');
        elseif ENV == 2
            [ M, N ] = LoadUtilities('/home/iankuoli/dataset/SmallToy_train.csv', '/home/iankuoli/dataset/SmallToy_test.csv', '/home/iankuoli/dataset/SmallToy_valid.csv');
        else
        end
        
    elseif TEST_TYPE == 2
        % Read JAIN
        
    elseif TEST_TYPE == 3
        % Read IRIS
        if ENV == 1
            vecLabel = Load_iris('/Users/iankuoli/Dataset/IRIS/iris.data');
        elseif ENV == 2
            vecLabel = Load_iris('/Users/iankuoli/Dataset/IRIS/iris.data');
        else
        end
        
    elseif TEST_TYPE == 4
        % Read YEAST
        if ENV == 1
            vecLabel = Load_yeast('/Users/iankuoli/Dataset/YEAST/yeast.data');
        elseif ENV == 2
            vecLabel = Load_yeast('/Users/iankuoli/Dataset/YEAST/yeast.data');
        else
        end
        
    elseif TEST_TYPE == 5
        % Read Last.fm data (User-Item-Word)
        if ENV == 1
            item_filepath = '/Users/iankuoli/Dataset/LastFm2/artists2.txt';
            word_filepath = '/Users/iankuoli/Dataset/LastFm2/tags.dat';
            UI_filepath = '/Users/iankuoli/Dataset/LastFm2/user_artists.dat';
            UIW_filepath = '/Users/iankuoli/Dataset/LastFm2/user_taggedartists.dat';
            
            [ M, N ] = LoadUtilities('/Users/iankuoli/Dataset/LastFm_train.csv', '/Users/iankuoli/Dataset/LastFm_test.csv', '/Users/iankuoli/Dataset/LastFm_valid.csv');
        elseif ENV == 2
            item_filepath = 'C:/Dataset/LastFm2/artists2.txt';
            word_filepath = 'C:/Dataset/LastFm2/tags.dat';
            UI_filepath = 'C:/Dataset/LastFm2/user_artists.dat';
            UIW_filepath = 'C:/Dataset/LastFm2/user_taggedartists.dat';
            
            %
            % 1585, 1879 are zero entries in test set.
            %
            [ M, N ] = LoadUtilities('/home/iankuoli/dataset/LastFm_train.csv', '/home/iankuoli/dataset/LastFm_test.csv', '/home/iankuoli/dataset/LastFm_valid.csv');
        else
        end
        
    elseif TEST_TYPE == 6

        % UCL Million Song Dataset
        % This dataset does not provide the user-item relationship
        meta_info = 'UCLSongDB';

        if ENV == 1
            [ M, N ] = LoadUtilities('/Users/iankuoli/Dataset/UCLSongDB_train.csv', '/Users/iankuoli/Dataset/UCLSongDB_test.csv', '/Users/iankuoli/Dataset/UCLSongDB_valid.csv');
        elseif ENV == 2
            [ M, N ] = LoadUtilities('/Users/iankuoli/Dataset/UCLSongDB_train.csv', '/Users/iankuoli/Dataset/UCLSongDB_test.csv', '/Users/iankuoli/Dataset/UCLSongDB_valid.csv');
        end
    elseif TEST_TYPE == 7

        % ----- The Echo Nest Taste Profile Subset -----
        % 1,019,318 unique users
        % 384,546 unique MSD songs
        % 48,373,586 user - song - play count triplets
        meta_info = 'EchoNest';

        if ENV == 1
            [ M, N ] = LoadUtilities('/Users/iankuoli/Dataset/EchoNest_train.csv', '/Users/iankuoli/Dataset/EchoNest_test.csv', '/Users/iankuoli/Dataset/EchoNest_valid.csv');
        elseif ENV == 2
            matX, matX_test, matX_valid = LoadFile.load_EchoNest('/home/iankuoli/dataset/EchoNest/train_triplets.txt');
        end

    elseif TEST_TYPE == 8
        % ----- MovieLens 20M Dataset -----
        % 138,000 users
        % 27, 000movies
        % 20 million ratings
        % 465, 000 tag applications
        meta_info = 'MovieLens';
        
        if ENV == 1
            [ M, N ] = LoadUtilities('/Users/iankuoli/Dataset/MovieLens_train.csv', '/Users/iankuoli/Dataset/MovieLens_test.csv', '/Users/iankuoli/Dataset/MovieLens_valid.csv');
        elseif ENV == 2
            [ M, N ] = LoadUtilities('C:/Dataset/small_toy/toy_graph.csv', 'C:/Dataset/small_toy/user.csv', 'C:/Dataset/small_toy/artist2.csv');
        else
        end
        
    end
end

%matW = Similarity(matX, matX, 'pfs2');   
%matW = full(matW - diag(diag(matW)));
%matX = matW;
[M, N] = size(matX);

%% Training Phase

if TEST_TYPE == 1
    K = 8;
    topK = [5 10 15 20];
    initialize = 1;
    testing = true;
    
    %CoordinateAscent_PMF_1(K, 1*ones(1,10), 1, topK);
    %CoordinateAscent_MPMF_3(K, 1*ones(1,10), 1, 1000, 100, 1, 0.1, topK, 1);
    %CoordinateAscent_GMPMF_1(K, 1*ones(1,10), 1, 100, 1, 1, 0.1, topK, 1);
    StochasticCoordinateAscent_MPMF_2(K, 1*ones(1,10), 1, 1000, 100, 1, 10, 0.001, 0.1, topK, testing, initialize);
    
    PPP = bsxfun(@times, matTheta, matPi);
    WWW = PPP * PPP';
    
    SSS = bsxfun(@times, matBeta, matGamma);
    SSS = SSS * SSS';
    
    matWW = full(matW);
    XXX = matTheta * matBeta';
    surf(WWW);
    %plot(matTheta);figure(gcf);
elseif TEST_TYPE == 2
    % The best settings for JAIN => 1.0
    %CoordinateAscent_MPF3(2, 1.0*ones(1,6), 0, 1, 0, 0.001, 5, 1);
    %CoordinateAscent_MPF_1(2, 1*ones(1,6), 0, 1, 0, 0.001, 5, 1, 0.1);
    CoordinateAscent_MPF_2(2, 1*ones(1,6), 0, 1, 0, 0.001, 5, 1, 0.1, 0.1);
elseif TEST_TYPE == 3
    % The best settings for IRIS => 0.966667
    CoordinateAscent_MPF3(3, 0.08*ones(1,4), 0, 1, 0, 0.1, 10, 1); 
elseif TEST_TYPE == 4
    % The best settings for YEAST => 0.384097 -> 20/0.1
    CoordinateAscent_MPF3(K, 1*ones(1,4), 0, 1, 0, 0.1, 20, 1); 
elseif TEST_TYPE == 5
    topK = 10;
    K = 100;
    initialize = 1;
    testing = true;
    
    % NMF: precision@10: 0.0648 , recall@10: 0.0616
    %[A, B] = nnmf(matX, 100);
    %B = B';
        
    % PNF by VI: precision@10: 0.081446 , recall@10: 0.079686
    %CoordinateAscent_PMF_1(K, 1*ones(1,10), 1, topK);  
    
    % PMF by SVI: precision@10: 0.095322 , recall@10: 0.092806
    %StochasticCoordinateAscent_MPMF_2(K, 1*ones(1,10), 1, 10e-20, 10e-20, 1, 600, 0.5, 1, topK, testing, initialize);
    
    % precision@10: 0.082350 , recall@10: 0.080102
    %CoordinateAscent_MPMF_3(K, 1*ones(1,10), 1, 1000, 0, 1, 0.1, topK, 0);
    
    % MPMF by SVI: precision@10: 0.104678  , recall@10: 0.101668 , similarity: 3
    %StochasticCoordinateAscent_MPMF_2(K, 1*ones(1,10), 1, 10e+2, 10e+1, 1, 600, 0.5, 0.1, topK, testing, initialize);
    
    % MPMF by SVI: precision@10: 0.113876  , recall@10: 0.109876 ,similarity: 3, alpha = 0.1
    StochasticCoordinateAscent_MPMF_2(K, 1*ones(1,10), 1, 10000, 1000, 1, 600, 0.5, 0.1, topK, testing, initialize);
    
    % MPMF by SVI: precision@10: 0.  , recall@10: 0. , similarity: 3 , alpha = 0.01
    %StochasticCoordinateAscent_MPMF_2(K, 1*ones(1,10), 1, 10e+2, 10e+2, 1, 600, 0.5, 0.01, topK, testing, initialize);
else
end
