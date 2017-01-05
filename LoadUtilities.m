function [ M, N ] = LoadUtilities(train_path, test_path, valid_path)
    
    global matX
    global matX_test
    global matX_valid
    
    %% Read file

    if ~isempty(train_path)
        fileUI = fopen(train_path, 'r', 'n', 'UTF-8');
        cellUserItem = textscan(fileUI, '%d%d%d', 'delimiter', ',');
        fclose(fileUI);
        matIndex_train = double((cell2mat(cellUserItem)));
    end    
    
    if ~isempty(test_path)
        fileUI = fopen(test_path, 'r', 'n', 'UTF-8');
        cellUserItem = textscan(fileUI, '%d%d%d', 'delimiter', ',');
        fclose(fileUI);
        matIndex_test = double((cell2mat(cellUserItem)));      
    end  
    
    if ~isempty(valid_path)
        fileUI = fopen(valid_path, 'r', 'n', 'UTF-8');
        cellUserItem = textscan(fileUI, '%d%d%d', 'delimiter', ',');
        fclose(fileUI);
        matIndex_valid = double((cell2mat(cellUserItem)));    
    end  
    
    M = max([max(matIndex_train(:,1)), max(matIndex_test(:,1)), max(matIndex_valid(:,1))]);
    N = max([max(matIndex_train(:,2)), max(matIndex_test(:,2)), max(matIndex_valid(:,2))]);
    
    matX = sparse(matIndex_train(:,1), matIndex_train(:,2), matIndex_train(:,3), M, N);
    matX_test = sparse(matIndex_test(:,1), matIndex_test(:,2), matIndex_test(:,3), M, N);
    matX_valid = sparse(matIndex_valid(:,1), matIndex_valid(:,2), matIndex_valid(:,3), M, N);
    
end

