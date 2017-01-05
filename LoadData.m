function [ U, D, V ] = LoadData( item_filepath, word_filepath, UI_filepath, UIW_filepath )

    % --- for Last.fm data ---
    % 
    % item_filepath = 'artists2.txt'
    % word_filepath = 'tags.dat'
    % UIT_filepath = 'user_taggedartists.dat'
    %
    
    global cellDocNames
    global cellWordNames
    global matW
    global matR
    
    %% Read file
    
    fileItems = fopen(item_filepath, 'w', 'n', 'UTF-8');
    cellDocNames = textscan(fileItems, '%d%s', 'delimiter', '\t');
    fclose(fileItems);
    
    fileWords = fopen(word_filepath, 'w', 'n', 'UTF-8');
    cellWordNames = textscan(fileWords, '%d%s', 'delimiter', '\t');
    fclose(fileWords);
    
    fileUI = fopen(UI_filepath);
    cellUserItem = textscan(fileUI, '%d%d%d', 'delimiter', '\t');
    fclose(fileUI);
    
    fileUIW = fopen(UIW_filepath);
    cellUserItemWord = textscan(fileUIW, '%d%d%d%d%d%d', 'delimiter', '\t');
    fclose(fileUIW);
    
    matUIW = (cell2mat(cellUserItemWord))';
    matUI = double((cell2mat(cellUserItem)))';
    matItemID = cell2mat(cellDocNames(1,1));
    matWordID = cell2mat(cellWordNames(1,1));
    
    % Size of WordID list
    U = double(max(matUI(1,:)));
    
    % Size of WordID list
    V = double(max(matUIW(3,:)));

    % Size of ItemID list
    D = max(double(max(matUIW(2,:))), double(max(matUI(2,:))));
    
    %% Construct matR
    for i = 1:size(matUI, 2)
        if find(matUI(2,size(matUI, 2)+1-i)) == 0
            matUI(:,size(matUI, 2)+1-i) = [];
        end
    end
    
    matR = sparse(matUI(1,:), matUI(2,:), matUI(3,:), U, D);
    
    
    %% Construct matW
    matW = sparse(D, V);

    for index = matUIW
        valRowIndex = index(2,1);
        valcolIndex = index(3,1);

        if find(matItemID == valRowIndex) * find(matWordID == valcolIndex) > 0
            matW(valRowIndex, valcolIndex) = matW(valRowIndex, valcolIndex) + 1;
        end
    end
    
    cellDocNames = [num2cell(cellDocNames{1,1}), cellDocNames{1,2}];
    cellWordNames = [num2cell(cellWordNames{1,1}), cellWordNames{1,2}];
    
end

