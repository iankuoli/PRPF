function [ D, W ] = LoadSmallToy( UI_path, user_profile_path, item_profile_path)

    % --- for Last.fm data ---
    % 
    % item_filepath = 'artists2.txt'
    % word_filepath = 'tags.dat'
    % UIT_filepath = 'user_taggedartists.dat'
    %
    
    global cellDocNames
    global cellWordNames
    global matX
    
    %% Read file
    
    if ~isempty(user_profile_path)
        fileUsers = fopen(user_profile_path, 'r', 'n', 'UTF-8');
        cellDocNames = textscan(fileUsers, '%d%s', 'delimiter', ',');
        fclose(fileUsers);
        matItemID = cell2mat(cellDocNames(1,1));
        cellDocNames = [num2cell(cellDocNames{1,1}), cellDocNames{1,2}];
        D = double(max(matItemID));
    end
    
    if ~isempty(item_profile_path)
        fileItems = fopen(item_profile_path, 'r', 'n', 'UTF-8');
        cellWordNames = textscan(fileItems, '%d%s', 'delimiter', ',');
        fclose(fileItems);
        matWordID = cell2mat(cellWordNames(1,1));
        cellWordNames = [num2cell(cellWordNames{1,1}), cellWordNames{1,2}];
        W = double(max(matWordID));
    end

    if ~isempty(item_profile_path)
        fileUI = fopen(UI_path, 'r', 'n', 'UTF-8');
        cellUserItem = textscan(fileUI, '%d%d%d', 'delimiter', ',');
        fclose(fileUI);
        matIndex = double((cell2mat(cellUserItem)));
        matX = sparse(matIndex(:,1), matIndex(:,2), matIndex(:,3), D, W);
    end    
end

