function [vecLabel] = Load_yeast( path )
    
    global matX
    global D
    global W

    % /Users/iankuoli/Dataset/iris/iris.data
    path = '/Users/iankuoli/Dataset/YEAST/yeast.data';

    file = fopen(path);
    cellData = textscan(file, '%s%f%f%f%f%f%f%f%f%s', 'delimiter', ',');
    fclose(file);

    A = {};
    for i = 1:length(cellData{1,1})
        cat = cellData{1,1}{i};
        if length(find(strcmp(A, cat))) == 0
            A{length(A)+1,1} = cat;
        end
    end
    
    matX = cell2mat(cellData(2:9));

    D = size(matX, 1);
    W = size(matX, 2);
    
    vecLabel = zeros(D,1);
    for i = 1:D
        name = cellData{1,10}{i};
        
        if strcmp(name, 'CYT')
            vecLabel(i, 1) = 1;
        elseif strcmp(name, 'NUC')
            vecLabel(i, 1) = 2;
        elseif strcmp(name, 'MIT')
            vecLabel(i, 1) = 3;
        elseif strcmp(name, 'ME3')
            vecLabel(i, 1) = 4;
        elseif strcmp(name, 'ME2')
            vecLabel(i, 1) = 5;
        elseif strcmp(name, 'ME1')
            vecLabel(i, 1) = 6;
        elseif strcmp(name, 'EXC')
            vecLabel(i, 1) = 7;
        elseif strcmp(name, 'VAC')
            vecLabel(i, 1) = 8;
        elseif strcmp(name, 'POX')
            vecLabel(i, 1) = 9;
        elseif strcmp(name, 'ERL')
            vecLabel(i, 1) = 10;
        else
        end
            
    end
    
end

