function [vecLabel] = Load_iris( path )
    
    global matX
    global D
    global W

    % /Users/iankuoli/Dataset/iris/iris.data
    path = '/Users/iankuoli/Dataset/IRIS/iris.data';

    file = fopen(path);
    cellData = textscan(file, '%f%f%f%f%s', 'delimiter', ',');
    fclose(file);

    matX = cell2mat(cellData(1:4));

    D = size(matX, 1);
    W = size(matX, 2);
    
    vecLabel = zeros(D,1);
    for i = 1:D
        name = cellData{1,5}{i};
        
        if strcmp(name, 'Iris-setosa')
            vecLabel(i, 1) = 1;
        elseif strcmp(name, 'Iris-versicolor')
            vecLabel(i, 1) = 2;
        elseif strcmp(name, 'Iris-virginica')
            vecLabel(i, 1) = 3;
        else
        end
            
    end
    
end

