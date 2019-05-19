  function [train,gnd_train,test,gnd_test]=random_data(sample_data,sample_labels,train_num)
[row_size, col_size] = size(sample_data);
    total_person = max(sample_labels);
    train_num    = train_num;   % # training data per person
    
    % initialization
    gnd_train  = [];
    train      = [];
    gnd_test   = [];
    test       = [];
    
    for j = 1 : total_person
        data      = double(sample_data(:,(sample_labels == j)));
        if size(data,2)< train_num
          train_num=size(data,2);
          fprintf('train_num is not enough');
        end
        test_num  = size(data,2) - train_num;    
        gnd_train = [gnd_train;  ones(train_num,1) * j];
        gnd_test  = [gnd_test;   ones(test_num,1) * j];
        
        col_num     = size(data, 2);
        col_indexs  = randperm(col_num);
        
        train  = [train data(:, col_indexs(1: train_num))];
        test   = [test data(:, col_indexs(train_num + 1: end))];      
        clear data
    end