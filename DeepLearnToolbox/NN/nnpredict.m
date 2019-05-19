function labels = nnpredict(nn, x)
% 找出打分最大值，得到预测标签
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    
    [dummy, i] = max(nn.a{end},[],2);%得分最大值
    labels = i;%预测值
end
