function output = nn_G_out(nn, x)
    nn.testing = 1;
     nn = nnff(nn, x, zeros(size(x,1), nn.size(end))); %nnff是前向传播函数，计算每一层的输出并保存在nn网络结构中,最后计算出error和loss保存在nn网络中。
    nn.testing = 0;
    output = nn.a{end};
end