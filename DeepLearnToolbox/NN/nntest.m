function [er, bad] = nntest(nn, x, y)
% 对比真实和预测标签。输入输出没有格式要求
    labels = nnpredict(nn, x);
    [dummy, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
end
