function output = nn_G_out(nn, x)
    nn.testing = 1;
     nn = nnff(nn, x, zeros(size(x,1), nn.size(end))); %nnff��ǰ�򴫲�����������ÿһ��������������nn����ṹ��,�������error��loss������nn�����С�
    nn.testing = 0;
    output = nn.a{end};
end