function [nn_G_t,nn_D,nn_G,fin_output]=cycle_train(test_x_inputt,train_xs,nn_G_t,nn_D,nn_G)

Num_xt_input_ST=size(train_xs,1);%Դ��������
Num_xs_input_ST=size(test_x_inputt,1);%Ŀ����������
num_ST = min(Num_xt_input_ST,Num_xs_input_ST); %ѡ������ѵ��������
train_y_domain = double(ones(size(train_xs,1),1));
test_y_domain = double(zeros(size(test_x_inputt,1),1));
test_y_rel = double(ones(size(test_x_inputt,1),1));
num=10;


opts.numepochs =  1;        %  Number of full sweeps through data
%opts.batchsize = min(Num_xt_input_ST,Num_xs_input_ST);       %  һ������������ͼƬ;�ú������
%num_ST = min(Num_xt_input_ST,Num_xs_input_ST);  %ѡ������ѵ��������
opts.batchsize = 10;

for each = 1:1480
    
    %----------����G�������������------------------- 
    for i = 1:length(nn_G_t.W)   %�����������
        nn_G_t.W{i} = nn_G.W{i};
    end
    G_output = nn_G_out(nn_G_t, test_x_inputt);
    %-----------ѵ��D------------------------------
    index = randperm(Num_xt_input_ST);%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    index1 = randperm(Num_xs_input_ST);
    train_data_D = [train_xs(index(1:num),:);G_output(index1(1:num),:)];
    train_y_D = [train_y_domain(index(1:num),:);test_y_domain(index1(1:num),:)];
    [nn_D,L] = nntrain(nn_D, train_data_D, train_y_D, opts);%ѵ��D  $����nntrain��������ѵ�������磬������յ�������� (nn.a, nn.e, nn.W, nn.b)��ѵ�����L:
    %ÿ��ѡ��һ��batch����ѵ����ÿ��ѵ����������������������
    %-----------ѵ��G-------------------------------
    for i = 1:length(nn_D.W)  %����ѵ����D���������
        nn_G.W{length(nn_G.W)-i+1} = nn_D.W{length(nn_D.W)-i+1};
    end
    %ѵ��G����ʱ��������ǩΪ1����Ϊ��������
    [nn_G] = nntrain(nn_G, test_x_inputt(index1(1:num),:), test_y_rel(index1(1:num),:), opts);
  
end
toc

% figure, ploterrcorr(error);%
for i = 1:length(nn_G_t.W)
    nn_G_t.W{i} = nn_G.W{i};
end
fin_output = nn_G_out(nn_G_t, test_x_inputt);