function [nn_G_t,nn_D,nn_G,fin_output]=cycle_train(test_x_inputt,train_xs,nn_G_t,nn_D,nn_G)

Num_xt_input_ST=size(train_xs,1);%源样本数量
Num_xs_input_ST=size(test_x_inputt,1);%目标样本数量
num_ST = min(Num_xt_input_ST,Num_xs_input_ST); %选择样本训练的数量
train_y_domain = double(ones(size(train_xs,1),1));
test_y_domain = double(zeros(size(test_x_inputt,1),1));
test_y_rel = double(ones(size(test_x_inputt,1),1));
num=10;


opts.numepochs =  1;        %  Number of full sweeps through data
%opts.batchsize = min(Num_xt_input_ST,Num_xs_input_ST);       %  一个批量多少张图片;好好设个数
%num_ST = min(Num_xt_input_ST,Num_xs_input_ST);  %选择样本训练的数量
opts.batchsize = 10;

for each = 1:1480
    
    %----------计算G的输出：假样本------------------- 
    for i = 1:length(nn_G_t.W)   %共享网络参数
        nn_G_t.W{i} = nn_G.W{i};
    end
    G_output = nn_G_out(nn_G_t, test_x_inputt);
    %-----------训练D------------------------------
    index = randperm(Num_xt_input_ST);%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    index1 = randperm(Num_xs_input_ST);
    train_data_D = [train_xs(index(1:num),:);G_output(index1(1:num),:)];
    train_y_D = [train_y_domain(index(1:num),:);test_y_domain(index1(1:num),:)];
    [nn_D,L] = nntrain(nn_D, train_data_D, train_y_D, opts);%训练D  $首先nntrain的作用是训练神经网络，输出最终的网络参数 (nn.a, nn.e, nn.W, nn.b)和训练误差L:
    %每次选择一个batch进行训练，每次训练都讲更新网络参数和误差
    %-----------训练G-------------------------------
    for i = 1:length(nn_D.W)  %共享训练的D的网络参数
        nn_G.W{length(nn_G.W)-i+1} = nn_D.W{length(nn_D.W)-i+1};
    end
    %训练G：此时假样本标签为1，认为是真样本
    [nn_G] = nntrain(nn_G, test_x_inputt(index1(1:num),:), test_y_rel(index1(1:num),:), opts);
  
end
toc

% figure, ploterrcorr(error);%
for i = 1:length(nn_G_t.W)
    nn_G_t.W{i} = nn_G.W{i};
end
fin_output = nn_G_out(nn_G_t, test_x_inputt);