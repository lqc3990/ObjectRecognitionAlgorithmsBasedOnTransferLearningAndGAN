function [nn_G_t_ST,nn_D_ST,nn_G_ST,fin_output_target]=GAN_noconditon(Xsc_train,Xtc_train,nn_G_t_ST,nn_D_ST,nn_G_ST)%c1 to c2
%%%%%%Ԥ����GAN_ST%%%%%%%%%%%%%%%%%%
mean_Xtc_train=mean(Xtc_train,1);
train_y_domain_ST = double(ones(size(Xtc_train,1),1));
test_y_domain_ST = double(zeros(size(Xsc_train,1),1));
test_y_rel_ST = double(ones(size(Xsc_train,1),1));

 train_x=Xtc_train;
 test_x_input=Xsc_train;
 Num_xt_input_ST=size(Xtc_train,1);
 Num_xs_input_ST=size(Xsc_train,1);
 dim_x_input_ST=size(Xsc_train,2);
 W_dim_ST=size(Xtc_train,2);%%%%% W_dim=784
 hiden_layer=W_dim_ST/2;
 
 %[ nn_G_t_ST,nn_D_ST,nn_G_ST ] = cycle_model(W_dim_ST ,dim_x_input_ST);

length_G=nn_G_t_ST.n;
opts.numepochs =  1;        %  Number of full sweeps through data
%opts.batchsize = min(Num_xt_input_ST,Num_xs_input_ST);       %  Take a mean gradient step over this many samples
%num_ST = min(Num_xt_input_ST,Num_xs_input_ST); 
opts.batchsize = 10;
num_ST=10;
for each = 1:1
    
    %----------����G�������������------------------- 
    for i = 1:length(nn_G_t_ST.W)   %�����������
        nn_G_t_ST.W{i} = nn_G_ST.W{i};
    end
    G_output = nn_G_out(nn_G_t_ST, test_x_input);
    %-----------ѵ��D------------------------------
    index = randperm(Num_xt_input_ST);
    index1 = randperm(Num_xs_input_ST); 
    train_data_D = [train_x(index(1:num_ST),:);G_output(index1(1:num_ST),:)];
    train_y_D = [train_y_domain_ST(index(1:num_ST),:);test_y_domain_ST(index1(1:num_ST),:)];
    nn_D_ST = nntrain(nn_D_ST, train_data_D, train_y_D, opts);%ѵ��D  ����nntrain��������ѵ�������磬������յ�������� (nn.a, nn.e, nn.W, nn.b)��ѵ�����L:
    %ÿ��ѡ��һ��batch����ѵ����ÿ��ѵ����������������������
    %-----------ѵ��G-------------------------------
	
	
	
    for i = 1:length(nn_D_ST.W)  %����ѵ����D���������
       nn_G_ST.W{length(nn_G_ST.W)-i+1} = nn_D_ST.W{length(nn_D_ST.W)-i+1};
    end
	
	
%     %ѵ��G����ʱ��������ǩΪ1����Ϊ��������
%     test_x_input_style_ST=repmat(mean_Xtc_train,Num_xs_input_ST,1);
%     [nn_G_ST,~,error_GST] = nntrain_style(length_G,test_x_input_style_ST(index(1:num_ST),:),nn_G_ST, test_x_input(index(1:num_ST),:), test_y_rel_ST(index(1:num_ST),:), opts);

end
   
for i = 1:length(nn_G_t_ST.W)
    nn_G_t_ST.W{i} = nn_G_ST.W{i};
end
fin_output_target = nn_G_out(nn_G_t_ST, test_x_input(index1(1:num_ST),:));
%  fin_output_target = fin_output_target./repmat(sqrt(sum(fin_output_target.^2)),[size(fin_output_target,1) 1]);  %//�й�һ�� 

