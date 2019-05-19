function [nn_G_t_ST,nn_D_ST,nn_G_ST,fin_output_target,Xtc_train]=GAN_noconditon_one(Xsc_train,Xtc_train,nn_G_t_ST,nn_D_ST,nn_G_ST)
%%%%%%Ԥ����GAN_ST%%%%%%%%%%%%%%%%%%
mean_Xtc_train=mean(Xtc_train,1);%Դƽ��
train_y_domain_ST = double(ones(size(Xtc_train,1),1));%Դ����һ��
test_y_domain_ST = double(zeros(size(Xsc_train,1),1));%Ŀ����������һ��
test_y_rel_ST = double(ones(size(Xsc_train,1),1));

 
 train_x=Xtc_train;%Դ
 test_x_input=[Xsc_train];%Ŀ��
 Num_xt_input_ST=size(Xtc_train,1);%Դ��������
 Num_xs_input_ST=size(Xsc_train,1);%Ŀ����������
 dim_x_input_ST=size(Xsc_train,2);%����ά��
 W_dim_ST=size(Xtc_train,2);%%%%% W_dim=784
 hiden_layer=W_dim_ST/2;
 
%  [ nn_G_t_ST,nn_D_ST,nn_G_ST ] = cycle_model(W_dim_ST ,dim_x_input_ST);
length_G=nn_G_t_ST.n;
%  %nn_G_t_ST = nnsetup([dim_x_input_ST W_dim_ST]);     %3 layers
% %nn_G_t_ST = nnsetup([dim_x_input_ST hiden_layer W_dim_ST]);   %4 layers
%  nn_G_t_ST = nnsetup([dim_x_input_ST  W_dim_ST]);   %5 layers
% %  nn_G_t_ST = nnsetup([dim_x_input_ST hiden_layer hiden_layer hiden_layer W_dim_ST]);   %6 layers
% % nn_G_t_ST.weightPenaltyL2 = 1e-4;  %  L2 weight decay
% % nn_G_t_ST.dropoutFraction = 0.5;   %  Dropout fraction 
% %  nn_G_t_ST.learningRate = 4;                %  Sigm require a lower learning rate
% nn_G_t_ST.activation_function = 'sigm';
% % nn_G_t_ST.activation_function = 'linear';
% nn_G_t_ST.output = 'sigm';
% %  nn_G_t_ST.output = 'linear';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
% length_G=nn_G_t_ST.n;
% 
% nn_D_ST = nnsetup([W_dim_ST 100 1]);
% nn_D_ST.weightPenaltyL2 = 1e-4;  %  L2 weight decay
% nn_D_ST.dropoutFraction = 0.5;   %  Dropout fraction 
% % nn_D_ST.learningRate = 0.01;                %  Sigm require a lower learning rate
% nn_D_ST.activation_function = 'sigm';
% % nn_D_ST.activation_function = 'linear';
% nn_D_ST.output = 'sigm';
% % nn_D_ST.output  = 'linear';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
% 
% %nn_G_ST = nnsetup([dim_x_input_ST  W_dim_ST 100 1]);  %3 layers
% % nn_G_ST = nnsetup([dim_x_input_ST hiden_layer W_dim_ST 100 1]);  %4 layers
%  nn_G_ST = nnsetup([dim_x_input_ST   W_dim_ST  100 1]);  %5 layers
% %  nn_G_ST = nnsetup([dim_x_input_ST hiden_layer hiden_layer hiden_layer W_dim_ST 100 1]);  %6 layers
% nn_G_ST.weightPenaltyL2 = 1e-4;  %  L2 weight decay
% nn_G_ST.dropoutFraction = 0.5;   %  Dropout fraction 
% % nn_G_ST.learningRate = 0.01;                %  Sigm require a lower learning rate
% nn_G_ST.activation_function = 'sigm';
% % nn_G_ST.activation_function = 'linear';
% nn_G_ST.output = 'sigm';
% %  nn_G_ST.output = 'linear';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'


opts.numepochs =  1;        %  Number of full sweeps through data
%opts.batchsize = min(Num_xt_input_ST,Num_xs_input_ST);       %  һ������������ͼƬ;�ú������
%num_ST = min(Num_xt_input_ST,Num_xs_input_ST);  %ѡ������ѵ��������
opts.batchsize = 10;
num_ST=10;
for each = 1:1
    %----------����G�������������------------------- 
    for i = 1:length(nn_G_t_ST.W)   %�����������
        nn_G_t_ST.W{i} = nn_G_ST.W{i};
    end
    G_output = nn_G_out(nn_G_t_ST, test_x_input);
    %-----------ѵ��D------------------------------
    index = randperm(Num_xt_input_ST);%xt---trainx
	index1 = randperm(Num_xs_input_ST);%xs---texinpput
    m=Num_xs_input_ST-num_ST + Num_xt_input_ST;
	index2=randperm(m);
    train_data_D = [train_x(index(1:num_ST),:);G_output(index1(1:num_ST),:)];
    train_y_D = [train_y_domain_ST(index(1:num_ST),:);test_y_domain_ST(index1(1:num_ST),:)];
    nn_D_ST = nntrain(nn_D_ST, train_data_D, train_y_D, opts);%ѵ��D  $����nntrain��������ѵ�������磬������յ�������� (nn.a, nn.e, nn.W, nn.b)��ѵ�����L:
    %ÿ��ѡ��һ��batch����ѵ����ÿ��ѵ����������������������
		
    %-----------ѵ��G-------------------------------
    for i = 1:length(nn_D_ST.W)  %����ѵ����D���������
        nn_G_ST.W{length(nn_G_ST.W)-i+1} = nn_D_ST.W{length(nn_D_ST.W)-i+1};
    end
    
    %ѵ��nn_G_condition����ʱ��������ǩΪ1����Ϊ��������
    test_x_input_style_ST_mean=repmat(mean_Xtc_train,(Num_xs_input_ST-num_ST),1);
     test_x_input_style_ST=[test_x_input_style_ST_mean;Xtc_train];%Xtc_train---mean_Xtc_train
%  test_x_input_style_ST=[G_output(1:(Num_xs_input_ST-Num_xt_input_ST),:);Xtc_train];
     [nn_G_ST,~,error_GST] = nntrain_style(length_G,test_x_input_style_ST(index2(1:num_ST),:),nn_G_ST, test_x_input(index1(1:num_ST),:), test_y_rel_ST(index1(1:num_ST),:), opts);
        %-----------ѵ��nn_G_ST��-------------------------------      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %-----------ѵ��G-------------------------------
end
%toc
for i = 1:length(nn_G_t_ST.W)
    nn_G_t_ST.W{i} = nn_G_ST.W{i};
end
fin_output_target = nn_G_out(nn_G_t_ST, Xsc_train(1:(Num_xs_input_ST-num_ST),:));
Xtc_train = nn_G_out(nn_G_t_ST, Xtc_train);
% fin_output_target = fin_output_target./repmat(sqrt(sum(fin_output_target.^2)),[size(fin_output_target,1) 1]);  %//�й�һ��
end