clc
clear
addpath(genpath('DeepLearnToolbox'))
addpath libsvm-new
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
src_str = {'MSRC','VOC'};
tgt_str = {'VOC','MSRC'};
for i_sam = 1:length(tgt_str)
    % i_sam=2;
    src = src_str{i_sam};
    tgt = tgt_str{i_sam};
    fprintf(' %s vs %s ', src, tgt);
 for iter=1:5
      load(['C:\Users\lenovo\Desktop\�ҵ�_2\xj������\datasets\MSRC_vs_VOC.mat']); 
   if strcmp(src,'MSRC')
     Xs = X_src;
     Xs_label = Y_src;
     Xt = X_tar;
     Xt_label = Y_tar; 
     train_num=50;
   else
     Xs = X_tar;
     Xs_label = Y_tar;
     Xt = X_src;
     Xt_label = Y_src;   
     train_num=50;
   end
   
    Xs_train=full(Xs)';   
    source_label_train=full(Xs_label); 
    Xt=full(Xt)';  
    Xt_label=full(Xt_label); 
       
    [Xt_train,target_label_train,Xt_test,target_label_test]=random_data(Xt,Xt_label,train_num);
    Xs_train = Xs_train./repmat(sqrt(sum(Xs_train.^2)),[size(Xs_train,1) 1]);  %//归一�?
    Xt_train = Xt_train./repmat(sqrt(sum(Xt_train.^2)),[size(Xt_train,1) 1]);  %//列归�?��   
    Xt_test = Xt_test./repmat(sqrt(sum(Xt_test.^2)),[size(Xt_test,1) 1]);  %//列归�?��  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
      Class= length(unique(Xs_label));%类别是重复的�?
      Xt_new_output=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%target_label_train改为onehot_label
  Ytat=[target_label_train];%target_label_train 
  Yt_train=zeros(length(Ytat),Class);
   for j=1:length(Ytat)   
       Yt_train(j,Ytat(j))=1; 
   end 
   %%%%%%%%%%%%%%%%%%%%%%%%%%source_label_train改为onehot_label
   Ysat=[source_label_train];%source_label_train
   Ys_train=zeros(length(Ysat),Class);
   for j=1:length(Ysat)   
       Ys_train(j,Ysat(j))=1; 
   end   
   %%%%%%%%%%%%%%%%%%%%%%%%%%
   tic;
   Y1=[];
 for c=1:Class  
     Xtc_train= Xt_train(:,target_label_train==c);
     Xsc_train= Xs_train(:,find(source_label_train==c));
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
     label_Xsc_train=double(zeros(size(Xsc_train,2),Class));
     label_Xsc_train(:,c) = 1;
     label_Xtc_train=double(zeros(size(Xtc_train,2),Class));
     label_Xtc_train(:,c) = 1;    
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% [nn_G_t_ST, fin_output]=oppo_discio_GAN_label_new(Xsc_train',label_Xsc_train,Xs_train',Ys_train,Xtc_train',label_Xtc_train,Xt_train',Yt_train); 
    Xt_new_output=[Xt_new_output;Xsc_train'];
    Y1=[Y1;label_Xsc_train];
 end
 toc;
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Xt_new_output = Xt_new_output./repmat(sqrt(sum(Xt_new_output.^2)),[size(Xt_new_output,1) 1]);  %//列归�?��     
     
          
%%%%%%%GAN%GAN%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%train_x=Xt_new_output;
 %train_x=[Xt_new_output];
  train_x=[Xs_train'];
 % train_x=[Xs_train'];
%  train_x = train_x./repmat(sqrt(sum(train_x.^2)),[size(train_x,1) 1]);  %//列归�?��  

%%%%%%分类%%%%%%



Y=[source_label_train];%target_label_train
%  Yat=[target_label_train];%target_label_train
   
%      Xat=train_x;  % Xt_new
   Xat=[train_x];  % Xt_new
%    Y=-1*ones(length(Yat),Class); 
%    for j=1:length(Yat)   
%        Y(j,Yat(j))=1; 
%    end  
%    search the best regularization parameter
   a=[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,100];
   for j=1:length(a)
       w=(Xat'*Xat+a(j)*eye(size(Xat,2)))\(Xat'*Y);
       yte1=Xt_test'*w;
       rate4_1(iter,j)=decision_class(yte1,target_label_test); 
   end
   rate_ls(iter)=max(rate4_1(iter,:));
%    fprintf('rate_ls(%d)=%2.2f%%\n',iter,rate_ls(iter)); 
     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
end
ave_ls=mean(rate_ls);
fprintf('ave_ls= %2.2f%%\n',ave_ls);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% 构�?真实训练样本 60000个样�?1*784维（28*28展开�?
% load(['E:\Program Files\MATLAB\R2015b\work\GAT\DeepLearnToolbox\data\mnist_uint8.mat']); 
% train_x = double(train_x(1:60000,:)) / 255;
% % 真实样本认为为标�?[1 0]�?生成样本为[0 1];
% train_y_domain = double(ones(size(train_x,1),1));
% % normalize
% train_x = mapminmax(train_x, 0, 1);   %归一�?
% 
% rand('state',0)
% %% 构�?模拟训练样本 10000个样�?1*200�?
% test_x_input = normrnd(0,1,[10000,200]); % 0-255的整�?
% test_x_input = mapminmax(test_x_input, 0, 1);   %归一�?
% 
% test_y_domain = double(zeros(size(test_x_input,1),1));
% test_y_rel = double(ones(size(test_x_input,1),1));

% %%
% nn_G_t = nnsetup([100 784]);  %% NNSETUP creates a Feedforward Backpropagate Neural Network  
% % nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)  
% % layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10] 
% nn_G_t.activation_function = 'sigm';
% nn_G_t.output = 'sigm';
% 
% nn_D = nnsetup([784 100 1]);
% nn_D.weightPenaltyL2 = 1e-4;  %  L2 weight decay
% nn.dropoutFraction = 0.5;   %  Dropout fraction 
% nn.learningRate = 0.01;                %  Sigm require a lower learning rate
% nn_D.activation_function = 'sigm';
% nn_D.output = 'sigm';
% % nn_D.weightPenaltyL2 = 1e-4;  %  L2 weight decay
% 
% nn_G = nnsetup([200 784 100 1]);
% nn_D.weightPenaltyL2 = 1e-4;  %  L2 weight decay
% nn.dropoutFraction = 0.5;   %  Dropout fraction 
% nn.learningRate = 0.01;                %  Sigm require a lower learning rate
% nn_G.activation_function = 'sigm';
% nn_G.output = 'sigm';
% % nn_G.weightPenaltyL2 = 1e-4;  %  L2 weight decay
% 
% opts.numepochs =  1;        %  Number of full sweeps through data
% opts.batchsize = 100;       %  Take a mean gradient step over this many samples
% %%
% num = 1000;
% tic
% for each = 1:2000
%      fprintf('each=%d\n',each);
%     %----------计算G的输出：假样�?------------------ )
%     for i = 1:length(nn_G_t.W)   %共享网络参数
%         nn_G_t.W{i} = nn_G.W{i};
%     end
%     G_output = nn_G_out(nn_G_t, test_x_input);
%     %-----------训练D------------------------------
%     index = randperm(60000);
%     index1 = randperm(10000);
%     train_data_D = [train_x(index(1:num),:);G_output(index1(1:num),:)];
%     train_y_D = [train_y_domain(index(1:num),:);test_y_domain(index1(1:num),:)];
%     nn_D = nntrain(nn_D, train_data_D, train_y_D, opts);%训练D  $首先nntrain的作用是训练神经网络，输出最终的网络参数 (nn.a, nn.e, nn.W, nn.b)和训练误差L:
%     %每次选择�?��batch进行训练，每次训练都讲更新网络参数和误差
%     %-----------训练G-------------------------------
%     for i = 1:length(nn_D.W)  %共享训练的D的网络参�?
%         nn_G.W{length(nn_G.W)-i+1} = nn_D.W{length(nn_D.W)-i+1};
%     end
%     %训练G：此时假样本标签�?，认为是真样�?
%     nn_G = nntrain(nn_G, test_x_input(index1(1:num),:), test_y_rel(index1(1:num),:), opts);
% end
% toc
% for i = 1:length(nn_G_t.W)
%     nn_G_t.W{i} = nn_G.W{i};
% end
% fin_output = nn_G_out(nn_G_t, test_x_input);

% for i = 1:3
%    testimage0=reshape(fin_output(i,:),[28 28]);%%%TEST
%    imshow(testimage0,[]);%%%TEST
%    imwrite(mat2gray(testimage0),strcat('MINIST_gen',num2str(i),'.JPG')); %%%TEST 
%    
%    testimage1=reshape(train_x(i,:),[28 28]);%%%TEST
%    imshow(testimage1,[]);%%%TEST
%    imwrite(mat2gray(testimage1),strcat('MINIST_train',num2str(i),'.JPG')); %%%TEST
%    
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train_label=[];
% test_y_label=[];
% test_x = double(test_x(1:10000,:)) / 255;
% test_x = mapminmax(test_x, 0, 1);   %归一�?
% for i=1:60000
%    % train_label(i,:)=find(train_y(i,:))-1;
%       train_label(i,:)=find(train_y(i,:));
% end
% for i=1:10000
%     %test_y_label(i,:)=find(test_y(i,:))-1;
%     test_y_label(i,:)=find(test_y(i,:));
% end
% Class= length(unique(train_label));
% 
% %   % ls
% %    Yat= train_label;%target_label_train
% %    Xat=train_x;  % Xt_new
% %    Y=-1*ones(length(Yat),Class); 
% %    for j=1:length(Yat)   
% %        Y(j,Yat(j))=1; 
% %    end  
% % %    search the best regularization parameter
% %    a=[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,100];
% %    for j=1:length(a)
% %        w=(Xat'*Xat+a(j)*eye(size(Xat,2)))\(Xat'*Y);
% %        yte1=test_x*w;
% %        rate4_1(i,j)=decision_class(yte1,test_y_label); 
% %    end
% %    rate_ls(i)=max(rate4_1(i,:));
% %    fprintf('rate_ls(%d)=%2.2f%%\n',i,rate_ls(i));             
% 
% 
%   % svm
%     counter=0;
%  for m_svm=-5:5
%         c=10^m_svm;
%         for n_svm=-5:5
%             gama=10^n_svm;
%             counter=counter+1;
%             tmd=['-s 0 -t t -g ',num2str(c), ' -c ',num2str(gama)];
%             model = svmtrain(train_label, train_x, tmd); 
%             [predict_label_test, accuracy_test] = svmpredict(test_y_label,test_x, model);
%             d2=diff([predict_label_test';test_y_label']);
%             N2 = numel(find(d2==0));
%             accur_test=N2/size(test_y_label,1);
%            result(counter,1)=accuracy_test(1);   
%            result(counter,2)=accur_test;
% %            result(counter,3)=predict_label_test;
%         end
% end
% [rate_svm(i),index]=max(result(:,1));
%  acc = rate_svm(i);
%  acc_a(i)=rate_svm(i);
% fprintf('acc(%d)= %2.2f%%\n',i,acc); 
% 
% %    %SRC  (内存不足)
% %   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% %    Yat= test_x';
% %    Xat=[train_x(1:length(train_label),:)]';  % Xt_new
% %   a=[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,100];
% %    for j=1:length(a)
% %    W=(Xat'*Xat+a(j)*eye(size(Xat,2)))\(Xat'*Yat);
% %    [accuracy(i,j),xp,r]=computaccuracy(train_x',Class,train_label,test_x',test_y_label,W,a(j));      
% %    end   
% %    rate_ls(i)=max(accuracy(i,:));
% %    fprintf('%2.2f%%\n',rate_ls(i));  




