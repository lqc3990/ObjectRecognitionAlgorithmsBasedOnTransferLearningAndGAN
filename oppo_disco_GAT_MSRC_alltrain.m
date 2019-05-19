
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
    load(['data\MSRC_vs_VOC.mat']); 
   if strcmp(src,'MSRC')
     Xs = X_src;
     Xs_label = Y_src;
     Xt = X_tar;
     Xt_label = Y_tar; 
     train_num=4;
   else
     Xs = X_tar;
     Xs_label = Y_tar;
     Xt = X_src;
     Xt_label = Y_src;   
     train_num=4;
   end
   
    Xs_train=full(Xs)';   
    source_label_train=full(Xs_label); 
    Xt=full(Xt)';  
    Xt_label=full(Xt_label); 
       
    [Xt_train,target_label_train,Xt_test,target_label_test]=random_data(Xt,Xt_label,train_num);
    Xs_train = Xs_train./repmat(sqrt(sum(Xs_train.^2)),[size(Xs_train,1) 1]);  %//归一化
    Xt_train = Xt_train./repmat(sqrt(sum(Xt_train.^2)),[size(Xt_train,1) 1]);  %//列归一化   
    Xt_test = Xt_test./repmat(sqrt(sum(Xt_test.^2)),[size(Xt_test,1) 1]);  %//列归一化  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
      Class= length(unique(Xs_label));%类别是重复的，
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
   y=[];
 for c=1:Class  
     Xtc_train= Xt_train(:,find(target_label_train==c));
     Xsc_train= Xs_train(:,find(source_label_train==c));%目标
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
     %label_Xsc_train=double(zeros(size(Xsc_train,2),Class));
     label_Xsc_train=double(zeros(size(Xsc_train,2),1));
     label_Xsc_train(:,1) = c;
    % label_Xsc_train(:,c) = 1;
     %label_Xtc_train=double(zeros(size(Xtc_train,2),Class));
     label_Xtc_train=double(zeros(size(Xtc_train,2),1));
     label_Xtc_train(:,1) = c;
     %label_Xtc_train(:,c) = 1;    
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 [nn_G_t_ST, fin_output]=oppo_discio_GAN_label_new(Xsc_train',label_Xsc_train,Xs_train',Ys_train,Xtc_train',label_Xtc_train,Xt_train',Yt_train); 
    Xt_new_output=[Xt_new_output;fin_output];
    y=[y;label_Xsc_train];
 end
 toc;
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Xt_new_output = Xt_new_output./repmat(sqrt(sum(Xt_new_output.^2)),[size(Xt_new_output,1) 1]);  %//列归一化     
     
          
%%%%%%%GAN%GAN%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
train_x=Xt_new_output;
train_x=[Xt_new_output];
%  train_x=[Xs_train'; Xt_train'];
%  train_x = train_x./repmat(sqrt(sum(train_x.^2)),[size(train_x,1) 1]);  %//列归一化  

%%%%%%分类%%%%%%
%Class= length(unique(source_label_train));

%   % ls
% Yat=[target_label_train];%target_label_train
   Yat=[source_label_train;target_label_train];%target_label_train
   Xat=train_x;  % Xt_new
  
 %  Xat=[train_x;Xt_train'];  % Xt_new
%    Y=-1*ones(length(Yat),Class); 
%    for j=1:length(Yat)   
%        Y(j,Yat(j))=1; 
%    end  
%    search the best regularization parameter 
Y=y;
%    a=[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,100];
%    for j=1:length(a)
%        w=(Xat'*Xat+a(j)*eye(size(Xat,2)))\(Xat'*Y);
%        yte1=Xt_test'*w;
%        rate4_1(iter,j)=decision_class(yte1,target_label_test); 
%    end
 Cls = knnclassify(Xt_test',Xat,Y,1);
   %rate_ls(iter)=max(rate4_1(iter,:));
    rate_ls(iter) = length(find(Cls==target_label_test))/length(target_label_test);

%    fprintf('rate_ls(%d)=%2.2f%%\n',iter,rate_ls(iter)); 
     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
end
ave_ls=mean(rate_ls);
fprintf('ave_ls= %2.2f%%\n',ave_ls);
end

