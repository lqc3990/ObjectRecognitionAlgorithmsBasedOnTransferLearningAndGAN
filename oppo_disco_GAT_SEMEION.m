clc
clear
addpath(genpath('DeepLearnToolbox'))
addpath libsvm-new
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%load 数据
src_str = {'MINIST_source','SEMEION_source','MINIST_source','USPS_source','USPS_source','SEMEION_source'};
tgt_str = {'USPS_target','USPS_target','SEMEION_target','SEMEION_target','MINIST_target','MINIST_target' };
 Class= 10;
for i_sam = 1:length(tgt_str)
     % i_sam=3;
    src = src_str{i_sam};
    tgt = tgt_str{i_sam};
    fprintf(' %s vs %s ', src, tgt); 
  
 for iter=1:5 %why 5
    %load Xs_train 源域
    load(['C:\Users\lenovo\Desktop\我的_2\xj的资料\datasets\mnist_picture\Handwritten_digits\' src '.mat']); 
     if strcmp(src,'MINIST_source')
      Xs_train =im2double(Xs);%target
      source_label_train = source_label;
     else    
      Xs_train = im2double(Xs{iter});
      source_label_train = source_label{iter};
      %%
      % 
      % * ITEM1
      % * ITEM2
      % 
     end
	 
	%load Xt_train 目标域
    load(['C:\Users\lenovo\Desktop\我的_2\xj的资料\datasets\mnist_picture\Handwritten_digits\' tgt '.mat']); 
    Xt_train = im2double(Xt{iter});
    target_label_train = target_label_train{iter};
	%load Xt_test
    Xt_test=im2double(Xt_test{iter});
    target_label_test = target_label_test{iter};
    tem_Xt_test=im2double(Xt_test);
    tem_target_label_test = target_label_test; 
    Xt_test=[];
    target_label_test = [];
    c_end=0;
    for j=1:Class 
    per_cn=length(find( tem_target_label_test==j)); 
    tem2_Xt_test=tem_Xt_test(:,1+c_end:per_cn+c_end);
    ind=randperm(per_cn);
    tem3_Xt_test=tem2_Xt_test(:,ind(1:10));
    tem3_target_label_test=ones(10,1)*j;
    Xt_test = [Xt_test, tem3_Xt_test];  
    target_label_test=[target_label_test;tem3_target_label_test]; 
     c_end=c_end+per_cn;
    end
    
%     load(['E:\Program Files\MATLAB\R2015b\work\HSIC_MC\random_of_minist_Xt_test3.mat']);   
%     load(['E:\Program Files\MATLAB\R2015b\work\HSIC_MC\random_of_minist_Xt_label_test3.mat']); 
    
    Xs_train = Xs_train./repmat(sqrt(sum(Xs_train.^2)),[size(Xs_train,1) 1]);  %//归一化
    Xt_train = Xt_train./repmat(sqrt(sum(Xt_train.^2)),[size(Xt_train,1) 1]);  %//列归一化   
    Xt_test = Xt_test./repmat(sqrt(sum(Xt_test.^2)),[size(Xt_test,1) 1]);  %//列归一化
      
%%%%%%%GAN%GAN%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
   y=[];
 for c=1:Class  
     Xtc_train= Xt_train(:,find(target_label_train==c));
     Xsc_train= Xs_train(:,find(source_label_train==c));
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
     label_Xsc_train=double(zeros(size(Xsc_train,2),Class));
     label_Xsc_train(:,c) = 1;  
     label_Xtc_train=double(zeros(size(Xtc_train,2),Class));
     label_Xtc_train(:,c) = 1;
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    [fin_output]=oppo_discio_GAN_label_new(Xsc_train',label_Xsc_train,Xs_train',Ys_train,Xtc_train',label_Xtc_train,Xt_train',Yt_train); 
    Xt_new_output=[Xt_new_output;fin_output];
    y=[y;label_Xsc_train];
 end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         
train_x=[Xt_new_output];%Xt_new_output:100*256;Xt_train':100*256合起来200*256

%%%%%%分类%%%%%%
%   % ls
  % Yat=[source_label_train(1:100,:);target_label_train(1:100,:)];%target_label_train
   Xat=train_x;  % Xt_new
   Y=y;
%    Y=-1*ones(length(Yat),Class); 
%    for j=1:length(Yat)   
%        Y(j,Yat(j))=1; 
%    end  
%    search the best regularization parameter
   a=[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,100];
   for j=1:length(a)
       w=(Xat'*Xat+a(j)*eye(size(Xat,2)))\(Xat'*Y);%eye(size(Xat,2))产生256*256
       yte1=Xt_test'*w;%转置后相乘
       rate4_1(iter,j)=decision_class(yte1,target_label_test); 
   end
   rate_ls(iter)=max(rate4_1(iter,:));
%    fprintf('rate_ls(%d)=%2.2f%%\n',iter,rate_ls(iter)); 
     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
end
ave_ls=mean(rate_ls);
fprintf('ave_ls= %2.2f%%\n',ave_ls);
end


