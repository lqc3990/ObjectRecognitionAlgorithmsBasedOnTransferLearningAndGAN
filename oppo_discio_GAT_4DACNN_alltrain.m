clc
clear
addpath(genpath('DeepLearnToolbox'))
addpath libsvm-new
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
src_strname = {'amazon','Caltech10','webcam','amazon','webcam','dslr','dslr','webcam','Caltech10','Caltech10','dslr','amazon'};
tgt_strname = {'dslr','dslr','dslr','Caltech10','Caltech10','Caltech10','amazon','amazon','amazon','webcam','webcam','webcam'};
Data=load('C:\Users\lenovo\Desktop\我的_2\xj的资料\datasets\4DA-CNN\DeCAF_Data.mat');
src_str = {Data.amazon_data,Data.caltech_data,Data.webcam_data,Data.amazon_data,Data.webcam_data,Data.dslr_data,Data.dslr_data,Data.webcam_data,Data.caltech_data,Data.caltech_data,Data.dslr_data,Data.amazon_data};
slabel_str={Data.amazon_label,Data.caltech_label,Data.webcam_label,Data.amazon_label,Data.webcam_label,Data.dslr_label,Data.dslr_label,Data.webcam_label,Data.caltech_label,Data.caltech_label,Data.dslr_label,Data.amazon_label};
tgt_str={Data.dslr_data,Data.dslr_data,Data.dslr_data,Data.caltech_data,Data.caltech_data,Data.caltech_data,Data.amazon_data,Data.amazon_data,Data.amazon_data,Data.webcam_data,Data.webcam_data,Data.webcam_data};
tlabel_str={Data.dslr_label,Data.dslr_label,Data.dslr_label,Data.caltech_label,Data.caltech_label,Data.caltech_label,Data.amazon_label,Data.amazon_label,Data.amazon_label,Data.webcam_label,Data.webcam_label,Data.webcam_label};

Xt_test_c=cell(10,1);
for i_sam = 5:length(src_strname)
%      i_sam=3;
    src = src_str{i_sam};
    tgt = tgt_str{i_sam};

    fts=src{2};
    labels=slabel_str{i_sam};

    Xs = fts';
    Xs_label = labels;
    clear fts;
    clear labels;

    fts=tgt{2};
    labels=tlabel_str{i_sam};
    Xt = fts';
    Xt_label = labels;
    clear fts;
    clear labels;
    
    src = src_strname{i_sam};
    tgt = tgt_strname{i_sam};
    fprintf(' %s vs %s ', src, tgt);
    
    %----------random-------------------------------
    load(strcat('C:\Users\lenovo\Desktop\我的_2\xj的资料\datasets\4DA\data\SameCategory_',src, '-',tgt, '_20RandomTrials_10Categories.mat'));
   if strcmp(src,'amazon')
        Xs_train_num=20;
     else
        Xs_train_num=8;
   end
    
   
%       Xs_r = Xs;
%       Xt_r = Xt;  
    Xs_r = Xs./repmat(sqrt(sum(Xs.^2)),[size(Xs,1) 1]);  %//归一化
    Xt_r = Xt./repmat(sqrt(sum(Xt.^2)),[size(Xt,1) 1]);  %//列归一化   
    
% Eigen_NUM=size(Xs_r,2);
% Pro_Matrix=my_pca(Xs_r,Eigen_NUM);
% Xs_r=Pro_Matrix'*Xs_r; 
% Xt_r=Pro_Matrix'*Xt_r;


    for iter=1:1
        Xs_train=Xs_r(:,train.source{iter});source_label_train= Xs_label(train.source{iter});
        Xt_train=Xt_r(:,train.target{iter});target_label_train=Xt_label(train.target{iter});
        Xt_test=Xt_r(:,test.target{iter});target_label_test=Xt_label(test.target{iter});
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      Class= length(unique(Xs_label));
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
  for c=1:Class  
     Xtc_train= Xt_train(:,find(target_label_train==c));
     Xsc_train= Xs_train(:,find(source_label_train==c));
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
     label_Xsc_train=double(zeros(size(Xsc_train,2),Class));
     label_Xsc_train(:,c) = 1;
     label_Xtc_train=double(zeros(size(Xtc_train,2),Class));
     label_Xtc_train(:,c) = 1;    
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
[nn_G_t_ST, fin_output]=oppo_discio_GAN_label_new(Xsc_train',label_Xsc_train,Xs_train',Ys_train,Xtc_train',label_Xtc_train,Xt_train',Yt_train); 
    Xt_new_output=[Xt_new_output;fin_output];
%       Xtc_train_c{c} = nn_G_out(nn_G_t_ST, Xtc_train');
%       Xt_test_c{c} = nn_G_out(nn_G_t_ST, Xt_test');
 end
%   [nn_G_t_ST,Xtc_ba]=GAN_noconditon(Xs_train',Xt_train');
%     Xt_new_output = nn_G_out(nn_G_t_ST, Xs_train');
%   Xt_train = nn_G_out(nn_G_t_ST, Xt_train');
%   Xt_test = nn_G_out(nn_G_t_ST, Xt_test');
%   Xt_test=Xt_test';
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%      Xs_normal=[Xt_new_output;Xt_train';Xt_test'];
%     Xs_normal = Xs_normal./repmat(sqrt(sum(Xs_normal.^2)),[size(Xs_normal,1) 1]);  %//列归一化
%     train_x=Xs_normal(1:110,:);
%      Xt_test= Xs_normal(111:end,:);  
%      Xt_test=Xt_test';
%%%%%%%GAN%GAN%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%  train_x=[Xt_new_output; Xt_train];
%   Xt_train_all=[];
%  Xt_test_all=zeros(127,4096);
%  for c=1:10
%      Xt_test_all=Xt_test_all+Xt_test_c{c};
%      Xt_train_all=[Xt_train_all;Xtc_train_c{c}]
%  end
%  Xt_test=Xt_test_all/10;
%    Xt_test=Xt_test';
   train_x=[Xt_new_output; Xt_train'];
%  train_x=[Xt_new_output; Xt_train_all];
%  train_x=[Xs_train'; Xt_train'];
%  train_x = train_x./repmat(sqrt(sum(train_x.^2)),[size(train_x,1) 1]);  %//列归一化  

%%%%%%分类%%%%%%
%Class= length(unique(source_label_train));

%   % ls
   Y=[source_label_train;target_label_train];%target_label_train
   Xat=train_x;  % Xt_new
%    Y=-1*ones(length(Yat),Class); 
%    for j=1:length(Yat)   
%        Y(j,Yat(j))=1; 
%    end  
%    search the best regularization parameter
Cls = knnclassify(Xt_test',Xat,Y,1);
rate_ls(iter) = length(find(Cls==target_label_test))/length(target_label_test);
%    a=[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,100];
%    for j=1:length(a)
%        w=(Xat'*Xat+a(j)*eye(size(Xat,2)))\(Xat'*Y);
%        yte1=Xt_test'*w;
%        rate4_1(iter,j)=decision_class(yte1,target_label_test); 
%    end
  % rate_ls(iter)=max(rate_ls(iter,:));
%    fprintf('rate_ls(%d)=%2.2f%%\n',iter,rate_ls(iter)); 
     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
end
ave_ls=mean(rate_ls);
fprintf('ave_ls= %2.2f%%\n',ave_ls);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

