
clc
clear
addpath(genpath('DeepLearnToolbox'))
addpath libsvm-new
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
src_str = {'COIL_1','COIL_2'};
tgt_str = {'COIL_2','COIL_1'};
for i_sam = 1:length(tgt_str)
    % i_sam=2;
    src = src_str{i_sam};
    tgt = tgt_str{i_sam};
    fprintf(' %s vs %s ', src, tgt);
 for iter=1:1
  load(['D:\Program Files (x86)\pytorch\COL20\COIL_1.mat']); 
   if strcmp(src,'COIL_1')
     Xs = X_src;
     Xs_label = Y_src;
%      Xt = X_tar;
%      Xt_label = Y_tar; 
  load(['D:\Program Files (x86)\pytorch\COL20\COIL_1.mat']); 
       Xt = X_tar;
     Xt_label = Y_tar; 
     train_num=10;
   else
%      Xs = X_tar;
%      Xs_label = Y_tar;
     Xt =X_tar;
     Xt_label = Y_tar;
     load(['D:\Program Files (x86)\pytorch\COL20\COIL_1.mat']); 
     Xs = X_src;
     Xs_label = Y_src;
     train_num=10;
   end
   
    Xs_train=full(Xs)';   
    source_label_train=full(Xs_label); 
    Xt=full(Xt)';  
    Xt_label=full(Xt_label); 
       
    [Xt_train,target_label_train,Xt_test,target_label_test]=random_data(Xt,Xt_label,train_num);
    Xs_train = Xs_train./repmat(sqrt(sum(Xs_train.^2)),[size(Xs_train,1) 1]);  %//ÂΩí‰∏ÄÂå?
    Xt_train = Xt_train./repmat(sqrt(sum(Xt_train.^2)),[size(Xt_train,1) 1]);  %//ÂàóÂΩí‰∏?åñ   
    Xt_test = Xt_test./repmat(sqrt(sum(Xt_test.^2)),[size(Xt_test,1) 1]);  %//ÂàóÂΩí‰∏?åñ  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
      Class= length(unique(Xs_label));%Á±ªÂà´ÊòØÈáçÂ§çÁöÑÔº?
      Xt_new_output=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%target_label_trainÊîπ‰∏∫onehot_label
  Ytat=[target_label_train];%target_label_train 
  Yt_train=zeros(length(Ytat),Class);
   for j=1:length(Ytat)   
       Yt_train(j,Ytat(j))=1; 
   end 
   %%%%%%%%%%%%%%%%%%%%%%%%%%source_label_trainÊîπ‰∏∫onehot_label
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
     Xsc_train= Xs_train(:,find(source_label_train==c));%ÁõÆÊ†á
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
  Xt_new_output = Xt_new_output./repmat(sqrt(sum(Xt_new_output.^2)),[size(Xt_new_output,1) 1]);  %//ÂàóÂΩí‰∏?åñ     
     
          
%%%%%%%GAN%GAN%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
train_x=Xt_new_output;
 Yat=[source_label_train];
 Xat=train_x;
 Xat = Xat./repmat(sqrt(sum(Xat.^2)),[size(Xat,1) 1]);  %//πÈ“ªªØ
Mdl = fitcknn(Xat,Yat,'NumNeighbors',6);  
Cls = predict(Mdl,Xt_test') ;
 rate_ls(iter) = length(find(Cls==target_label_test))/length(target_label_test)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
end
ave_ls=mean(rate_ls);
fprintf('ave_ls= %2.2f%%\n',ave_ls);
end

