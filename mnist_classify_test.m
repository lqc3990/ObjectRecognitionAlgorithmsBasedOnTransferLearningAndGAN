clc
clear
addpath(genpath('DeepLearnToolbox'))
addpath libsvm-new
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
src_str = {'svhn'};
tgt_str = {'MINIST_target'};
 Class= 10;

for i_sam = 1:length(tgt_str)
     src_name='mnist_fake_16';
     src = src_str{i_sam};
     tgt = tgt_str{i_sam};
    
    fprintf(' %s vs %s ', src, tgt); 
 for i=1:5 
    load(['C:\Users\lenovo\Desktop\我的_2\xj的资料\CatGAN\' src_name '.mat']); 
     if strcmp(src,'svhn')
      Xs_train =im2double(svhn);   
     else    
      Xs_train = im2double(mnist);
     end
     
      source_label_train = fake_label;
    load(['C:\Users\lenovo\Desktop\我的_2\xj的资料\datasets\mnist_picture\Handwritten_digits\' tgt '.mat']); 
    Xt_train = im2double(Xt{i});
    target_label_train = target_label_train{i};
    
    Xt_test=im2double(Xt_test{i});
    target_label_test = target_label_test{i};
    
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
    
    Xs_train = Xs_train./repmat(sqrt(sum(Xs_train.^2)),[size(Xs_train,1) 1]);  %//归一化
    Xt_train = Xt_train./repmat(sqrt(sum(Xt_train.^2)),[size(Xt_train,1) 1]);  %//列归一化   
    Xt_test = Xt_test./repmat(sqrt(sum(Xt_test.^2)),[size(Xt_test,1) 1]);  %//列归一化
    %-------------------------------------------------
    %test分类
    %------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % -------------------------------------------
    %               Classification
    % -------------------------------------------
    % SRC  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
   Class= length(unique(source_label_train)); 
  % ls
  
    Yat=[source_label_train];%target_label_train
  Xat=[Xs_train(:,1:length(source_label_train))]';  % Xt_new

%   Yat=[source_label_train;target_label_train];%target_label_train
%   Xat=[Xs_train(:,1:length(source_label_train)),Xt_train(:,1:length(target_label_train))]';  % Xt_new
   Y=-1*ones(length(Yat),Class); 
   for j=1:length(Yat)   
       Y(j,Yat(j))=1; 
   end  
%    search the best regularization parameter
   a=[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,100];
   for j=1:length(a)
       w=(Xat'*Xat+a(j)*eye(size(Xat,2)))\(Xat'*Y);
       yte1=Xt_test'*w;
       rate4_1(i,j)=decision_class(yte1,target_label_test); 
   end
   rate_ls(i)=max(rate4_1(i,:));
   fprintf('rate_ls(%d)=%2.2f%%\n',i,rate_ls(i));  
end            
ave_ls=mean(rate_ls);
fprintf('ave_ls= %2.2f%%\n',ave_ls);

%   % svm
%     counter=0;
%   for m_svm=-5:5
%         c=10^m_svm;
%         for n_svm=-5:5
%             gama=10^n_svm;
%             counter=counter+1;
%             tmd=['-s 0 -t t -g ',num2str(c), ' -c ',num2str(gama)];
%             model = svmtrain(Xs_train_label_sample, X_train', tmd); 
%             [predict_label_test, accuracy_test] = svmpredict(target_label_test, Y_test', model);
%             d2=diff([predict_label_test';target_label_test']);
%             N2 = numel(find(d2==0));
%             accur_test=N2/size(target_label_test,1);
%            result(counter,1)=accuracy_test(1);   
%            result(counter,2)=accur_test;
% %            result(counter,3)=predict_label_test;
%    end
% end
% [rate_svm(i),index]=max(result(:,1));
%  acc = rate_svm(i);
%  acc_a(i)=rate_svm(i);
% fprintf('acc(%d)= %2.2f%%\n',i,acc); 
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%  end
%     ave_acc=mean(acc_a);
%     fprintf('ave_svm= %2.2f%%\n',ave_acc);
%     end
 end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


