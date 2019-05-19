
function[ nn_G_t_ST2,fin_output_target]=oppo_discio_GAN_label_new(Xsc_train,label_Xsc_train,Xs_train,Ys_train,Xtc_train,label_Xtc_train,Xt_train,Yt_train)
   Num_Xtc_train=size(Xtc_train,1);%2
   Num_Xsc_train=size(Xsc_train,1);%1
   dim_x_input_ST=size(Xsc_train,2);%1dim
   W_dim_ST=size(Xtc_train,2);%2dim
    
   Xtc_ba_new=[];
   %[nn_G_t_ST,Xtc_ba]=GAN_noconditon(Xsc_train,Xtc_train);
%G，D，G+D
[ nn_G_t_ST1,nn_D_ST1,nn_G_ST1 ] = cycle_model(W_dim_ST ,dim_x_input_ST);%2 to 1,xt--xs
[ nn_G_t_ST2,nn_D_ST2,nn_G_ST2 ] = cycle_model(dim_x_input_ST,W_dim_ST );%xs--xt

[ nn_G_t_ST3,nn_D_ST3,nn_G_ST3 ] = cycle_model(W_dim_ST ,dim_x_input_ST);%2 to 1,xt--xs

[ nn_G_t_ST4,nn_D_ST4,nn_G_ST4 ] = cycle_model(dim_x_input_ST,W_dim_ST );%xs--xt%
 %  [nn_G_t_TS,Xsc_ba]=GAN_noconditon(Xtc_train,Xsc_train);

%    [nn_G_t_ST1,nn_D_ST1,nn_G_ST1,fin_output]=cycle_train(Xtc_train,Xsc_train,nn_G_t_ST1,nn_D_ST1,nn_G_ST1);
%    [nn_G_t_ST2,nn_D_ST2,nn_G_ST2,fin_output]=cycle_train(Xsc_train,Xtc_train,nn_G_t_ST2,nn_D_ST2,nn_G_ST2);
%%%%%%%%%%%%%%%%%%%%%%%%%预处理过后拿XS和XS_new训练GST&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

label_Xsc_train_two=[label_Xsc_train;label_Xtc_train];
label_Xtc_train_two=[label_Xtc_train;label_Xsc_train];


for item=1:1800
%Xsc_train_two=[Xsc_train;Xsc_ba];
  [nn_G_t_ST2,nn_D_ST2,nn_G_ST2,Xsc_ba]=GAN_noconditon(Xsc_train,Xtc_train, nn_G_t_ST2,nn_D_ST2,nn_G_ST2 );%xs--xt D
 [nn_G_t_ST1,nn_D_ST1,nn_G_ST1,Xtc_ba]=GAN_noconditon(Xtc_train,Xsc_train, nn_G_t_ST1,nn_D_ST1,nn_G_ST1 );%xt--xs D

Xsc_train_two=[Xsc_train;Xtc_ba];%xs循环
Xtc_train_two=[Xtc_train;Xsc_ba];
  [nn_G_t_ST1,nn_D_ST1,nn_G_ST1,xs_cycle]=GAN_noconditon_style(Xtc_train_two,Xsc_train,nn_G_t_ST1,nn_D_ST1,nn_G_ST1);
  [nn_G_t_ST2,nn_D_ST2,nn_G_ST2,xt_cycle]=GAN_noconditon_style(Xsc_train_two,Xtc_train,nn_G_t_ST2,nn_D_ST2,nn_G_ST2);
 
Xsc_train_two=[Xsc_train;xs_cycle];%xs循环
Xtc_train_two=[Xtc_train;xt_cycle];
[nn_G_t_ST1,nn_D_ST1,nn_G_ST1,xs_cycle]=GAN_noconditon_style(Xtc_train_two,Xsc_train,nn_G_t_ST1,nn_D_ST1,nn_G_ST1);
[nn_G_t_ST2,nn_D_ST2,nn_G_ST2,xt_cycle]=GAN_noconditon_style(Xsc_train_two,Xtc_train,nn_G_t_ST2,nn_D_ST2,nn_G_ST2);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fin_output_target = nn_G_out(nn_G_t_ST2,Xsc_train);



