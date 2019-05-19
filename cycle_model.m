function [ nn_G_t_ST,nn_D_ST,nn_G_ST ] = cycle_model(W_dim_ST ,dim_t_input_ST)%souce,target dimension


hiden_layer=100;%W_dim_ST¡;%Òþº¬²ã
nn_G_t_ST = nnsetup([dim_t_input_ST   W_dim_ST]);  %
nn_G_t_ST.activation_function = 'sigm';
nn_G_t_ST.output = 'sigm';

nn_D_ST = nnsetup([W_dim_ST  W_dim_ST]);
nn_D_ST.weightPenaltyL2 = 1e-4;  %  L2 weight decay
nn_D_ST.dropoutFraction = 0.5;   %  Dropout fraction 
nn_D_ST.learningRate = 0.01;                %  Sigm require a lower learning rate
nn_D_ST.activation_function = 'sigm';
nn_D_ST.output = 'sigm';


nn_G_ST = nnsetup([dim_t_input_ST   W_dim_ST   100 1]);
nn_G_ST.weightPenaltyL2 = 1e-4;  %  L2 weight decay
nn_G_ST.dropoutFraction = 0.5;   %  Dropout fraction 
nn_G_ST.learningRate = 0.01;                %  Sigm require a lower learning rate
nn_G_ST.activation_function = 'sigm';
nn_G_ST.output = 'sigm';

end

