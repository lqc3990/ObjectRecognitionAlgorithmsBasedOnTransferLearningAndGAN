function nn = nnbp_style(x,test_x_input_style,length_G,nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    dim_x_input=size(test_x_input_style,2); 
    m = size(x, 1); 
    n = nn.n;
    sparsityError = 0;
    switch nn.output
        case 'sigm'
          %  d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
		  d{n} = -nn.e .*( 1- nn.a{n}.^2) ;
        case {'softmax','linear'}
            d{n} = - nn.e;
    end
    for i = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn.activation_function 
            case 'sigm'
              %  d_act = nn.a{i} .* (1 - nn.a{i});
			  if(nn.a{i}>0)d_act=1;
			  else d_act=0.01;
			  end
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
              case 'linear'
                d_act = 1;
        end
        
        if(nn.nonSparsityPenalty>0)
            pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
            sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
        end
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)
        else % in this case in d{i} the bias term has to be removed
            d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError) .* d_act;
           if i==length_G 
%               nn.a_style= nn.a{i}(:,2:(dim_x_input+1)); 
%              gradient_error_style=nn.error_style.* (nn.a_style .* (1 - nn.a_style));
              gradient_error_style=-2*nn.error_style.* (nn.z_error_style .* (1 - nn.z_error_style));
              gradient_error_style = [ones(m,1) gradient_error_style];
             d{i} = d{i}+gradient_error_style;          
           end              
        end
        
        if(nn.dropoutFraction>0)
            d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
        end

    end

    for i = 1 : (n - 1)
        if i+1==n
            nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
        else
            nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);      
        end
    end
end
