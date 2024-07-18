import torch
from torch import nn
from tools import nul_matrix_except_one_column_of_ones, normalize, denormalize, mean_error_percentage, init_weights_xavier

class Pinn(nn.Module):
    """
    Class of physicals-informed neural network.
    
    Attributes
    ----------
    t : torch.Tensor
        temporal data given at initialization
    t_batch : torch.Tensor
        temporal data reshape for batch
    nb_variables : int
        total number of variables
    variables_data : dict (str : torch.Tensor)
        dictionary with known variable name as key and the corresponding data
        as associated values
    variables_max : dict (str : torch.Tensor)
        dictionary with known variable name as key and the maximum on the data
        for this variable as value 
    variables_min : dict (str : torch.Tensor)
        dictionary with known variable name as key and the minimum on the data
        for this variable as value 
    variables_norm : dict (str : torch.Tensor)
        dictionary with known variable name as key and the normalized data for
        this variable as value 
    variables : dict (str : torch.Tensor or int)
        dictionary with all variable name as key and the data tensor if known
        and the integer 1 if not known
    ode_residual_dict : dict (str : function)
        dictionary of function that compute the residual for every equation
    true_parameters : list (float)
        list of the true parameters used to compute error on parameters
    ode_parameters_ranges : list (tuple)
        list of given ranges for parameters of ode
    ode_parameters : dict (str : torch.Tensor)
        estimated parameters for ode learned by the pinn training
    params : list (torch.Tensor)
        list of parameter from the neural network and ode parameters
    constants_dict : dict (str : torch.Tensor)
        dictionary with the parameters that we already know and consider as constants
    neural_net : NeuralNet
        multi-layer neural network used to learn the variables form temporal
        data
    
    Methods
    -------
    net_f : (t_batch) -> (residual,neural_output)
        returns the residual given by ode system and the output of the neural
        network for a given batch. This method also returns the
        output of the neural layer. 
    
    train : (n_epochs) -> (r2_store, last_pred_unorm,losses, learned_parameters)
        train the network for a given number of epochs. At every
        epoch the loss on the variables and the residual loss are computed and
        stored in losses. Similarly an mean percentage of error on the learned
        parameters is computed at each epoch and stored in parameters_error.
        At the end of training this methods return also the last predicted
        variables output of the neural layer, and the learned parameters.


    output_param_range : (param, index) -> (framed parameter)
        this method send the given parameter into the range of
        ode_parameter_ranges of corresponding index
    """

    def __init__(self,
                 ode_residual_dict,
                 ranges,
                 data_t,
                 variables_data,
                 variables_no_data,
                 parameter_names,
                 true_parameters=[],
                 constants_dict={}, 
                 net_hidden =7):
        super(Pinn,self).__init__()

        # Temporal data
        self.t = torch.tensor(data_t, requires_grad=True,dtype=torch.float32)
        self.t_batch = torch.reshape(self.t, (len(self.t),1))

        # Variable used to fit the neural network
        self.nb_variables = len(variables_data) + len(variables_no_data)
        self.variables_data = {k : torch.tensor(v) for (k,v) in variables_data.items()}
        self.variables_max = {k : max(v) for (k,v) in self.variables_data.items()}
        self.variables_min = {k : min(v) for (k,v) in self.variables_data.items()}
        self.variables_norm = {k : normalize(self.variables_data[k],
                                             self.variables_min[k],
                                             self.variables_max[k])
                                    for k in self.variables_data.keys()}

        # on no_data value normalization is identity
        self.variables_min.update({k:0 for k in variables_no_data.keys()})
        self.variables_max.update({k:1 for k in variables_no_data.keys()})

        # All variables
        self.variables = dict(variables_data,
                              **variables_no_data)

        # ODE residual computation
        self.ode_residual_dict = ode_residual_dict

        # Original parameter used to compute error on parameter prediction
        self.true_parameters=true_parameters

        # Ranges of parameters
        self.ode_parameters_ranges = ranges

        # Parameters of ODE learned by training with residual loss
        self.ode_parameters = {param: torch.nn.Parameter(torch.rand(1, requires_grad=True))
                           for param in parameter_names}

        # Neural network with time as input and predict variables as output
        self.neural_net = self.NeuralNet(net_hidden,self.nb_variables)
        self.neural_net.apply(init_weights_xavier)
        self.params = list(self.neural_net.parameters())
        self.params.extend(self.ode_parameters.values())
        self.constants_dict = constants_dict


    class NeuralNet(nn.Module): # input = [[t1], [t2]...[t100]] -- that is, a batch of timesteps

        """
        Multi-layers neural network. The number of hidden layer is chosen as
        variable. Every layer have 20 neurons.

        Attributes
        ----------
        linear_relu_stack : 
            the multi_layer neural network

        Methods
        -------
        forward :
            the forward method defined by passing through the neural network
        """
        def __init__(self,net_hidden,output_size):
            super(Pinn.NeuralNet, self).__init__()

            layers = [nn.Linear(1, 20), nn.LeakyReLU()]
            for _ in range(net_hidden):
                layers.append(nn.Linear(20, 20))
                layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(20, output_size))
            layers.append(nn.LeakyReLU())
            self.linear_leaky_relu_stack = nn.Sequential(*layers)

        def forward(self, t_batch):
            logits = self.linear_leaky_relu_stack(t_batch)
            return logits

    # Residual ODE from the output of neural network
    def net_f(self, t_batch):
        """
        Returns the residual given by ode system and the output of the neural
        network for a given batch. This method also returns the
        output of the neural layer. 

        Parameters
        ----------
        t_batch : torch.Tensor
            temporal data reshape for batch
        
        Returns
        -------
        residual : list (toch.Tensor)
            residual computed on the neural network output with ode equation
        neural_output : torch.Tensor
            output of the neural network for all times

        """

        # Output for all variables for all time
        # shape is (time_batch,nb_of_variables)
        net_output = self.neural_net(t_batch)

        # Compute derivative of variable on every time of t_batch
        d_dt_var_dict = {}
        for i,k in enumerate(self.variables.keys()):
            m = nul_matrix_except_one_column_of_ones((len(self.t),self.nb_variables),i)
            net_output.backward(m, retain_graph=True)
            d_dt_var_dict[k]=self.t.grad.clone()
            self.t.grad.zero_()

        # denormalize variables output
        var_output_net_dict= {k : denormalize(net_output[:,i],
                                              self.variables_min[k],
                                              self.variables_max[k])
                              for (i,k) in enumerate(self.variables.keys())}

        # Predicted parameters
        params_dict = {k: self.output_param_range(v,i)
                       for (i,(k,v)) in enumerate(self.ode_parameters.items())}

        # Get residual according to ODE
        value_dict = self.constants_dict | params_dict 
        residual = [res(var_output_net_dict,
                        d_dt_var_dict,
                        value_dict,
                        self.variables_min,
                        self.variables_max)
                    for res in self.ode_residual_dict.values()]
        return residual, [net_output[:,i] for i in range(self.nb_variables)]


    def train(self, n_epochs):
        """
        Train the network for a given number of epochs. At every
        epoch the loss on the variables and the residual loss are computed and
        stored in losses. Similarly the mean of the percentage of error in the
        parameter prediction is computed and stored in parameter_errors. At
        the end of training this methods return also the last predicted
        variables output of the neural layer, and the learned parameters.

        Parameters
        ----------
        n_epochs : int
            number of epochs
        
        Returns
        -------
        parameter_errors : list (numpy.float64)
            errors as percentage mean for every epoch

        last_pred_unorm : list (torch.Tensor)
            last output variables of the neural network

        losses : list (float)
            losses for every epoch

        all_learned_parameters : list (list (float))
            learned parameters for every epochs
        """


        print('\nstarting training...\n')
        # r2_store = []
        parameter_errors = []
        all_learned_parameters = []
        losses = []
        residudal_losses = []
        variable_fit_losses = []
        learning_rates = []
        loss_res = False

        for epoch in range(n_epochs):
            if epoch % 1000 == 0:          
                print('\nEpoch ', epoch)
                print('#################################')

            res, net_output = self.net_f(self.t_batch)
            self.optimizer.zero_grad()

            loss_residual = sum([torch.mean(torch.square(r)) for r in res])


            loss_variable_fit = sum([torch.mean(torch.square(v - net_output[i]))
                                     for (i,v) in enumerate(self.variables_norm.values())])

            if loss_residual<10*loss_variable_fit: loss_res = True

            if loss_res:
                loss = loss_variable_fit + loss_residual
            else:
                loss = loss_variable_fit

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            learned_parameters = [self.output_param_range(v,i).item()
                             for (i,v) in enumerate(self.ode_parameters.values())]

            losses.append(loss.item())
            residudal_losses.append(loss_residual.item())
            variable_fit_losses.append(loss_variable_fit.item())
            learning_rates.append(self.scheduler.get_last_lr())
            all_learned_parameters.append(learned_parameters)
            if self.true_parameters:
                parameter_errors.append(mean_error_percentage(self.true_parameters, learned_parameters))

        last_pred_unorm = [self.variables_min[k] + (self.variables_max[k] -
                                                    self.variables_min[k]) * net_output[i]
                           for (i,k) in enumerate(self.variables.keys())]

        return parameter_errors, last_pred_unorm, losses, variable_fit_losses, residudal_losses, all_learned_parameters, learning_rates


    def output_param_range(self, param, index):
        """
        Returns the parameter of given index into the range given in 
        ode_parameter_ranges. If the range is [a,b] the former parameter x is
        sent to (tanh(x)+1)/2 * (b-a) + a.

        Parameters
        ----------
        param : float
            parameter to send in the given range
        index : int
            index of the parameter

        Returns
        -------
        framed_parameters : float
            the result of the framing function describe above
        """
        return (torch.tanh(param) + 1) / 2 * (self.ode_parameters_ranges[index][1] - self.ode_parameters_ranges[index][0]) + self.ode_parameters_ranges[index][0]