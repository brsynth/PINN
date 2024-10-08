import torch
import optuna
import numpy as np
from torch import nn
from numpy import isnan
from tqdm import tqdm
from scipy.special import softmax
from .tools import nul_matrix_except_one_column_of_ones, normalize, denormalize, mean_error_percentage, init_weights_xavier
import torch.optim as optim

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

    optimizer_type : torch.optim.Optimizer
        optimizer object used to update the parameters during training, e.g.,
        torch.optim.Adam

    scheduler_type : torch.optim.lr_scheduler._LRScheduler
        learning rate scheduler used to adjust the learning rate during
        training, e.g., torch.optim.lr_scheduler.CyclicLR
    
    constants_dict : dict (str : float)
        dictionary with every parameter in the ode : those that we use as
        constants and those that we want to find by using the PINN
    
    neural_net : NeuralNet
        multi-layer neural network used to learn the variables from temporal
        data
    
    residual_weights : list (float)
        list of weights to ponder every part of the residual loss associated
        to the different equations in ode system
    
    variable_fit_weights : list (float)
        list of weights to ponder every part of the variable fit loss
        associated to the different observables
    
    soft_adapt_beta : float
        hyperparameter employed when using the SoftAdapt method for balancing
        losses
    
    soft_adapt_t : int
        hyperparameter employed when using the SoftAdapt method for balancing
        losses
    
    soft_adapt_normalize : bool
        use normalization with the method soft adapt
    
    soft_adapt_by_type : bool
        if we use the method soft adapt on residual losses on one side and on
        variable losses on the other side or all together
    
    prior_losses_t : int 
        similar to soft_adapt_t but using the prior loss method

    wang_residual : bool
        if True then the balancing is applied to the residual loss,
        else it is to the variable fit loss.

    wang_t : int
        similar to soft_adapt_t but using the wang method
    
    wang_alpha : float
        same as soft_adapt_beta vut for wang method

    Methods
    -------
    net_f : (t_batch) -> (residual,neural_output)
        returns the residual given by ode system and the output of the neural
        network for a given batch. This method also returns the
        output of the neural layer. 
    
    train : (n_epochs) -> (r2_store, last_pred_unorm,losses, 
                           learned_parameters)
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
                 optimizer_type,
                 optimizer_hyperparameters,
                 scheduler_hyperparameters,
                 constants_dict,
                 multi_loss_method=None,
                 residual_weights=None,
                 variable_fit_weights=None,
                 soft_adapt_beta=0.1,
                 soft_adapt_t=1,
                 soft_adapt_normalize=True,
                 soft_adapt_by_type=True,
                 prior_losses_t=100,
                 wang_residual = True,
                 wang_t=1,
                 wang_alpha=0.9,
                 net_hidden=7,
                 optuna=False,
                 ):
        super(Pinn,self).__init__()

        # Making sure that there is no unknown constant
        for c in list(constants_dict.items()):
            key, value = c
            if (value is None) and not (key in parameter_names):
                raise ValueError("You did not provide a value for " + key + \
                                 ". Please provide its value or define it as \
                                 a parameter to be searched.")

        # Temporal data
        self.t = torch.tensor(data_t, requires_grad=True,dtype=torch.float32)
        self.t_batch = torch.reshape(self.t, (len(self.t),1))

        # Variable used to fit the neural network
        self.nb_variables = len(variables_data) + len(variables_no_data)
        self.variables_data = {k : torch.tensor(v)
                               for (k,v) in variables_data.items()}
        self.variables_max = {k : max(v)
                              for (k,v) in self.variables_data.items()}
        self.variables_min = {k : min(v)
                              for (k,v) in self.variables_data.items()}
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
        self.true_parameters=[]
        for key in parameter_names:
            self.true_parameters.append(constants_dict[key])
        if None in self.true_parameters:
            self.true_parameters = []

        # Ranges of parameters
        self.ode_parameters_ranges = ranges

        # Parameters of ODE learned by training with residual loss
        self.ode_parameters = {param: torch.nn.Parameter(torch.rand(1,
                                                                    requires_grad=True))
                               for param in parameter_names}

        # Neural network with time as input and predict variables as output
        self.neural_net = self.NeuralNet(net_hidden,self.nb_variables)
        self.neural_net.apply(init_weights_xavier)
        params = list(self.neural_net.parameters())
        params.extend(self.ode_parameters.values())
        self.optimizer=self.set_optimizer(optimizer_type,params,optimizer_hyperparameters)
        self.scheduler = self.set_scheduler(scheduler_hyperparameters)
        self.constants_dict = constants_dict
        self.multi_loss_method = multi_loss_method


        # initialize loss weights at 1 if not given
        if residual_weights is None :
            self.residual_weights = [1]*self.nb_variables
        else :
            self.residual_weights = residual_weights

        if variable_fit_weights is None :
            self.variable_fit_weights = [1]*len(self.variables_data)
        else :
            self.variable_fit_weights = variable_fit_weights


        self.nb_observables = len(variables_data)
        self.nb_res=len(self.ode_residual_dict)

        # parameter use into method for weight loss component
        self.soft_adapt_beta = soft_adapt_beta
        self.soft_adapt_t = soft_adapt_t
        self.soft_adapt_normalize=soft_adapt_normalize
        self.soft_adapt_by_type=soft_adapt_by_type
        self.prior_losses_t = prior_losses_t
        self.wang_residual = wang_residual
        self.wang_t = wang_t
        self.wang_alpha = wang_alpha

        # Is this object is used inside optuna trial
        self.optuna=optuna


    def set_optimizer(self,opimizer_type,parameters,hyperparameters):
        if opimizer_type == "Adam":
            optimizer = optim.Adam
            return optimizer(**({"params":parameters} |
                                           hyperparameters))
        else:
            print("Please enter optimizer type")
    
    def set_scheduler(self,hyperparameters):
        scheduler= optim.lr_scheduler.CyclicLR
        return scheduler(**({"optimizer":self.optimizer} |
                                       hyperparameters))


    class NeuralNet(nn.Module): # input: [[t1], [t2]...[t100]] batch of timesteps

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
        residual : list (torch.Tensor)
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

        residual_losses : list (float)
            residual losses for every epoch

        variable_fit_losses : list (float)
            variable fit losses for every epoch

        all_learned_parameters : list (list (float))
            learned parameters for every epochs

        learning_rates : list (float)
            learning rate for every epoch
        """

        # Monitor the training
        parameter_errors = []
        all_learned_parameters = []
        losses = []
        residual_losses = []
        variable_fit_losses = []
        learning_rates = []

        # Losses vectors for all epochs
        all_loss_residual_list = []
        all_loss_variable = []

        # Weights vectors for all epochs
        all_weights_residual_list = []
        all_weights_variable = []

        for epoch in tqdm(range(n_epochs), desc="Training the neural network", ncols=150):
            res, net_output = self.net_f(self.t_batch)
            self.optimizer.zero_grad()

            # Actual epoch losses for every residual equation (mse on temporal
            # data)
            loss_residual_list = [torch.mean(torch.square(r)) for r in res]
            # Actual epoch losses for every known variables (mse on temporal
            # data)
            loss_variable_fit_list = [torch.mean(torch.square(v - net_output[i]))
                                      for (i,v) in enumerate(self.variables_norm.values())]

            # Losses for all epochs
            all_loss_residual_list.append([l.item() for l in loss_residual_list])
            all_loss_variable.append([l.item() for l in loss_variable_fit_list])

            # Weights on losses component depending on the method
            if self.multi_loss_method is None:
                residual_method_weights = [1]*self.nb_res
                variable_method_weights = [1]*self.nb_observables

            elif self.multi_loss_method=="soft_adapt":
                residual_method_weights,variable_method_weights=\
                    self.soft_adapt(epoch,
                                    all_loss_residual_list,
                                    all_loss_variable,
                                    )

            elif self.multi_loss_method == "prior_losses":
                index=max(0,epoch-self.prior_losses_t)
                residual_method_weights = [1/(all_loss_residual_list[index][i])
                                           for i in range(self.nb_res)]
                variable_method_weights = [1/(all_loss_variable[index][i])
                                           for i in range(self.nb_observables)]

            elif self.multi_loss_method == "wang":
                residual_method_weights,variable_method_weights=\
                    self.wang(epoch,
                              loss_residual_list,
                              loss_variable_fit_list,
                              all_weights_residual_list,
                              all_weights_variable,
                              )

            # Weights vectors for all epochs
            all_weights_residual_list.append(residual_method_weights)
            all_weights_variable.append(variable_method_weights)

            # loss multiply by manual weights and weights from method above
            loss_residual = sum([loss_residual_list[i]*
                                 residual_method_weights[i]*
                                 self.residual_weights[i]
                                 for i in range(self.nb_res)])
            loss_variable_fit = sum([loss_variable_fit_list[i]*
                                     variable_method_weights[i]*
                                     self.variable_fit_weights[i]
                                     for i in range(self.nb_observables)])

            loss = loss_residual + loss_variable_fit

            if isnan(loss.detach().numpy()):
                if self.optuna:
                    raise optuna.exceptions.TrialPruned()
                raise ValueError("loss is not a number (nan) anymore. Consider changing the hyperparameters. This happened at epoch " + str(epoch) +".")

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            learned_parameters = [self.output_param_range(v,i).item()
                             for (i,v) in enumerate(self.ode_parameters.values())]

            losses.append(loss.item())
            residual_losses.append(loss_residual.item())
            variable_fit_losses.append(loss_variable_fit.item())
            learning_rates.append(self.scheduler.get_last_lr())
            all_learned_parameters.append(learned_parameters)
            if self.true_parameters:
                parameter_errors.append(mean_error_percentage(self.true_parameters,
                                                              learned_parameters))

        last_pred_unorm = [self.variables_min[k] + (self.variables_max[k] -
                                                    self.variables_min[k]) * 
                                                    net_output[i]
                           for (i,k) in enumerate(self.variables.keys())]

        return parameter_errors, \
               last_pred_unorm, \
               losses, \
               variable_fit_losses, \
               residual_losses, \
               all_learned_parameters, \
               learning_rates


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
        return (torch.tanh(param) + 1) / 2 * \
                (self.ode_parameters_ranges[index][1] - 
                 self.ode_parameters_ranges[index][0]) + \
                self.ode_parameters_ranges[index][0]


    def soft_adapt(self,
                   epoch,
                   all_loss_residual_list,
                   all_loss_variable_list,
                   ):
        """
        This method return the weight given by the soft adapt method. For
        every component of the loss, the method compare last loss with the
        soft_adapt_t losses before. The higher this difference is, in 
        comparison with other component, the higher will be the bigger
        weights.

        Parameters
        ----------
        epoch : int
            actual epoch

        all_loss_residual_list : list(list(float))
            list at all epoch of loss for all residual component

        all_loss_variable_list : list(list(float))
            list at all epoch of loss for all variable component

        Returns
        -------
        residual_method_weights: array(float)
            weight for residual component of the loss

        variable_method_weights: array(float)
            weight for residual component of the loss
        """

        beta=self.soft_adapt_beta

        if epoch >= self.soft_adapt_t:
            index = epoch-self.soft_adapt_t
            # last loss vector minus the loss vector self.soft_adapt_t before
            x = np.concatenate((np.array(all_loss_residual_list[-1][:])-
                                np.array(all_loss_residual_list[index][:]),
                                np.array(all_loss_variable_list[-1][:]-
                                np.array(all_loss_variable_list[index][:]))))
            # Soft adapt between residual loss on one side and variable loss
            # on the other side.
            if self.soft_adapt_by_type:
                if self.soft_adapt_normalize:
                    x[:self.nb_res]=x[:self.nb_res]/ \
                                    np.linalg.norm(x[:self.nb_res])
                    x[self.nb_res:]=x[self.nb_res:]/ \
                                    np.linalg.norm(x[self.nb_res:])

                return softmax(x[:self.nb_res]*beta),\
                    softmax(x[self.nb_res:]*beta)
            # Soft adapt on all losses.
            else:
                if self.soft_adapt_normalize:
                    x = x/np.linalg.norm(x)
                return softmax(beta * x)[:self.nb_res], \
                       softmax(beta * x)[self.nb_res:]
        else:
            return np.array([1] *self.nb_res), \
                   np.array([1] * self.nb_observables)


    def wang(self,
             epoch,
             loss_residual_list,
             loss_variable_fit_list,
             all_weights_residual_list,
             all_weights_variable,
            ):
        """
        This method return the weights given by the method described in Wang et al.

        Parameters
        ----------
        epoch : int
            actual epoch

        loss_residual_list : list(tensor)
            list of the residual loss components

        loss_variable_fit_list : list(tensor)
            list of the variable_fit loss components  

        all_weights_residual_list : list(list(float))
            list of the residual weights vectors

        all_weights_variable : list(list(float))
            list of the variable weights vectors

        Returns
        -------
        residual_method_weights: array(float)
            weight for residual component of the loss

        variable_method_weights: array(float)
            weight for residual component of the loss
        """
        if epoch >= self.wang_t :
            # Defining which part of the loss has to be balanced
            if self.wang_residual :
                sick_loss_list = loss_residual_list
                sane_loss = sum(loss_variable_fit_list)
            else :
                sick_loss_list = loss_variable_fit_list
                sane_loss = sum(loss_residual_list)

            # Computing the maximum of the absolute values of the gradient loss coordinates for
            # the term that do not need balancing
            sane_loss.backward(retain_graph=True)
            grads = torch.cat([param.grad.flatten()
                               for param in self.params if param.grad is not None])
            max_grad = torch.max(torch.abs(grads)).item()
            self.optimizer.zero_grad()

            # Computing the mean of the absolute values of the gradient loss coordinates for
            # every term that needs balancing
            means_grad = []
            for loss in sick_loss_list:
                loss.backward(retain_graph=True)
                grads = torch.cat([param.grad.flatten()
                               for param in self.params if param.grad is not None])
                means_grad.append(torch.mean(torch.abs(grads)).item())
                self.optimizer.zero_grad()

            lambda_hat = max_grad/np.array(means_grad)

            # Giving the weights in function of the loss terms we want to balance
            # The final weights are a weighted average between previous weights and lambda_hat
            if self.wang_residual :
                residual_method_weights = \
                    (1 - self.wang_alpha)*np.array(all_weights_residual_list[epoch - self.wang_t]) \
                        + self.wang_alpha*lambda_hat
                variable_method_weights = [1]*self.nb_observables
            else :
                residual_method_weights = [1]*self.nb_res
                variable_method_weights = \
                    (1 - self.wang_alpha)*np.array(all_weights_variable[epoch - self.wang_t]) \
                        + self.wang_alpha*lambda_hat
        else :
            residual_method_weights = [1]*self.nb_res
            variable_method_weights = [1]*self.nb_observables


        return residual_method_weights, variable_method_weights
