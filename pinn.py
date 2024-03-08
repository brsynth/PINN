import torch
from torch import nn
from sklearn.metrics import r2_score
from tools import nul_matrix_except_one_column_of_ones, normalize, denormalize

class Pinn(nn.Module):

    def __init__(self,
                 ODE_residual_dict,
                 true_parameters,
                 ranges,
                 data_t,
                 data_dict,
                 parameter_names,
                 net_hidden =7):
        super(Pinn,self).__init__()

        # Temporal data
        self.t = torch.tensor(data_t, requires_grad=True,dtype=torch.float32)
        self.t_batch = torch.reshape(self.t, (len(self.t),1)) #reshape for batch 

        # Variable used to fit the neural network
        self.variables = {k : torch.tensor(v) for (k,v) in data_dict.items()}
        self.variables_max = {k : max(v) for (k,v) in self.variables.items()}
        self.variables_min = {k : min(v) for (k,v) in self.variables.items()}
        self.variables_norm = {k : normalize(self.variables[k],self.variables_min[k],self.variables_max[k]) 
                              for k in self.variables.keys()}

        # ODE residual computation
        self.ODE_residual_dict = ODE_residual_dict

        # Original parameter used to compute score
        self.true_parameters=true_parameters

        # Ranges of parameters
        self.ode_parameters_ranges = ranges

        # Parameters of ODE learned by training with residual loss
        self.ode_parameters = {param: torch.nn.Parameter(torch.rand(1, requires_grad=True))
                           for param in parameter_names}

        # Neural network with time as input and predict variables as output
        self.neural_net = self.Neural_net(net_hidden,len(self.variables))
        self.params = list(self.neural_net.parameters())
        self.params.extend(self.ode_parameters.values())


    class Neural_net(nn.Module): # input = [[t1], [t2]...[t100]] -- that is, a batch of timesteps 
        def __init__(self,net_hidden,output_size):
            super(Pinn.Neural_net, self).__init__()

            layers = [nn.Linear(1, 20), nn.ReLU()]
            for _ in range(net_hidden):
                layers.append(nn.Linear(20, 20))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(20, output_size))
            layers.append(nn.ReLU())
            self.linear_relu_stack = nn.Sequential(*layers)

        def forward(self, t_batch):
            logits = self.linear_relu_stack(t_batch)
            return logits

    # Residual ODE from the output of neural network 
    def net_f(self, t_batch):
            # Output for all variables for all time 
            # shape is (time_batch,nb_of_variables)
            net_output = self.neural_net(t_batch)

            # Compute derivative of variable on every time of t_batch
            d_dt_var_dict = {}
            for i,k in enumerate(self.variables.keys()):
                m = nul_matrix_except_one_column_of_ones((len(self.t),len(self.variables)),i)
                net_output.backward(m, retain_graph=True)
                d_dt_var_dict[k]=self.t.grad.clone()
                self.t.grad.zero_()

            # denormalize variables output
            var_output_net_dict= {k : denormalize(net_output[:,i],self.variables_min[k],self.variables_max[k]) 
                       for (i,k) in enumerate(self.variables.keys())}
            
            # Predicted parameters
            params_dict = {k: self.output_param_range(v,i) for (i,(k,v)) in enumerate(self.ode_parameters.items())}
            
            # Get residual according to ODE
            residual = [res(var_output_net_dict, d_dt_var_dict, params_dict, self.variables_min,self.variables_max)
                   for res in self.ODE_residual_dict.values()]

            return residual, [net_output[:,i] for i in range(len(self.variables))]

    

    def train(self, n_epochs):
        print('\nstarting training...\n')
        r2_store = []
        all_learned_parameters = []
        losses = [] 

        for epoch in range(n_epochs):
            if epoch % 1000 == 0:          
                print('\nEpoch ', epoch)
                print('#################################')   

            res, net_output = self.net_f(self.t_batch)
            self.optimizer.zero_grad()

            loss_residual = sum([torch.mean(torch.square(r)) for r in res])            
            loss_variable_fit = sum([torch.mean(torch.square(v - net_output[i]))
                                     for (i,v) in enumerate(self.variables_norm.values())])
            loss = loss_residual + loss_variable_fit
            loss.backward()
            self.optimizer.step()
            self.scheduler.step() 

            learned_parameters = [self.output_param_range(v,i).item() 
                             for (i,v) in enumerate(self.ode_parameters.values())]
            
            losses.append(loss.item())
            all_learned_parameters.append(learned_parameters)        
            r2_store.append(r2_score(self.true_parameters, learned_parameters))

        last_pred_unorm = [self.variables_min[k] + (self.variables_max[k] - self.variables_min[k]) * net_output[i]
              for (i,k) in enumerate(self.variables.keys())]
 
        return r2_store, last_pred_unorm, losses, all_learned_parameters
    

    def output_param_range(self, param, index):
        return (torch.tanh(param) + 1) / 2 * (self.ode_parameters_ranges[index][1] - self.ode_parameters_ranges[index][0]) + self.ode_parameters_ranges[index][0]
    
