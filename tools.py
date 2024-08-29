import torch 
from torch import nn
import numpy as np
import random as rd

def nul_matrix_except_one_column_of_ones(shape,index_col):
    """
    Returns a pytorch matrix m of given shape with zero for every entry
    except on the columns of given index where the value of m is one.     
    Parameters
    ----------
    shape : tuple (int)
        shape of the matrix to return
    index_col : int
        index of the columns full of one    
    Returns
    -------
    m : torch.Tensor
        matrix with the given shape and the given non null column
    """
    m = torch.zeros(shape)
    m[:,index_col]=1
    return m

def normalize(x, x_min,x_max):
    """
    Normalize the float x with the given max and min.
    Parameters
    ----------
    x : torch.Tensor
        a tensor with just a float number 
    x_min : torch.Tensor
        minimum value for x
    x_max : torch.Tensor
        maximum value for x
    Returns
    -------
    x_norm : torch.Tensor
        normalization of x
    """
    return (x - x_min)/(x_max-x_min)

def denormalize(x_norm, x_min,x_max):
    """
    Denormalize the float x with the given max and min. Inverse function of
    the normalize function.

    Parameters
    ----------
    x_norm : torch.Tensor
        a tensor with just a float number 
    x_min : torch.Tensor
        minimum value for x
    x_max : torch.Tensor
        maximum value for x

    Returns
    -------
    x_denorm : torch.Tensor
        denormalization of x_norm
    """
    return x_min + (x_max - x_min)*x_norm

def param_error_percentages(true_parameters,learned_parameters):
    """
    Return the list of percentage of error of the learned parameters compared
    to the true parameters.
    Parameters
    ----------
    true_parameters : list (float)
        list of true parameters
    learned_parameters : list (float)
        list of learned parameters

    Returns
    -------
    errors : list (float)
        list of error of learned parameter compared to the true parameters
    """
    errors = []
    for i,true_parameter in enumerate(true_parameters):
        p = abs(true_parameter - learned_parameters[i])/true_parameter
        errors.append(p)
    return errors

def mean_error_percentage(true_parameters,learned_parameters):
    """
    Return the mean of percentage of error of the learned parameters compared
    to the true parameters.
    Parameters
    ----------
    true_parameters : list (float)
        list of true parameters
    learned_parameters : list (float)
        list of learned parameters

    Returns
    -------
    mean_error : float
        mean of error of learned parameter compared to the true parameters
    """
    errors = param_error_percentages(true_parameters,learned_parameters)
    return np.array(errors).mean()

def random_ranges(true_parameters, scale):
    """Returns random ranges for each of the parameters in the list"""
    return [(rd.random()*p/scale,scale*(1+rd.random())*p) for p in true_parameters]

def init_weights_xavier(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

# takes in a module and applies the specified weight initialization
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
    # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0,1.0)
        if m.bias is not None:
            m.bias.data.fill_(0)

# calculates the sum of squared residuals error as described in Millard's paper (objective function)
def ssr_error(standard_deviations_dict, observables, variable_data, variable_res):
    ssr = 0
    for key in observables:
        for i in range(len(variable_data[key])):
            ssr += ((variable_data[key][i] - variable_res[key][i])/standard_deviations_dict[key])**2
    return ssr

# calculates the loss 
def loss_calculator(epoch,
                    multi_loss_method, 
                    loss_residual_list, 
                    loss_variable_fit_list, 
                    losses_residual_list,
                    losses_variable_fit_list,
                    residual_weights,
                    variable_fit_weights,
                    nb_variables,
                    nb_observables,
                    prior_losses_t,
                    SoftAdapt_t,
                    SoftAdapt_beta,
                    ):
    loss_residual = sum(loss_residual_list)
    loss_variable_fit = sum(loss_variable_fit_list)

    # multiply each term of the loss by weights defined by user
    if multi_loss_method == "my_weights":
        loss_residual = sum([loss_residual_list[i]*residual_weights[i] 
                                for i in range(nb_variables)])
        loss_variable_fit = sum([loss_variable_fit_list[i]*variable_fit_weights[i] 
                                    for i in range(nb_observables)])
    
    # divide each term of the loss by its own initial value
    if multi_loss_method == "initial_losses":
        loss_residual = sum([loss_residual_list[i]/(losses_residual_list[0][i]) 
                                for i in range(nb_variables)])
        loss_variable_fit = sum([loss_variable_fit_list[i]/(losses_variable_fit_list[0][i]) 
                                    for i in range(nb_observables)])
    
    # divide each term of the loss by its prior value
    if multi_loss_method == "prior_losses":
        if epoch >= prior_losses_t:
            loss_residual = sum([loss_residual_list[i]/(losses_residual_list[epoch-prior_losses_t][i]) 
                                    for i in range(nb_variables)])
            loss_variable_fit = sum([loss_variable_fit_list[i]/(losses_variable_fit_list[epoch-prior_losses_t][i]) 
                                        for i in range(nb_observables)])
    
    # using the SoftAdapt method
    if multi_loss_method[0:9] == "SoftAdapt":
        loss_residual = sum([residual_weights[i]*loss_residual_list[i] for i in range(nb_variables)])
        loss_variable_fit = sum([variable_fit_weights[i]*loss_variable_fit_list[i] for i in range(nb_observables)])
        if epoch >= SoftAdapt_t:
            s_list_residual = np.array([(loss_residual_list[i] - losses_residual_list[epoch-SoftAdapt_t][i]).item() 
                                        for i in range(nb_variables)])
            s_list_variable_fit = np.array([(loss_variable_fit_list[i] - losses_variable_fit_list[epoch-SoftAdapt_t][i]).item() 
                                            for i in range(nb_observables)])
            if len(multi_loss_method) > 9 :
                if multi_loss_method[9:] == "_normalized":
                    s_list_residual *= 1/np.sqrt(sum([x**2 for x in s_list_residual])+sum([x**2 for x in s_list_variable_fit]))
                    s_list_residual *= 1/np.sqrt(sum([x**2 for x in s_list_residual])+sum([x**2 for x in s_list_variable_fit]))
            
            alpha_residual = np.array([np.exp(SoftAdapt_beta*s_list_residual[i]).item() for i in range(nb_variables)])
            alpha_residual *= 1/sum(alpha_residual)

            alpha_variable_fit = np.array([np.exp(SoftAdapt_beta*s_list_residual[i]).item() for i in range(nb_observables)])
            alpha_variable_fit *= 1/sum(alpha_variable_fit)

            loss_residual = sum([residual_weights[i]*loss_residual_list[i]*alpha_residual[i] for i in range(nb_variables)])
            loss_variable_fit = sum([variable_fit_weights[i]*loss_variable_fit_list[i]*alpha_variable_fit[i] for i in range(nb_observables)])

    return loss_variable_fit + loss_residual, loss_residual, loss_variable_fit