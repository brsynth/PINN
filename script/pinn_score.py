import sys
sys.path.insert(0, '..')

import os
import torch
import random
from numpy import genfromtxt
from lib.pinn import Pinn
from lib.tools import random_ranges, ssr_error
from scipy.integrate import solve_ivp
from ode_equation.deriv_equations_Millard import ODE_residual_dict_Millard, deriv_Millard
from ode_equation.Millard_dicts import ode_parameters_dict, ode_parameter_ranges_dict, variable_standard_deviations_dict


def pinn_score(training_dict, data_dict, ode_dict,seed=42):
    """
    This function create, train and score a physic-informed neural network.
    
    Parameters
    ----------
    training_dict : dictionary
    dictionary with parameters for training pinn; optimizer, scheduler, and
    choice of method to weight the different losses.

    data_dict : dictionary
    dictionary with the data used to train the pinn with instruction about
    the use of pinn; the variable considered as observables, ode parameters we
    seek to find with pinn.

    ode_dict : dictionary
    dictionary with known values for all parameters, and ranges for some of
    the ode parameters. 

    
    Returns
    -------
    
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # Load data
    ode_data = genfromtxt(data_dict["file"], delimiter=',')

    # Setting variables (known and unknown), parameters, parameters ranges, constants, weights
    observables = data_dict["observables"]
    variable_data = {v : ode_data[i+1] for i,v in enumerate(observables)}
    variable_no_data  = {}

    data_t = ode_data[0]
    residual_weights=training_dict["multiple_loss_method"]["manual_residual_weights"]
    variable_fit_weights=training_dict["multiple_loss_method"]["manual_variable_weights"]

    # Creating the ranges: using random_ranges or the ranges provided by Millard
    ranges = random_ranges([ode_dict["ode_parameter_dict"][key] 
                            for key in data_dict["parameter_names"]],scale=20)
    for i,name in enumerate(data_dict["parameter_names"]):
        if name in ode_dict["ode_parameter_dict"]:
            ranges[i]=ode_dict["ode_parameter_ranges_dict"][name]


    # training pinn
    # Create the PINN
    pinn_cell = Pinn(ode_residual_dict=ODE_residual_dict_Millard,
                     ranges=ranges,
                     data_t=data_t,
                     variables_data=variable_data,
                     variables_no_data=variable_no_data,
                     parameter_names=data_dict["parameter_names"],
                     optimizer_type=training_dict["optimizer"]["name"],
                     optimizer_hyperparameters=training_dict["optimizer"]["parameters"],
                     scheduler_hyperparameters=training_dict["scheduler"]["parameters"],
                     constants_dict=ode_dict["ode_parameter_dict"],
                     multi_loss_method=training_dict["multiple_loss_method"]["name"],
                     residual_weights=residual_weights,
                     variable_fit_weights=variable_fit_weights,
                     )


    # Training
    training_result = pinn_cell.train(training_dict["epoch"])
    _, _, _, _, _, all_learned_parameters, _ = training_result

    # Score by reconstruct data from model
    new_ode_parameters = ode_dict["ode_parameter_dict"] | dict(zip(data_dict["parameter_names"],
                                                                   all_learned_parameters[-1]))
    net_res = solve_ivp(fun=deriv_Millard,
                    t_span=(0,4.25),
                    y0=[d[0] for d in ode_data[1:]],
                    method='LSODA',
                    args=(new_ode_parameters,),
                    t_eval=data_t,
                    dense_output=True)

    variable_res = {name:net_res.y[i] for i,name in enumerate(observables)}
    error = ssr_error(standard_deviations_dict=variable_standard_deviations_dict, observables=observables, variable_data=variable_data, variable_res=variable_res)
    return error


if __name__ == "__main__":

    DATA_FILE = "../data/"
    RESULT_FILE = "../result/"


    training_dict = {"epoch":150000,
                     "optimizer" : {"name": "Adam",
                                    "parameters": {"lr":1e-4},
                                    },
                     "scheduler": {"name": "CyclicLR",
                                    "parameters":{"base_lr":1e-4, 
                                                  "max_lr":1e-4, 
                                                  "step_size_up":100,
                                                  "scale_mode":"exp_range",
                                                  "gamma":0.999,
                                                  "cycle_momentum":False,
                                                  }
                                  },
                     "multiple_loss_method": {"name": "soft_adapt",
                                              "method_parameters": {"soft_adapt_t":100,
                                                                    "soft_adapt_beta":10,
                                                                    "soft_adapt_by_type":True,
                                                                    "soft_adapt_normalize":True,
                                                                    },
                                              "manual_variable_weights":None,
                                              "manual_residual_weights":[1e-4,1e-1,1e-1,1e-14,1e-15,1e-8],
                                              },
                     }

    data_dict = {"file":os.path.join(DATA_FILE,'generated_Millard_data.csv'),
                 "observables":["GLC","ACE_env","X","ACCOA","ACP","ACE_cell"],
                 "unknown_variable":{},
                 "parameter_names": ["v_max_AckA",
                                     "v_max_Pta",
                                     "v_max_glycolysis",
                                     "Ki_ACE_glycolysis",
                                     "Km_ACCOA_TCA_cycle",
                                     "v_max_TCA_cycle",
                                     "Ki_ACE_TCA_cycle", 
                                     "Y","v_max_acetate_exchange",
                                     "Km_ACE_acetate_exchange"],
                 }

    ode_dict = {"ode_parameter_dict":ode_parameters_dict,
                "ode_parameter_ranges_dict":ode_parameter_ranges_dict,
                }

    score = pinn_score(training_dict, data_dict, ode_dict,seed=42)
    print(score)
