# PINN

This is an example of using a physically informed neural network in the context of cell modeling. The code is adapted from the tutorial of the code join to the article : 
[Data-Driven Approach for Predicting Spread of Infectious Diseases Through DINNs: Disease Informed Neural Networks](https://arxiv.org/pdf/2110.05445)

Here the [code](https://github.com/Shaier/DINN) associated.

Here we use a physically informed neural network to predict parameters of a cell ODE model. The code have been generalized and made modular to be able to work on any number of differential equation and parameters. A first example in the file `pinn_toy.ipynb` find parameters of a minimalist physiological model. The differential equations use in the toy file were proposed by Ihab Boulas. For a complete introduction of this model see the file `Equation_latex.ipynb`.


# Getting started

- **Clone** the git ([how to clone a git repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository))

- Install a distribution of **conda** if not already installed ([how to install conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation))

- Import the **environment** `environment.yaml` (stored at the root of the repository) with the following command:

`conda env create -n <local-env-name> --file environment.yml`

# Used with other ODE model
The main class of this project is Pinn. It needs several thing to make it work :
* A system of differential equation with unknown parameters.
* Data supposed to follows differential equations.
* A range for every parameter.
* A list of true parameters to compute the score. 

The purpose of this class is to be trained on the data to get an estimation of parameters, in the given ranges. The system of differential equation is given to the model through a dictionary of function which encode the partial derivation to compute residual loss. In the case of the toy example `pinn_toy.ipynb`, this dictionary is in the file `deriv_equations.py`. In the toy example, the data are generated via ODE solver odeint of [Scipy](https://scipy.org/) library, equations are also in the file `deriv_equations.py`. The list of true parameters is optional and only used to get a score on parameters fitting one simulated situations more to evaluate the model.
