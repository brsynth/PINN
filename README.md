# PINN

This is an example of using a physically informed neural network in the context of cell modeling. The code is adapted from the tutorial of the code join to the article : 
[Data-Driven Approach for Predicting Spread of Infectious Diseases Through DINNs: Disease Informed Neural Networks](https://arxiv.org/pdf/2110.05445)

Here the [code](https://github.com/Shaier/DINN) associated.

Here we use a physically informed neural network to predict parameters of a cell ODE model. The code have been generalized and made modular to be able to work on any number of differential equation and parameters. A first example in the file `pinn_toy` find parameters of a minimalist physiological model. THe model was proposed by Ihab Boulas. For a complete introduction of this model see the file `Equation_latex.ipynb`.